from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pandas as pd
import requests

FRED_SERIES = {
    "sp500": "SP500", "vix": "VIXCLS", "eurusd": "DEXUSEU",
    "usdjpy": "DEXJPUS", "ust2y": "DGS2", "ust10y": "DGS10",
}
REQUIRED_COLUMNS = ("date", *FRED_SERIES.keys(), "wti", "brent")
STRICTLY_POSITIVE_COLUMNS = ("sp500", "eurusd", "usdjpy", "brent")


def validate_market_data(frame: pd.DataFrame, *, require_oil: bool = True) -> None:
    """Validate the full panel, or the FRED-only intermediate before EIA oil is merged."""
    fred_intermediate = tuple(FRED_SERIES)
    required = REQUIRED_COLUMNS if require_oil else ("date", *fred_intermediate)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"missing required market columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("market data is empty")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("date column contains invalid dates")
    if dates.duplicated().any():
        raise ValueError("duplicate market dates are not allowed")
    if not dates.is_monotonic_increasing:
        raise ValueError("market dates must be strictly increasing")
    numeric_columns = [name for name in required if name != "date"]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("market data contains missing or non-numeric values")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("market data contains non-finite values")
    strictly_positive = [name for name in STRICTLY_POSITIVE_COLUMNS if name in numeric.columns]
    if (numeric[strictly_positive] <= 0).any().any():
        raise ValueError(
            "equity index levels and FX rates must be positive; Brent spot must be positive; WTI spot may be zero or negative"
        )
    if (numeric["vix"] <= 0).any():
        raise ValueError("VIX observations must be positive")
    if ((numeric["ust2y"] < -5) | (numeric["ust2y"] > 30)).any() or (
        (numeric["ust10y"] < -5) | (numeric["ust10y"] > 30)
    ).any():
        raise ValueError("Treasury yields outside sanity bounds")


def load_market_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        for candidate in ("DATE", "observation_date"):
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: "date"})
                break
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    validate_market_data(frame)
    return frame.sort_values("date").reset_index(drop=True)


def _fred_url(series_id: str, start: str, end: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}&coed={end}"


def download_fred_series(start: str = "2007-01-01", end: str | None = None,
                         series_ids: dict[str, str] | None = None,
                         timeout: int = 30, output_dir: str | Path = "data/raw"):
    """Download public FRED CSV series and return aligned frame plus provenance manifest.

    FRED public graph CSV is used without an API key. Every raw response is hashed.
    Short holiday gaps are forward-filled on business dates; long gaps are not.
    """
    end = end or datetime.now(timezone.utc).date().isoformat()
    ids = series_ids or FRED_SERIES
    session = requests.Session()
    raw_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    series_frames, manifest = [], []
    for name, series_id in ids.items():
        url = _fred_url(series_id, start, end)
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        digest = sha256(response.content).hexdigest()
        path = raw_dir / f"{series_id}_{start}_{end}.csv"
        path.write_bytes(response.content)
        from io import BytesIO
        frame = pd.read_csv(BytesIO(response.content))
        date_col = next((c for c in frame.columns if c.lower() in ("date", "observation_date")), None)
        if date_col is None or series_id not in frame.columns:
            raise ValueError(f"unexpected FRED response schema for {series_id}: {list(frame.columns)}")
        frame = frame[[date_col, series_id]].rename(columns={date_col: "date", series_id: name})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame[name] = pd.to_numeric(frame[name].replace(".", np.nan), errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date")
        series_frames.append(frame)
        manifest.append({
            "field": name, "series_id": series_id, "url": url,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "raw_sha256": digest, "raw_path": str(path),
            "rows": int(len(frame)), "missing_values": int(frame[name].isna().sum()),
            "first_date": frame["date"].min().date().isoformat(),
            "last_date": frame["date"].max().date().isoformat(),
        })
    merged = series_frames[0]
    for item in series_frames[1:]:
        merged = merged.merge(item, on="date", how="outer", validate="one_to_one")
    merged = merged.sort_values("date").set_index("date")
    business = pd.bdate_range(merged.index.min(), merged.index.max())
    merged = merged.reindex(business).ffill(limit=4).dropna().reset_index(names="date")
    validate_market_data(merged, require_oil=False)
    provenance = {
        "source": "Federal Reserve Bank of St. Louis FRED public graph CSV",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "start_requested": start, "end_requested": end,
        "alignment": "business-day calendar; forward-fill maximum four business days",
        "series": manifest,
        "aligned_rows": int(len(merged)),
        "aligned_start": merged.date.min().date().isoformat(),
        "aligned_end": merged.date.max().date().isoformat(),
        "aligned_sha256": sha256(merged.to_csv(index=False).encode()).hexdigest(),
    }
    return merged, provenance



def merge_eia_petroleum_into_market_data(
    fred_frame: pd.DataFrame,
    eia_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final risk panel using EIA WTI/Brent spot observations.

    FRED contributes equity, FX and rates. EIA bulk is the canonical source
    for both oil spot benchmarks, avoiding a mixed WTI/Brent source definition.
    Short business-day publication gaps are forward-filled by at most four days.
    """
    required_eia = {"date", "series", "series_id", "value"}
    missing = required_eia - set(eia_frame.columns)
    if missing:
        raise ValueError(f"EIA frame is missing columns: {sorted(missing)}")
    expected = {
        "wti_cushing_spot": "PET.RWTC.D",
        "brent_europe_spot": "PET.RBRTE.D",
    }
    observed = eia_frame[["series", "series_id"]].drop_duplicates()
    for field, series_id in expected.items():
        matches = observed.loc[observed["series"] == field, "series_id"].tolist()
        if matches != [series_id]:
            raise ValueError(
                f"EIA frame must contain exactly {field} mapped to {series_id}; got {matches}"
            )

    oil = eia_frame.loc[eia_frame["series"].isin(expected)].copy()
    oil["date"] = pd.to_datetime(oil["date"], errors="coerce")
    oil["value"] = pd.to_numeric(oil["value"], errors="coerce")
    if oil["date"].isna().any() or oil["value"].isna().any():
        raise ValueError("EIA oil frame contains invalid dates or values")
    if not np.isfinite(oil["value"].to_numpy(dtype=float)).all():
        raise ValueError("EIA oil frame contains non-finite values")
    if oil.duplicated(["date", "series"]).any():
        raise ValueError("duplicate EIA oil observations are not allowed")

    pivot = oil.pivot(index="date", columns="series", values="value").rename(
        columns={
            "wti_cushing_spot": "wti",
            "brent_europe_spot": "brent",
        }
    )
    if set(pivot.columns) != {"wti", "brent"}:
        raise ValueError("EIA oil frame must contain both WTI and Brent")

    fred = fred_frame.copy()
    fred["date"] = pd.to_datetime(fred["date"], errors="coerce")
    if fred["date"].isna().any() or fred["date"].duplicated().any():
        raise ValueError("FRED intermediate frame has invalid or duplicate dates")
    # EIA is the canonical source for both oil factors. Do not silently retain a
    # second WTI column from a caller-supplied FRED frame.
    fred = fred.drop(columns=["wti", "brent"], errors="ignore").set_index("date")
    merged = fred.join(pivot, how="outer", validate="one_to_one").sort_index()
    business = pd.bdate_range(merged.index.min(), merged.index.max())
    merged = merged.reindex(business).ffill(limit=4).dropna().reset_index(names="date")
    validate_market_data(merged)
    return merged


def save_market_data(frame: pd.DataFrame, provenance: dict, csv_path: str | Path,
                     manifest_path: str | Path) -> None:
    validate_market_data(frame)
    csv_path, manifest_path = Path(csv_path), Path(manifest_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    manifest_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
