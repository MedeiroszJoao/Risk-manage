from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .risk import historical_var_es, select_stress_window


EXPECTED_OIL_SERIES = {
    "wti_cushing_spot": "PET.RWTC.D",
    "brent_europe_spot": "PET.RBRTE.D",
}
DEFAULT_WTI_QUANTITY = 150_000.0
DEFAULT_BRENT_QUANTITY = -100_000.0
TRADING_HORIZON_10D = 10
STRESS_WINDOW_DAYS = 250


def build_commodity_spot_pnl(
    eia_frame: pd.DataFrame,
    wti_quantity: float = DEFAULT_WTI_QUANTITY,
    brent_quantity: float = DEFAULT_BRENT_QUANTITY,
    confidence_levels: tuple[float, ...] = (0.975, 0.99),
) -> tuple[pd.DataFrame, dict]:
    """Build a standalone hypothetical WTI/Brent spot-proxy P&L history.

    Uses only dates on which both EIA spot series have an actual observation.
    It performs no forward fill, interpolation, futures-basis conversion or roll.
    HPL and RTPL are identical because both positions are linear spot exposures.
    """
    required = {"date", "series", "series_id", "value", "units"}
    missing = required - set(eia_frame.columns)
    if missing:
        raise ValueError(f"EIA frame is missing columns: {sorted(missing)}")
    quantities = (float(wti_quantity), float(brent_quantity))
    if not all(math.isfinite(value) for value in quantities):
        raise ValueError("commodity quantities must be finite")
    if not confidence_levels or any(not 0.5 < level < 1 for level in confidence_levels):
        raise ValueError("confidence levels must be between 0.5 and 1")

    selected = eia_frame.loc[
        eia_frame["series"].isin(EXPECTED_OIL_SERIES)
    ].copy()
    if selected.empty:
        raise ValueError("EIA frame contains no WTI/Brent spot observations")

    selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    if selected["date"].isna().any() or selected["value"].isna().any():
        raise ValueError("EIA oil observations contain invalid dates or values")
    if not np.isfinite(selected["value"].to_numpy(dtype=float)).all():
        raise ValueError("EIA oil observations contain non-finite values")
    if selected.duplicated(["date", "series"]).any():
        raise ValueError("duplicate EIA oil observations are not allowed")

    observed = selected[["series", "series_id"]].drop_duplicates()
    for field, series_id in EXPECTED_OIL_SERIES.items():
        ids = observed.loc[observed["series"] == field, "series_id"].tolist()
        if ids != [series_id]:
            raise ValueError(f"{field} must map to {series_id}; got {ids}")

    units_by_series = {
        series: sorted({str(value) for value in group["units"].dropna().unique()})
        for series, group in selected.groupby("series")
    }
    if any(len(values) != 1 for values in units_by_series.values()):
        raise ValueError(f"EIA oil series have inconsistent units: {units_by_series}")
    wti_units = units_by_series.get("wti_cushing_spot", [])
    brent_units = units_by_series.get("brent_europe_spot", [])
    if not wti_units or not brent_units or wti_units[0] != brent_units[0]:
        raise ValueError(
            f"WTI and Brent units must match before aggregation: {units_by_series}"
        )

    selected["series"] = selected["series"].map(
        {"wti_cushing_spot": "wti", "brent_europe_spot": "brent"}
    )
    wide = selected.pivot(index="date", columns="series", values="value").sort_index()
    if not {"wti", "brent"}.issubset(wide.columns):
        raise ValueError("both WTI and Brent spot series are required")

    available_wti = int(wide["wti"].notna().sum())
    available_brent = int(wide["brent"].notna().sum())
    aligned = wide.dropna(subset=["wti", "brent"]).copy()
    if len(aligned) < 2:
        raise ValueError("at least two common WTI/Brent observation dates are required")

    changes = aligned[["wti", "brent"]].diff().iloc[1:]
    dates = changes.index
    wti_pnl = changes["wti"].to_numpy(dtype=float) * quantities[0]
    brent_pnl = changes["brent"].to_numpy(dtype=float) * quantities[1]
    total_hpl = wti_pnl + brent_pnl

    pnl = pd.DataFrame({
        "date": dates,
        "wti_previous": aligned["wti"].iloc[:-1].to_numpy(dtype=float),
        "wti_current": aligned["wti"].iloc[1:].to_numpy(dtype=float),
        "brent_previous": aligned["brent"].iloc[:-1].to_numpy(dtype=float),
        "brent_current": aligned["brent"].iloc[1:].to_numpy(dtype=float),
        "wti_pnl": wti_pnl,
        "brent_pnl": brent_pnl,
        "hpl": total_hpl,
        "rtpl": total_hpl.copy(),
        "residual": np.zeros_like(total_hpl),
        "loss": -total_hpl,
    })
    pnl["date"] = pd.to_datetime(pnl["date"])

    loss_factors = {
        "wti": -wti_pnl,
        "brent": -brent_pnl,
    }
    risk = {}
    ten_day_loss = np.convolve(
        pnl["loss"].to_numpy(dtype=float),
        np.ones(TRADING_HORIZON_10D),
        mode="valid",
    ) if len(pnl) >= TRADING_HORIZON_10D else None

    for confidence in confidence_levels:
        key = f"{confidence:.1%}"
        daily = historical_var_es(pnl["loss"].to_numpy(dtype=float), confidence)
        daily_factor = {
            factor: historical_var_es(values, confidence)
            for factor, values in loss_factors.items()
            if len(values) >= 2
        }
        risk[key] = {
            "one_observation_horizon": daily,
            "ten_observation_horizon": (
                historical_var_es(ten_day_loss, confidence)
                if ten_day_loss is not None and len(ten_day_loss) >= 2
                else None
            ),
            "factor_daily_risk": daily_factor,
        }

    stress = None
    if len(pnl) >= STRESS_WINDOW_DAYS:
        stress = select_stress_window(
            loss_factors,
            window_days=STRESS_WINDOW_DAYS,
            confidence=0.975,
        )
        stress["start_date"] = pnl.iloc[stress["start_index"]]["date"].date().isoformat()
        stress["end_date"] = pnl.iloc[stress["end_index"]]["date"].date().isoformat()

    summary = {
        "portfolio": {
            "classification": "hypothetical_linear_spot_proxy_book",
            "wti_quantity_barrels": quantities[0],
            "brent_quantity_barrels": quantities[1],
            "wti_units": wti_units[0],
            "brent_units": brent_units[0],
        },
        "source": {
            "provider": "U.S. Energy Information Administration",
            "series": EXPECTED_OIL_SERIES,
            "raw_alignment": "inner join on actual observation dates; no forward fill or interpolation",
            "wti_observations_available": available_wti,
            "brent_observations_available": available_brent,
            "common_observation_dates": int(len(aligned)),
            "wti_observations_without_brent": int(available_wti - len(aligned)),
            "brent_observations_without_wti": int(available_brent - len(aligned)),
            "unmatched_source_observations": int(
                available_wti + available_brent - 2 * len(aligned)
            ),
            "first_common_date": aligned.index.min().date().isoformat(),
            "last_common_date": aligned.index.max().date().isoformat(),
        },
        "pnl": {
            "observations": int(len(pnl)),
            "first_pnl_date": pnl["date"].min().date().isoformat(),
            "last_pnl_date": pnl["date"].max().date().isoformat(),
            "cumulative_hpl_usd": float(pnl["hpl"].sum()),
            "mean_daily_hpl_usd": float(pnl["hpl"].mean()),
            "daily_hpl_volatility_usd": float(pnl["hpl"].std(ddof=1)),
            "worst_daily_hpl_usd": float(pnl["hpl"].min()),
            "best_daily_hpl_usd": float(pnl["hpl"].max()),
            "max_absolute_hpl_rtpl_residual": float(pnl["residual"].abs().max()),
            "negative_wti_observations": int((aligned["wti"] < 0).sum()),
            "zero_or_negative_wti_observations": int((aligned["wti"] <= 0).sum()),
            "minimum_wti_spot": float(aligned["wti"].min()),
            "minimum_wti_date": aligned["wti"].idxmin().date().isoformat(),
            "spot_price_change_correlation": float(changes["wti"].corr(changes["brent"])),
        },
        "historical_risk": risk,
        "stress_window": stress,
        "methodology_limits": [
            "The positions are hypothetical linear spot-linked exposures, not actual trades.",
            "This is a standalone oil-spot desk diagnostic, not an aggregated four-desk portfolio.",
            "WTI and Brent spot prices are not listed futures settlements or executable bid/ask quotes.",
            "No futures basis, forward curve, contract month, expiry, roll, transaction costs or financing is modeled.",
            "Ten-observation risk aggregates consecutive common EIA observation days, not calendar days.",
            "Historical VaR/ES are descriptive empirical estimates and are not supervisory-approved FRTB capital.",
            "EIA data can be revised; retrieved archive hash does not recreate historical publication vintages.",
        ],
        "normalized_pnl_sha256": sha256(
            pnl.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")
        ).hexdigest(),
    }
    return pnl, summary


def save_commodity_spot_case(
    pnl: pd.DataFrame,
    summary: dict,
    pnl_path: str | Path = "reports/generated/commodity_spot_pnl.csv",
    summary_path: str | Path = "reports/generated/commodity_spot_risk.json",
) -> None:
    pnl_target, summary_target = Path(pnl_path), Path(summary_path)
    pnl_target.parent.mkdir(parents=True, exist_ok=True)
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    pnl.to_csv(pnl_target, index=False, date_format="%Y-%m-%d", lineterminator="\n")
    summary_target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
