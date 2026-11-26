from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests


EIA_PETROLEUM_SPOT_ENDPOINT = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
DEFAULT_PETROLEUM_SERIES = {
    "wti_cushing_spot": "RWTC",
    "brent_europe_spot": "RBRTE",
}
MAX_EIA_JSON_PAGE_SIZE = 5000


def _iso_date(value: str | date | None, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    else:
        try:
            parsed = date.fromisoformat(str(value))
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)") from exc
    return parsed.isoformat()


def _api_key(value: str | None) -> str:
    key = value or os.getenv("EIA_API_KEY")
    if not key or not key.strip():
        raise RuntimeError(
            "EIA API key not found. Set the EIA_API_KEY environment variable; "
            "do not pass credentials on the command line or commit them."
        )
    return key.strip()


def _parse_page(payload: Mapping[str, Any], series_id: str) -> tuple[list[dict[str, Any]], int]:
    if payload.get("error"):
        raise RuntimeError(f"EIA API error for {series_id}: {payload.get('error')}")
    response = payload.get("response")
    if not isinstance(response, Mapping):
        raise ValueError(f"EIA response for {series_id} has no response object")
    if response.get("error"):
        raise RuntimeError(f"EIA API error for {series_id}: {response.get('error')}")
    rows = response.get("data")
    if not isinstance(rows, list):
        raise ValueError(f"EIA response for {series_id} has no data array")
    try:
        total = int(response["total"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"EIA response for {series_id} has no valid total") from exc
    if total < 0:
        raise ValueError(f"EIA response for {series_id} has a negative total")
    return rows, total


def download_eia_petroleum_spot_prices(
    start: str | date | None = None,
    end: str | date | None = None,
    series: Mapping[str, str] | None = None,
    api_key: str | None = None,
    session: requests.Session | None = None,
    timeout: int = 30,
    page_size: int = MAX_EIA_JSON_PAGE_SIZE,
    raw_dir: str | Path = "data/raw/eia/petroleum_spot",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download daily EIA WTI/Brent spot prices with raw-page provenance.

    The API key is supplied only to the HTTP request and is never written to
    the returned frame, raw response files, manifest, or logs. Each series is
    paginated independently and missing observations are not forward-filled.
    """
    key = _api_key(api_key)
    start_date, end_date = _iso_date(start, "start"), _iso_date(end, "end")
    if start_date and end_date and start_date > end_date:
        raise ValueError("start must be on or before end")
    if not 1 <= page_size <= MAX_EIA_JSON_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_EIA_JSON_PAGE_SIZE}")
    selected = dict(series or DEFAULT_PETROLEUM_SERIES)
    if not selected or any(not name.strip() or not sid.strip() for name, sid in selected.items()):
        raise ValueError("series must contain non-empty field names and EIA series IDs")
    if len(set(selected.values())) != len(selected):
        raise ValueError("duplicate EIA series IDs are not allowed")

    http = session or requests.Session()
    retrieved_utc = datetime.now(timezone.utc).isoformat()
    snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_root = Path(raw_dir)
    snapshot_raw_root = raw_root / snapshot_id
    snapshot_raw_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    series_manifest: list[dict[str, Any]] = []

    for field, series_id in selected.items():
        offset = 0
        total: int | None = None
        page_manifest: list[dict[str, Any]] = []
        series_records: list[dict[str, Any]] = []

        while total is None or offset < total:
            params: list[tuple[str, str]] = [
                ("api_key", key),
                ("frequency", "daily"),
                ("data[]", "value"),
                ("facets[series][]", series_id),
                ("sort[0][column]", "period"),
                ("sort[0][direction]", "asc"),
                ("length", str(page_size)),
                ("offset", str(offset)),
            ]
            if start_date:
                params.append(("start", start_date))
            if end_date:
                params.append(("end", end_date))

            try:
                response = http.get(EIA_PETROLEUM_SPOT_ENDPOINT, params=params, timeout=timeout)
            except requests.RequestException:
                raise RuntimeError(
                    f"EIA request failed for {series_id}; request details omitted to protect credentials"
                ) from None
            try:
                response.raise_for_status()
            except requests.HTTPError:
                status = getattr(response, "status_code", "unknown")
                raise RuntimeError(
                    f"EIA returned HTTP {status} for {series_id}; response details omitted"
                ) from None

            raw_bytes = response.content
            page_path = snapshot_raw_root / f"{series_id}_offset_{offset:08d}.json"
            page_path.write_bytes(raw_bytes)
            try:
                payload = response.json()
            except ValueError as exc:
                raise ValueError(f"EIA returned invalid JSON for {series_id}, offset {offset}") from exc

            page_rows, page_total = _parse_page(payload, series_id)
            if total is None:
                total = page_total
                if total == 0:
                    raise ValueError(f"EIA returned no observations for requested series {series_id}")
            elif page_total != total:
                raise ValueError(
                    f"EIA total changed during pagination for {series_id}: {total} -> {page_total}"
                )
            if not page_rows and offset < total:
                raise ValueError(
                    f"EIA returned an empty page before total was reached for {series_id} "
                    f"at offset {offset}"
                )

            page_manifest.append({
                "offset": offset,
                "rows": len(page_rows),
                "raw_path": str(page_path),
                "raw_sha256": sha256(raw_bytes).hexdigest(),
            })

            for item in page_rows:
                period = item.get("period")
                value_raw = item.get("value")
                try:
                    observation_date = date.fromisoformat(str(period))
                    value = float(value_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid EIA period/value for {series_id}: {period!r}/{value_raw!r}"
                    ) from exc
                if start_date and observation_date.isoformat() < start_date:
                    raise ValueError(f"EIA returned {series_id} observation before requested start date")
                if end_date and observation_date.isoformat() > end_date:
                    raise ValueError(f"EIA returned {series_id} observation after requested end date")
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Non-positive or non-finite EIA value for {series_id}: {value}")
                returned_series = item.get("series")
                if returned_series is not None and returned_series != series_id:
                    raise ValueError(
                        f"Unexpected EIA series in {series_id} response: {returned_series!r}"
                    )
                units = (
                    item.get("value-units")
                    or item.get("value_units")
                    or item.get("units")
                    or item.get("unit")
                )
                description = item.get("seriesDescription") or item.get("series-description")
                series_records.append({
                    "date": observation_date.isoformat(),
                    "series": field,
                    "series_id": series_id,
                    "value": value,
                    "units": units,
                    "series_description": description,
                })

            offset += len(page_rows)
            if offset > total:
                raise ValueError(f"EIA returned more rows than its reported total for {series_id}")

        frame_series = pd.DataFrame(series_records)
        if len(frame_series) != total:
            raise ValueError(
                f"EIA row count mismatch for {series_id}: received {len(frame_series)}, total {total}"
            )
        if not frame_series.empty:
            parsed_dates = pd.to_datetime(frame_series["date"], errors="raise")
            if parsed_dates.duplicated().any():
                raise ValueError(f"Duplicate daily observations returned for EIA series {series_id}")
            if not parsed_dates.is_monotonic_increasing:
                raise ValueError(f"EIA series {series_id} is not ordered by ascending date")
        records.extend(series_records)
        series_manifest.append({
            "field": field,
            "series_id": series_id,
            "frequency": "daily",
            "requested_start": start_date,
            "requested_end": end_date,
            "rows": len(series_records),
            "first_date": series_records[0]["date"] if series_records else None,
            "last_date": series_records[-1]["date"] if series_records else None,
            "units": next((r["units"] for r in series_records if r["units"]), None),
            "pages": page_manifest,
        })

    columns = ["date", "series", "series_id", "value", "units", "series_description"]
    frame = pd.DataFrame(records, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["series_id", "date"], kind="stable").reset_index(drop=True)

    manifest = {
        "source": "U.S. Energy Information Administration Open Data API v2",
        "endpoint": EIA_PETROLEUM_SPOT_ENDPOINT,
        "dataset": "petroleum/pri/spt",
        "retrieved_utc": retrieved_utc,
        "snapshot_id": snapshot_id,
        "raw_snapshot_directory": str(snapshot_raw_root),
        "request": {
            "frequency": "daily",
            "data": ["value"],
            "sort": [{"column": "period", "direction": "asc"}],
            "page_size": page_size,
            "start": start_date,
            "end": end_date,
        },
        "series": series_manifest,
        "row_count": int(len(frame)),
        "normalized_sha256": sha256(
            frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        ).hexdigest(),
        "credential_policy": "API key supplied through EIA_API_KEY; credential omitted from manifest",
        "missing_observations": "No forward fill or synthetic interpolation is applied",
    }
    return frame, manifest


def save_eia_petroleum_spot_prices(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    csv_path: str | Path = "data/processed/eia_petroleum_spot.csv",
    manifest_path: str | Path = "data/processed/eia_petroleum_spot_manifest.json",
) -> None:
    """Write normalized EIA observations and the sanitized provenance manifest."""
    required = {"date", "series", "series_id", "value", "units", "series_description"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"EIA frame is missing columns: {sorted(missing)}")
    if not frame.empty:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        values = pd.to_numeric(frame["value"], errors="coerce")
        if dates.isna().any() or values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError("EIA frame contains invalid dates or values")
        if (values <= 0).any():
            raise ValueError("EIA spot prices must be positive")
    csv_target, manifest_target = Path(csv_path), Path(manifest_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_target, index=False, lineterminator="\n")
    manifest_target.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
