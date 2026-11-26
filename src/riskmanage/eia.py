from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import zipfile
from typing import Any, Mapping

import pandas as pd
import requests


EIA_PETROLEUM_SPOT_BULK_URL = "https://www.eia.gov/opendata/bulk/PET.zip"
DEFAULT_PETROLEUM_SERIES = {
    "wti_cushing_spot": "PET.RWTC.D",
    "brent_europe_spot": "PET.RBRTE.D",
}
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


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


def _bulk_period(value: Any, series_id: str) -> date:
    """Parse the compact YYYYMMDD or ISO YYYY-MM-DD period used in PET.txt."""
    raw = str(value).strip()
    try:
        if len(raw) == 8 and raw.isdigit():
            return datetime.strptime(raw, "%Y%m%d").date()
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid EIA period for {series_id}: {value!r}") from exc


def _download_petroleum_bulk_file(
    session: requests.Session,
    destination: Path,
    timeout: int | tuple[int, int],
) -> dict[str, Any]:
    """Stream the public EIA PET.zip to disk and return file/HTTP provenance."""
    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {
        "Accept": "application/zip, application/octet-stream;q=0.9, */*;q=0.8",
        "User-Agent": "risk-manage-research/1.0 (public EIA bulk-data ingestion)",
    }
    response = None
    digest = sha256()
    bytes_written = 0
    try:
        try:
            response = session.get(
                EIA_PETROLEUM_SPOT_BULK_URL,
                headers=headers,
                stream=True,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"EIA PET bulk download failed ({type(exc).__name__}); check connectivity and retry"
            ) from None

        try:
            response.raise_for_status()
        except requests.HTTPError:
            status = getattr(response, "status_code", "unknown")
            raise RuntimeError(f"EIA PET bulk download returned HTTP {status}") from None

        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                bytes_written += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())

        if bytes_written == 0:
            raise ValueError("EIA PET bulk download returned an empty file")

        response_headers = getattr(response, "headers", {}) or {}
        content_length = response_headers.get("Content-Length")
        if content_length is not None:
            try:
                expected_bytes = int(content_length)
            except (TypeError, ValueError):
                expected_bytes = None
            if expected_bytes is not None and bytes_written != expected_bytes:
                raise ValueError(
                    "EIA PET bulk download size mismatch: "
                    f"received {bytes_written} bytes, expected {expected_bytes}"
                )

        os.replace(temporary, destination)
        return {
            "url": EIA_PETROLEUM_SPOT_BULK_URL,
            "final_url": getattr(response, "url", None) or EIA_PETROLEUM_SPOT_BULK_URL,
            "http_status": int(getattr(response, "status_code", 200)),
            "content_type": response_headers.get("Content-Type"),
            "content_length_header": content_length,
            "last_modified": response_headers.get("Last-Modified"),
            "etag": response_headers.get("ETag"),
            "bytes": bytes_written,
            "sha256": digest.hexdigest(),
            "archive_path": str(destination),
        }
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if response is not None and hasattr(response, "close"):
            response.close()


def _read_petroleum_series_from_zip(
    archive_path: Path,
    selected: Mapping[str, str],
    start_date: str | None,
    end_date: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Read EIA's line-delimited PET.txt without extracting the 300+ MB text file."""
    by_id = {series_id: field for field, series_id in selected.items()}
    found: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    line_count = 0
    stats = {"source_observations": 0, "selected_observations": 0, "null_observations": 0}

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = [
                info for info in archive.infolist()
                if PurePosixPath(info.filename).name.casefold() == "pet.txt"
            ]
            if len(members) != 1:
                raise ValueError(
                    "EIA PET.zip must contain exactly one PET.txt member; "
                    f"found {len(members)}"
                )
            member = members[0]
            if member.file_size <= 0:
                raise ValueError("EIA PET.txt member is empty")

            with archive.open(member, "r") as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    if not raw_line.strip():
                        continue
                    line_count += 1
                    try:
                        item = json.loads(raw_line.decode("utf-8-sig"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"Invalid JSON in EIA PET.txt at line {line_number}"
                        ) from exc
                    if not isinstance(item, dict):
                        raise ValueError(f"Unexpected JSON object in PET.txt at line {line_number}")

                    series_id = item.get("series_id")
                    if series_id not in by_id:
                        continue
                    if series_id in found:
                        raise ValueError(f"Duplicate EIA series metadata for {series_id}")

                    data = item.get("data")
                    if not isinstance(data, list):
                        raise ValueError(f"EIA series {series_id} has no data array")

                    field = by_id[series_id]
                    metadata = {
                        "field": field,
                        "series_id": series_id,
                        "series_name": item.get("name"),
                        "series_description": item.get("description"),
                        "units": item.get("units"),
                        "units_short": item.get("unitsshort"),
                        "frequency": item.get("f"),
                        "source_start": item.get("start"),
                        "source_end": item.get("end"),
                        "source_last_updated": item.get("last_updated"),
                        "source_line_number": line_number,
                        "source_observations": len(data),
                        "null_observations_in_range": 0,
                        "filtered_out_of_range": 0,
                    }
                    found[series_id] = metadata
                    stats["source_observations"] += len(data)

                    for observation in data:
                        if not isinstance(observation, (list, tuple)) or len(observation) < 2:
                            raise ValueError(
                                f"Malformed observation in EIA series {series_id}"
                            )
                        observation_date = _bulk_period(observation[0], series_id)
                        iso_period = observation_date.isoformat()
                        if start_date and iso_period < start_date:
                            metadata["filtered_out_of_range"] += 1
                            continue
                        if end_date and iso_period > end_date:
                            metadata["filtered_out_of_range"] += 1
                            continue

                        raw_value = observation[1]
                        if raw_value is None or raw_value == "":
                            metadata["null_observations_in_range"] += 1
                            stats["null_observations"] += 1
                            continue
                        try:
                            value = float(raw_value)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"Invalid EIA value for {series_id} on {iso_period}: {raw_value!r}"
                            ) from exc
                        # Negative prices are economically possible for crude oil; only non-finite
                        # values are rejected. In particular, preserve the WTI April 2020 event.
                        if not math.isfinite(value):
                            raise ValueError(
                                f"Non-finite EIA value for {series_id} on {iso_period}"
                            )
                        records.append({
                            "date": iso_period,
                            "series": field,
                            "series_id": series_id,
                            "value": value,
                            "units": metadata["units"],
                            "series_description": metadata["series_description"],
                        })
                        stats["selected_observations"] += 1
    except zipfile.BadZipFile as exc:
        raise ValueError("Downloaded EIA PET file is not a valid ZIP archive") from exc

    missing = sorted(set(selected.values()) - set(found))
    if missing:
        raise ValueError(f"EIA PET.zip is missing requested daily series: {', '.join(missing)}")

    for field, series_id in selected.items():
        series_rows = [row for row in records if row["series_id"] == series_id]
        if not series_rows:
            raise ValueError(
                f"EIA series {series_id} has no observations in the requested date range"
            )
        if found[series_id]["frequency"] not in (None, "D"):
            raise ValueError(
                f"EIA series {series_id} is not daily: {found[series_id]['frequency']!r}"
            )
        dates = pd.to_datetime([row["date"] for row in series_rows], errors="raise")
        if dates.duplicated().any():
            raise ValueError(f"Duplicate daily observations returned for EIA series {series_id}")
        series_rows.sort(key=lambda row: row["date"])
        if start_date and series_rows[0]["date"] < start_date:
            raise ValueError(f"EIA returned {series_id} observation before requested start date")
        if end_date and series_rows[-1]["date"] > end_date:
            raise ValueError(f"EIA returned {series_id} observation after requested end date")

    records.sort(key=lambda row: (row["series_id"], row["date"]))
    series_manifest = []
    for field, series_id in selected.items():
        metadata = found[series_id]
        rows = [row for row in records if row["series_id"] == series_id]
        series_manifest.append({
            **metadata,
            "rows": len(rows),
            "first_date": rows[0]["date"],
            "last_date": rows[-1]["date"],
        })

    stats["json_lines"] = line_count
    stats["matched_series_lines"] = len(found)
    stats["pet_txt_uncompressed_bytes"] = member.file_size
    stats["pet_txt_member"] = member.filename
    return records, series_manifest, stats


def download_eia_petroleum_spot_prices(
    start: str | date | None = None,
    end: str | date | None = None,
    series: Mapping[str, str] | None = None,
    session: requests.Session | None = None,
    timeout: int | tuple[int, int] = (20, 180),
    raw_dir: str | Path = "data/raw/eia/petroleum_spot",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download and normalize EIA daily WTI/Brent spot prices from public PET.zip.

    EIA's bulk ZIP does not require an API key. The archive is retained exactly,
    hashed before parsing, and its single PET.txt member is streamed line by line.
    No observations are forward-filled; negative spot prices are preserved.
    """
    start_date, end_date = _iso_date(start, "start"), _iso_date(end, "end")
    if start_date and end_date and start_date > end_date:
        raise ValueError("start must be on or before end")

    selected = dict(series or DEFAULT_PETROLEUM_SERIES)
    if not selected or any(not name.strip() or not sid.strip() for name, sid in selected.items()):
        raise ValueError("series must contain non-empty field names and EIA series IDs")
    if len(set(selected.values())) != len(selected):
        raise ValueError("duplicate EIA series IDs are not allowed")
    if any(not sid.startswith("PET.") or not sid.endswith(".D") for sid in selected.values()):
        raise ValueError("only daily EIA Petroleum series IDs (PET.*.D) are supported")

    http = session or requests.Session()
    retrieved_utc = datetime.now(timezone.utc).isoformat()
    snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot_root = Path(raw_dir) / snapshot_id
    snapshot_root.mkdir(parents=True, exist_ok=False)
    archive_path = snapshot_root / "PET.zip"

    download_metadata = _download_petroleum_bulk_file(http, archive_path, timeout)
    records, series_manifest, parse_stats = _read_petroleum_series_from_zip(
        archive_path, selected, start_date, end_date
    )

    columns = ["date", "series", "series_id", "value", "units", "series_description"]
    frame = pd.DataFrame(records, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["series_id", "date"], kind="stable").reset_index(drop=True)

    manifest = {
        "source": "U.S. Energy Information Administration (EIA), Open Data Bulk Download",
        "dataset": "Petroleum",
        "file": "PET.zip",
        "retrieved_utc": retrieved_utc,
        "snapshot_id": snapshot_id,
        "download": download_metadata,
        "raw_snapshot_directory": str(snapshot_root),
        "request": {
            "start": start_date,
            "end": end_date,
            "series_ids": list(selected.values()),
            "api_key_required": False,
        },
        "parser": {
            "format": "ZIP containing PET.txt; one complete JSON series object per line",
            "streaming": "PET.txt is parsed line by line without extracting the full text file",
            **parse_stats,
        },
        "series": series_manifest,
        "row_count": int(len(frame)),
        "normalized_sha256": sha256(
            frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        ).hexdigest(),
        "missing_observations": (
            "Null source values are omitted and counted; no forward fill or interpolation is applied"
        ),
        "negative_prices": "Preserved when finite; negative crude-oil spot prices are economically possible",
        "vintage_limit": (
            "The hash identifies the downloaded archive snapshot; it does not reconstruct historical "
            "publication vintages or the information set available on each observation date"
        ),
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
            raise ValueError("EIA frame contains invalid dates or non-finite values")
        # Do not reject zero or negative spot prices. WTI Cushing was negative in April 2020.
    csv_target, manifest_target = Path(csv_path), Path(manifest_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_target, index=False, lineterminator="\n")
    manifest_target.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
