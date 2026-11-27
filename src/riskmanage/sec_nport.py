from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any


TABLES = {
    "submission": "SUBMISSION",
    "registrant": "REGISTRANT",
    "fund_info": "FUND_REPORTED_INFO",
    "holdings": "FUND_REPORTED_HOLDING",
    "identifiers": "IDENTIFIERS",
}


def _normalize_accession(value: str) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    if len(digits) != 18:
        raise ValueError("accession must contain exactly 18 digits (hyphens are optional)")
    return digits


def _decimal(value: Any, field: str) -> Decimal | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid decimal in {field}: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"non-finite decimal in {field}: {value!r}")
    return number


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


def _find_table(dataset_dir: Path, table_name: str) -> Path:
    candidates = []
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        stem = path.stem.upper()
        if stem == table_name or stem.endswith("_" + table_name):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {table_name}.tsv under {dataset_dir}; "
            "extract the official SEC N-PORT quarterly data set first"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple {table_name} tables found under {dataset_dir}; "
            "use a directory containing exactly one quarterly data set"
        )
    return candidates[0]


def _iter_tsv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} has no TSV header")
        reader.fieldnames = [name.strip().upper() for name in reader.fieldnames]
        for row in reader:
            normalized = {
                str(key).strip().upper(): (value.strip() if isinstance(value, str) else value)
                for key, value in row.items()
                if key is not None
            }
            yield normalized


def _rows_for_accession(path: Path, accession_digits: str) -> list[dict[str, str]]:
    rows = []
    for row in _iter_tsv(path):
        value = row.get("ACCESSION_NUMBER", "")
        try:
            matches = _normalize_accession(value) == accession_digits
        except ValueError:
            matches = False
        if matches:
            rows.append(row)
    return rows


def _require_fields(row: dict[str, str], required: set[str], table: str) -> None:
    missing = sorted(field for field in required if field not in row)
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_nport_dataset(
    dataset_dir: str | Path,
    accession: str,
    series_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse one explicitly selected SEC N-PORT filing from extracted TSV data.

    The SEC data set is disseminated quarterly and contains as-filed submissions.
    The caller must select the exact accession (and series when needed); this
    function does not silently choose among amendments or combine multiple funds.
    """
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    accession_digits = _normalize_accession(accession)

    table_paths = {
        key: _find_table(root, table_name)
        for key, table_name in TABLES.items()
    }
    for key in ("submission", "registrant", "fund_info", "holdings"):
        if not any(True for _ in _iter_tsv(table_paths[key])):
            raise ValueError(f"{table_paths[key].name} is empty")

    submission_rows = _rows_for_accession(table_paths["submission"], accession_digits)
    if len(submission_rows) != 1:
        raise ValueError(
            f"expected exactly one SUBMISSION row for accession {accession}, "
            f"found {len(submission_rows)}"
        )
    submission = submission_rows[0]
    _require_fields(
        submission, {"ACCESSION_NUMBER", "FILING_DATE", "REPORT_DATE"}, "SUBMISSION"
    )

    registrant_rows = _rows_for_accession(table_paths["registrant"], accession_digits)
    if len(registrant_rows) != 1:
        raise ValueError(
            f"expected exactly one REGISTRANT row for accession {accession}, "
            f"found {len(registrant_rows)}"
        )
    registrant = registrant_rows[0]
    _require_fields(registrant, {"CIK", "REGISTRANT_NAME"}, "REGISTRANT")

    all_fund_rows = _rows_for_accession(table_paths["fund_info"], accession_digits)
    fund_rows = all_fund_rows
    if series_id is not None:
        fund_rows = [row for row in all_fund_rows if row.get("SERIES_ID") == series_id]
    if len(fund_rows) != 1:
        raise ValueError(
            f"expected exactly one FUND_REPORTED_INFO row for accession {accession} "
            f"and series {series_id!r}, found {len(fund_rows)}; "
            "supply --series-id when the filing contains multiple series"
        )
    fund = fund_rows[0]
    _require_fields(
        fund, {"SERIES_ID", "SERIES_NAME", "NET_ASSETS"}, "FUND_REPORTED_INFO"
    )

    holding_rows = _rows_for_accession(table_paths["holdings"], accession_digits)
    if not holding_rows:
        raise ValueError(f"no holdings found for accession {accession}")
    if len(all_fund_rows) > 1:
        if "SERIES_ID" not in holding_rows[0]:
            raise ValueError(
                "the accession contains multiple fund series, but the holdings table "
                "has no SERIES_ID field; refusing to mix positions across series"
            )
        holding_rows = [
            row for row in holding_rows if row.get("SERIES_ID") == fund["SERIES_ID"]
        ]
        if not holding_rows:
            raise ValueError(
                f"no holdings linked to selected series {fund['SERIES_ID']}"
            )
    elif "SERIES_ID" in holding_rows[0]:
        holding_rows = [
            row for row in holding_rows if row.get("SERIES_ID") == fund["SERIES_ID"]
        ]
        if not holding_rows:
            raise ValueError(
                f"no holdings linked to selected series {fund['SERIES_ID']}"
            )

    required_holding_fields = {
        "HOLDING_ID",
        "ISSUER_NAME",
        "ISSUER_TITLE",
        "ISSUER_CUSIP",
        "BALANCE",
        "UNIT",
        "OTHER_UNIT_DESC",
        "CURRENCY_CODE",
        "CURRENCY_VALUE",
        "EXCHANGE_RATE",
        "PERCENTAGE",
        "PAYOFF_PROFILE",
        "ASSET_CAT",
        "DERIVATIVE_CAT",
        "INVESTMENT_COUNTRY",
        "IS_RESTRICTED_SECURITY",
        "FAIR_VALUE_LEVEL",
    }
    _require_fields(holding_rows[0], required_holding_fields, "FUND_REPORTED_HOLDING")

    holding_ids = {row.get("HOLDING_ID", "") for row in holding_rows}
    identifiers_by_holding: dict[str, dict[str, list[str]]] = {}
    for row in _iter_tsv(table_paths["identifiers"]):
        holding_id = row.get("HOLDING_ID", "")
        if holding_id not in holding_ids:
            continue
        identifiers = identifiers_by_holding.setdefault(
            holding_id, {"isins": [], "tickers": [], "other": []}
        )
        for field, target in (
            ("IDENTIFIER_ISIN", "isins"),
            ("IDENTIFIER_TICKER", "tickers"),
            ("OTHER_IDENTIFIER", "other"),
        ):
            value = row.get(field, "")
            if value and value not in identifiers[target]:
                identifiers[target].append(value)

    positions: list[dict[str, Any]] = []
    seen_holding_ids: set[str] = set()
    for row in holding_rows:
        holding_id = row["HOLDING_ID"]
        if not holding_id:
            raise ValueError("FUND_REPORTED_HOLDING contains an empty HOLDING_ID")
        if holding_id in seen_holding_ids:
            raise ValueError(f"duplicate HOLDING_ID in selected filing: {holding_id}")
        seen_holding_ids.add(holding_id)

        balance = _decimal(row.get("BALANCE"), "BALANCE")
        currency_value = _decimal(row.get("CURRENCY_VALUE"), "CURRENCY_VALUE")
        percentage = _decimal(row.get("PERCENTAGE"), "PERCENTAGE")
        exchange_rate = _decimal(row.get("EXCHANGE_RATE"), "EXCHANGE_RATE")
        payoff = row.get("PAYOFF_PROFILE", "").strip().lower()
        if payoff == "long":
            direction = "long"
            signed_value = currency_value
        elif payoff == "short":
            direction = "short"
            signed_value = -currency_value if currency_value is not None else None
        else:
            direction = "unknown"
            signed_value = None

        linked_ids = identifiers_by_holding.get(
            holding_id, {"isins": [], "tickers": [], "other": []}
        )
        positions.append(
            {
                "accession_number": submission["ACCESSION_NUMBER"],
                "filing_date": submission["FILING_DATE"],
                "report_date": submission["REPORT_DATE"],
                "registrant_cik": registrant["CIK"],
                "registrant_name": registrant["REGISTRANT_NAME"],
                "series_id": fund["SERIES_ID"],
                "series_name": fund["SERIES_NAME"],
                "net_assets": fund["NET_ASSETS"],
                "holding_id": holding_id,
                "issuer_name": row.get("ISSUER_NAME", ""),
                "issuer_title": row.get("ISSUER_TITLE", ""),
                "cusip": row.get("ISSUER_CUSIP", ""),
                "isins": ";".join(linked_ids["isins"]),
                "tickers": ";".join(linked_ids["tickers"]),
                "other_identifiers": ";".join(linked_ids["other"]),
                "balance": _decimal_text(balance),
                "unit": row.get("UNIT", ""),
                "other_unit_desc": row.get("OTHER_UNIT_DESC", ""),
                "currency_code": row.get("CURRENCY_CODE", ""),
                "currency_value": _decimal_text(currency_value),
                "exchange_rate": _decimal_text(exchange_rate),
                # SEC PERCENTAGE is a percent value, e.g. 0.061 means 0.061%,
                # not the decimal fraction 6.1% or 0.061 of NAV.
                "percentage_of_nav_pct": _decimal_text(percentage),
                "payoff_profile": row.get("PAYOFF_PROFILE", ""),
                "position_direction": direction,
                # Signed value remains denominated in CURRENCY_CODE. It is not
                # converted to USD unless the filing's currency is explicitly USD.
                "signed_currency_value": _decimal_text(signed_value),
                "asset_category": row.get("ASSET_CAT", ""),
                "derivative_category": row.get("DERIVATIVE_CAT", ""),
                "investment_country": row.get("INVESTMENT_COUNTRY", ""),
                "is_restricted_security": row.get("IS_RESTRICTED_SECURITY", ""),
                "fair_value_level": row.get("FAIR_VALUE_LEVEL", ""),
            }
        )

    manifest = {
        "schema": "sec_nport_extracted_tsv_v1",
        "source": "SEC Form N-PORT Data Sets",
        "accession_number": submission["ACCESSION_NUMBER"],
        "accession_digits": accession_digits,
        "filing_date": submission["FILING_DATE"],
        "report_date": submission["REPORT_DATE"],
        "registrant_cik": registrant["CIK"],
        "registrant_name": registrant["REGISTRANT_NAME"],
        "series_id": fund["SERIES_ID"],
        "series_name": fund["SERIES_NAME"],
        "net_assets": fund["NET_ASSETS"],
        "position_count": len(positions),
        "selected_holding_ids": len(seen_holding_ids),
        "tables": {
            key: {
                "filename": path.name,
                "sha256": _sha256(path),
            }
            for key, path in table_paths.items()
        },
        "classification": "publicly_disclosed_fund_holdings_snapshot",
        "not_equivalent_to": [
            "bank trading desk positions",
            "transaction-level trading history",
            "actual P&L (APL)",
            "regulatory RFET evidence",
            "regulatory FRTB IMA validation",
        ],
        "amendment_handling": (
            "The exact accession is selected by the caller. Amendments are not "
            "automatically resolved; inspect the SEC submission history and select "
            "the intended as-filed or amended accession."
        ),
    }
    return positions, manifest


def write_nport_snapshot(
    positions: list[dict[str, Any]],
    manifest: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    positions_path = destination / "positions.csv"
    manifest_path = destination / "snapshot_manifest.json"
    if not positions:
        raise ValueError("refusing to write an empty N-PORT snapshot")
    with positions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(positions[0].keys()))
        writer.writeheader()
        writer.writerows(positions)
    manifest = dict(manifest)
    manifest["output"] = {
        "positions_filename": positions_path.name,
        "positions_sha256": _sha256(positions_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return positions_path, manifest_path
