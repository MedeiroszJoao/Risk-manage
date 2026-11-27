from __future__ import annotations

import csv
from pathlib import Path

import pytest

from riskmanage.sec_nport import parse_nport_dataset, write_nport_snapshot


ACCESSION = "0002071691-26-019760"
ACCESSION_COMPACT = "000207169126019760"


def _write_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


@pytest.fixture
def nport_dataset(tmp_path: Path) -> Path:
    _write_table(
        tmp_path / "SUBMISSION.tsv",
        ["ACCESSION_NUMBER", "FILING_DATE", "REPORT_DATE"],
        [[ACCESSION_COMPACT, "2026-08-25", "2026-06-30"]],
    )
    _write_table(
        tmp_path / "REGISTRANT.tsv",
        ["ACCESSION_NUMBER", "CIK", "REGISTRANT_NAME"],
        [[ACCESSION_COMPACT, "0001100663", "iShares Trust"]],
    )
    _write_table(
        tmp_path / "FUND_REPORTED_INFO.tsv",
        ["ACCESSION_NUMBER", "SERIES_ID", "SERIES_NAME", "NET_ASSETS"],
        [[ACCESSION_COMPACT, "S000004310", "iShares Core S&P 500 ETF", "888128937468.17"]],
    )
    holding_headers = [
        "ACCESSION_NUMBER", "HOLDING_ID", "ISSUER_NAME", "ISSUER_TITLE",
        "ISSUER_CUSIP", "BALANCE", "UNIT", "OTHER_UNIT_DESC", "CURRENCY_CODE",
        "CURRENCY_VALUE", "EXCHANGE_RATE", "PERCENTAGE", "PAYOFF_PROFILE",
        "ASSET_CAT", "DERIVATIVE_CAT", "INVESTMENT_COUNTRY",
        "IS_RESTRICTED_SECURITY", "FAIR_VALUE_LEVEL",
    ]
    holding_rows = [
        [
            ACCESSION_COMPACT, "fixture-h1", "CBRE Group, Inc.",
            "CBRE Group, Inc., Class A", "12504L109", "4027521.00000000",
            "NS", "", "USD", "542466803.49000000", "1.00000000",
            "0.061079735228", "Long", "EC", "", "US", "N", "1",
        ],
        [
            ACCESSION_COMPACT, "fixture-h2", "Brown & Brown, Inc.",
            "Brown & Brown, Inc.", "115236101", "4009031.00000000",
            "NS", "", "USD", "257179338.65000000", "1.00000000",
            "0.028957432620", "Long", "EC", "", "US", "N", "1",
        ],
        [
            ACCESSION_COMPACT, "fixture-h3", "Oracle Corp.", "Oracle Corp.",
            "68389X105", "23340523.00000000", "NS", "", "USD",
            "3420553645.65000000", "1.00000000", "0.385141560120",
            "Long", "EC", "", "US", "N", "1",
        ],
    ]
    _write_table(tmp_path / "FUND_REPORTED_HOLDING.tsv", holding_headers, holding_rows)
    _write_table(
        tmp_path / "IDENTIFIERS.tsv",
        ["HOLDING_ID", "IDENTIFIERS_ID", "IDENTIFIER_ISIN", "IDENTIFIER_TICKER",
         "OTHER_IDENTIFIER", "OTHER_IDENTIFIER_DESC"],
        [
            ["fixture-h1", "1", "US12504L1098", "CBRE", "", ""],
            ["fixture-h2", "2", "US1152361010", "BRO", "", ""],
            ["fixture-h3", "3", "US68389X1054", "ORCL", "", ""],
            ["other", "4", "US0000000000", "OTHER", "", ""],
        ],
    )
    return tmp_path


def test_parse_official_sec_nport_filing_subset_without_rescaling_percentage(
    nport_dataset: Path,
):
    positions, manifest = parse_nport_dataset(nport_dataset, ACCESSION)
    assert len(positions) == 3
    assert manifest["registrant_name"] == "iShares Trust"
    assert manifest["series_name"] == "iShares Core S&P 500 ETF"
    assert manifest["report_date"] == "2026-06-30"

    first = positions[0]
    assert first["cusip"] == "12504L109"
    assert first["isins"] == "US12504L1098"
    assert first["tickers"] == "CBRE"
    assert first["balance"] == "4027521.00000000"
    assert first["currency_value"] == "542466803.49000000"
    assert first["percentage_of_nav_pct"] == "0.061079735228"
    assert first["signed_currency_value"] == "542466803.49000000"
    assert first["position_direction"] == "long"


def test_nport_parser_requires_explicit_series_if_filing_has_multiple(
    nport_dataset: Path,
):
    _write_table(
        nport_dataset / "FUND_REPORTED_INFO.tsv",
        ["ACCESSION_NUMBER", "SERIES_ID", "SERIES_NAME", "NET_ASSETS"],
        [
            [ACCESSION_COMPACT, "S000004310", "iShares Core S&P 500 ETF", "888128937468.17"],
            [ACCESSION_COMPACT, "S000099999", "Second series", "1000.00"],
        ],
    )
    with pytest.raises(ValueError, match="supply --series-id"):
        parse_nport_dataset(nport_dataset, ACCESSION)
    with pytest.raises(ValueError, match="no SERIES_ID field"):
        parse_nport_dataset(nport_dataset, ACCESSION, series_id="S000004310")

    holding_path = nport_dataset / "FUND_REPORTED_HOLDING.tsv"
    lines = holding_path.read_text(encoding="utf-8").splitlines()
    lines[0] += "\\tSERIES_ID"
    lines = [lines[0]] + [line + "\\tS000004310" for line in lines[1:]]
    holding_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    positions, manifest = parse_nport_dataset(
        nport_dataset, ACCESSION, series_id="S000004310"
    )
    assert len(positions) == 3
    assert manifest["series_id"] == "S000004310"


def test_nport_unknown_payoff_direction_does_not_receive_signed_value(
    nport_dataset: Path,
):
    holding_path = nport_dataset / "FUND_REPORTED_HOLDING.tsv"
    content = holding_path.read_text(encoding="utf-8")
    content = content.replace("\tLong\tEC\t", "\tUnspecified\tEC\t", 1)
    holding_path.write_text(content, encoding="utf-8")
    positions, _ = parse_nport_dataset(nport_dataset, ACCESSION)
    assert positions[0]["position_direction"] == "unknown"
    assert positions[0]["signed_currency_value"] is None


def test_nport_snapshot_writes_hash_linked_manifest(nport_dataset: Path, tmp_path: Path):
    positions, manifest = parse_nport_dataset(nport_dataset, ACCESSION)
    output = tmp_path / "output"
    csv_path, manifest_path = write_nport_snapshot(positions, manifest, output)
    saved = manifest_path.read_text(encoding="utf-8")
    assert csv_path.exists()
    assert manifest_path.exists()
    assert '"positions_sha256"' in saved
    assert '"publicly_disclosed_fund_holdings_snapshot"' in saved


def test_nport_rejects_wrong_accession(nport_dataset: Path):
    with pytest.raises(ValueError, match="expected exactly one SUBMISSION"):
        parse_nport_dataset(nport_dataset, "000000000000000001")
