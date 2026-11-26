# EIA petroleum spot-price ingestion

## Purpose and data contract

The EIA adapter retrieves daily observations from the official API v2 route:

- Route: `petroleum/pri/spt/data`
- WTI Cushing spot: `RWTC`
- Europe Brent spot: `RBRTE`
- Frequency: daily
- Data column: `value`

The normalized output has one row per observation and these fields:

| Field | Meaning |
|---|---|
| `date` | EIA period, represented as ISO date |
| `series` | Stable project field name |
| `series_id` | EIA series identifier |
| `value` | Numeric spot-price observation |
| `units` | Unit returned by EIA, when supplied |
| `series_description` | Description returned by EIA, when supplied |

The observations are daily spot benchmarks, not NYMEX/ICE futures settlements, executable bid/ask quotes, a full forward curve, or the actual positions of a trading desk. WTI Cushing and Europe Brent are useful reference risk factors for the commodities desk, but a futures-book HPL must not mark listed futures directly from these spot observations without an explicit basis/curve model and documented contract rolls.

## Authentication and local execution

Set the API key as a user-level environment variable in Windows PowerShell. The command prompts without echoing the key:

```powershell
$secure = Read-Host "EIA API Key" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)

try {
    $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    [Environment]::SetEnvironmentVariable("EIA_API_KEY", $key, "User")
    Write-Host "EIA_API_KEY configured for this Windows user."
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    $secure.Dispose()
    Remove-Variable key, secure, ptr -ErrorAction SilentlyContinue
}
```

Open a new PowerShell window, then run:

```powershell
python scripts/download_eia_petroleum_spot.py --start 2007-01-01
```

Optional bounds and output paths:

```powershell
python scripts/download_eia_petroleum_spot.py --start 2012-01-01 --end 2025-12-31
```

The downloader reads `EIA_API_KEY` from the environment. It has no command-line API-key argument. The credential is not written to the output CSV, raw response files, manifest, or normal success output. Request and HTTP error messages are deliberately redacted because the EIA documentation requires the key as a query parameter.

## Reproducibility and provenance

For each series, the downloader:
1. Calls EIA API v2 with explicit daily frequency, series facet, ascending period sort and bounded date range.
2. Retrieves pages using `offset` and `length`; the page size is capped at 5,000 JSON rows.
3. Saves each exact raw JSON response page separately.
4. Stores the SHA-256 digest, offset, row count and raw path for every page.
5. Produces a normalized CSV and manifest with retrieval time (UTC), requested period, actual coverage, series metadata, units and normalized CSV SHA-256.

No missing daily observations are forward-filled, interpolated or synthesized. Weekends and market holidays therefore remain absent. A changed result on a later retrieval is a new snapshot, not a silent rewrite of an earlier one. Preserve the raw pages and manifest for each research run.

The EIA API may be updated as underlying source data are revised. The retrieval timestamp and raw-page hashes identify the exact response snapshot retrieved; they do not establish point-in-time vintage history for what was known to the market on each historical date. For strict historical information-set backtests, source publication/vintage timestamps are a separate requirement.

The EIA API returns at most 5,000 JSON rows per request. Pagination is explicit rather than assuming a single response contains the full history. API usage should remain measured; do not recursively enumerate large parts of the EIA data hierarchy.

## Tests and validation boundary

The unit tests use deterministic mocked EIA responses. They validate pagination, row/total reconciliation, credential omission from saved manifests, SHA-256 provenance, duplicate-date rejection, invalid-value rejection, date bounds and output serialization. CI does not need an EIA credential and does not make live API calls.

A successful local run is still required before marking live ingestion as verified. The run must retain the CSV, all raw pages and the manifest together. Review the actual returned units and data coverage before connecting these observations to portfolio valuation.

## Official references

- [EIA API technical documentation](https://www.eia.gov/opendata/documentation.php)
- [EIA petroleum spot-prices route](https://www.eia.gov/opendata/browser/petroleum/pri/spt)
- [EIA API FAQ](https://www.eia.gov/opendata/faqs.php)

This adapter is a research data collector, not a representation of any institution's actual trading book and not a certified regulatory data feed.
