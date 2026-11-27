# EIA petroleum spot-price ingestion

## Purpose and source

The collector downloads the official, public EIA Petroleum bulk archive:

- Download: https://www.eia.gov/opendata/bulk/PET.zip
- Dataset: PET.zip (Petroleum)
- Selected daily WTI Cushing spot series: PET.RWTC.D
- Selected daily Europe Brent spot series: PET.RBRTE.D
- Units reported by EIA: dollars per barrel
- Authentication: **no API key is required**

The EIA bulk-download documentation states that each ZIP contains one same-named .txt file and that each line in the text file is a complete JSON object. Series objects contain metadata and a data array of period/value pairs. The collector streams the ZIP to disk, then parses PET.txt one line at a time directly from the archive. It does not extract the full text file and does not load the entire Petroleum dataset into memory.

The selected identifiers are the full legacy series IDs used in the Petroleum bulk file. They are not the shorter API v2 facet values (RWTC and RBRTE).

## Local execution

For the complete empirical market panel, use the integrated downloader:

```powershell
python scripts/download_market_data.py --start 2007-01-01
```

That command downloads FRED equity, FX and rates, retrieves this EIA bulk archive for both WTI and Brent, writes the standalone EIA CSV/manifest, and merges the observations into:

- `data/market_data.csv` — final aligned market panel consumed by `scripts/run_case.py`
- `data/source_manifest.json` — combined FRED/EIA source manifest

The main panel uses EIA as the canonical source for both oil benchmarks. The FRED intermediate deliberately excludes the WTI series so that WTI and Brent are not silently sourced from different feeds.

The standalone EIA collector is also available:

```powershell
python scripts/download_eia_petroleum_spot.py --start 2007-01-01
```

## Standalone commodity risk diagnostic

The EIA-only history can be used independently of the full four-desk panel. FRED's current SP500 daily feed has a shorter history, so the common multi-asset panel may start much later than 2007. To retain the longer oil history, run:

```powershell
python scripts/run_commodity_spot_case.py
```

This reads the normalized EIA CSV and uses only dates with actual WTI **and** Brent observations (inner join; no forward fill). The default hypothetical book is long 150,000 barrels of WTI spot and short 100,000 barrels of Europe Brent spot. It writes daily HPL/RTPL and residuals to `reports/generated/commodity_spot_pnl.csv`, and daily / 10-observation historical VaR and ES plus a 250-observation stress window selected by maximum 97.5% 10-observation ES to `reports/generated/commodity_spot_risk.json`.

This standalone output is a descriptive spot-risk diagnostic, not an aggregated four-desk portfolio or regulatory IMA result. Quantities are hypothetical; no futures basis, forward curve, expiry, roll, financing or transaction costs are modeled.

Python 3.11+ and the project dependencies are required. No EIA credential or API environment variable is needed.

Optional bounds and paths:

- `python scripts/download_eia_petroleum_spot.py --start 2012-01-01 --end 2025-12-31`
- `python scripts/download_eia_petroleum_spot.py --start 2007-01-01 --timeout 240`

Default outputs:

- `data/raw/eia/petroleum_spot/<snapshot_id>/PET.zip` — exact downloaded archive, retained and never overwritten by a later run
- `data/processed/eia_petroleum_spot.csv` — normalized long-form observations
- `data/processed/eia_petroleum_spot_manifest.json` — provenance and validation metadata

The integrated downloader writes the same standalone EIA outputs before building the combined market panel. The raw and processed data directories are local artifacts excluded from Git.

The data/raw/ and data/processed/ outputs are local artifacts and are excluded from Git. Keep the archive, CSV and manifest together when archiving a research run.

## Data contract

The normalized output has one row per actual observation:

| Field | Meaning |
|---|---|
| date | EIA period normalized to ISO YYYY-MM-DD |
| series | Stable project field name |
| series_id | Full EIA identifier, e.g. PET.RWTC.D |
| value | Numeric spot-price observation |
| units | Unit string supplied by EIA |
| series_description | Series description supplied by EIA |

The manifest records the source URL, final URL, HTTP status and selected response headers, UTC retrieval timestamp, snapshot ID, archive byte count and SHA-256, ZIP member name and uncompressed size, parsed line count, selected series metadata, source coverage, requested date bounds, null counts, normalized row count and normalized CSV SHA-256.

No weekend/holiday observations are synthesized. EIA's documented special values are handled explicitly: `null` (missing), `w` (withheld) and `*` (statistically insignificant) are omitted from the numeric panel and counted separately in the manifest. Unknown nonnumeric codes fail closed rather than being silently coerced. There is no forward fill or interpolation. Prices are required to be finite, but **zero and negative prices are not rejected**: negative crude-oil spot prices are economically possible, and the WTI Cushing series recorded a negative observation on April 20, 2020. This is a deliberate data-integrity rule, not an exception or data-cleaning workaround.

## Reproducibility and validation

Each run receives a unique timestamped directory. The ZIP is streamed in fixed-size chunks to a temporary file, hashed while downloading, checked against Content-Length when provided, and atomically renamed to PET.zip only after the transfer completes. The ZIP is then opened without extracting its large text member. The parser verifies:

1. The downloaded file is a valid ZIP containing exactly one PET.txt member.
2. The two requested daily series occur exactly once in the archive.
3. Series metadata identifies daily frequency (D).
4. Each observation is a valid daily date and a finite numeric value.
5. Duplicate dates are rejected; returned rows are normalized to ascending date order.
6. Requested date bounds are applied without generating observations.
7. The raw archive digest and normalized CSV digest are recorded.

Unit tests use small synthetic ZIP archives matching the documented EIA line-oriented JSON format. They cover filtering, metadata, missing values, duplicate dates, invalid values, missing series, incomplete downloads, invalid ZIPs, immutable snapshots, output serialization and negative WTI prices. A live ingestion was also completed in GitHub Actions on 2026-11-26 (run 45897080993): HTTP 200, 55,121,219-byte PET.zip, SHA-256 `8f3a6b2e4768a274f52af5f4e52b5e50a0c44c6be7bfc956eccd85ca3396dfc2`, 10,034 normalized rows (5,002 WTI and 5,032 Brent) covering 2007-01-02 through 2026-11-24. The 2020-04-20 WTI value of -$36.98/bbl was retained. The run artifact contains the raw ZIP, normalized CSV, manifest, commodity P&L and risk JSON; it is not committed to Git. The integrated FRED/EIA panel and four-desk runner were also executed successfully in GitHub Actions on 2026-11-26 (run 45897867676): 2,607 aligned business-day observations from 2016-11-28 through 2026-11-24, four desk outputs, seven source-coverage warnings, and an explicit finding that the 2007 stress-history span is unavailable. RFET status remained `evidence_review_required`; public closes do not establish qualifying transaction / committed-quote evidence. The integrated panel and its derived multi-desk outputs were not uploaded or committed because FRED's S&P 500 source has licensing restrictions. The artifact retains only the EIA source archive, normalized EIA CSV/manifest and standalone commodity risk outputs. Local runs should retain and inspect their own source manifests.

## Interpretation and limits

WTI Cushing and Europe Brent are daily spot benchmarks. They are not NYMEX/ICE futures settlements, executable bid/ask quotes, a full forward curve or actual trading-desk positions. They can be used as spot risk-factor histories, but a listed-futures HPL must not mark futures directly from these spot observations without an explicit basis/curve model, contract months, expiry and documented roll rules.

The ZIP hash identifies the exact archive retrieved for a run. It does not reconstruct historical EIA publication vintages or the information set available to traders on each historical observation date. Strict point-in-time backtests require vintage/publication-time data separately.

The collector is a research data pipeline, not a certified regulatory feed or a representation of any institution's actual trading book.

## Official references

- EIA bulk download facility and file format: https://www.eia.gov/opendata/v1/bulkfiles.php
- EIA Petroleum bulk archive: https://www.eia.gov/opendata/bulk/PET.zip
- EIA WTI Cushing daily spot-price series: https://www.eia.gov/dnav/pet/hist/rwtcD.htm
- EIA Europe Brent daily spot-price series: https://www.eia.gov/dnav/pet/hist/rbrteD.htm
- EIA explanation of negative oil prices and Cushing storage constraints: https://www.eia.gov/todayinenergy/detail.php?id=43875
