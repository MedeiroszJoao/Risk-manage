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

Python 3.11+ and the project dependencies are required. No EIA credential or API environment variable is needed.

Command: python scripts/download_eia_petroleum_spot.py --start 2007-01-01

Optional bounds and paths:

- python scripts/download_eia_petroleum_spot.py --start 2012-01-01 --end 2025-12-31
- python scripts/download_eia_petroleum_spot.py --start 2007-01-01 --timeout 240

Outputs:

- data/raw/eia/petroleum_spot/<snapshot_id>/PET.zip — exact downloaded archive, retained and never overwritten by a later run
- data/processed/eia_petroleum_spot.csv — normalized long-form observations
- data/processed/eia_petroleum_spot_manifest.json — provenance and validation metadata

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

No weekend/holiday observations are synthesized. Null source values are omitted and counted. There is no forward fill or interpolation. Prices are required to be finite, but **zero and negative prices are not rejected**: negative crude-oil spot prices are economically possible, and the WTI Cushing series recorded a negative observation on April 20, 2020. This is a deliberate data-integrity rule, not an exception or data-cleaning workaround.

## Reproducibility and validation

Each run receives a unique timestamped directory. The ZIP is streamed in fixed-size chunks to a temporary file, hashed while downloading, checked against Content-Length when provided, and atomically renamed to PET.zip only after the transfer completes. The ZIP is then opened without extracting its large text member. The parser verifies:

1. The downloaded file is a valid ZIP containing exactly one PET.txt member.
2. The two requested daily series occur exactly once in the archive.
3. Series metadata identifies daily frequency (D).
4. Each observation is a valid daily date and a finite numeric value.
5. Duplicate dates are rejected; returned rows are normalized to ascending date order.
6. Requested date bounds are applied without generating observations.
7. The raw archive digest and normalized CSV digest are recorded.

Unit tests use small synthetic ZIP archives matching the documented EIA line-oriented JSON format. They cover filtering, metadata, missing values, duplicate dates, invalid values, missing series, incomplete downloads, invalid ZIPs, immutable snapshots, output serialization and negative WTI prices. These deterministic tests do not replace the first live download from EIA.

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
