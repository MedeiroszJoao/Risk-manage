# Data provenance

## Public source pipeline

The downloader uses Federal Reserve Bank of St. Louis FRED public CSV graph endpoints:

| Field | FRED series | Description |
|---|---|---|
| sp500 | SP500 | S&P 500 index |
| vix | VIXCLS | CBOE Volatility Index |
| eurusd | DEXUSEU | U.S. dollars to one euro |
| usdjpy | DEXJPUS | Japanese yen to one U.S. dollar |
| ust2y | DGS2 | 2-year Treasury constant maturity yield |
| ust10y | DGS10 | 10-year Treasury constant maturity yield |

The ingestion script stores each run in a unique timestamped raw-snapshot directory, retaining the exact CSV bytes and a SHA-256 digest per series. Its manifest records the first and last dates with actual numeric observations (not merely the requested or first calendar date), the total source rows, missing-value count and source URL. FRED public graph CSV does not require an API key. Short calendar gaps are aligned onto business dates with a maximum four-business-day forward fill; remaining missing values are rejected. The series do not all share the same historical start: the combined panel begins at the first date on which every required factor is valid.

## EIA daily petroleum spot prices

The integrated scripts/download_market_data.py pipeline uses this EIA snapshot as the canonical source for both crude-oil risk factors. It deliberately excludes the FRED WTI column from the main request, so WTI and Brent come from the same source and definition. FRED contributes S&P 500, VIX, FX and Treasury rates; the EIA daily spot series contribute WTI and Brent.

The EIA collector downloads the public Petroleum bulk archive PET.zip from https://www.eia.gov/opendata/bulk/PET.zip. This bulk download does not require an API key. It parses the archive's PET.txt member as line-delimited JSON and selects exactly:

| Project field | EIA bulk series ID | Frequency | Interpretation |
|---|---|---|---|
| wti_cushing_spot | PET.RWTC.D | Daily | WTI Cushing spot benchmark |
| brent_europe_spot | PET.RBRTE.D | Daily | Europe Brent spot benchmark |

Each timestamped run preserves the exact ZIP archive and its SHA-256 digest, HTTP retrieval metadata, parsed series metadata, source coverage, null counts, normalized CSV digest and a sanitized manifest. Parsing is performed line by line from the ZIP member; the much larger uncompressed text file is not extracted to disk or loaded into memory. No forward fill or interpolation is performed. Null source values are counted and omitted; finite zero and negative spot prices are preserved.

These are spot-price benchmarks, not exchange-traded futures settlements, executable bid/ask quotes or a futures forward curve. They are not sufficient to value a listed futures book without contract-month settlements, expiry and roll rules, and an explicit spot/futures basis or curve methodology.

The archive hash identifies the retrieved snapshot. It does not recreate historical EIA publication vintages or the information set available at every historical date. Strict point-in-time backtests need publication/vintage information separately.

See EIA_DATA_INGESTION.md for the source format and collector details. Run python scripts/download_market_data.py --start 2007-01-01 to build the unified market panel and combined source manifest. Mocked unit tests do not replace the first live local download and manifest review.

## Reference run

The reference outputs linked from the README are generated from the deterministic synthetic fixture (seed 20261124), not downloaded market observations. The reference history has 1,800 business-day rows from 2016-01-04 through 2022-11-25.

The public FRED ingestion pipeline is implemented. A successfully downloaded public-data run must retain its source manifest alongside generated reports before any values are described as public empirical results.

## RFET distinction

Public daily closes do not prove qualifying transaction or committed-quote evidence. The RFET module refuses to certify modellability from close-only inputs. An independently checked evidence ledger is required.

## Licensing

Observe the FRED / source-provider terms for redistribution and derivative use. This repository does not claim ownership of underlying market observations.
