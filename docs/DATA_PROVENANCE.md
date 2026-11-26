# Data provenance

## Public source pipeline

The downloader uses Federal Reserve Bank of St. Louis FRED public CSV graph endpoints:

| Field | FRED series | Description |
|---|---|---|
| sp500 | SP500 | S&P 500 index |
| vix | VIXCLS | CBOE Volatility Index |
| eurusd | DEXUSEU | U.S. dollars to one euro |
| usdjpy | DEXJPUS | Japanese yen to one U.S. dollar |
| wti | DCOILWTICO | WTI spot price |
| ust2y | DGS2 | 2-year Treasury constant maturity yield |
| ust10y | DGS10 | 10-year Treasury constant maturity yield |

The ingestion script stores raw CSV bytes and records series IDs, source URLs, UTC retrieval timestamps, row counts, missingness, date coverage and SHA-256 digests. Short calendar gaps are aligned onto business dates with a maximum four-business-day forward fill; remaining missing values are rejected.

## EIA daily petroleum spot prices

A separate API v2 adapter is available in src/riskmanage/eia.py and scripts/download_eia_petroleum_spot.py.

| Project field | EIA route | Series | Frequency | Interpretation |
|---|---|---|---|---|
| wti_cushing_spot | petroleum/pri/spt | RWTC | Daily | WTI Cushing spot benchmark |
| brent_europe_spot | petroleum/pri/spt | RBRTE | Daily | Europe Brent spot benchmark |

The collector paginates each series independently with an explicit ascending period sort and a maximum 5,000 JSON rows per page. It retains every raw JSON page, SHA-256 page digest, offset, row count, requested date bounds, retrieval timestamp, returned units, normalized CSV digest and a sanitized manifest. No forward filling or interpolation is performed. The API key is read from EIA_API_KEY and excluded from files and normal output.

These are spot-price benchmarks, not exchange-traded futures settlements, executable bid/ask quotes or a futures forward curve. They are not sufficient to value a listed futures book without contract-month settlements, expiry and roll rules, and an explicit spot/futures basis or curve methodology.

The API's current data response is a retrieval snapshot. Its retrieval time and raw-response hashes support snapshot reproducibility, but do not recreate historical EIA data vintages or the information set available to market participants at every historical date. Historical point-in-time backtests need publication/vintage information separately.

See EIA_DATA_INGESTION.md for execution, tests and limitations. Mocked unit tests do not replace the first live local download and manifest review.

## Reference run

The reference outputs linked from the README are generated from the deterministic synthetic fixture (seed 20261124), not downloaded market observations. The reference history has 1,800 business-day rows from 2016-01-04 through 2022-11-25.

The public FRED ingestion pipeline is implemented. A successfully downloaded public-data run must retain its source manifest alongside generated reports before any values are described as public empirical results.

## RFET distinction

Public daily closes do not prove qualifying transaction or committed-quote evidence. The RFET module refuses to certify modellability from close-only inputs. An independently checked evidence ledger is required.

## Licensing

Observe the FRED / source-provider terms for redistribution and derivative use. This repository does not claim ownership of underlying market observations.
