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

## Reference run

The reference outputs linked from the README are generated from the deterministic synthetic fixture (seed 20261124), not downloaded market observations. The reference history has 1,800 business-day rows from 2016-01-04 through 2022-11-25.

The public FRED ingestion pipeline is implemented. A successfully downloaded public-data run must retain its source manifest alongside generated reports before any values are described as public empirical results.

## RFET distinction

Public daily closes do not prove qualifying transaction or committed-quote evidence. The RFET module refuses to certify modellability from close-only inputs. An independently checked evidence ledger is required.

## Licensing

Observe the FRED / source-provider terms for redistribution and derivative use. This repository does not claim ownership of underlying market observations.
