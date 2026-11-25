# Data provenance

## Public data
The downloader uses Federal Reserve Bank of St. Louis FRED public CSV graph endpoints for:
- SP500 — S&P 500 index
- VIXCLS — CBOE Volatility Index
- DEXUSEU — U.S. dollars to one euro
- DEXJPUS — Japanese yen to one U.S. dollar
- DCOILWTICO — WTI spot price
- DGS2 / DGS10 — 2-year / 10-year Treasury constant-maturity yields

The ingestion script records data series IDs, endpoint URLs, retrieval UTC timestamp, row count, missingness and SHA-256 hashes of downloaded CSV bytes. The current market data files are generated locally and are not silently bundled into repository history.

## Synthetic test data
The fixture is generated deterministically from a fixed seed and regime-switching correlated shocks. It is labelled synthetic in metadata and reports. It is used only to verify software behavior, not to infer real risk.

## Important distinction
Historical closing prices are not proof of qualifying RFET transactions / committed quotes. The model's RFET output must remain “not evidenced” without a separately supplied real-price evidence ledger.
