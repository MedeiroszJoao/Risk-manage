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

The ingestion script stores each run in a unique timestamped raw-snapshot directory, retaining the exact CSV bytes and a SHA-256 digest per series. Its manifest records the first and last dates with actual numeric observations (not merely the requested or first calendar date), the total source rows, missing-value count, source URL and coverage warnings when a series starts later than requested. FRED public graph CSV does not require an API key. Short calendar gaps are aligned onto business dates with a maximum four-business-day forward fill; remaining missing values are rejected. The series do not all share the same historical start: the combined panel begins at the first date on which every required factor is valid.

**Important SP500 limitation:** FRED states that its current SP500 series includes only 10 years of daily history. A request starting in 2007 therefore does not produce a 2007 daily S&P 500 series from this FRED feed. The final common market panel begins at the first date on which every factor is available; the downloader records this actual date and emits a coverage warning. FRED's SP500 series notes identify S&P Dow Jones Indices LLC as the source and mark the data as copyrighted; check licensing/permission before sharing SP500 observations or derived material. The repository excludes the local raw files, market panel and generated reports from Git by default.

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

See EIA_DATA_INGESTION.md for the source format and collector details. The standalone EIA bulk collector and commodity diagnostic were live-validated in GitHub Actions on 2026-11-26 (run 45897080993), including the raw archive hash and 10,034 normalized observations. The integrated `python scripts/download_market_data.py --start 2007-01-01` FRED/EIA panel and `scripts/run_case.py` four-desk runner were live-validated in GitHub Actions run 45897867676: 2,607 observations from 2016-11-28 through 2026-11-24, four desks, seven source-coverage warnings, no full 2007 stress history, and RFET `evidence_review_required`. The combined panel and derived multi-desk outputs were ephemeral and are not committed or redistributed; the FRED S&P 500 source has licensing restrictions. Run locally and inspect the combined source manifest before relying on any local result. The separate `python scripts/run_commodity_spot_case.py` runner consumes the standalone EIA normalized CSV using common actual WTI/Brent dates only, so the commodity desk can retain its longer source history without pretending the full multi-desk panel has the same coverage.

## Reference run

The reference outputs linked from the README are generated from the deterministic synthetic fixture (seed 20261124), not downloaded market observations. The reference history has 1,800 business-day rows from 2016-01-04 through 2022-11-25.

The public FRED ingestion pipeline is implemented. A successfully downloaded public-data run must retain its source manifest alongside generated reports before any values are described as public empirical results.

## RFET distinction

Public daily closes do not prove qualifying transaction or committed-quote evidence. The RFET module refuses to certify modellability from close-only inputs. An independently checked evidence ledger is required.

## Licensing

Observe the FRED / source-provider terms for redistribution and derivative use. This repository does not claim ownership of underlying market observations.


## SEC Form N-PORT disclosed fund holdings

The repository includes a parser for the SEC's quarterly Form N-PORT flat-file data sets. The parser consumes the extracted `SUBMISSION`, `REGISTRANT`, `FUND_REPORTED_INFO`, `FUND_REPORTED_HOLDING` and `IDENTIFIERS` tables. It requires the caller to select one exact accession and, when the filing has multiple series rows, an exact series ID. When holdings cannot be linked unambiguously to that series, the parser fails closed instead of mixing funds. It does not silently choose among amendments or merge unrelated series.

Official sources:
- SEC Form N-PORT Data Sets: https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets
- SEC filing data set README / schemas: https://www.sec.gov/files/dera/data/form-n-port-data-sets/readme.pdf
- Test filing (iShares Core S&P 500 ETF, report date 2026-06-30, accession 0002071691-26-019760): https://www.sec.gov/Archives/edgar/data/1100663/000207169126019760/xslFormNPORT-P_X01/primary_doc.xml

Run the parser after extracting one official quarterly N-PORT data set:

```bash
python scripts/parse_sec_nport.py \
  --dataset-dir path/to/extracted_nport_quarter \
  --accession 0002071691-26-019760 \
  --series-id S000004310 \
  --output-dir data/processed/sec_nport/ivv_2026-06-30
```

The output is a position-level CSV plus a manifest containing the selected accession, filing/report dates, fund/series metadata, SHA-256 hashes for each input table, selected holding count and SHA-256 for the normalized output. Local raw and processed data remain excluded from Git by `.gitignore`.

**Interpretation boundary:** this produces a snapshot of positions disclosed by a registered fund, not a bank desk's actual positions, not a trade blotter, not daily position history and not Actual P&L. It does not generate market returns or a risk backtest by itself. Reconstructing historical P&L requires appropriately licensed instrument-level market data, corporate-action/contract handling, point-in-time identifiers, and subsequent portfolio snapshots. The exact accession matters because amendments are separate as-filed submissions. The SEC data sets are supplied by filers and may contain extraction errors; verify any material result against the original filing.
