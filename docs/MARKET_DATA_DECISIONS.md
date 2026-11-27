# Market-data source decisions and coverage policy

Decision date: 2026-11-26

This note records why the repository uses the current public-data feeds, what each feed actually represents, and where licensing or history prevents an apparently simple substitution.

## 1. EIA crude-oil spot benchmarks

**Selected source:** EIA public Petroleum bulk archive, https://www.eia.gov/opendata/bulk/PET.zip.

- WTI Cushing daily: PET.RWTC.D.
- Europe Brent daily: PET.RBRTE.D.
- No API key is required.
- The archive snapshot and normalized CSV are retained locally with SHA-256 hashes and source metadata.
- The raw EIA observations are not forward-filled. The standalone commodity diagnostic uses only common dates with actual WTI and Brent observations.
- Finite zero and negative WTI values are valid and retained.

The archive is a spot benchmark history. It is not an exchange-traded futures settlement history, executable quote feed or futures forward curve.

## 2. Equity, volatility, FX and rates

**Selected public feed:** FRED public graph CSV, without an API key.

- S&P 500 daily close: SP500.
- VIX daily close: VIXCLS.
- EUR/USD: DEXUSEU.
- USD/JPY: DEXJPUS.
- US 2-year Treasury constant-maturity yield: DGS2.
- US 10-year Treasury constant-maturity yield: DGS10.

Each series is stored in an immutable local retrieval snapshot with its raw SHA-256, valid observation coverage and missing-value count. The combined panel begins on the first date on which every required series is available; the requested start date is not represented as actual coverage when a series has no observation.

### S&P 500 coverage and rights boundary

FRED's current SP500 daily series provides only a ten-year daily history. Its notes identify S&P Dow Jones Indices LLC as the source and state that the observations are copyrighted. A request for 2007 therefore does not create a 2007 S&P 500 daily history. The final aligned four-desk panel reports its actual start date and source-coverage warnings in data/source_manifest.json.

Do not commit, publish or redistribute raw S&P 500 observations, or data-derived outputs, until the applicable S&P Dow Jones/FRED rights have been checked. The downloader keeps raw observations and generated reports local and gitignored.

## 3. If full-history official SPX is required

Cboe DataShop lists a Main Channel End-of-Day Summary product that includes popular S&P indices including SPX and VIX, with OHLC and option-volume fields where available. Cboe states that its historical data availability for active symbols starts in January 1975. The dataset is proprietary, and Cboe explicitly says external redistribution is strictly prohibited. A purchase requires selection of historical/subscription use and an internal-usage classification. This is a licensing decision, not a free API-key problem.

Official product page: https://datashop.cboe.com/main-channel-end-of-day-summary

The Fama/French daily Rm-Rf series is not an S&P 500 substitute. It is a CRSP value-weight return of eligible NYSE, AMEX and NASDAQ firms, and it is an excess-return factor, not the S&P 500 index level. It may support a separately specified broad-US-equity return study, but the model must label it differently and should not silently use it to mark S&P 500 options.

Official description: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/f-f_factors.html

SPY is an ETF, not the SPX index. It can be used only after explicitly redefining the equity underlier and validating ETF distributions, corporate actions, tracking basis and contract specifications.

## 4. Current implementation policy

- Keep the current FRED/EIA public pipeline free of credentials.
- Preserve the EIA-only commodity desk history separately so the SP500 coverage limit does not discard the longer oil history.
- Do not silently replace SPX with SPY, CRSP market returns, a third-party reconstruction or a scraped quote page.
- If the project requires exact daily SPX history back to 2007 and public presentation of derived output, obtain an appropriate license that allows the intended use before integrating the feed.
- Do not commit provider data, API keys or licensed raw observations to the public repository.

## 5. Scope and validation

The current four-desk common panel is a public-source research dataset, not a real bank portfolio. The standalone long-history oil analysis is a separate spot-risk diagnostic, not a full four-desk aggregate. Neither is a certified regulatory data feed or supervisory-approved FRTB IMA implementation.

A successful CI run validates code against deterministic synthetic fixtures. It does not prove that a live download succeeded, that provider data licensing allows public redistribution, or that historical observations are point-in-time vintages.
