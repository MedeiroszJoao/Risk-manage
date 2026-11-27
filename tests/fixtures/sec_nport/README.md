# SEC N-PORT fixture

The numeric holding values in the test fixture are a deliberately small subset of the official SEC Form N-PORT filing for iShares Core S&P 500 ETF:

- CIK: 0001100663
- Accession: 0002071691-26-019760
- Filing date: 2026-08-25
- Report date: 2026-06-30
- Series: S000004310 — iShares Core S&P 500 ETF
- Official filing: https://www.sec.gov/Archives/edgar/data/1100663/000207169126019760/xslFormNPORT-P_X01/primary_doc.xml

The test fixture contains only three selected positions, not the full fund portfolio. Fixture-specific HOLDING_ID values exist only to test the identifier join; they are not claimed to be SEC-issued IDs. The issuer identifiers, reported balances, currency values, and reported percentages were transcribed from the filing. The parser preserves SEC PERCENTAGE as percentage points: 0.061079735228 means 0.061079735228% of NAV.
