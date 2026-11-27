# Architecture

```text
SEC N-PORT (fund holdings) ─────┐
FRED CSV (equity / FX / rates) ─┼
                               ├─> aligned panel + provenance ─> market states
EIA PET.zip (WTI / Brent) ──────┤
Synthetic fixture ─────────────┘
                                              v
                                  hypothetical desk positions
                                              |
                                  +-----------+-----------+
                                  v                       v
                             Full revaluation          Risk model
                                  |                       |
                                  v                       v
                                 HPL                     RTPL
                                  |                       |
                     +------------+-----------------------+
                     v                                    v
             VaR / ES / backtest                     PLA / residual analysis
                     |                                    |
                     +----------------+-------------------+
                                      v
                Liquidity horizons / stress / reduced set / IMCC
                                      |
                                      v
                        RFET warnings + reports + figures
```

| Module | Responsibility |
|---|---|
| sec_nport.py | Parse one selected SEC N-PORT accession into disclosed fund-holdings positions, identifiers, and input/output hash manifest; no bank-book inference |\n| sec_nport.py | Parse one selected SEC N-PORT accession into disclosed fund-holdings positions, linked identifiers and input/output hash manifest |\n| data.py / eia.py / fixtures.py | FRED equity/FX/rates and EIA WTI/Brent bulk ingestion, unified data contract, provenance and seeded synthetic factor fixture |
| pricing.py | Black–Scholes, Garman–Kohlhagen and zero-coupon proxy pricing |
| portfolio.py | Hypothetical constant-maturity desk positions |
| pnl.py | Full HPL, sensitivity-based RTPL and factor-level HPL attribution |
| risk.py | Historical VaR / ES and liquidity-adjusted ES |
| capital.py | Descriptive historical stress-window selector, in-sample reduced-factor diagnostic and non-regulatory IMCC/NMRF formula primitives |
| validation.py | HPL-only VaR diagnostics, Kupiec/Christoffersen, HPL-vs-RTPL PLA diagnostic and RFET evidence-count screen |
| scripts/parse_sec_nport.py | Parse an extracted SEC N-PORT quarter for a specific accession and series |\n| scripts/parse_sec_nport.py | Parse an extracted SEC N-PORT quarter for one accession and series |\n| scripts/download_market_data.py | FRED + EIA bulk acquisition, alignment and combined source manifest |
| scripts/download_eia_petroleum_spot.py | Standalone EIA PET archive collector |
| scripts/run_case.py | End-to-end reproducible run, scenarios, reports and figures |

The data, valuation, P&L and validation layers are deliberately separated so RTPL cannot reuse HPL outputs. Factor P&L attribution reconciles exactly to aggregate desk HPL, with theta and cross-factor interactions reported separately.


The architecture intentionally separates computational status from evidence status. capital.py refuses short histories and incomplete selected risk-class aggregates, but every capital-shaped value remains research-only. validation.py does not substitute an undefined PLA correlation with zero and never certifies RFET modellability.
