# Architecture

```text
FRED CSV (equity / FX / rates) ─┐
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
| data.py / eia.py / fixtures.py | FRED equity/FX/rates and EIA WTI/Brent bulk ingestion, unified data contract, provenance and seeded synthetic factor fixture |
| pricing.py | Black–Scholes, Garman–Kohlhagen and zero-coupon proxy pricing |
| portfolio.py | Hypothetical constant-maturity desk positions |
| pnl.py | Full HPL, sensitivity-based RTPL and factor-level HPL attribution |
| risk.py | Historical VaR / ES and liquidity-adjusted ES |
| capital.py | Historical stress window, rolling 12-week reduced-factor coverage, IMCC and NMRF aggregation |
| validation.py | VaR backtesting, Kupiec, Christoffersen, PLA and RFET evidence diagnostics |
| scripts/download_market_data.py | FRED + EIA bulk acquisition, alignment and combined source manifest |
| scripts/download_eia_petroleum_spot.py | Standalone EIA PET archive collector |
| scripts/run_case.py | End-to-end reproducible run, scenarios, reports and figures |

The data, valuation, P&L and validation layers are deliberately separated so RTPL cannot reuse HPL outputs. Factor P&L attribution reconciles exactly to aggregate desk HPL, with theta and cross-factor interactions reported separately.
