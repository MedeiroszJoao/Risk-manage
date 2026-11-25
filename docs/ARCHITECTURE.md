# Architecture

```text
FRED CSV ───┐
            ├─> data contract / provenance ─> market states
Synthetic ──┘                                 │
                                              v
                                 positions / instrument definitions
                                              │
                                  ┌───────────┴───────────┐
                                  v                       v
                             Full revaluation          Risk model
                                  │                       │
                                  v                       v
                                 HPL                     RTPL
                                  │                       │
                     ┌────────────┴───────────────┐       │
                     v                            v       v
                VaR / ES / backtest            PLA / residual analysis
                     │                            │
                     └────────────┬───────────────┘
                                  v
                Stress window / LH ES / IMCC / RFET warnings
                                  │
                                  v
                      machine-readable reports + figures
```

| Module | Responsibility |
|---|---|
| data.py / fixtures.py | FRED ingestion, validation, provenance and deterministic market fixture |
| pricing.py | Black–Scholes, Garman–Kohlhagen, rates / commodity valuation |
| portfolio.py | Hypothetical desk instruments and positions |
| pnl.py | Static HPL and independent RTPL generation |
| risk.py | Historical VaR, ES, factor subset ES and liquidity adjustment |
| capital.py | Stressed window, reduced set, illustrative IMCC and NMRF stress |
| validation.py | Backtesting, Kupiec, Christoffersen, PLA, RFET evidence diagnostics |
| scripts/run_case.py | Reproducible workflow and generated report |

The data, valuation, P&L and validation layers are deliberately separate so the RTPL implementation cannot silently reuse HPL outputs.
