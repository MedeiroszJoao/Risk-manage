# Generated reports

Run `python scripts/run_case.py --synthetic` for deterministic offline outputs or `python scripts/download_market_data.py --start 2007-01-01` followed by `python scripts/run_case.py` for the unified FRED equity/FX/rates and EIA WTI/Brent spot panel. The public EIA bulk download does not require an API key.

Generated report files are not source-of-truth and are overwritten by the case runner. Each report includes input provenance and configuration.


## Standalone commodity spot diagnostic

After downloading the EIA Petroleum archive with `python scripts/download_eia_petroleum_spot.py --start 2007-01-01`, run `python scripts/run_commodity_spot_case.py`. This produces the independent WTI/Brent spot-proxy HPL/RTPL history, daily and ten-observation historical VaR/ES, and a 250-observation stress-window diagnostic. It uses actual common EIA dates only and is not an aggregated multi-desk IMA result.
