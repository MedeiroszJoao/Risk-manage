from __future__ import annotations

import argparse
from hashlib import sha256
import json

from riskmanage.data import (
    FRED_SERIES,
    download_fred_series,
    merge_eia_petroleum_into_market_data,
    save_market_data,
)
from riskmanage.eia import (
    download_eia_petroleum_spot_prices,
    save_eia_petroleum_spot_prices,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the empirical market panel from public FRED equity/FX/rates "
            "and EIA WTI/Brent spot benchmarks."
        )
    )
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="data/market_data.csv")
    parser.add_argument("--manifest", default="data/source_manifest.json")
    parser.add_argument("--fred-timeout", type=int, default=45)
    parser.add_argument("--eia-timeout", type=int, default=240)
    args = parser.parse_args()

    if args.fred_timeout <= 0 or args.eia_timeout <= 0:
        parser.error("timeouts must be positive")

    # FRED supplies equity, volatility, FX and rates. Both crude-oil spot
    # benchmarks are deliberately taken from the same EIA bulk snapshot.
    fred_ids = {name: series_id for name, series_id in FRED_SERIES.items() if name != "wti"}
    fred_frame, fred_manifest = download_fred_series(
        start=args.start,
        end=args.end,
        series_ids=fred_ids,
        timeout=args.fred_timeout,
    )

    eia_frame, eia_manifest = download_eia_petroleum_spot_prices(
        start=args.start,
        end=args.end,
        timeout=(20, args.eia_timeout),
    )
    save_eia_petroleum_spot_prices(eia_frame, eia_manifest)

    market = merge_eia_petroleum_into_market_data(fred_frame, eia_frame)
    provenance = {
        "source": (
            "FRED public graph CSV for equity/FX/rates plus EIA Petroleum bulk archive "
            "for WTI Cushing and Europe Brent spot prices"
        ),
        "retrieved_utc": eia_manifest["retrieved_utc"],
        "start_requested": args.start,
        "end_requested": args.end,
        "fred": fred_manifest,
        "eia": eia_manifest,
        "alignment": (
            "Outer join on observation date, then business-day calendar with a maximum "
            "four-business-day forward fill; no interpolation"
        ),
        "aligned_rows": int(len(market)),
        "aligned_start": market["date"].min().date().isoformat(),
        "aligned_end": market["date"].max().date().isoformat(),
        "aligned_sha256": sha256(
            market.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")
        ).hexdigest(),
        "classification": "public_observed_source_data_with_limited_calendar_alignment",
        "api_key_required": False,
    }

    save_market_data(market, provenance, args.output, args.manifest)
    print(
        json.dumps(
            {
                "rows": int(len(market)),
                "start": provenance["aligned_start"],
                "end": provenance["aligned_end"],
                "columns": list(market.columns),
                "eia_archive_sha256": eia_manifest["download"]["sha256"],
                "eia_csv": "data/processed/eia_petroleum_spot.csv",
                "fred_series": [item["series_id"] for item in fred_manifest["series"]],
                "market_csv": args.output,
                "manifest": args.manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
