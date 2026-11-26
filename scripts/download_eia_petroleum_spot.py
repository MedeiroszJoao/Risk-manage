from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskmanage.eia import (  # noqa: E402
    EIA_PETROLEUM_SPOT_BULK_URL,
    download_eia_petroleum_spot_prices,
    save_eia_petroleum_spot_prices,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download daily EIA WTI and Brent spot prices from the public PET.zip bulk file."
    )
    parser.add_argument("--start", default="2007-01-01", help="First observation date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Last observation date (YYYY-MM-DD); defaults to latest.")
    parser.add_argument(
        "--output",
        default="data/processed/eia_petroleum_spot.csv",
        help="Normalized CSV output path.",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/eia_petroleum_spot_manifest.json",
        help="Sanitized provenance manifest path.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/eia/petroleum_spot",
        help="Directory for immutable timestamped PET.zip snapshots.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP connect/read timeout in seconds (default: 180).",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    print(f"Downloading public EIA bulk archive (no API key required): {EIA_PETROLEUM_SPOT_BULK_URL}")
    try:
        frame, manifest = download_eia_petroleum_spot_prices(
            start=args.start,
            end=args.end,
            timeout=(20, args.timeout),
            raw_dir=args.raw_dir,
        )
        save_eia_petroleum_spot_prices(frame, manifest, args.output, args.manifest)
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    summary = {
        "rows": int(len(frame)),
        "series": [
            {
                "field": item["field"],
                "series_id": item["series_id"],
                "rows": item["rows"],
                "first_date": item["first_date"],
                "last_date": item["last_date"],
                "units": item["units"],
                "negative_or_zero_values": int(
                    (frame.loc[frame["series_id"] == item["series_id"], "value"] <= 0).sum()
                ),
            }
            for item in manifest["series"]
        ],
        "csv": args.output,
        "manifest": args.manifest,
        "raw_archive": manifest["download"]["archive_path"],
        "archive_bytes": manifest["download"]["bytes"],
        "archive_sha256": manifest["download"]["sha256"],
        "normalized_sha256": manifest["normalized_sha256"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
