from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskmanage.sec_nport import parse_nport_dataset, write_nport_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse one explicitly selected filing from an extracted SEC N-PORT "
            "quarterly data set into an auditable disclosed fund-holdings snapshot."
        )
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--series-id", default=None)
    parser.add_argument("--output-dir", default="data/processed/sec_nport")
    args = parser.parse_args()

    positions, manifest = parse_nport_dataset(
        args.dataset_dir,
        accession=args.accession,
        series_id=args.series_id,
    )
    positions_path, manifest_path = write_nport_snapshot(
        positions, manifest, args.output_dir
    )
    print(
        json.dumps(
            {
                "positions": len(positions),
                "report_date": manifest["report_date"],
                "filing_date": manifest["filing_date"],
                "registrant": manifest["registrant_name"],
                "series": manifest["series_name"],
                "series_id": manifest["series_id"],
                "positions_csv": str(positions_path),
                "manifest": str(manifest_path),
                "classification": manifest["classification"],
                "regulatory_status": "not_assessed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
