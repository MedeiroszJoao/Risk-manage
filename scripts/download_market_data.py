from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from riskmanage.data import download_fred_series, save_market_data

def main():
    parser = argparse.ArgumentParser(description="Download public FRED market-risk time series.")
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="data/market_data.csv")
    parser.add_argument("--manifest", default="data/source_manifest.json")
    args = parser.parse_args()
    frame, manifest = download_fred_series(args.start, args.end)
    save_market_data(frame, manifest, args.output, args.manifest)
    print(json.dumps({"rows": len(frame), "start": str(frame.date.min().date()),
                      "end": str(frame.date.max().date()), "manifest": args.manifest}, indent=2))
if __name__ == "__main__":
    main()
