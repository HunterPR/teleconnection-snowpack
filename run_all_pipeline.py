"""
Run the full Snoqualmie data pipeline in one command.

Steps:
1) build_snoqualmie_weather_pipeline.py
2) organize_data.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weather fetch + organize pipeline.")
    parser.add_argument("--start-date", default="2000-01-01", help="Historical start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Historical end date (YYYY-MM-DD)")
    parser.add_argument("--wsdot-station-limit", type=int, default=8, help="Nearest WSDOT station count")
    parser.add_argument("--wsdot-chunk-days", type=int, default=7, help="WSDOT request chunk size in days")
    parser.add_argument("--forecast-days", type=int, default=16, help="Forecast horizon in days")
    parser.add_argument(
        "--forecast-models",
        default="ecmwf_ifs025,gfs_seamless,hrrr_conus",
        help="Comma-separated Open-Meteo forecast models",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Skip weather fetch stage")
    parser.add_argument("--skip-organize", action="store_true", help="Skip organize_data stage")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print(f"\n> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()

    if not args.skip_fetch:
        fetch_cmd = [
            sys.executable,
            "build_snoqualmie_weather_pipeline.py",
            "--start-date",
            str(args.start_date),
            "--wsdot-station-limit",
            str(args.wsdot_station_limit),
            "--wsdot-chunk-days",
            str(args.wsdot_chunk_days),
            "--forecast-days",
            str(args.forecast_days),
            "--forecast-models",
            str(args.forecast_models),
        ]
        if args.end_date:
            fetch_cmd.extend(["--end-date", str(args.end_date)])
        run_cmd(fetch_cmd)

    if not args.skip_organize:
        run_cmd([sys.executable, "organize_data.py"])

    print("\nDone.")
    print("Outputs are in data/pipeline/ and data/processed/.")


if __name__ == "__main__":
    main()
