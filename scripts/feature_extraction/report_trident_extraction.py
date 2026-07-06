#!/usr/bin/env python
# report_trident_extraction.py
#
# Summarize the state of a TRIDENT feature-extraction run: how many slides have
# a .h5, which are still missing, and any per-slide failures logged by the
# array tasks.

import glob
import argparse
from pathlib import Path

# repo root is two levels up: scripts/feature_extraction/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SLIDES_DIR = PROJECT_ROOT / "data" / "raw" / "slides"
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features_uni_v1_full" / "features"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report TRIDENT extraction progress/failures.")
    p.add_argument("--slides-dir", type=Path, default=DEFAULT_SLIDES_DIR,
                   help="Root holding <file_id>/<file>.svs (the expected set).")
    p.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR,
                   help="Dir of produced per-slide .h5 files.")
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                   help="Dir with the SLURM/run .out and .log files to scan for failures.")
    p.add_argument("--show", type=int, default=20,
                   help="How many missing/failed names to print (default: 20).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    expected = {Path(p).stem for p in
                glob.glob(str(args.slides_dir / "**" / "*.svs"), recursive=True)}
    produced = {p.stem for p in args.features_dir.glob("*.h5")}
    missing = sorted(expected - produced)

    print(f"Slides expected (.svs on disk): {len(expected)}")
    print(f"Features produced (.h5)       : {len(produced)}")
    print(f"Still missing                 : {len(missing)}")
    if expected:
        pct = 100.0 * len(produced & expected) / len(expected)
        print(f"Coverage                      : {pct:.1f}%")

    if missing:
        print(f"\nMissing (first {min(args.show, len(missing))}):")
        for stem in missing[: args.show]:
            print(f"  - {stem}")
        if len(missing) > args.show:
            print(f"  ... and {len(missing) - args.show} more")

    # Collect per-slide failures logged by the extraction script.
    failures = []
    for log_path in sorted(glob.glob(str(args.log_dir / "*.out"))) + \
                    sorted(glob.glob(str(args.log_dir / "*.log"))):
        try:
            for line in Path(log_path).read_text(errors="ignore").splitlines():
                if "FAILED:" in line or "] failed:" in line.lower():
                    failures.append((Path(log_path).name, line.strip()))
        except OSError:
            continue

    print(f"\nFailure lines found in logs   : {len(failures)}")
    for name, line in failures[: args.show]:
        print(f"  [{name}] {line}")
    if len(failures) > args.show:
        print(f"  ... and {len(failures) - args.show} more")

    # Non-zero exit if the run is incomplete, so it's easy to gate on.
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
