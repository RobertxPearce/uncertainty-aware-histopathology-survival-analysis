#!/usr/bin/env python
# full_trident_feature_extraction.py
#
# Runs the TRIDENT pipeline (segmentation -> patch coords -> patch features)
# over the slides in the survival table via run_batch_of_slides.py.
#
# Slide layout on disk (one directory per GDC file_id):
#     <slides_dir>/<file_id>/<file_name>.svs
#
# Survival table columns used:
#     file_ids   - stringified list, e.g. "['<uuid>']"
#     file_names - stringified list, e.g. "['TCGA-...-DX1.<uuid>.svs']"
#
# Examples:
#     python scripts/full_trident_feature_extraction.py --gpus 0
#     python scripts/full_trident_feature_extraction.py --gpus 0 1 2 3

import ast
import sys
import argparse
import subprocess
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRIDENT_ROOT = PROJECT_ROOT / "TRIDENT"
BATCH_RUNNER = TRIDENT_ROOT / "run_batch_of_slides.py"

# uni_v1 expects 256px patches at 20x.
PATCH_ENCODER = "uni_v1"
TARGET_MAG = 20
PATCH_SIZE = 256


def build_slide_list(survival_table_path: Path, slides_dir: Path, out_csv: Path) -> int:
    """Expand the survival table into a TRIDENT slide list CSV.

    Writes a CSV with a single `wsi` column of paths relative to `slides_dir`
    (i.e. "<file_id>/<file_name>"), restricted to slides that actually exist on
    disk. Returns the number of slides written.
    """
    table = pd.read_csv(survival_table_path)

    rel_paths = []
    missing = []
    for _, row in table.iterrows():
        file_ids = ast.literal_eval(row["file_ids"])
        file_names = ast.literal_eval(row["file_names"])
        for file_id, file_name in zip(file_ids, file_names):
            rel_path = Path(file_id) / file_name
            if (slides_dir / rel_path).exists():
                rel_paths.append(rel_path.as_posix())
            else:
                missing.append(rel_path.as_posix())

    if missing:
        print(f"WARNING: {len(missing)} slide(s) from the survival table are "
              f"missing on disk and will be skipped. First few:")
        for rel_path in missing[:5]:
            print(f"  - {rel_path}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"wsi": rel_paths}).to_csv(out_csv, index=False)
    return len(rel_paths)


def resolve_gpus(gpus_arg) -> list:
    """Resolve --gpus to device indices.

    'auto' expands to the visible CUDA devices (0..N-1), or [-1] (CPU) if none.
    """
    if [str(g).lower() for g in gpus_arg] == ["auto"]:
        try:
            import torch
            n = torch.cuda.device_count()
        except Exception:
            n = 0
        if n == 0:
            print("WARNING: no CUDA devices visible -- falling back to CPU.")
            return [-1]
        return list(range(n))
    return [int(g) for g in gpus_arg]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run TRIDENT feature extraction over the full slide dataset."
    )
    parser.add_argument(
        "--survival-table",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "matched_clinical_pilot100.csv",
        help="CSV with file_ids/file_names columns (default: pilot100).",
    )
    parser.add_argument(
        "--slides-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "slides" / "pilot100_slides",
        help="Directory containing the <file_id>/<file_name>.svs subdirectories.",
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "trident_full",
        help="Output directory for all TRIDENT artifacts (resume = same dir).",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["auto"],
        help="GPU indices to shard slides across, or 'auto' (default) to use "
             "every GPU SLURM allocated (via CUDA_VISIBLE_DEVICES). -1 = CPU.",
    )
    parser.add_argument(
        "--patch-encoder", default=PATCH_ENCODER,
        help=f"Patch encoder (default: {PATCH_ENCODER}).",
    )
    parser.add_argument(
        "--mag", type=int, default=TARGET_MAG,
        help=f"Target magnification (default: {TARGET_MAG}).",
    )
    parser.add_argument(
        "--patch-size", type=int, default=PATCH_SIZE,
        help=f"Patch size in px at target mag (default: {PATCH_SIZE}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build the slide list and print the command without running it.",
    )
    args = parser.parse_args()

    if not BATCH_RUNNER.exists():
        print(f"ERROR: TRIDENT batch runner not found at {BATCH_RUNNER}")
        return 1
    if not args.survival_table.exists():
        print(f"ERROR: survival table not found at {args.survival_table}")
        return 1
    if not args.slides_dir.exists():
        print(f"ERROR: slides directory not found at {args.slides_dir}")
        return 1

    args.job_dir.mkdir(parents=True, exist_ok=True)
    slide_list_csv = args.job_dir / "trident_slide_list.csv"
    n_slides = build_slide_list(args.survival_table, args.slides_dir, slide_list_csv)
    if n_slides == 0:
        print("ERROR: no slides found on disk for the survival table.")
        return 1
    print(f"Wrote slide list with {n_slides} slide(s) to {slide_list_csv}")

    gpus = resolve_gpus(args.gpus)
    print(f"Using GPU indices: {gpus}")

    cmd = [
        sys.executable, str(BATCH_RUNNER),
        "--task", "all",
        "--wsi_dir", str(args.slides_dir),
        "--custom_list_of_wsis", str(slide_list_csv),
        "--job_dir", str(args.job_dir),
        "--patch_encoder", args.patch_encoder,
        "--mag", str(args.mag),
        "--patch_size", str(args.patch_size),
        "--gpus", *[str(g) for g in gpus],
        "--skip_errors",
    ]

    print("Running:\n  " + " ".join(cmd))
    if args.dry_run:
        print("(dry run -- not executing)")
        return 0

    # Run from TRIDENT/ so its local `trident` package is importable.
    result = subprocess.run(cmd, cwd=str(TRIDENT_ROOT))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
