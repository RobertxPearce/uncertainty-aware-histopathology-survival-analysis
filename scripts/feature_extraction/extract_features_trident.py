#!/usr/bin/env python
# extract_features_trident.py
#
# Shared, cluster-agnostic worker for TRIDENT UNI feature extraction.
#
# Runs the TRIDENT pipeline (segmentation -> patch coords -> UNI patch features)
# over every .svs slide under the slides directory, writing one flat .h5 feature
# file per slide.
#
# All cluster-specific settings live in the paired SLURM wrappers:
#   scripts/feature_extraction/slurm/extract_features.rebelx.sh
#
# Slide layout on disk (one directory per GDC file_id):
#     <slides_dir>/<file_id>/<file_name>.svs

import os
import sys
import glob
import time
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime

# repo root is two levels up: scripts/feature_extraction/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRIDENT_ROOT = PROJECT_ROOT / "TRIDENT"

# Make the local TRIDENT shadow the site-packages package of the
# same name. Must happen before `import trident`.
sys.path.insert(0, str(TRIDENT_ROOT))

# Keep downloaded model weights inside the repo (gitignored: model_cache/)
_CACHE_ROOT = PROJECT_ROOT / "model_cache"
os.environ.setdefault("HF_HUB_CACHE", str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "torch"))

import torch
from trident import OpenSlideWSI
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory

# uni_v1 expects 256px patches at 20x; segmentation is run at 10x.
PATCH_ENCODER = "uni_v1"
TARGET_MAG = 20
PATCH_SIZE = 256
SEG_MAG = 10

DEFAULT_SLIDES_DIR = PROJECT_ROOT / "data" / "raw" / "slides"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "features_uni_v1_full"
DEFAULT_FEATURES_DIR = DEFAULT_OUTPUT_DIR / "features"
DEFAULT_GEOJSON_DIR = DEFAULT_OUTPUT_DIR / "geojson"
DEFAULT_WORK_DIR = DEFAULT_OUTPUT_DIR / "work"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def resolve_device(choice: str) -> str:
    """Resolve --device 'auto' to cuda/mps/cpu, or pass an explicit choice through."""
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / datetime.now().strftime("trident_extract_%Y%m%d_%H%M%S.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_path}")
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TRIDENT UNI feature extraction over all slides under slides-dir. "
                    "Cluster-agnostic; driven by scripts/feature_extraction/slurm/extract_features.<cluster>.sh."
    )
    p.add_argument("--slides-dir", type=Path, default=DEFAULT_SLIDES_DIR,
                   help=f"Root holding <file_id>/<file>.svs (default: {DEFAULT_SLIDES_DIR}).")
    p.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR,
                   help="Output dir for flat per-slide .h5 feature files.")
    p.add_argument("--geojson-dir", type=Path, default=DEFAULT_GEOJSON_DIR,
                   help="Output dir for per-slide tissue-segmentation .geojson files.")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
                   help="Dir for throwaway intermediates (patch coords, thumbnails, contours).")
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                   help="Dir for run logs.")
    p.add_argument("--patch-encoder", default=PATCH_ENCODER,
                   help=f"Patch encoder (default: {PATCH_ENCODER}).")
    p.add_argument("--mag", type=int, default=TARGET_MAG,
                   help=f"Target magnification for patching (default: {TARGET_MAG}).")
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE,
                   help=f"Patch size in px at target mag (default: {PATCH_SIZE}).")
    p.add_argument("--seg-mag", type=int, default=SEG_MAG,
                   help=f"Magnification for tissue segmentation (default: {SEG_MAG}).")
    p.add_argument("--device", default="auto",
                   help="cuda:0 | mps | cpu | auto (default: auto).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers for all stages. On CUDA pass a positive "
                        "value (the SLURM wrappers use 6) for parallel loading. 0 "
                        "disables multiprocessing (single-process; also avoids fork() "
                        "worker segfaults on macOS/MPS). -1 lets TRIDENT auto-scale. "
                        "Default: 0.")
    p.add_argument("--shard", default=None,
                   help="Process only shard i of n, format 'i/n' (0-indexed), "
                        "e.g. '0/8'. Slides are split deterministically by sorted "
                        "order, so shards are disjoint. Use with SLURM job arrays "
                        "to run one GPU per shard in parallel.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N slides (smoke test).")
    p.add_argument("--custom-mpp", type=float, default=None,
                   help="Force microns-per-pixel for every slide in this run, "
                        "bypassing metadata. Use only when a slide's header is "
                        "missing MPP (OpenSlide 'Unable to extract MPP'); a full "
                        "re-run with this set re-processes just the failed slides, "
                        "since already-extracted ones are skipped. Must match the "
                        "slides' true scale (Aperio 40x=0.25, 20x=0.50).")
    p.add_argument("--no-cleanup", action="store_true",
                   help="Keep thumbnail/contour PNGs (deleted by default).")
    p.add_argument("--dry-run", action="store_true",
                   help="List work to do and exit without loading models.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = build_logger(args.log_dir)

    logger.info("TRIDENT feature extraction started (outputs: .h5, .geojson, logs).")
    logger.info(f"Slides dir  : {args.slides_dir}")
    logger.info(f"Features dir: {args.features_dir}")
    logger.info(f"Geojson dir : {args.geojson_dir}")
    logger.info(f"Work dir    : {args.work_dir}")

    if not args.slides_dir.exists():
        logger.error(f"Slides dir not found: {args.slides_dir}")
        return 1

    args.features_dir.mkdir(parents=True, exist_ok=True)
    args.geojson_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # Recurse into <file_id>/ subdirectories to find every slide under slides-dir.
    svs_list = sorted(glob.glob(str(args.slides_dir / "**" / "*.svs"), recursive=True))
    logger.info(f"Found {len(svs_list)} .svs slide(s) under {args.slides_dir}.")
    if not svs_list:
        logger.warning("No .svs files found. Nothing to do.")
        return 0

    if args.shard is not None:
        try:
            shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            logger.error(f"--shard must be 'i/n' (e.g. '0/8'); got {args.shard!r}")
            return 1
        if not (0 <= shard_i < shard_n):
            logger.error(f"--shard i/n requires 0 <= i < n; got {args.shard!r}")
            return 1
        svs_list = svs_list[shard_i::shard_n]
        logger.info(f"Shard {shard_i}/{shard_n}: this task owns {len(svs_list)} slide(s).")

    if args.limit is not None:
        svs_list = svs_list[: args.limit]
        logger.info(f"--limit set: processing first {len(svs_list)} slide(s).")

    # Split into work-to-do vs already-done for a clean report / resume.
    todo, already = [], 0
    for svs_path in svs_list:
        h5_path = args.features_dir / (Path(svs_path).stem + ".h5")
        if h5_path.exists():
            already += 1
        else:
            todo.append(svs_path)
    logger.info(f"{already} slide(s) already have features; {len(todo)} to process.")

    if args.dry_run:
        for svs_path in todo[:10]:
            logger.info(f"  would process: {Path(svs_path).name}")
        if len(todo) > 10:
            logger.info(f"  ... and {len(todo) - 10} more")
        logger.info("(dry run -- exiting before loading models)")
        return 0

    if not todo:
        logger.info("Everything is already extracted. Done.")
        return 0

    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")

    # max_workers feeds get_num_workers() for every stage (segment, patch coords,
    # feature extraction). On CUDA a positive value enables parallel DataLoader
    # loading; 0 disables multiprocessing entirely (single-process, and avoids
    # fork()ed-worker segfaults on macOS/MPS). -1 -> None -> TRIDENT auto-scales.
    max_workers = None if args.num_workers < 0 else args.num_workers
    logger.info(f"DataLoader workers (max_workers): "
                f"{'auto' if max_workers is None else max_workers}")

    logger.info("Loading segmentation + patch encoder models...")
    try:
        segmentation_model = segmentation_model_factory("hest")
        patch_encoder = encoder_factory(args.patch_encoder).eval().to(device)
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return 1

    work_dir = str(args.work_dir)
    features_dir = str(args.features_dir)
    total = len(todo)
    success = 0
    failures = []

    for i, svs_path in enumerate(todo, 1):
        filename = os.path.basename(svs_path)
        logger.info(f"[{i}/{total}] {filename}")
        start = time.time()
        try:
            wsi_kwargs = {"mpp": args.custom_mpp} if args.custom_mpp is not None else {}
            slide = OpenSlideWSI(slide_path=svs_path, lazy_init=False,
                                 max_workers=max_workers, **wsi_kwargs)

            logger.info("\t-> 1/3 segmentation...")
            slide.segment_tissue(
                segmentation_model=segmentation_model,
                target_mag=args.seg_mag,
                job_dir=work_dir,
                device=device,
            )

            logger.info("\t-> 2/3 patch coordinates...")
            coords_path = slide.extract_tissue_coords(
                target_mag=args.mag,
                patch_size=args.patch_size,
                save_coords=work_dir,
            )

            logger.info("\t-> 3/3 UNI feature extraction (.h5)...")
            slide.extract_patch_features(
                patch_encoder=patch_encoder,
                coords_path=coords_path,
                save_features=features_dir,
                device=device,
            )

            # Collect this slide's segmentation geojson next to the features.
            # TRIDENT writes it to <work_dir>/contours_geojson/<stem>.geojson.
            # Best-effort: a missing/failed move must not fail an otherwise-good slide.
            try:
                stem = Path(svs_path).stem
                src_geojson = args.work_dir / "contours_geojson" / f"{stem}.geojson"
                if src_geojson.exists():
                    shutil.move(str(src_geojson), str(args.geojson_dir / f"{stem}.geojson"))
            except Exception as move_err:
                logger.warning(f"\t-> geojson not collected: {move_err}")

            logger.info(f"\t-> OK ({time.time() - start:.1f}s)")
            success += 1
        except Exception as e:
            logger.error(f"\t-> FAILED: {e}")
            failures.append((filename, str(e)))

    # Drop the visualization PNGs TRIDENT writes; we only keep geojson/h5.
    if not args.no_cleanup:
        for sub in ("thumbnails", "contours"):
            path = args.work_dir / sub
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

    logger.info("=" * 60)
    logger.info(f"Done. Success: {success}/{total} "
                f"(+{already} previously extracted = "
                f"{success + already}/{success + already + len(failures)} total).")
    if failures:
        logger.warning(f"{len(failures)} slide(s) failed:")
        for name, err in failures:
            logger.warning(f"  - {name}: {err}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
