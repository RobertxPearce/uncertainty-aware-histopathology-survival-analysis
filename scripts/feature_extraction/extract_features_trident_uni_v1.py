"""
Extract TRIDENT UNI v1 patch features for every slide in a cohort.

Per slide the TRIDENT pipeline runs: tissue segmentation -> patch coords ->
UNI v1 patch features, writing one flat .h5 feature bag per slide.

Example Run on a GPU node:
    python scripts/feature_extraction/extract_features_trident_uni_v1.py \
        2>&1 | tee logs/trident_extract_$(date +%Y%m%d_%H%M%S).log

For SLURM job-array sharding submit with e.g. `sbatch --array=0-7`; the shard is
read from SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT (see SHARD_IDX below).
"""

import os
import sys
import glob
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRIDENT_ROOT = PROJECT_ROOT / "TRIDENT"

# Local TRIDENT must shadow any site-packages package of the same name; this
# must happen before `import trident`.
sys.path.insert(0, str(TRIDENT_ROOT))

# Keep downloaded model weights inside the repo (gitignored: model_cache/); must
# be set before importing trident/huggingface.
CACHE_ROOT = PROJECT_ROOT / "model_cache"
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))

import torch
from trident import OpenSlideWSI
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory


def resolve_device():
    """Pick cuda on the GPU nodes, mps on Apple Silicon, else cpu."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Cohort / Encoder
COHORT  = "TCGA_LUAD"
ENCODER = "uni_v1"

# Patching (uni_v1 expects 256px patches at 20x; segmentation runs at 10x).
TARGET_MAG = 20
PATCH_SIZE = 256
SEG_MAG    = 10

DATA = PROJECT_ROOT / "data"

# Input Paths
SLIDES_DIR = DATA / "raw/slides"                         # <file_id>/<file>.svs
# Output Paths (current repo layout, matches build_survival_table_local.py).
# geojson/work sit under trident/<cohort>/ with no encoder level because tissue
# segmentation and patch coords are encoder-independent (shared across uni_v1/v2).
FEATURES_DIR = DATA / "processed/features" / ENCODER / COHORT
GEOJSON_DIR  = DATA / "processed/trident" / COHORT / "geojson"
WORK_DIR     = DATA / "processed/trident" / COHORT / "work"

# Runtime
SEG_MODEL   = "hest"
DEVICE      = resolve_device()
NUM_WORKERS = 6        # DataLoader workers per stage; set 0 locally on macOS/MPS
CLEANUP     = False    # If True, delete TRIDENT's thumbnail/contour QC PNGs after the run
CUSTOM_MPP  = None     # Force microns-per-pixel when a slide header lacks MPP

# SLURM job-array sharding (env-driven; defaults to a single shard = all slides).
SHARD_IDX = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
N_SHARDS  = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))


def main():
    for d in (FEATURES_DIR, GEOJSON_DIR, WORK_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Cohort: {COHORT} | Encoder: {ENCODER} | Device: {DEVICE}")
    print(f"Slides dir  : {SLIDES_DIR}")
    print(f"Features dir: {FEATURES_DIR}")
    print(f"Geojson dir : {GEOJSON_DIR}")
    print(f"Work dir    : {WORK_DIR}")
    print()

    # Recurse into <file_id>/ subdirs to find every slide under SLIDES_DIR.
    svs_list = sorted(glob.glob(str(SLIDES_DIR / "**" / "*.svs"), recursive=True))
    if N_SHARDS > 1:
        svs_list = svs_list[SHARD_IDX::N_SHARDS]
        print(f"Shard {SHARD_IDX}/{N_SHARDS}: this task owns {len(svs_list)} slide(s).")
    print(f"Found {len(svs_list)} slide(s) to consider.")

    # Skip slides that already have a feature bag (resume-friendly).
    todo = [p for p in svs_list if not (FEATURES_DIR / (Path(p).stem + ".h5")).exists()]
    print(f"{len(svs_list) - len(todo)} already extracted; {len(todo)} to process.")
    print()
    if not todo:
        print("Everything is already extracted. Done.")
        return

    print("Loading segmentation + patch encoder models...")
    segmentation_model = segmentation_model_factory(SEG_MODEL)
    patch_encoder = encoder_factory(ENCODER).eval().to(DEVICE)

    total = len(todo)
    failures = []
    for i, svs_path in enumerate(todo, 1):
        name = os.path.basename(svs_path)
        print(f"[{i}/{total}] {name}")
        try:
            wsi_kwargs = {"mpp": CUSTOM_MPP} if CUSTOM_MPP is not None else {}
            slide = OpenSlideWSI(slide_path=svs_path, lazy_init=False,
                                 max_workers=NUM_WORKERS, **wsi_kwargs)

            print("\t-> 1/3 segmentation...")
            slide.segment_tissue(segmentation_model=segmentation_model,
                                 target_mag=SEG_MAG, job_dir=str(WORK_DIR), device=DEVICE)

            print("\t-> 2/3 patch coordinates...")
            coords_path = slide.extract_tissue_coords(target_mag=TARGET_MAG,
                                                      patch_size=PATCH_SIZE, save_coords=str(WORK_DIR))

            print("\t-> 3/3 feature extraction (.h5)...")
            slide.extract_patch_features(patch_encoder=patch_encoder, coords_path=coords_path,
                                         save_features=str(FEATURES_DIR), device=DEVICE)

            # Collect this slide's segmentation geojson next to the features.
            # TRIDENT writes it to <work>/contours_geojson/<stem>.geojson. Best-
            # effort: a failed move must not fail an otherwise-good slide.
            try:
                src_geojson = WORK_DIR / "contours_geojson" / f"{Path(svs_path).stem}.geojson"
                if src_geojson.exists():
                    shutil.move(str(src_geojson), str(GEOJSON_DIR / src_geojson.name))
            except Exception as move_err:
                print(f"\t-> geojson not collected: {move_err}")

            print("\t-> OK")
        except Exception as e:
            print(f"\t-> FAILED: {e}")
            failures.append((name, str(e)))

    # Drop the visualization PNGs TRIDENT writes; we only keep geojson/h5.
    if CLEANUP:
        for sub in ("thumbnails", "contours"):
            shutil.rmtree(WORK_DIR / sub, ignore_errors=True)

    print()
    print(f"Done. Success: {total - len(failures)}/{total}.")
    for name, err in failures:
        print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
