"""
Build the survival table locally using the TCGA-LUAD clinical data from GDC.

make_survival_metadata  (patient-level survival table)
attach_feature_paths    (join to .h5 feature bags)
make_splits             (frozen patient-level splits)

Example Run on Head Node:
    python scripts/data_prep/build_survival_table_local.py 2>&1 | tee logs/data_prep/build_$(date +%Y%m%d_%H%M%S).log
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    make_survival_metadata,
    attach_feature_paths,
    make_splits,
    load_survival_table,
    seed_everything,
    make_generator,
    pick_device,
)

# Experiment/Run Name
EXPERIMENT_NAME = "uni_v1_luad"

# Input Paths
SAMPLE_SHEET = PROJECT_ROOT / "data/raw/gdc_sample_sheet.2026-06-24.tsv"
CLINICAL_TSV = PROJECT_ROOT / "data/raw/clinical/clinical_supplement/clinical.tsv"
FEATURE_DIR = PROJECT_ROOT / "data/processed/features/uni_v1/TCGA_LUAD"
# Output Paths
RUN_DIR = PROJECT_ROOT / "runs" / EXPERIMENT_NAME
CLINICAL_CSV = RUN_DIR / "clinical_survival.csv"
METADATA_CSV = RUN_DIR / "survival_metadata.csv"
SPLIT_CSV = RUN_DIR / "splits.csv"
# Log Directory
LOG_DIR = PROJECT_ROOT / "logs" / EXPERIMENT_NAME

# Seed
SEED = 42

# Data Loading
FEATURE_KEY = "features"   # Dataset key inside each .h5 bag
MAX_PATCHES = 4096         # Cap patches per bag on the train split only
BATCH_SIZE = 16            # Cox risk set is the batch -> prefer larger batches
NUM_WORKERS = 0


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    seed_everything(SEED)
    print(f"Seed: {SEED}")
    
    print("Building patient-level survival table...")
    make_survival_metadata(
        sample_sheet_path=SAMPLE_SHEET,
        clinical_tsv_path=CLINICAL_TSV,
        out_path=CLINICAL_CSV,
    )
    print("Done.")
    print()
    print("Joining clinical table to feature bags...")
    attach_feature_paths(
        clinical_csv=CLINICAL_CSV,
        feature_dir=FEATURE_DIR,
        out_path=METADATA_CSV,
    )
    print("Done.")
    print()
    print("Freezing patient-level, stratified train/val/test splits...")
    make_splits(
        metadata_csv=METADATA_CSV,
        out_path=SPLIT_CSV,
        val_fraction=0.20,
        test_fraction=0.10,
        seed=SEED,
    )
    print("Done.")
    print()
    print("Loading patient-level survival table...")
    table = load_survival_table(SPLIT_CSV)
    print(f"Slides: {len(table)} | Patients: {table['case_id'].nunique()}")
    print(f"Split Sizes:\n{table['split'].value_counts()}")
    print()

if __name__ == "__main__":
    main()