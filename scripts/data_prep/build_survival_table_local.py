"""
Build the survival table locally using the clinical data from GDC.

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
)

# Experiment/Run Name
COHORT          = "TCGA_GBMLGG"
ENCODER         = "uni_v2"
EXPERIMENT_NAME = f"{ENCODER}_{COHORT}"

DATA = PROJECT_ROOT / "data"

# Input Paths
SAMPLE_SHEET = DATA / "raw" / COHORT / "gdc_sample_sheet.2026-07-11.tsv"
CLINICAL_TSV = DATA / "raw" / COHORT / "clinical/clinical_supplement/clinical.tsv"
FEATURE_DIR  = DATA / "processed/features" / ENCODER / COHORT
# Output Paths
SURV_TABLE_CSV  = DATA / f"interim/survival_table_{COHORT}.csv"
EXP_DIR         = DATA / "processed/experiments" / EXPERIMENT_NAME
METADATA_CSV    = EXP_DIR / "survival_metadata.csv"
SPLIT_CSV       = EXP_DIR / "splits.csv"
# Log Directory
LOG_DIR = PROJECT_ROOT / "logs" / EXPERIMENT_NAME

# Seed
SEED = 42


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(SEED)
    print(f"Seed: {SEED}")

    print("Building patient-level survival table...")
    make_survival_metadata(
        sample_sheet_path=SAMPLE_SHEET,
        clinical_tsv_path=CLINICAL_TSV,
        out_path=SURV_TABLE_CSV,
    )
    print("Done.")
    print()
    print("Joining clinical table to feature bags...")
    attach_feature_paths(
        clinical_csv=SURV_TABLE_CSV,
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