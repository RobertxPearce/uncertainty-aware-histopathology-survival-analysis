"""Evaluate one trained ABMIL survival checkpoint on a frozen data split.

Edit the constants in the configuration section, then run:
    python scripts/eval/evaluate_abmil_model.py

The script reports both slide-level and patient-level Harrell C-index values
with patient-bootstrap confidence intervals. Patient risk is the mean risk
across all slides belonging to that patient.
"""

import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    bootstrap_cindex,
    collate_bags,
    load_survival_table,
    make_datasets,
    pick_device,
    predict,
    seed_everything,
)
from src.eval import load_model  # noqa: E402


# ----------------------------------------------------------------------
# Evaluation configuration
# ----------------------------------------------------------------------

# Model and data paths
EXPERIMENT_NAME = "abmil_cox_survrnc_evaluation_uni_v2_TCGA_LUAD"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "runs/abmil_cox_survrnc_tuning_uni_v2_TCGA_LUAD"
    / "dropout_0.25__lambda_rnc_0.1"
    / "best.pt"
)
SPLIT_CSV = PROJECT_ROOT / "data/processed/experiments/uni_v2_luad/splits.csv"
OUTPUT_DIR = PROJECT_ROOT / "results/evaluation" / EXPERIMENT_NAME

# Data split and feature settings
SPLIT = "test"
FEATURE_KEY = "features"
EVAL_BATCH_SIZE = 8
NUM_WORKERS = 4

# Runtime and statistics
DEVICE = "cuda"
SEED = 42
N_BOOTSTRAP = 1000
CI_ALPHA = 0.05


def aggregate_patient_predictions(slide_predictions):
    """Average slide risks while retaining one survival label per patient."""
    inconsistent = (
        slide_predictions.groupby("case_id")[["time", "event"]].nunique() > 1
    ).any(axis=1)
    if inconsistent.any():
        case_ids = inconsistent[inconsistent].index.tolist()
        raise ValueError(
            f"Patients have inconsistent survival labels across slides: {case_ids[:5]}"
        )

    return (
        slide_predictions.groupby("case_id", as_index=False)
        .agg(
            risk=("risk", "mean"),
            time=("time", "first"),
            event=("event", "first"),
            n_slides=("slide_id", "size"),
        )
    )


def score_predictions(predictions):
    """Return C-index and its patient-bootstrap confidence interval."""
    point, low, high, n_valid = bootstrap_cindex(
        predictions,
        n_boot=N_BOOTSTRAP,
        seed=SEED,
        alpha=CI_ALPHA,
        group_col="case_id",
    )
    return {
        "c_index": point,
        "ci_low": low,
        "ci_high": high,
        "n_bootstrap_valid": n_valid,
    }


def main():
    seed_everything(SEED)
    device = pick_device(DEVICE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Split: {SPLIT}")

    model, checkpoint = load_model(CHECKPOINT_PATH, device=device)
    table = load_survival_table(SPLIT_CSV, project_root=PROJECT_ROOT)
    datasets = make_datasets(table, feature_key=FEATURE_KEY, splits=(SPLIT,))
    if SPLIT not in datasets:
        raise ValueError(f"{SPLIT_CSV} contains no rows for split {SPLIT!r}.")

    loader = DataLoader(
        datasets[SPLIT],
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_bags,
    )

    slide_predictions = predict(model, loader, device)
    patient_predictions = aggregate_patient_predictions(slide_predictions)

    slide_metrics = score_predictions(slide_predictions)
    patient_metrics = score_predictions(patient_predictions)
    summary = {
        "checkpoint": str(CHECKPOINT_PATH),
        "trained_epoch": checkpoint.get("epoch"),
        "selected_validation_cindex": checkpoint.get("val_cindex"),
        "split": SPLIT,
        "n_slides": len(slide_predictions),
        "n_patients": len(patient_predictions),
        "n_patient_events": int(patient_predictions["event"].sum()),
        "slide_level": slide_metrics,
        "patient_level": patient_metrics,
    }

    slide_predictions.to_csv(OUTPUT_DIR / "slide_predictions.csv", index=False)
    patient_predictions.to_csv(OUTPUT_DIR / "patient_predictions.csv", index=False)
    with open(OUTPUT_DIR / "evaluation_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    confidence = int(round((1.0 - CI_ALPHA) * 100))
    print(
        f"Slide-level C-index: {slide_metrics['c_index']:.4f} "
        f"({confidence}% CI {slide_metrics['ci_low']:.4f}-"
        f"{slide_metrics['ci_high']:.4f})"
    )
    print(
        f"Patient-level C-index: {patient_metrics['c_index']:.4f} "
        f"({confidence}% CI {patient_metrics['ci_low']:.4f}-"
        f"{patient_metrics['ci_high']:.4f})"
    )
    print(f"Saved evaluation to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
