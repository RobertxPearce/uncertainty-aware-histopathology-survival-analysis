"""Tune a regularized ABMIL Cox + SurvRNC model on one frozen train/val split.

This script intentionally does only model tuning: no uncertainty estimators,
test-set scoring, or plots. It trains a compact grid, evaluates each selected
checkpoint at the patient level on validation data, and writes a ranked CSV.

Example:
    python scripts/training/tune_abmil_cox_survrnc.py
"""

import json
import sys
from functools import partial
from itertools import product
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    build_model,
    build_optimizer,
    collate_bags,
    concordance_index,
    load_survival_table,
    make_dataloaders,
    make_datasets,
    make_generator,
    pick_device,
    predict,
    seed_everything,
    survrnc_cox_loss,
    train,
    worker_init_fn,
)
from src.train.train_loop import cox_loss_step


# ----------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------

# Experiment and paths
COHORT = "TCGA_LUAD"
ENCODER = "uni_v2"
EXPERIMENT_NAME = f"abmil_cox_survrnc_tuning_{ENCODER}_{COHORT}"
SPLIT_CSV = PROJECT_ROOT / "data/processed/experiments/uni_v2_luad/splits.csv"
RUN_DIR = PROJECT_ROOT / "runs" / EXPERIMENT_NAME

# Runtime
DEVICE = "cuda"
SEED = 42
NUM_WORKERS = 4

# Data loading
FEATURE_KEY = "features"
MAX_PATCHES = 1024
TRAIN_BATCH_SIZE = 96
# Validation bags are uncapped, so this must remain much smaller than training.
EVAL_BATCH_SIZE = 8

# Fixed model settings
MODEL_CONFIG = {
    "input_dim": 1536,
    "embed_dim": 128,
    "attention_dim": 128,
    "gated": True,
}

# Values whose Cartesian product will be tuned.
TUNING_GRID = {
    "dropout": [0.25, 0.40],
    "lambda_rnc": [0.0, 0.01, 0.05, 0.10],
}

# Optimization and loss
OPTIMIZER = "adamw"
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-3
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
GRAD_CLIP = 1.0
SURVRNC_TEMPERATURE = 2.0


def patient_level_cindex(predictions):
    """Average slide risks per patient, then calculate one-patient-one-vote C-index."""
    patient_predictions = (
        predictions.groupby("case_id", as_index=False)
        .agg(risk=("risk", "mean"), time=("time", "first"), event=("event", "first"))
    )
    score = concordance_index(
        patient_predictions["risk"],
        patient_predictions["time"],
        patient_predictions["event"],
    )
    return score, patient_predictions


def make_loaders(table, trial_seed):
    """Rebuild loaders per trial so shuffling and patch sampling are comparable."""
    datasets = make_datasets(
        table,
        feature_key=FEATURE_KEY,
        max_patches=MAX_PATCHES,
        seed=trial_seed,
    )
    train_loader, _val_loader, test_loader = make_dataloaders(
        datasets,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        generator=make_generator(trial_seed),
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_bags,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, test_loader


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device(DEVICE)
    table = load_survival_table(SPLIT_CSV, project_root=PROJECT_ROOT)

    if not {"train", "val"}.issubset(set(table["split"])):
        raise ValueError("The split CSV must contain both train and val rows.")

    settings = {
        "cohort": COHORT,
        "encoder": ENCODER,
        "split_csv": str(SPLIT_CSV),
        "run_dir": str(RUN_DIR),
        "device": str(device),
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "feature_key": FEATURE_KEY,
        "max_patches": MAX_PATCHES,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "model_config": MODEL_CONFIG,
        "tuning_grid": TUNING_GRID,
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "grad_clip": GRAD_CLIP,
        "survrnc_temperature": SURVRNC_TEMPERATURE,
    }
    with open(RUN_DIR / "tuning_config.json", "w") as handle:
        json.dump(settings, handle, indent=2, sort_keys=True)

    trials = list(product(TUNING_GRID["dropout"], TUNING_GRID["lambda_rnc"]))
    print(f"Device: {device}")
    print(f"Trials: {len(trials)} (test split will not be evaluated)")

    rows = []
    for trial_index, (dropout, lambda_rnc) in enumerate(trials):
        # Hold initialization, shuffling, and patch sampling constant so differences
        # between trials come from the hyperparameters rather than seed variance.
        trial_seed = SEED
        seed_everything(trial_seed)
        trial_name = f"dropout_{dropout:g}__lambda_rnc_{lambda_rnc:g}"
        trial_dir = RUN_DIR / trial_name

        train_loader, val_loader, _ = make_loaders(table, trial_seed)
        model_config = {
            **MODEL_CONFIG,
            "dropout": dropout,
        }
        model = build_model(**model_config)
        optimizer = build_optimizer(
            model,
            name=OPTIMIZER,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        loss_fn = (
            cox_loss_step
            if lambda_rnc == 0
            else partial(
                survrnc_cox_loss,
                lambda_rnc=lambda_rnc,
                temperature=SURVRNC_TEMPERATURE,
            )
        )

        print(f"\n[{trial_index + 1}/{len(trials)}] {trial_name}")
        history = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn=loss_fn,
            epochs=EPOCHS,
            device=device,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            checkpoint_dir=trial_dir,
            model_config=model_config,
            grad_clip=GRAD_CLIP,
        )

        val_predictions = predict(model, val_loader, device)
        patient_cindex, patient_predictions = patient_level_cindex(val_predictions)
        patient_predictions.to_csv(
            trial_dir / "val_patient_predictions.csv", index=False
        )

        row = {
            "trial": trial_name,
            "dropout": dropout,
            "lambda_rnc": lambda_rnc,
            "seed": trial_seed,
            "best_epoch": history["best_epoch"],
            "slide_val_cindex": history["best_cindex"],
            "patient_val_cindex": patient_cindex,
            "checkpoint": str(trial_dir / "best.pt"),
        }
        rows.append(row)
        pd.DataFrame(rows).sort_values(
            "patient_val_cindex", ascending=False, na_position="last"
        ).to_csv(RUN_DIR / "tuning_summary.csv", index=False)
        print(f"Patient-level val C-index: {patient_cindex:.4f}")

    summary = pd.DataFrame(rows).sort_values(
        "patient_val_cindex", ascending=False, na_position="last"
    )
    best = summary.iloc[0]
    print("\nTuning complete")
    print(summary[["trial", "best_epoch", "patient_val_cindex"]].to_string(index=False))
    print(f"\nBest checkpoint: {best['checkpoint']}")
    print("Keep the test split untouched until the tuning decision is final.")


if __name__ == "__main__":
    main()
