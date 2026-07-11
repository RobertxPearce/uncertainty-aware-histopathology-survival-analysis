"""Screen ABMIL Cox + SurvRNC architectures under K-fold cross-validation.

Each architecture is evaluated over event-stratified K-fold CV: for every fold it
is trained on that fold's train patients (early-stopped on the fold's val
patients) and scored on the fold's held-out test patients, so every patient is
scored once per architecture. Architectures are ranked on the mean held-out fold
C-index - a stronger protocol than a single frozen val split, because the
estimate does not hinge on one lucky split.

Loss and attention knobs are held fixed here (lambda_rnc, SurvRNC temperature,
batch size); only the model shape varies. Those knobs are tuned separately, on
the chosen architecture, by tune_loss_abmil_cox_survrnc.py.

Example:
    python scripts/tune/screen_architecture_abmil_cox_survrnc.py
"""

import json
import sys
from functools import partial
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
    make_cv_folds,
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
COHORT = "TCGA_GBMLGG"
ENCODER = "uni_v2"
EXPERIMENT_NAME = f"abmil_cox_survrnc_arch_screen_{ENCODER}_{COHORT}"
# Any table with all patients works; the split file is re-folded here, so its
# train/val/test labels are ignored and every patient is used.
SPLIT_CSV = PROJECT_ROOT / "data/processed/experiments/uni_v2_TCGA_GBMLGG/splits.csv"
RUN_DIR = PROJECT_ROOT / "runs" / EXPERIMENT_NAME

# Cross-validation
N_SPLITS = 5
# Val carved from each fold's non-test patients, used only for early stopping.
VAL_FRACTION = 0.25

# Runtime
DEVICE = "cuda"
SEED = 42
NUM_WORKERS = 4

# Data loading
FEATURE_KEY = "features"
MAX_PATCHES = 1024
TRAIN_BATCH_SIZE = 96
# Val/test bags are uncapped and large (median ~15k patches); keep this small.
EVAL_BATCH_SIZE = 4

# Model settings shared by every architecture below.
MODEL_CONFIG = {
    "input_dim": 1536,
    "gated": True,
}

# Named architectures, each a set of overrides on MODEL_CONFIG.
ARCHITECTURES = {
    "baseline": dict(embed_dim=128, attention_dim=128),
    "baseline_input_norm": dict(embed_dim=128, attention_dim=128, input_norm=True),
    "deep_proj": dict(embed_dim=128, attention_dim=128, input_norm=True, hidden_dims=[512, 256]),
    "wide": dict(
        embed_dim=512,
        attention_dim=256,
        input_norm=True,
        pool_norm=False,
        risk_hidden_dim=256,
    ),
}

# Fixed loss/regularization knobs; only the architecture varies. These are tuned
# separately by tune_loss_abmil_cox_survrnc.py once an architecture is chosen.
DROPOUT = 0.25
LAMBDA_RNC = 0.05
SURVRNC_TEMPERATURE = 2.0

# Optimization
OPTIMIZER = "adamw"
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-3
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
# No checkpoint is selected before this epoch, so a lucky initialisation can't win.
MIN_EPOCHS = 3
GRAD_CLIP = 1.0


def make_loss_fn():
    """Cox + LAMBDA_RNC * SurvRNC, or plain Cox when LAMBDA_RNC == 0."""
    if LAMBDA_RNC == 0:
        return cox_loss_step
    return partial(
        survrnc_cox_loss,
        lambda_rnc=LAMBDA_RNC,
        temperature=SURVRNC_TEMPERATURE,
    )


def aggregate_patient_predictions(slide_predictions):
    """Average slide risks per patient, keeping one survival label per patient."""
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


def make_fold_loaders(fold_table, seed):
    """Build (train, val, test) loaders for one fold table from make_cv_folds."""
    datasets = make_datasets(
        fold_table,
        feature_key=FEATURE_KEY,
        max_patches=MAX_PATCHES,
        seed=seed,
    )
    train_loader, _, _ = make_dataloaders(
        datasets,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        generator=make_generator(seed),
        worker_init_fn=worker_init_fn,
    )
    eval_loaders = {}
    for name in ("val", "test"):
        eval_loaders[name] = DataLoader(
            datasets[name],
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_bags,
            worker_init_fn=worker_init_fn,
        )
    return train_loader, eval_loaders["val"], eval_loaders["test"]


def patient_cindex_on(model, loader, device):
    """Patient-level C-index of `model` on `loader` (slide risks averaged per case)."""
    predictions = predict(model, loader, device)
    patients = aggregate_patient_predictions(predictions)
    return concordance_index(patients["risk"], patients["time"], patients["event"])


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device(DEVICE)
    table = load_survival_table(SPLIT_CSV, project_root=PROJECT_ROOT)

    settings = {
        "cohort": COHORT,
        "encoder": ENCODER,
        "split_csv": str(SPLIT_CSV),
        "run_dir": str(RUN_DIR),
        "device": str(device),
        "seed": SEED,
        "n_splits": N_SPLITS,
        "val_fraction": VAL_FRACTION,
        "num_workers": NUM_WORKERS,
        "feature_key": FEATURE_KEY,
        "max_patches": MAX_PATCHES,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "model_config": MODEL_CONFIG,
        "architectures": ARCHITECTURES,
        "dropout": DROPOUT,
        "lambda_rnc": LAMBDA_RNC,
        "survrnc_temperature": SURVRNC_TEMPERATURE,
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "min_epochs": MIN_EPOCHS,
        "grad_clip": GRAD_CLIP,
    }
    with open(RUN_DIR / "arch_screen_config.json", "w") as handle:
        json.dump(settings, handle, indent=2, sort_keys=True)

    loss_fn = make_loss_fn()
    n_runs = len(ARCHITECTURES) * N_SPLITS
    print(f"Device: {device}")
    print(
        f"{N_SPLITS}-fold CV over the full cohort: "
        f"{len(ARCHITECTURES)} architectures x {N_SPLITS} folds = {n_runs} training runs"
    )

    # Fold the cohort once so every architecture is scored on the same splits.
    folds = list(make_cv_folds(table, n_splits=N_SPLITS, val_fraction=VAL_FRACTION, seed=SEED))

    rows = []
    for arch_name, overrides in ARCHITECTURES.items():
        print(f"\n[architecture: {arch_name}]")
        for fold_index, fold_table in folds:
            # Reseed per run so weight init and data order match across architectures.
            seed_everything(SEED)
            run_dir = RUN_DIR / arch_name / f"fold_{fold_index}"

            train_loader, val_loader, test_loader = make_fold_loaders(fold_table, SEED)
            model_config = {**MODEL_CONFIG, **overrides, "dropout": DROPOUT}
            model = build_model(**model_config)
            n_params = sum(p.numel() for p in model.parameters())
            optimizer = build_optimizer(
                model, name=OPTIMIZER, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )

            history = train(
                model,
                train_loader,
                val_loader,
                optimizer,
                loss_fn=loss_fn,
                epochs=EPOCHS,
                device=device,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                min_epochs=MIN_EPOCHS,
                checkpoint_dir=run_dir,
                model_config=model_config,
                grad_clip=GRAD_CLIP,
                verbose=False,
            )
            # Score the held-out fold -- data this model never trained or stopped on.
            test_cindex = patient_cindex_on(model, test_loader, device)

            rows.append({
                "architecture": arch_name,
                "fold": fold_index,
                "params": n_params,
                "best_epoch": history["best_epoch"],
                "slide_val_cindex": history["best_cindex"],
                "patient_test_cindex": test_cindex,
                "checkpoint": str(run_dir / "best.pt"),
            })
            pd.DataFrame(rows).to_csv(RUN_DIR / "arch_screen_trials.csv", index=False)
            print(
                f"  fold {fold_index + 1}/{N_SPLITS}: best_epoch={history['best_epoch']} "
                f"held-out patient C-index={test_cindex:.4f}"
            )

    trials_frame = pd.DataFrame(rows)

    # Rank architectures on the mean held-out fold C-index. std is the fold
    # noise; a gap between two architectures smaller than it means nothing.
    summary = (
        trials_frame.groupby(["architecture", "params"])
        .agg(
            mean_test_cindex=("patient_test_cindex", "mean"),
            std_test_cindex=("patient_test_cindex", "std"),
            min_test_cindex=("patient_test_cindex", "min"),
            max_test_cindex=("patient_test_cindex", "max"),
            folds=("fold", "count"),
        )
        .reset_index()
        .sort_values("mean_test_cindex", ascending=False, na_position="last")
    )
    summary.to_csv(RUN_DIR / "arch_screen_summary.csv", index=False)

    best = summary.iloc[0]
    print("\nArchitecture screen complete")
    print(
        summary[
            ["architecture", "params", "folds",
             "mean_test_cindex", "std_test_cindex", "min_test_cindex", "max_test_cindex"]
        ].to_string(index=False)
    )

    # A gap smaller than the fold noise is not a result.
    if len(summary) > 1:
        gap = best["mean_test_cindex"] - summary.iloc[1]["mean_test_cindex"]
        noise = trials_frame["patient_test_cindex"].std()
        print(f"\nTop-two gap: {gap:.4f} | fold noise (std over all runs): {noise:.4f}")
        if gap < noise:
            print(
                "The top two architectures are within fold noise of each other. Treat "
                "this as 'no architecture separated', not as a winner."
            )

    print(
        f"\nBest architecture: {best['architecture']} "
        f"({best['mean_test_cindex']:.4f} +/- {best['std_test_cindex']:.4f} over {int(best['folds'])} folds)"
    )
    print(f"\nSaved architecture-screen results to: {RUN_DIR}")


if __name__ == "__main__":
    main()
