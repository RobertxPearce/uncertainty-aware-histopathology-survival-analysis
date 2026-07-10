"""Nested cross-validation of ABMIL Cox (+ optional SurvRNC) over the whole cohort.

Motivation: a single 45-patient / 16-event holdout gives a C-index whose 95%
interval is ~0.28 wide it cannot tell a good model from a lucky split. This
script runs patient-level, event-stratified K-fold CV over all patients: every
patient is scored exactly once, in the fold where it is held out.

Nested, not flat. Inside each outer fold the inner grid is trained on that fold's
training patients and selected on its inner validation patients; only then is the
winner scored on the held-out fold. The outer fold never influences the choice,
so the reported number estimates the whole procedure (tune, then fit) rather
than a configuration that was chosen with the test data in view. The inner
selection may be noisy it is one seed on a small val split -- and that is
fine: a noisy-but-honest selector still yields an unbiased outer estimate,
because the noise is part of what is being measured.

What the inner grid sweeps: three loss/attention knobs left open after batch
size and architecture came back flat (a 4-architecture x 4-seed screen moved the
mean C-index by 0.018 against seed noise 0.028; a batch-size sweep over
{16, 32, 96} moved the inner val by 0.008):
  - lambda_rnc          SurvRNC strength; 0.0 is pure Cox, currently the best
                        result (Cox-only outer 0.591 vs Cox+SurvRNC 0.561).
  - temperature         attention sharpness; 0.05 sharpens, 1.0 is standard.
  - survrnc_temperature SurvRNC contrastive temperature; 0.1 vs 2.0.
The 0.05 attention temperature and 0.1 SurvRNC temperature are the partner
implementation's settings (it scored higher on a different data pipeline); this
grid ports those knobs into the honest nested-CV harness to see which actually
help here. Batch size is fixed at 32; architecture at the baseline.

Example:
    python scripts/training/cross_validate_abmil_cox_survrnc.py
"""

import json
import os
import sys
from functools import partial
from pathlib import Path

# GPU selection
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    bootstrap_cindex,
    build_model,
    build_optimizer,
    collate_bags,
    concordance_index,
    load_survival_table,
    make_cv_folds,
    make_datasets,
    make_dataloaders,
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
EXPERIMENT_NAME = f"abmil_cox_survrnc_cv_{ENCODER}_{COHORT}"
# Any table with all patients works; the existing split file is re-folded here,
# so its train/val/test labels are ignored and every patient is used.
SPLIT_CSV = PROJECT_ROOT / "data/processed/experiments/uni_v2_luad/splits.csv"
RUN_DIR = PROJECT_ROOT / "runs" / EXPERIMENT_NAME

# Cross-validation
N_SPLITS = 5
# Inner val carved from each fold's non-test patients: used for early stopping and
# for choosing the inner-grid winner. Its size sets how well that choice can be
# made. A random-risk model on one fold's val patients scores 0.5 +/- this much:
#     0.15 -> 54 patients, sd 0.086   (a coin flip between the candidates)
#     0.20 -> 73 patients, sd 0.068
#     0.25 -> 91 patients, sd 0.062   (matches the outer test fold's 0.056)
#     0.30 -> 109 patients, sd 0.056  (costs 55 training patients for +0.006)
# 0.25 buys a selector as precise as the thing being estimated, and stops there.
VAL_FRACTION = 0.25

# Runtime
DEVICE = "cuda"
SEED = 42
NUM_WORKERS = 4

# Data loading
FEATURE_KEY = "features"
MAX_PATCHES = 1024
# Fixed at the middle of the old inner grid. A prior nested run swept this over
# {16, 32, 96} and found no separation (inner val 0.601 / 0.609 / 0.604, spread
# 0.008 against selector noise ~0.062), so batch size is no longer a variable.
TRAIN_BATCH_SIZE = 32
# Val/test bags are uncapped and large (median ~15k patches), so this must stay
# much smaller than training.
EVAL_BATCH_SIZE = 4

# Fixed model settings. The architecture screen found no separation between
# baseline / input_norm / deep_proj / wide, so the smallest is kept. Attention
# `temperature` is NOT set here -- it is swept per candidate in the inner grid.
MODEL_CONFIG = {
    "input_dim": 1536,
    "embed_dim": 128,
    "attention_dim": 128,
    "gated": True,
    "dropout": 0.25,
}

# Inner grid: every candidate is trained on the fold's train patients and the
# winner is chosen on its val patients. Sweeps lambda_rnc x temperature x
# survrnc_temperature (see module docstring). When lambda_rnc == 0 there is no
# SurvRNC term, so its temperature is irrelevant and those redundant
# combinations are dropped (survrnc_temperature = None).
#
# Other partner-vs-ours differences are confounded in the partner's reported
# gain (no patch cap, wider projector, no dropout, weight_decay 1e-4); they are
# held fixed here to keep the grid readable and can be added as further knobs.
def _inner_grid():
    grid = []
    for lambda_rnc in (0.0, 0.1, 0.5):
        for temperature in (1.0, 0.05):
            survrnc_temps = (None,) if lambda_rnc == 0 else (0.1, 2.0)
            for survrnc_temperature in survrnc_temps:
                grid.append({
                    "lambda_rnc": lambda_rnc,
                    "temperature": temperature,
                    "survrnc_temperature": survrnc_temperature,
                })
    return grid


INNER_GRID = _inner_grid()


def config_label(candidate):
    """Short stable id for a grid candidate, used in logs and result tables."""
    label = f"rnc{candidate['lambda_rnc']:g}_T{candidate['temperature']:g}"
    if candidate["survrnc_temperature"] is not None:
        label += f"_st{candidate['survrnc_temperature']:g}"
    return label


# Optimization
OPTIMIZER = "adamw"
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-3
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
# No checkpoint is selected before this epoch. Without it the initialisation can
# win: in the architecture screen one run's best epoch was 1, four gradient steps.
MIN_EPOCHS = 3
GRAD_CLIP = 1.0


def make_loss_fn(lambda_rnc, survrnc_temperature):
    """Cox + lambda_rnc * SurvRNC, or plain Cox when lambda_rnc == 0."""
    if lambda_rnc == 0:
        return cox_loss_step
    return partial(
        survrnc_cox_loss,
        lambda_rnc=lambda_rnc,
        temperature=survrnc_temperature,
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
    """Build (train, val, test) loaders for one fold table from make_cv_folds.

    The fold table already carries train/val/test labels, so the standard
    make_datasets -> make_dataloaders path applies. Val and test bags are uncapped
    and scored at the smaller EVAL_BATCH_SIZE, mirroring the tuning script.
    """
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


def select_inner_config(fold_table, fold_dir, device):
    """
    Train every inner-grid candidate on this fold's train patients, pick on val.

    Sweeps the loss/attention grid (INNER_GRID). Returns (best, rows). The outer
    fold's test patients are never touched here -- that is the whole point of
    nesting. `rows` records every candidate so the selection can be audited.
    """
    best = None
    rows = []
    for candidate in INNER_GRID:
        label = config_label(candidate)
        lambda_rnc = candidate["lambda_rnc"]
        # Reseed and rebuild the loaders per candidate so weight init and data
        # order are identical across the grid; only the swept knobs differ.
        seed_everything(SEED)
        candidate_dir = fold_dir / f"inner_{label}"

        train_loader, val_loader, test_loader = make_fold_loaders(fold_table, SEED)
        model_config = {**MODEL_CONFIG, "temperature": candidate["temperature"]}
        model = build_model(**model_config)
        optimizer = build_optimizer(
            model, name=OPTIMIZER, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        loss_fn = make_loss_fn(lambda_rnc, candidate["survrnc_temperature"])

        steps_per_epoch = len(train_loader)
        print(f"  inner: {label} ({steps_per_epoch} steps/epoch)")
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
            checkpoint_dir=candidate_dir,
            model_config=model_config,
            grad_clip=GRAD_CLIP,
            verbose=False,
        )
        val_cindex = patient_cindex_on(model, val_loader, device)
        # history["best_cindex"] is the slide-level score train() actually selected
        # on. Printing both separates "the model never learned" from "the patient
        # aggregation or this val split is the problem".
        print(
            f"    best_epoch={history['best_epoch']} "
            f"({history['best_epoch'] * steps_per_epoch} steps)  "
            f"inner val C-index: slide={history['best_cindex']:.4f} "
            f"patient={val_cindex:.4f}"
        )

        rows.append({
            "config": label,
            "lambda_rnc": lambda_rnc,
            "temperature": candidate["temperature"],
            "survrnc_temperature": candidate["survrnc_temperature"],
            "steps_per_epoch": steps_per_epoch,
            "best_epoch": history["best_epoch"],
            "total_steps": history["best_epoch"] * steps_per_epoch,
            "first_epoch_val_cindex": history["val_cindex"][0],
            "final_epoch_val_cindex": history["val_cindex"][-1],
            "inner_val_slide_cindex": history["best_cindex"],
            "inner_val_patient_cindex": val_cindex,
        })
        if best is None or val_cindex > best["val_cindex"]:
            best = {
                "val_cindex": val_cindex,
                "config": label,
                "candidate": candidate,
                "model": model,
                "test_loader": test_loader,
                "history": history,
                "dir": candidate_dir,
            }

    return best, rows


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
        "inner_grid": INNER_GRID,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "model_config": MODEL_CONFIG,
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "min_epochs": MIN_EPOCHS,
        "grad_clip": GRAD_CLIP,
    }
    with open(RUN_DIR / "cv_config.json", "w") as handle:
        json.dump(settings, handle, indent=2, sort_keys=True)

    print(f"Device: {device}")
    print(f"Inner grid: {len(INNER_GRID)} candidates over "
          f"lambda_rnc x temperature x survrnc_temperature")
    print(f"{N_SPLITS}-fold CV over the full cohort (every patient held out once)")

    fold_rows = []
    inner_rows = []
    pooled_patient_predictions = []
    for fold_index, fold_table in make_cv_folds(
        table, n_splits=N_SPLITS, val_fraction=VAL_FRACTION, seed=SEED
    ):
        fold_dir = RUN_DIR / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[fold {fold_index + 1}/{N_SPLITS}]")
        best, candidate_rows = select_inner_config(fold_table, fold_dir, device)
        for row in candidate_rows:
            row["fold"] = fold_index
            row["selected"] = row["config"] == best["config"]
        inner_rows.extend(candidate_rows)

        model = best["model"]
        test_loader = best["test_loader"]
        history = best["history"]
        print(f"  selected config: {best['config']}")

        # Score the held-out fold -- data this model never saw or selected on.
        test_predictions = predict(model, test_loader, device)
        test_predictions.insert(0, "fold", fold_index)
        patient_predictions = aggregate_patient_predictions(test_predictions)
        patient_predictions.insert(0, "fold", fold_index)

        slide_cindex = concordance_index(
            test_predictions["risk"], test_predictions["time"], test_predictions["event"]
        )
        patient_cindex = concordance_index(
            patient_predictions["risk"],
            patient_predictions["time"],
            patient_predictions["event"],
        )

        test_predictions.to_csv(fold_dir / "test_slide_predictions.csv", index=False)
        patient_predictions.to_csv(fold_dir / "test_patient_predictions.csv", index=False)
        pooled_patient_predictions.append(patient_predictions)

        fold_rows.append(
            {
                "fold": fold_index,
                "selected_config": best["config"],
                "selected_lambda_rnc": best["candidate"]["lambda_rnc"],
                "selected_temperature": best["candidate"]["temperature"],
                "selected_survrnc_temperature": best["candidate"]["survrnc_temperature"],
                "inner_val_patient_cindex": best["val_cindex"],
                "best_epoch": history["best_epoch"],
                "n_test_patients": len(patient_predictions),
                "n_test_events": int(patient_predictions["event"].sum()),
                "slide_test_cindex": slide_cindex,
                "patient_test_cindex": patient_cindex,
                "checkpoint": str(best["dir"] / "best.pt"),
            }
        )
        print(
            f"held-out fold C-index: slide={slide_cindex:.4f} "
            f"patient={patient_cindex:.4f} "
            f"(n={len(patient_predictions)}, events={int(patient_predictions['event'].sum())})"
        )

    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(RUN_DIR / "cv_fold_summary.csv", index=False)

    inner_summary = pd.DataFrame(inner_rows)
    inner_summary.to_csv(RUN_DIR / "cv_inner_selection.csv", index=False)

    # Primary report: mean +/- std of the per-fold held-out C-index.
    patient_scores = fold_summary["patient_test_cindex"].to_numpy()
    slide_scores = fold_summary["slide_test_cindex"].to_numpy()
    patient_mean, patient_std = float(patient_scores.mean()), float(patient_scores.std(ddof=1))
    slide_mean, slide_std = float(slide_scores.mean()), float(slide_scores.std(ddof=1))

    # Secondary: pool the per-fold held-out patient predictions (each patient once,
    # whole cohort) and bootstrap a single CI. Ranks mix scores from the 5 fold
    # models, so read this alongside -- not instead of -- the mean +/- std above.
    pooled = pd.concat(pooled_patient_predictions, ignore_index=True)
    pooled.to_csv(RUN_DIR / "cv_pooled_patient_predictions.csv", index=False)
    pooled_point, pooled_lo, pooled_hi, n_valid = bootstrap_cindex(
        pooled, n_boot=1000, seed=SEED, alpha=0.05, group_col="case_id"
    )

    summary = {
        "experiment": EXPERIMENT_NAME,
        "n_splits": N_SPLITS,
        "inner_grid": INNER_GRID,
        "selected_config_per_fold": fold_summary["selected_config"].tolist(),
        "n_patients_total": int(pooled["case_id"].nunique()),
        "n_events_total": int(pooled["event"].sum()),
        "patient_cindex_mean": patient_mean,
        "patient_cindex_std": patient_std,
        "slide_cindex_mean": slide_mean,
        "slide_cindex_std": slide_std,
        "pooled_patient_cindex": pooled_point,
        "pooled_ci_low": pooled_lo,
        "pooled_ci_high": pooled_hi,
        "pooled_bootstrap_valid": n_valid,
        "per_fold_patient_cindex": patient_scores.tolist(),
    }
    with open(RUN_DIR / "cv_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nNested cross-validation complete")
    print(fold_summary[["fold", "selected_config", "best_epoch",
                        "n_test_patients", "n_test_events",
                        "inner_val_patient_cindex", "patient_test_cindex"]].to_string(index=False))

    # Inner val C-index by candidate, averaged over folds. This is what the
    # selector saw; it does not license a claim about the held-out data, but if
    # one config wins on most folds that is worth knowing.
    print("\nInner val C-index by candidate (mean over folds, best first):")
    print(
        inner_summary.groupby("config")
        .agg(slide=("inner_val_slide_cindex", "mean"),
             patient=("inner_val_patient_cindex", "mean"),
             total_steps=("total_steps", "mean"),
             times_selected=("selected", "sum"))
        .sort_values("patient", ascending=False)
        .to_string()
    )
    # If no candidate ever beats chance on the inner val, the selection below is
    # meaningless and so is any claim about SurvRNC strength.
    if inner_summary["inner_val_slide_cindex"].max() < 0.55:
        print(
            "\nWARNING: no inner candidate exceeded 0.55 slide C-index on its own "
            "validation split. The models are not learning; the fold scores below "
            "describe noise, not a survival signal."
        )
    chosen = fold_summary["selected_config"]
    if chosen.nunique() == 1:
        print(f"\nEvery fold selected {chosen.iloc[0]}.")
    else:
        print(
            f"\nFolds disagreed on the config ({chosen.tolist()}). The outer "
            "estimate below is still valid -- it measures the procedure, not one "
            "config -- but no single config is established as best."
        )
    print(f"\nPatient-level C-index: {patient_mean:.4f} +/- {patient_std:.4f} (mean +/- std over {N_SPLITS} folds)")
    print(f"Slide-level C-index:   {slide_mean:.4f} +/- {slide_std:.4f}")
    print(
        f"Pooled patient C-index: {pooled_point:.4f} "
        f"(95% CI {pooled_lo:.4f}-{pooled_hi:.4f}, {int(pooled['case_id'].nunique())} patients)"
    )
    print(f"\nSaved CV results to: {RUN_DIR}")


if __name__ == "__main__":
    main()
