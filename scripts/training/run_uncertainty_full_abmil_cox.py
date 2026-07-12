"""Full uncertainty-method run: K-fold CV selection, then a default-split refit.

Two stages, one architecture (the arch-screen winner: baseline_input_norm + pure
Cox):

  1. Selection. Fold the cohort once, then for every fold train the shared
     backbones once (an ensemble of ABMIL members + one SNGP model) and score
     all three UQ methods on that fold's held-out patients:
         * MC-dropout     - stochastic passes through ensemble member 0
         * Deep ensemble  - spread across the ENSEMBLE_SIZE members
         * SNGP           - GP posterior variance from the SNGP model
     Every method sees identical folds, so their CV C-index / selective-AUC are
     directly comparable. The methods share each fold's trained backbones because
     they share the backbone by construction -- the CV measures the UQ estimator,
     not a fresh model per method.

  2. Refit. Take the method with the best CV patient C-index, retrain it once on
     the default train/val split from SPLIT_CSV, and report final test metrics
     (patient C-index + bootstrap CI, mean risk_std, selective AUC). Per-slide
     prediction frames are written so the comparison-plot script can render
     figures later.

The architecture and optimisation knobs match screen_architecture_abmil_cox_survrnc.py
so the numbers line up with the screen that chose this architecture.

Example:
    python scripts/training/run_uncertainty_full_abmil_cox.py
"""

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    build_model,
    build_optimizer,
    build_sngp_model,
    bootstrap_cindex,
    collate_bags,
    concordance_index,
    deep_ensemble_predict,
    fit_sngp_covariance,
    load_survival_table,
    make_cv_folds,
    make_dataloaders,
    make_datasets,
    make_generator,
    mc_dropout_predict,
    pick_device,
    seed_everything,
    sngp_predict,
    train,
    worker_init_fn,
)
from src.train.train_loop import cox_loss_step

# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ----------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------

# Experiment and paths
COHORT = "TCGA_GBMLGG"
ENCODER = "uni_v2"
EXPERIMENT_NAME = f"abmil_cox_uncertainty_full_{ENCODER}_{COHORT}"
SPLIT_CSV = PROJECT_ROOT / "data/processed/experiments/uni_v2_TCGA_GBMLGG/splits.csv"
RUN_DIR = PROJECT_ROOT / "runs" / EXPERIMENT_NAME

# Which UQ methods to screen (and choose between).
METHODS = ("mc_dropout", "deep_ensemble", "sngp")
# Metric used to pick the winning method: "c_index" (discrimination) or
# "selective_auc" (how useful the uncertainty is). Discrimination by default.
SELECTION_METRIC = "c_index"

# Cross-validation
N_SPLITS = 5
VAL_FRACTION = 0.25  # carved from each fold's non-test patients, early stopping only

# Runtime
DEVICE = "cuda"
SEED = 42
NUM_WORKERS = 4

# Data loading (matches the architecture screen)
FEATURE_KEY = "features"
MAX_PATCHES = 1024  # caps the train split only
TRAIN_BATCH_SIZE = 96  # Cox risk set = one batch -> prefer large batches
EVAL_BATCH_SIZE = 4  # val/test bags are uncapped and large; keep this small

# Architecture: the arch-screen winner (baseline_input_norm), pure Cox.
MODEL_CONFIG = dict(
    input_dim=1536,
    embed_dim=128,
    attention_dim=128,
    input_norm=True,
    gated=True,
    dropout=0.25,
)
LAMBDA_RNC = 0.0  # 0 = pure Cox (SurvRNC did not help on GBMLGG)
SURVRNC_TEMPERATURE = 2.0  # unused while LAMBDA_RNC == 0

# Optimization (matches the architecture screen)
OPTIMIZER = "adamw"
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-3
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
MIN_EPOCHS = 3
GRAD_CLIP = 1.0

# Uncertainty-method knobs
MC_DROPOUT_SAMPLES = 100  # stochastic forward passes per slide
ENSEMBLE_SIZE = 5  # independently trained members
SNGP_NUM_FEATURES = 1024  # random Fourier features in the GP head
SNGP_RIDGE_PENALTY = 1.0  # Laplace precision ridge
SNGP_NORM_BOUND = 0.95  # spectral-norm cap on the encoder
N_BOOT = 1000  # bootstrap resamples for the final C-index CI

# SNGP extends the shared architecture with its GP-head hyper-parameters.
SNGP_CONFIG = dict(
    **MODEL_CONFIG,
    num_features=SNGP_NUM_FEATURES,
    gp_ridge_penalty=SNGP_RIDGE_PENALTY,
    spectral_norm_bound=SNGP_NORM_BOUND,
)


# ----------------------------------------------------------------------
# Loss, data, scoring helpers
# ----------------------------------------------------------------------


def make_loss_fn():
    """Cox + LAMBDA_RNC * SurvRNC, or plain Cox when LAMBDA_RNC == 0."""
    if LAMBDA_RNC == 0:
        return cox_loss_step
    from src import survrnc_cox_loss

    return partial(
        survrnc_cox_loss, lambda_rnc=LAMBDA_RNC, temperature=SURVRNC_TEMPERATURE
    )


def make_split_loaders(table, seed):
    """(train, val, test) loaders from a table carrying train/val/test split labels.

    Works for both a make_cv_folds fold table and the raw SPLIT_CSV table, since
    both use the same split vocabulary. Train is capped/shuffled; val and test are
    uncapped and iterated in a stable order (needed so ensemble members line up).
    """
    datasets = make_datasets(
        table, feature_key=FEATURE_KEY, max_patches=MAX_PATCHES, seed=seed
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


def to_patient_frame(uq_df):
    """Collapse a per-slide UQ frame to one row per patient.

    Averages risk_mean and risk_std over a patient's slides and keeps the
    patient's single (time, event) label. On a one-slide-per-patient cohort this
    is a pass-through; it keeps the scoring correct if that ever changes.
    """
    inconsistent = (
        uq_df.groupby("case_id")[["time", "event"]].nunique() > 1
    ).any(axis=1)
    if inconsistent.any():
        bad = inconsistent[inconsistent].index.tolist()
        raise ValueError(f"Patients have inconsistent survival labels: {bad[:5]}")
    return uq_df.groupby("case_id", as_index=False).agg(
        risk_mean=("risk_mean", "mean"),
        risk_std=("risk_std", "mean"),
        time=("time", "first"),
        event=("event", "first"),
    )


def selective_auc(patient_df, min_patients=10, n_points=25):
    """Area under the selective-prediction curve (patient level).

    Drops the most-uncertain patients (highest risk_std) progressively and
    recomputes the C-index over the most-confident k; returns the normalised area
    under C-index-vs-coverage. Higher means the uncertainty is more informative
    (confident patients are ranked better). nan if too few comparable points.
    """
    d = patient_df.sort_values("risk_std").reset_index(drop=True)
    n = len(d)
    ks = np.unique(np.linspace(min(min_patients, n), n, n_points).astype(int))
    cov, cidx = [], []
    for k in ks:
        sub = d.iloc[:k]
        c = concordance_index(sub["risk_mean"], sub["time"], sub["event"])
        if np.isfinite(c):
            cov.append(k / n)
            cidx.append(c)
    if len(cov) < 2:
        return float("nan")
    return float(_trapz(cidx, cov) / (cov[-1] - cov[0]))


def score_uq_frame(uq_df):
    """Patient-level metrics for one UQ frame: C-index, selective AUC, mean std."""
    patients = to_patient_frame(uq_df)
    return {
        "c_index": concordance_index(
            patients["risk_mean"], patients["time"], patients["event"]
        ),
        "selective_auc": selective_auc(patients),
        "mean_risk_std": float(patients["risk_std"].mean()),
        "n_patients": len(patients),
        "n_events": int(patients["event"].sum()),
    }


# ----------------------------------------------------------------------
# Backbone training and per-method prediction
# ----------------------------------------------------------------------


def n_members_needed(methods):
    """How many ABMIL members the requested methods require.

    Deep ensemble needs the full ENSEMBLE_SIZE; MC-dropout needs only member 0;
    SNGP needs none. So the union is ENSEMBLE_SIZE if the ensemble is requested,
    else 1 if only MC-dropout is, else 0.
    """
    if "deep_ensemble" in methods:
        return ENSEMBLE_SIZE
    if "mc_dropout" in methods:
        return 1
    return 0


def fit_backbones(methods, train_loader, val_loader, device, checkpoint_dir, loss_fn):
    """Train exactly the backbones the requested methods need.

    Returns {"members": [...], "sngp": model_or_None, "histories": {...}}.
    Members get distinct seeds (SEED + i) for ensemble diversity; SNGP additionally
    has its GP covariance fit over the train loader so sngp_predict works.
    """
    fitted = {"members": [], "sngp": None, "histories": {}}

    for i in range(n_members_needed(methods)):
        seed_everything(SEED + i)  # different init -> ensemble diversity
        model = build_model(**MODEL_CONFIG)
        history = train(
            model,
            train_loader,
            val_loader,
            build_optimizer(model, OPTIMIZER, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
            loss_fn=loss_fn,
            epochs=EPOCHS,
            device=device,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            min_epochs=MIN_EPOCHS,
            checkpoint_dir=checkpoint_dir / f"member_{i}",
            model_config=MODEL_CONFIG,
            grad_clip=GRAD_CLIP,
            verbose=False,
        )
        fitted["members"].append(model)
        fitted["histories"][f"member_{i}"] = history

    if "sngp" in methods:
        seed_everything(SEED)
        sngp_model = build_sngp_model(**SNGP_CONFIG)
        history = train(
            sngp_model,
            train_loader,
            val_loader,
            build_optimizer(sngp_model, OPTIMIZER, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
            loss_fn=loss_fn,
            epochs=EPOCHS,
            device=device,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            min_epochs=MIN_EPOCHS,
            checkpoint_dir=checkpoint_dir / "sngp",
            model_config=SNGP_CONFIG,
            grad_clip=GRAD_CLIP,
            verbose=False,
        )
        # The GP covariance describes where the training data lives -> fit on train.
        fit_sngp_covariance(sngp_model, train_loader, device)
        fitted["sngp"] = sngp_model
        fitted["histories"]["sngp"] = history

    return fitted


def predict_method(method, fitted, loader, device):
    """Per-slide (risk_mean, risk_std) frame for one UQ method from fitted backbones."""
    if method == "mc_dropout":
        return mc_dropout_predict(
            fitted["members"][0], loader, device, n_samples=MC_DROPOUT_SAMPLES
        )
    if method == "deep_ensemble":
        return deep_ensemble_predict(fitted["members"], loader, device)
    if method == "sngp":
        return sngp_predict(fitted["sngp"], loader, device)
    raise ValueError(f"Unknown UQ method: {method!r}")


# ----------------------------------------------------------------------
# Stage 1: K-fold CV selection
# ----------------------------------------------------------------------


def run_cv_selection(table, device, loss_fn):
    """Score every method under shared K-fold CV; return (per-fold rows, pooled preds)."""
    folds = list(
        make_cv_folds(table, n_splits=N_SPLITS, val_fraction=VAL_FRACTION, seed=SEED)
    )
    rows = []
    pooled = {m: [] for m in METHODS}  # held-out patient frames, concatenated over folds

    for fold_index, fold_table in folds:
        print(f"\n[fold {fold_index + 1}/{N_SPLITS}]")
        train_loader, val_loader, test_loader = make_split_loaders(fold_table, SEED)
        fitted = fit_backbones(
            METHODS, train_loader, val_loader, device,
            RUN_DIR / "cv" / f"fold_{fold_index}", loss_fn,
        )
        for method in METHODS:
            uq_df = predict_method(method, fitted, test_loader, device)
            result = score_uq_frame(uq_df)
            rows.append({"method": method, "fold": fold_index, **result})
            pooled[method].append(to_patient_frame(uq_df))
            print(
                f"  {method:<14} held-out patient C-index={result['c_index']:.4f}  "
                f"selective_auc={result['selective_auc']:.4f}  "
                f"mean_risk_std={result['mean_risk_std']:.4f}"
            )
        pd.DataFrame(rows).to_csv(RUN_DIR / "cv_trials.csv", index=False)

    pooled_frames = {m: pd.concat(pooled[m], ignore_index=True) for m in METHODS}
    return pd.DataFrame(rows), pooled_frames


def summarize_cv(trials, pooled_frames):
    """Aggregate per-fold scores per method and add a pooled (whole-cohort) C-index."""
    summary = (
        trials.groupby("method")
        .agg(
            mean_cindex=("c_index", "mean"),
            std_cindex=("c_index", "std"),
            min_cindex=("c_index", "min"),
            max_cindex=("c_index", "max"),
            mean_selective_auc=("selective_auc", "mean"),
            mean_risk_std=("mean_risk_std", "mean"),
            folds=("fold", "count"),
        )
        .reset_index()
    )
    # Pooled C-index: every patient scored once across the folds, in one C-index.
    pooled_cindex = {
        m: concordance_index(f["risk_mean"], f["time"], f["event"])
        for m, f in pooled_frames.items()
    }
    summary["pooled_cindex"] = summary["method"].map(pooled_cindex)
    sort_key = "mean_cindex" if SELECTION_METRIC == "c_index" else "mean_selective_auc"
    return summary.sort_values(sort_key, ascending=False, na_position="last")


# ----------------------------------------------------------------------
# Stage 2: default-split refit of the winning method
# ----------------------------------------------------------------------


def refit_and_evaluate(method, table, device, loss_fn):
    """Retrain the winning method on the default split and report final test metrics."""
    train_loader, val_loader, test_loader = make_split_loaders(table, SEED)
    fitted = fit_backbones(
        (method,), train_loader, val_loader, device, RUN_DIR / "final", loss_fn
    )
    uq_df = predict_method(method, fitted, test_loader, device)
    uq_df.to_csv(RUN_DIR / f"final_{method}_predictions.csv", index=False)

    result = score_uq_frame(uq_df)
    # Patient-grouped bootstrap CI (bootstrap_cindex ranks a "risk" column).
    point, lo, hi, n_valid = bootstrap_cindex(
        uq_df.rename(columns={"risk_mean": "risk"}), n_boot=N_BOOT, seed=SEED
    )
    result.update(c_index=point, ci_low=lo, ci_high=hi, n_boot_valid=n_valid)
    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device(DEVICE)
    table = load_survival_table(SPLIT_CSV, project_root=PROJECT_ROOT)
    loss_fn = make_loss_fn()

    settings = {
        name: (str(value) if isinstance(value, Path) else value)
        for name, value in globals().items()
        if name.isupper() and not name.startswith("_")
    }
    with open(RUN_DIR / "run_config.json", "w") as handle:
        json.dump(settings, handle, indent=2, sort_keys=True, default=str)

    n_train = (n_members_needed(METHODS) + (1 if "sngp" in METHODS else 0)) * N_SPLITS
    print(f"Device: {device}")
    print(
        f"Objective: {'Cox' if LAMBDA_RNC == 0 else f'Cox + {LAMBDA_RNC}*SurvRNC'} | "
        f"architecture: baseline_input_norm ({MODEL_CONFIG['embed_dim']}d)"
    )
    print(
        f"Stage 1: {N_SPLITS}-fold CV over {len(METHODS)} methods "
        f"({n_train} backbone trainings)"
    )

    # ---- Stage 1: CV selection ----
    trials, pooled_frames = run_cv_selection(table, device, loss_fn)
    summary = summarize_cv(trials, pooled_frames)
    summary.to_csv(RUN_DIR / "cv_summary.csv", index=False)
    for method, frame in pooled_frames.items():
        frame.to_csv(RUN_DIR / f"cv_pooled_{method}_predictions.csv", index=False)

    print("\nCV selection complete")
    print(
        summary[
            ["method", "folds", "mean_cindex", "std_cindex",
             "pooled_cindex", "mean_selective_auc", "mean_risk_std"]
        ].to_string(index=False)
    )

    # A gap smaller than the fold noise is not a real separation.
    metric_col = "mean_cindex" if SELECTION_METRIC == "c_index" else "mean_selective_auc"
    best = summary.iloc[0]
    if len(summary) > 1:
        gap = best[metric_col] - summary.iloc[1][metric_col]
        noise = trials.groupby("method")["c_index"].std().mean()
        print(f"\nTop-two gap ({SELECTION_METRIC}): {gap:.4f} | fold noise (std): {noise:.4f}")
        if gap < noise:
            print(
                "The top two methods are within fold noise. They share the backbone, "
                "so treat this as 'no method separated on discrimination' -- decide on "
                "selective_auc / calibration / cost instead."
            )

    winner = best["method"]
    print(f"\nSelected method: {winner} ({SELECTION_METRIC}={best[metric_col]:.4f})")

    # ---- Stage 2: default-split refit + final metrics ----
    print(f"\nStage 2: refitting {winner} on the default train/val/test split...")
    final = refit_and_evaluate(winner, table, device, loss_fn)
    final_row = {"method": winner, **final}
    pd.DataFrame([final_row]).to_csv(RUN_DIR / "final_metrics.csv", index=False)

    ci = (
        f"95% CI {final['ci_low']:.4f}-{final['ci_high']:.4f}"
        if np.isfinite(final["ci_low"])
        else "CI unavailable"
    )
    print(
        f"\nFinal {winner} (test split, n={final['n_patients']} patients, "
        f"{final['n_events']} events):\n"
        f"  patient C-index : {final['c_index']:.4f}  ({ci})\n"
        f"  selective AUC   : {final['selective_auc']:.4f}\n"
        f"  mean risk std   : {final['mean_risk_std']:.4f}"
    )
    print(f"\nSaved full-run results to: {RUN_DIR}")


if __name__ == "__main__":
    main()
