"""Deep-ensemble uncertainty for the ABMIL + Cox model, in two phases.

Sibling of mc_dropout_abmil_cox.py -- same cohort, folds, architecture, and
optimisation, so the two methods are directly comparable. The ONLY difference is
the uncertainty mechanism: here we train ENSEMBLE_SIZE independent members
(different seeds) and read the uncertainty off their disagreement, rather than
sampling dropout from one model.

    Phase 1 -- 5-fold cross-validation
        Fold the cohort once (make_cv_folds), and for every fold train an
        ensemble of members and run the ensemble on that fold's held-out
        patients. Every patient is scored exactly once across the folds.

    Phase 2 -- frozen-split refit
        Train one ensemble on the frozen train/val split stored in the CSV, then
        score it on that CSV's held-out test set. The single-member (member 0)
        C-index is reported alongside the ensemble as a "point prediction"
        baseline.

Comparison contract (keep identical across the three UQ scripts):
    * make_cv_folds(table, n_splits=5, val_fraction=0.25, seed=42)
    * MODEL_CONFIG + optimizer block below
    * predictions are scored patient-level with the same concordance_index /
      selective_auc / bootstrap_cindex helpers.
Only compare the methods on scale-free quantities (C-index, selective-AUC); the
raw risk_std magnitudes are NOT commensurable across methods.

Example:
    python scripts/training/deep_ensembles_abmil_cox.py
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
    bootstrap_cindex,
    build_model,
    build_optimizer,
    collate_bags,
    concordance_index,
    deep_ensemble_predict,
    evaluate,
    load_survival_table,
    make_cv_folds,
    make_dataloaders,
    make_datasets,
    make_generator,
    pick_device,
    save_checkpoint,
    seed_everything,
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
SPLIT_CSV = PROJECT_ROOT / "data/processed/experiments/uni_v2_TCGA_GBMLGG/splits.csv"
RESULT_DIR = PROJECT_ROOT / "results" / "deep_ensembles_abmil_cox_tcga_gbmlgg"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"  # one .pt per member (folds + frozen split)

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
    dropout=0.25,  # training-time regularization; inference is deterministic per member
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

# Deep-ensemble / scoring knobs
ENSEMBLE_SIZE = 5  # independently trained members; disagreement = the uncertainty
N_BOOT = 1000  # bootstrap resamples for the pooled / test-set C-index CI


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
    """(train, val, test) loaders for a table carrying a train/val/test split.

    Works for both phases: make_cv_folds writes the same train/val/test
    vocabulary as the frozen CSV, so one loader builder serves both. Train is
    capped/shuffled; val and test are uncapped and iterated in a stable order
    (needed so the ensemble members line up per slide).
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
        if name not in datasets:
            eval_loaders[name] = None
            continue
        eval_loaders[name] = DataLoader(
            datasets[name],
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_bags,
            worker_init_fn=worker_init_fn,
        )
    return train_loader, eval_loaders["val"], eval_loaders["test"]


def fit_model(train_loader, val_loader, device, loss_fn, seed):
    """Train one ABMIL member (best-epoch weights kept in memory).

    Returns (model, history, optimizer); the optimizer is handed back so the
    caller can persist a resumable checkpoint via save_model / save_checkpoint.
    """
    seed_everything(seed)
    model = build_model(**MODEL_CONFIG)
    optimizer = build_optimizer(model, OPTIMIZER, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
        grad_clip=GRAD_CLIP,
        verbose=False,
    )
    return model, history, optimizer


def fit_ensemble(train_loader, val_loader, device, loss_fn, base_seed):
    """Train ENSEMBLE_SIZE members with distinct seeds (base_seed + m).

    Distinct seeds give the members different initialisations (and data-shuffle
    order), which is the entire source of ensemble diversity / disagreement.
    Returns (models, histories, optimizers), each a length-ENSEMBLE_SIZE list.
    """
    models, histories, optimizers = [], [], []
    for m in range(ENSEMBLE_SIZE):
        model, history, optimizer = fit_model(
            train_loader, val_loader, device, loss_fn, base_seed + m
        )
        models.append(model)
        histories.append(history)
        optimizers.append(optimizer)
    return models, histories, optimizers


def save_model(model, optimizer, history, path):
    """Checkpoint a trained member so src.eval.load_model can rebuild and score it.

    Stores MODEL_CONFIG under the "config" key (the build_model kwargs), so the
    architecture round-trips exactly; best epoch / val C-index come from the
    train() history. This is the same checkpoint format the CLI trainer writes.
    """
    save_checkpoint(
        path,
        model,
        optimizer,
        history.get("best_epoch", 0),
        history.get("best_cindex", float("nan")),
        MODEL_CONFIG,
    )


def to_patient_frame(uq_df):
    """Collapse a per-slide UQ frame to one row per patient (mean risk; its label)."""
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


def selective_curve(patient_df, min_patients=10, n_points=25):
    """C-index vs coverage as the most-uncertain patients are dropped.

    Sorts by ascending risk_std (most confident first) and recomputes the C-index
    over the most-confident k. Returns (coverage, cindex) arrays; points with no
    comparable pairs are skipped.
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
    return np.array(cov), np.array(cidx)


def selective_auc(patient_df):
    """Normalised area under the selective-prediction curve (higher = more useful)."""
    cov, cidx = selective_curve(patient_df)
    if len(cov) < 2:
        return float("nan")
    return float(_trapz(cidx, cov) / (cov[-1] - cov[0]))


def histories_to_long_frame(histories):
    """Flatten {source: train() history} into a tidy per-epoch frame for plotting.

    Each training run early-stops at a different epoch, so the per-run lists have
    different lengths; long form (one row per source-epoch) sidesteps that. The
    selected (best-val) epoch is flagged so a plot can mark it.
    """
    rows = []
    for source, h in histories.items():
        train_loss = h.get("train_loss", [])
        val_loss = h.get("val_loss", [])
        val_cindex = h.get("val_cindex", [])
        best_epoch = h.get("best_epoch")
        for i in range(len(train_loss)):
            epoch = i + 1
            rows.append({
                "source": source,
                "epoch": epoch,
                "train_loss": train_loss[i],
                "val_loss": val_loss[i] if i < len(val_loss) else float("nan"),
                "val_cindex": val_cindex[i] if i < len(val_cindex) else float("nan"),
                "is_best_epoch": epoch == best_epoch,
            })
    return pd.DataFrame(
        rows,
        columns=["source", "epoch", "train_loss", "val_loss", "val_cindex", "is_best_epoch"],
    )


def save_histories(histories, out_dir):
    """Persist training histories as a tidy CSV (for plotting) and raw JSON."""
    histories_to_long_frame(histories).to_csv(out_dir / "learning_curves.csv", index=False)
    with open(out_dir / "histories.json", "w") as handle:
        json.dump(histories, handle, indent=2)


def split_counts(table):
    """(name, patients, slides, events) per train/val/test in a resolved table."""
    out = []
    for name in ("train", "val", "test"):
        sub = table[table["split"] == name]
        if sub.empty:
            continue
        per_patient = sub.groupby("case_id")["event"].max()
        out.append((name, len(per_patient), len(sub), int(per_patient.sum())))
    return out


def score_uq_frame(uq_df):
    """Patient-level metrics for one ensemble frame: C-index, selective AUC, mean std."""
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
# Phase 1: 5-fold cross-validation
# ----------------------------------------------------------------------


def run_cross_validation(table, device, loss_fn):
    """Deep ensemble under shared K-fold CV.

    Returns (trials, pooled_patients, histories):
        trials         - one row per fold with that fold's held-out metrics
        pooled_patients- held-out patient frames concatenated over folds
                         (every patient scored once)
        histories      - {"cv_fold_{i}_member_{m}": train() history} for the curves
    """
    folds = list(
        make_cv_folds(table, n_splits=N_SPLITS, val_fraction=VAL_FRACTION, seed=SEED)
    )
    rows = []
    pooled = []
    histories = {}

    sizes = {s: int((folds[0][1]["split"] == s).sum()) for s in ("train", "val", "test")}
    print(f"Per Fold: train {sizes['train']}, val {sizes['val']}, test {sizes['test']}"
          f"  ({ENSEMBLE_SIZE} members each)\n")
    print(f"{'Fold':<6}{'C-index':>9}{'Sel-AUC':>9}{'Risk-std':>10}"
          f"{'Patients':>10}{'Events':>8}{'Best-Epoch':>12}")
    for fold_index, fold_table in folds:
        train_loader, val_loader, test_loader = make_split_loaders(fold_table, SEED)
        # Unique seed per (fold, member) so every training run is independent.
        base_seed = SEED + fold_index * ENSEMBLE_SIZE
        models, member_histories, optimizers = fit_ensemble(
            train_loader, val_loader, device, loss_fn, base_seed
        )
        for m, (model, history, optimizer) in enumerate(
            zip(models, member_histories, optimizers)
        ):
            histories[f"cv_fold_{fold_index}_member_{m}"] = history
            save_model(model, optimizer, history,
                       CHECKPOINT_DIR / f"cv_fold_{fold_index}_member_{m}.pt")

        uq_df = deep_ensemble_predict(models, test_loader, device)
        result = score_uq_frame(uq_df)
        rows.append({"fold": fold_index, **result})
        # Tag the fold: the five folds are five separately-fit models whose Cox scores
        # share no common scale, so anything pooling them needs to know which is which.
        pooled.append(to_patient_frame(uq_df).assign(fold=fold_index))
        mean_best = int(round(np.mean([h.get("best_epoch", 0) for h in member_histories])))
        print(
            f"{f'{fold_index + 1}/{N_SPLITS}':<6}{result['c_index']:>9.4f}"
            f"{result['selective_auc']:>9.4f}{result['mean_risk_std']:>10.4f}"
            f"{result['n_patients']:>10}{result['n_events']:>8}"
            f"{mean_best:>12}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(RESULT_DIR / "cv_trials.csv", index=False)

    return pd.DataFrame(rows), pd.concat(pooled, ignore_index=True), histories


def summarize_cv(trials, pooled_patients):
    """Aggregate per-fold scores, plus a pooled (whole-cohort) C-index + bootstrap CI."""
    point, lo, hi, _ = bootstrap_cindex(
        pooled_patients.rename(columns={"risk_mean": "risk"}), n_boot=N_BOOT, seed=SEED
    )
    return {
        "mean_cindex": float(trials["c_index"].mean()),
        "std_cindex": float(trials["c_index"].std()),
        "min_cindex": float(trials["c_index"].min()),
        "max_cindex": float(trials["c_index"].max()),
        "mean_selective_auc": float(trials["selective_auc"].mean()),
        "mean_risk_std": float(trials["mean_risk_std"].mean()),
        "pooled_cindex": point,
        "pooled_ci_low": lo,
        "pooled_ci_high": hi,
        "folds": int(len(trials)),
    }


# ----------------------------------------------------------------------
# Phase 2: frozen-split refit
# ----------------------------------------------------------------------


def run_frozen_split(table, device, loss_fn):
    """Train one ensemble on the frozen train/val split; score its held-out test.

    Returns (metrics, test_predictions, member_histories). metrics carries the
    single-member (member 0) C-index + bootstrap CI as a point-prediction
    baseline and the ensemble mean-risk C-index + selective-prediction /
    uncertainty numbers on the same test patients.
    """
    train_loader, val_loader, test_loader = make_split_loaders(table, SEED)
    if test_loader is None:
        raise ValueError(f"{SPLIT_CSV} has no 'test' rows to evaluate.")

    print(
        f"Fitting {ENSEMBLE_SIZE} members on train {len(train_loader.dataset)}, "
        f"val {len(val_loader.dataset)}, test {len(test_loader.dataset)} ...",
        flush=True,
    )
    # Seeds disjoint from every CV (fold, member) seed above.
    base_seed = SEED + N_SPLITS * ENSEMBLE_SIZE
    models, member_histories, optimizers = fit_ensemble(
        train_loader, val_loader, device, loss_fn, base_seed
    )
    for m, (model, history, optimizer) in enumerate(
        zip(models, member_histories, optimizers)
    ):
        save_model(model, optimizer, history,
                   CHECKPOINT_DIR / f"frozen_split_member_{m}.pt")

    # Single-member baseline (member 0): a point prediction without ensembling.
    single = evaluate(models[0], test_loader, device, n_boot=N_BOOT, seed=SEED)

    # Ensemble: mean risk + between-member disagreement on the same test patients.
    uq_df = deep_ensemble_predict(models, test_loader, device)
    patients = to_patient_frame(uq_df)
    ens_point, ens_lo, ens_hi, _ = bootstrap_cindex(
        patients.rename(columns={"risk_mean": "risk"}), n_boot=N_BOOT, seed=SEED
    )

    metrics = {
        "n_test_patients": len(patients),
        "n_test_events": int(patients["event"].sum()),
        "single_cindex": single["c_index"],
        "single_ci_low": single["ci_low"],
        "single_ci_high": single["ci_high"],
        "ensemble_cindex": ens_point,
        "ensemble_ci_low": ens_lo,
        "ensemble_ci_high": ens_hi,
        "ensemble_selective_auc": selective_auc(patients),
        "ensemble_mean_risk_std": float(patients["risk_std"].mean()),
    }
    return metrics, uq_df, member_histories


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device(DEVICE)
    table = load_survival_table(SPLIT_CSV, project_root=PROJECT_ROOT)
    loss_fn = make_loss_fn()

    settings = {
        name: (str(value) if isinstance(value, Path) else value)
        for name, value in globals().items()
        if name.isupper() and not name.startswith("_")
    }
    with open(RESULT_DIR / "config.json", "w") as handle:
        json.dump(settings, handle, indent=2, sort_keys=True, default=str)

    # Build one throwaway member just to report the architecture + parameter count.
    display_model = build_model(**MODEL_CONFIG)
    n_params = sum(p.numel() for p in display_model.parameters())
    head_kind = "mlp" if MODEL_CONFIG.get("risk_hidden_dim") else "linear-cox"
    loss_name = (
        "cox" if LAMBDA_RNC == 0
        else f"cox+{LAMBDA_RNC}*survrnc(T={SURVRNC_TEMPERATURE})"
    )

    print(f"{'Device':<14}{device}   seed {SEED}")
    print(f"{'Cohort':<14}{COHORT} / {ENCODER}")
    print(f"{'CSV':<14}{SPLIT_CSV.relative_to(PROJECT_ROOT)}")
    print()
    print(f"{'Split':<8}{'Patients':>10}{'Slides':>9}{'Events':>9}")
    for name, n_pat, n_sl, n_ev in split_counts(table):
        print(f"{name:<8}{n_pat:>10}{n_sl:>9}{n_ev:>9}")

    print(f"\n{'Model':<14}{display_model.__class__.__name__}  "
          f"({n_params:,} params/member, {head_kind} head)")
    print(f"{'Config':<14}" + "  ".join(f"{k}={v}" for k, v in MODEL_CONFIG.items()))
    print(display_model)

    print(f"\n{'Loss':<14}{loss_name}")
    print(f"{'Optimizer':<14}{OPTIMIZER}  lr={LEARNING_RATE}  wd={WEIGHT_DECAY}  "
          f"epochs={EPOCHS}  patience={EARLY_STOPPING_PATIENCE}  grad_clip={GRAD_CLIP}")
    print(f"{'Ensemble':<14}members={ENSEMBLE_SIZE}  bootstrap={N_BOOT}")

    histories = {}  # every train() run, keyed by source, for the learning curves

    # --- Phase 1: cross-validation ---
    print(f"\n--- Phase 1: {N_SPLITS}-Fold Cross-Validation ---\n")
    trials, pooled_patients, cv_histories = run_cross_validation(table, device, loss_fn)
    histories.update(cv_histories)
    cv_summary = summarize_cv(trials, pooled_patients)

    pd.DataFrame([cv_summary]).to_csv(RESULT_DIR / "cv_summary.csv", index=False)
    pooled_patients.to_csv(RESULT_DIR / "cv_pooled_predictions.csv", index=False)

    print(
        f"\n{'Mean':<14}C-index {cv_summary['mean_cindex']:.4f} +/- "
        f"{cv_summary['std_cindex']:.4f}   (min {cv_summary['min_cindex']:.4f}, "
        f"max {cv_summary['max_cindex']:.4f})"
    )
    print(
        f"{'Pooled':<14}C-index {cv_summary['pooled_cindex']:.4f}   "
        f"95% CI [{cv_summary['pooled_ci_low']:.4f}, {cv_summary['pooled_ci_high']:.4f}]"
    )
    print(
        f"{'':<14}Sel-AUC {cv_summary['mean_selective_auc']:.4f}   "
        f"risk-std {cv_summary['mean_risk_std']:.4f}"
    )

    # --- Phase 2: frozen-split refit + held-out test ---
    print("\n--- Phase 2: Frozen-Split Refit ---\n")
    test_metrics, test_predictions, frozen_histories = run_frozen_split(table, device, loss_fn)
    for m, history in enumerate(frozen_histories):
        histories[f"frozen_split_member_{m}"] = history

    pd.DataFrame([test_metrics]).to_csv(RESULT_DIR / "test_summary.csv", index=False)
    test_predictions.to_csv(RESULT_DIR / "test_predictions.csv", index=False)

    # Learning curves for every training run (both phases), for plotting later.
    save_histories(histories, RESULT_DIR)

    print(
        f"\n{'Test set':<14}{test_metrics['n_test_patients']} patients, "
        f"{test_metrics['n_test_events']} events"
    )
    print(
        f"{'Single':<14}C-index {test_metrics['single_cindex']:.4f}   "
        f"95% CI [{test_metrics['single_ci_low']:.4f}, "
        f"{test_metrics['single_ci_high']:.4f}]"
    )
    print(
        f"{'Ensemble':<14}C-index {test_metrics['ensemble_cindex']:.4f}   "
        f"95% CI [{test_metrics['ensemble_ci_low']:.4f}, {test_metrics['ensemble_ci_high']:.4f}]"
    )
    print(
        f"{'':<14}sel-AUC {test_metrics['ensemble_selective_auc']:.4f}   "
        f"risk-std {test_metrics['ensemble_mean_risk_std']:.4f}"
    )

    print(f"\n{'Results':<14}{RESULT_DIR.relative_to(PROJECT_ROOT)}")
    print(
        f"{'checkpoints':<14}{CHECKPOINT_DIR.relative_to(PROJECT_ROOT)}  "
        f"({N_SPLITS * ENSEMBLE_SIZE} CV + {ENSEMBLE_SIZE} frozen members)"
    )


if __name__ == "__main__":
    main()
