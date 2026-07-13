"""K-fold cross-validation of three uncertainty methods on one ABMIL + Cox model.

Folds the cohort once, then for every fold trains the shared backbones once (an
ensemble of ABMIL members + one SNGP model) and scores all three UQ methods on
that fold's held-out patients:

    * MC-dropout     - stochastic passes through ensemble member 0
    * Deep ensemble  - spread across the ENSEMBLE_SIZE members
    * SNGP           - GP posterior variance from the SNGP model

Every method sees identical folds, so their per-fold C-index / selective-AUC are
directly comparable. The methods share each fold's trained backbones because they
share the backbone by construction -- the CV measures the UQ estimator, not a
fresh model per method.

Architecture and optimisation knobs match screen_architecture_abmil_cox_survrnc.py
(the arch-screen winner, baseline_input_norm + pure Cox), so the numbers line up.

Outputs (metric CSVs + figures) are written to RESULT_DIR.

Example:
    python scripts/training/cross_validate_abmil_cox_per_uq_method.py
"""

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
SPLIT_CSV = PROJECT_ROOT / "data/processed/experiments/uni_v2_TCGA_GBMLGG/splits.csv"
# CSVs and figures both land here, so the results directory is self-contained.
RESULT_DIR = PROJECT_ROOT / "results" / "cv_abmil_cox_tcga_gbmlgg"

# Which UQ methods to cross-validate.
METHODS = ("mc_dropout", "deep_ensemble", "sngp")

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
N_BOOT = 1000  # bootstrap resamples for the pooled C-index CI

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


def make_fold_loaders(fold_table, seed):
    """(train, val, test) loaders for one make_cv_folds fold table.

    Train is capped/shuffled; val and test are uncapped and iterated in a stable
    order (needed so ensemble members line up per slide).
    """
    datasets = make_datasets(
        fold_table, feature_key=FEATURE_KEY, max_patches=MAX_PATCHES, seed=seed
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
    """ABMIL members the methods need: ENSEMBLE_SIZE for the ensemble, else 1 for
    MC-dropout, else 0."""
    if "deep_ensemble" in methods:
        return ENSEMBLE_SIZE
    if "mc_dropout" in methods:
        return 1
    return 0


def fit_backbones(methods, train_loader, val_loader, device, loss_fn):
    """Train the backbones the methods need; keep best-epoch weights in memory.

    No checkpoints are written -- each model is used for prediction immediately
    after training. Members get distinct seeds (SEED + i) for ensemble diversity;
    SNGP additionally has its GP covariance fit over the train loader.

    Returns {"members": [...], "sngp": model_or_None, "histories": {...}}.
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
# Cross-validation
# ----------------------------------------------------------------------


def run_cross_validation(table, device, loss_fn):
    """Score every method under shared K-fold CV.

    Returns (trials, pooled_frames, histories):
        trials       - one row per (method, fold) with the fold's metrics
        pooled_frames- per method, the held-out patient frames concatenated over
                       folds (every patient scored once)
        histories    - {fold_index: {model_name: train() history}} for the curves
    """
    folds = list(
        make_cv_folds(table, n_splits=N_SPLITS, val_fraction=VAL_FRACTION, seed=SEED)
    )
    rows = []
    pooled = {m: [] for m in METHODS}
    histories = {}

    for fold_index, fold_table in folds:
        print(f"\n[fold {fold_index + 1}/{N_SPLITS}]")
        train_loader, val_loader, test_loader = make_fold_loaders(fold_table, SEED)
        fitted = fit_backbones(METHODS, train_loader, val_loader, device, loss_fn)
        histories[fold_index] = fitted["histories"]

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
        pd.DataFrame(rows).to_csv(RESULT_DIR / "cv_trials.csv", index=False)

    pooled_frames = {m: pd.concat(pooled[m], ignore_index=True) for m in METHODS}
    return pd.DataFrame(rows), pooled_frames, histories


def summarize(trials, pooled_frames):
    """Aggregate per-fold scores per method, plus a pooled (whole-cohort) C-index + CI."""
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
    pooled_stats = {}
    for method, frame in pooled_frames.items():
        point, lo, hi, _ = bootstrap_cindex(
            frame.rename(columns={"risk_mean": "risk"}), n_boot=N_BOOT, seed=SEED
        )
        pooled_stats[method] = (point, lo, hi)
    summary["pooled_cindex"] = summary["method"].map(lambda m: pooled_stats[m][0])
    summary["pooled_ci_low"] = summary["method"].map(lambda m: pooled_stats[m][1])
    summary["pooled_ci_high"] = summary["method"].map(lambda m: pooled_stats[m][2])
    return summary.sort_values("mean_cindex", ascending=False, na_position="last")


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


METHOD_LABELS = {
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble",
    "sngp": "SNGP",
}
METHOD_COLORS = {
    "mc_dropout": "#0072B2",     # blue
    "deep_ensemble": "#E69F00",  # orange
    "sngp": "#009E73",           # green
}


def _style(ax, grid_axis="y"):
    """Recessive chrome: drop top/right spines, light gridline."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_cindex_by_method(trials, out_path):
    """Per-fold held-out C-index per method: fold points + mean/std."""
    methods = [m for m in METHODS if m in set(trials["method"])]
    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for i, m in enumerate(methods):
        vals = trials.loc[trials["method"] == m, "c_index"].to_numpy(float)
        xj = i + (rng.random(len(vals)) - 0.5) * 0.22
        ax.scatter(xj, vals, s=42, color=METHOD_COLORS[m], edgecolor="white",
                   linewidth=0.5, zorder=3, alpha=0.9)
        mean, std = np.nanmean(vals), np.nanstd(vals)
        ax.errorbar(i, mean, yerr=std, fmt="_", color="0.2", markersize=26,
                    markeredgewidth=2.2, capsize=6, elinewidth=1.4, zorder=4)
        ax.text(i, mean + std + 0.006, f"{mean:.3f}", ha="center", va="bottom",
                fontsize=9, color="0.2")

    ax.axhline(0.5, color="0.6", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(len(methods) - 0.5, 0.505, "random", color="0.5", fontsize=8,
            ha="right", va="bottom")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel("Held-out patient C-index")
    ax.set_title(f"{N_SPLITS}-fold CV discrimination by method")
    _style(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_error_per_fold(trials, out_path):
    """Grouped bars of held-out error (1 - C-index) per fold, one group per method."""
    methods = [m for m in METHODS if m in set(trials["method"])]
    folds = sorted(trials["fold"].unique())
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for j, m in enumerate(methods):
        by_fold = trials[trials["method"] == m].set_index("fold")["c_index"]
        err = [1.0 - float(by_fold.get(f, np.nan)) for f in folds]
        x = np.arange(len(folds)) + (j - (len(methods) - 1) / 2) * width
        ax.bar(x, err, width=width, color=METHOD_COLORS[m], edgecolor="0.2",
               linewidth=0.6, label=METHOD_LABELS[m])

    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f"fold {f + 1}" for f in folds])
    ax.set_ylabel("Held-out error (1 - C-index)")
    ax.set_title("Per-fold error by method")
    ax.legend(frameon=False)
    _style(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_selective_prediction(pooled_frames, out_path):
    """Selective-prediction curves (pooled over folds): C-index vs coverage."""
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for m in METHODS:
        cov, cidx = selective_curve(pooled_frames[m])
        auc = _trapz(cidx, cov) / (cov[-1] - cov[0]) if len(cov) > 1 else float("nan")
        ax.plot(cov, cidx, marker="o", markersize=4, linewidth=1.8,
                color=METHOD_COLORS[m], label=f"{METHOD_LABELS[m]} (area {auc:.3f})")
    ax.set_xlabel("Coverage (fraction of most-confident patients retained)")
    ax.set_ylabel("C-index on retained patients")
    ax.set_title("Selective prediction: is the uncertainty informative?")
    ax.legend(frameon=False)
    _style(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_hist(pooled_frames, out_path):
    """Per-method histogram of per-patient predictive uncertainty (risk_std).

    Small multiples because the absolute scales are method-specific (dropout
    sampling std vs ensemble disagreement vs GP posterior std) and should not be
    read on a shared axis.
    """
    methods = list(METHODS)
    fig, axes = plt.subplots(1, len(methods), figsize=(4.0 * len(methods), 3.6))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, methods):
        vals = pooled_frames[m]["risk_std"].to_numpy(float)
        ax.hist(vals, bins=30, color=METHOD_COLORS[m], alpha=0.8,
                edgecolor="white", linewidth=0.4)
        ax.axvline(float(np.mean(vals)), color="0.2", linestyle="--", linewidth=1.2)
        ax.set_title(f"{METHOD_LABELS[m]} (mean {np.mean(vals):.3f})", fontsize=10)
        ax.set_xlabel("Predictive uncertainty (risk std)")
        _style(ax, grid_axis="both")
    axes[0].set_ylabel("Patients")
    fig.suptitle("Predictive-uncertainty distribution by method (pooled folds)", y=1.03)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(histories, out_path):
    """Training dynamics across folds: val C-index (left) and train loss (right).

    Every ensemble member is a light grey line, every SNGP model a green line,
    overlaid across folds so the spread shows training stability. Dots mark the
    selected (best-val) epoch.
    """
    fig, (ax_c, ax_l) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for per_model in histories.values():
        for name, h in per_model.items():
            is_sngp = name == "sngp"
            color = METHOD_COLORS["sngp"] if is_sngp else "0.6"
            alpha = 0.8 if is_sngp else 0.4
            vc = h.get("val_cindex", [])
            if vc:
                ax_c.plot(range(1, len(vc) + 1), vc, color=color, linewidth=1.6,
                          alpha=alpha, zorder=3 if is_sngp else 2)
                if h.get("best_epoch"):
                    ax_c.scatter([h["best_epoch"]], [h["best_cindex"]], s=22,
                                 color=color, zorder=4, alpha=alpha)
            tl = h.get("train_loss", [])
            if tl:
                ax_l.plot(range(1, len(tl) + 1), tl, color=color, linewidth=1.6,
                          alpha=alpha, zorder=3 if is_sngp else 2)

    # Proxy legend handles (one per model type).
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="0.6", lw=1.6, label="ensemble member"),
        Line2D([0], [0], color=METHOD_COLORS["sngp"], lw=1.6, label="SNGP"),
    ]
    ax_c.axhline(0.5, color="0.7", linestyle="--", linewidth=1.0, zorder=0)
    ax_c.set_xlabel("Epoch")
    ax_c.set_ylabel("Validation C-index")
    ax_c.set_title("Validation C-index (dots = selected epoch)")
    ax_c.legend(handles=handles, frameon=False)
    _style(ax_c)

    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Train loss")
    ax_l.set_title("Training loss")
    _style(ax_l)

    fig.suptitle("Training dynamics across folds", y=1.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_plots(trials, pooled_frames, histories, out_dir):
    """Render every CV figure as a vector PDF into out_dir."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "pdf.fonttype": 42,  # embed editable TrueType text (Illustrator-friendly)
    })
    plot_cindex_by_method(trials, out_dir / "cindex_by_method.pdf")
    plot_error_per_fold(trials, out_dir / "error_per_fold.pdf")
    plot_selective_prediction(pooled_frames, out_dir / "selective_prediction.pdf")
    plot_uncertainty_hist(pooled_frames, out_dir / "uncertainty_hist.pdf")
    plot_learning_curves(histories, out_dir / "learning_curves.pdf")
    print(f"Saved figures to: {out_dir}")


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
    with open(RESULT_DIR / "cv_config.json", "w") as handle:
        json.dump(settings, handle, indent=2, sort_keys=True, default=str)

    n_train = (n_members_needed(METHODS) + (1 if "sngp" in METHODS else 0)) * N_SPLITS
    print(f"Device: {device}")
    print(
        f"Objective: {'Cox' if LAMBDA_RNC == 0 else f'Cox + {LAMBDA_RNC}*SurvRNC'} | "
        f"architecture: baseline_input_norm ({MODEL_CONFIG['embed_dim']}d)"
    )
    print(
        f"{N_SPLITS}-fold CV over {len(METHODS)} methods "
        f"({n_train} backbone trainings)"
    )

    trials, pooled_frames, histories = run_cross_validation(table, device, loss_fn)
    summary = summarize(trials, pooled_frames)

    summary.to_csv(RESULT_DIR / "cv_summary.csv", index=False)
    for method, frame in pooled_frames.items():
        frame.to_csv(RESULT_DIR / f"cv_pooled_{method}_predictions.csv", index=False)

    print("\nCross-validation complete")
    print(
        summary[
            ["method", "folds", "mean_cindex", "std_cindex",
             "pooled_cindex", "mean_selective_auc", "mean_risk_std"]
        ].to_string(index=False)
    )

    # Honest-noise note: shared backbone -> discrimination will be near-identical.
    if len(summary) > 1:
        gap = summary.iloc[0]["mean_cindex"] - summary.iloc[1]["mean_cindex"]
        noise = trials.groupby("method")["c_index"].std().mean()
        print(f"\nTop-two C-index gap: {gap:.4f} | fold noise (std): {noise:.4f}")
        if gap < noise:
            print(
                "The methods are within fold noise on discrimination (they share the "
                "backbone). Compare them on selective_auc / uncertainty instead."
            )

    print("\nGenerating figures...")
    generate_plots(trials, pooled_frames, histories, RESULT_DIR)
    print(f"\nSaved CV results to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
