"""
Compare three uncertainty-quantification methods on one ABMIL + Cox (+ SurvRNC)
survival model, all evaluated on the same test split:

    * MC Dropout     - stochastic forward passes through one trained model
    * Deep Ensemble  - spread across ENSEMBLE_SIZE independently trained models
    * SNGP           - Gaussian-process posterior variance from one SNGP model

Each method returns a per-slide (risk_mean, risk_std) frame on the same Cox risk
scale, so they are scored with the same C-index + patient-level bootstrap CI and
reported side by side (mean risk_std summarises how much uncertainty each assigns).

Example Run:
    python scripts/training/train_abmil_cox_survrnc_uncertanty_comparison.py
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    # data
    load_survival_table,
    make_datasets,
    make_dataloaders,
    # model + optimizer
    build_model,
    build_sngp_model,
    build_optimizer,
    # loss
    survrnc_cox_loss,
    # training
    train,
    # evaluation
    bootstrap_cindex,
    concordance_index,
    # uncertainty estimation
    mc_dropout_predict,
    deep_ensemble_predict,
    fit_sngp_covariance,
    sngp_predict,
    # utilities
    seed_everything,
    make_generator,
    worker_init_fn,
    pick_device,
)


# ----------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------


# Experiment/Run Name
COHORT = "TCGA_LUAD"
ENCODER = "uni_v2"
EXPERIMENT_NAME = f"abmil_cox_survrnc_uncertainty_comparison_{ENCODER}_{COHORT}"

# Input Paths
DATA = PROJECT_ROOT / "data"
FEATURE_DIR = DATA / "processed/features" / ENCODER / COHORT
SPLIT_CSV = DATA / "processed/experiments/uni_v2_luad/splits.csv"
# Output Paths
RUN_DIR = PROJECT_ROOT / "runs" / EXPERIMENT_NAME
RESULT_DIR = PROJECT_ROOT / "results/figures" / EXPERIMENT_NAME

# Seed
SEED = 42

# Data Loading
FEATURE_KEY = "features"  # dataset key inside each .h5 bag
MAX_PATCHES = 4096  # cap patches per bag on the train split only
BATCH_SIZE = 16  # Cox risk set is the batch -> prefer larger batches
NUM_WORKERS = 0

# Model architecture (kwargs are stored verbatim in the checkpoint's config).
MODEL_CONFIG = dict(
    input_dim=1536,
    embed_dim=512,
    attention_dim=256,
    dropout=0.5,
    gated=True,
)

# Training
OPTIMIZER = "adamw"
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
GRAD_CLIP = 1.0

# SurvRNC Auxiliary Loss: L = Cox + LAMBDA_RNC * SurvRNC
LAMBDA_RNC = 0.0
SURVRNC_TEMPERATURE = 2.0

# Uncertainty
MC_DROPOUT_SAMPLES = 100  # Stochastic forward passes per slide (MC Dropout)
ENSEMBLE_SIZE = 5  # Independently trained members (Deep Ensemble)
SNGP_NUM_FEATURES = 1024  # Random Fourier features in the GP head (SNGP)
SNGP_RIDGE_PENALTY = 1.0  # Laplace precision ridge (SNGP)
SNGP_NORM_BOUND = 0.95  # Spectral-norm cap on the encoder (SNGP)
N_BOOT = 1000  # Bootstrap resamples for the C-index CI
OOD_NOISE_STD = 0.5  # Distance-awareness check: feature noise as a fraction of feature std
REFERENCE_METHOD = "deep_ensemble"  # risk_mean source for KM / risk-distribution plots

# SNGP extends the shared architecture with its GP-head hyper-parameters.
SNGP_CONFIG = dict(
    **MODEL_CONFIG,
    num_features=SNGP_NUM_FEATURES,
    gp_ridge_penalty=SNGP_RIDGE_PENALTY,
    spectral_norm_bound=SNGP_NORM_BOUND,
)


def make_loss_fn():
    """Cox partial likelihood, plus the SurvRNC term when LAMBDA_RNC > 0."""
    return partial(
        survrnc_cox_loss,
        lambda_rnc=LAMBDA_RNC,
        temperature=SURVRNC_TEMPERATURE,
    )


def score(pred_df):
    """C-index + patient-level bootstrap CI + mean uncertainty for a UQ frame."""
    # bootstrap_cindex ranks on a "risk" column; the UQ frames call it risk_mean.
    ranked = pred_df.rename(columns={"risk_mean": "risk"})
    point, lo, hi, n_valid = bootstrap_cindex(ranked, n_boot=N_BOOT, seed=SEED)
    # Area under the selective-prediction curve: a single "is the uncertainty
    # useful?" number (mean C-index as the most-uncertain slides are dropped).
    cov, cidx = _selective_curve(pred_df)
    sel_auc = float(np.trapz(cidx, cov) / (cov[-1] - cov[0])) if len(cov) > 1 else float("nan")
    return {
        "c_index": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot_valid": n_valid,
        "mean_risk_std": float(pred_df["risk_std"].mean()),
        "selective_auc": sel_auc,
    }


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
EVENT_COLORS = {0: "#999999", 1: "#D55E00"}  # censored (grey) / event (vermillion)


def _style(ax, grid_axis="y"):
    """Recessive chrome: drop the top/right spines and use a light gridline."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_cindex_comparison(summary, out_path):
    """Bar chart of test C-index per method with 95% bootstrap CI error bars."""
    methods = list(summary["method"])
    c = summary["c_index"].to_numpy(float)
    lower = np.nan_to_num(c - summary["ci_low"].to_numpy(float), nan=0.0)
    upper = np.nan_to_num(summary["ci_high"].to_numpy(float) - c, nan=0.0)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    x = np.arange(len(methods))
    ax.bar(
        x, c, width=0.6,
        color=[METHOD_COLORS[m] for m in methods],
        edgecolor="0.2", linewidth=0.8,
        yerr=[lower, upper], capsize=5, error_kw=dict(ecolor="0.3", lw=1.2),
    )
    ax.axhline(0.5, color="0.55", linestyle="--", linewidth=1.0, zorder=0)  # random baseline
    ax.text(len(methods) - 0.5, 0.505, "random", color="0.5", fontsize=8,
            ha="right", va="bottom")
    for xi, ci in zip(x, c):
        if np.isfinite(ci):
            ax.text(xi, ci + 0.012, f"{ci:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel("Concordance index (C-index)")
    # Floor at 0.4 by convention, but drop lower if a bar/CI would otherwise clip.
    finite = np.isfinite(c)
    y_low = min(0.4, float(np.min((c - lower)[finite])) - 0.02) if finite.any() else 0.4
    ax.set_ylim(y_low, 1.0)
    ax.set_title("Discrimination by uncertainty method (test split)")
    _style(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_distributions(predictions, out_path):
    """
    Box + jittered points of per-slide risk_std for each method.

    Note the absolute scales are method-specific (MC-dropout sampling std vs
    ensemble disagreement vs GP posterior std), so read this as the *spread and
    shape* of each method's uncertainty, not a direct magnitude comparison.
    """
    methods = list(predictions)
    data = [predictions[m]["risk_std"].to_numpy() for m in methods]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    bp = ax.boxplot(
        data, widths=0.55, patch_artist=True, showfliers=False,
        medianprops=dict(color="0.15", linewidth=1.4),
    )
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS[m])
        patch.set_alpha(0.75)
        patch.set_edgecolor("0.2")

    rng = np.random.default_rng(SEED)
    for i, m in enumerate(methods, start=1):
        y = predictions[m]["risk_std"].to_numpy()
        xj = i + (rng.random(len(y)) - 0.5) * 0.18
        ax.scatter(xj, y, s=10, color="0.25", alpha=0.4, linewidths=0, zorder=3)

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel("Predictive uncertainty (risk std)")
    ax.set_title("Per-slide uncertainty spread by method (test split)")
    _style(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _selective_curve(pred_df, min_slides=10, n_points=25):
    """
    C-index vs coverage as the most-uncertain slides are progressively dropped.

    Sorts slides by ascending risk_std (most confident first) and recomputes the
    C-index over the most-confident k. If uncertainty is informative the curve
    rises as coverage shrinks. Points with no comparable pairs (nan) are skipped.
    """
    d = pred_df.sort_values("risk_std").reset_index(drop=True)
    n = len(d)
    ks = np.unique(np.linspace(min(min_slides, n), n, n_points).astype(int))

    cov, cidx = [], []
    for k in ks:
        sub = d.iloc[:k]
        c = concordance_index(sub["risk_mean"], sub["time"], sub["event"])
        if np.isfinite(c):
            cov.append(k / n)
            cidx.append(c)
    return np.array(cov), np.array(cidx)


def plot_selective_prediction(predictions, out_path):
    """Selective-prediction curves: C-index vs coverage, one line per method."""
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for m in predictions:
        cov, cidx = _selective_curve(predictions[m])
        auc = np.trapz(cidx, cov) / (cov[-1] - cov[0]) if len(cov) > 1 else float("nan")
        ax.plot(cov, cidx, marker="o", markersize=4, linewidth=1.8,
                color=METHOD_COLORS[m], label=f"{METHOD_LABELS[m]} (area {auc:.3f})")

    ax.set_xlabel("Coverage (fraction of most-confident slides retained)")
    ax.set_ylabel("C-index on retained slides")
    ax.set_title("Selective prediction: is the uncertainty informative? (test split)")
    ax.legend(frameon=False)
    _style(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_risk_vs_uncertainty(predictions, out_path):
    """Small multiples: predicted risk vs its uncertainty, coloured by outcome."""
    methods = list(predictions)
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.8))
    axes = np.atleast_1d(axes)

    for ax, m in zip(axes, methods):
        df = predictions[m]
        for ev, label in [(0, "censored"), (1, "event")]:
            sub = df[df["event"] == ev]
            ax.scatter(sub["risk_mean"], sub["risk_std"], s=24,
                       color=EVENT_COLORS[ev], edgecolor="white", linewidth=0.4,
                       alpha=0.85, label=label)
        ax.set_title(METHOD_LABELS[m])
        ax.set_xlabel("Predicted risk (risk mean)")
        _style(ax, grid_axis="both")

    axes[0].set_ylabel("Predictive uncertainty (risk std)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Risk vs. predictive uncertainty by method (test split)", y=1.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


class PerturbedLoader:
    """
    Re-iterable loader wrapper that adds fixed Gaussian noise to each batch's features.

    Used for the SNGP distance-awareness / OOD check: feeding off-manifold inputs
    should raise a distance-aware method's predictive uncertainty. The noise is
    deterministic (seeded) and scaled to a fraction of the batch's real-patch feature
    std, so every ensemble member sees the *same* perturbed inputs. Only the features
    are changed; mask/time/event/ids pass through unchanged.
    """

    def __init__(self, loader, noise_std, seed=0):
        self.loader = loader
        self.noise_std = noise_std
        self.seed = seed

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)
        for batch in self.loader:
            features = batch["features"]
            real = features[~batch["mask"]]
            scale = real.std() if real.numel() else features.new_tensor(1.0)
            noise = torch.randn(features.shape, generator=gen) * (self.noise_std * scale)
            out = dict(batch)
            out["features"] = features + noise
            yield out


def bag_sizes_from_loader(loader):
    """Map slide_id -> number of real (unpadded) patches, read off the loader's masks."""
    sizes = {}
    for batch in loader:
        counts = (~batch["mask"]).sum(dim=1)
        for slide_id, count in zip(batch["slide_id"], counts):
            sizes[slide_id] = int(count)
    return sizes


def _to_patient_level(pred_df):
    """Collapse per-slide rows to one row per patient (mean risk; the patient's label)."""
    return (
        pred_df.groupby("case_id")
        .agg(risk_mean=("risk_mean", "mean"), time=("time", "first"), event=("event", "first"))
        .reset_index()
    )


def _km_curve(time, event):
    """Kaplan-Meier survival estimate; returns step arrays (t, S) starting at (0, 1)."""
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    ts, surv, s = [0.0], [1.0], 1.0
    for ut in np.unique(time[event == 1]):
        at_risk = int((time >= ut).sum())
        deaths = int(((time == ut) & (event == 1)).sum())
        if at_risk > 0:
            s *= 1.0 - deaths / at_risk
        ts.append(float(ut))
        surv.append(s)
    return np.array(ts), np.array(surv)


def _logrank_p(t1, e1, t2, e2):
    """Two-group log-rank test; returns the p-value (nan if undefined)."""
    from scipy.stats import chi2

    t1, e1, t2, e2 = (np.asarray(a, float) for a in (t1, e1, t2, e2))
    event_times = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))
    obs1, exp1, var = 0.0, 0.0, 0.0
    for tt in event_times:
        n1, n2 = (t1 >= tt).sum(), (t2 >= tt).sum()
        n = n1 + n2
        d1 = ((t1 == tt) & (e1 == 1)).sum()
        d = d1 + ((t2 == tt) & (e2 == 1)).sum()
        if n <= 1:
            continue
        obs1 += d1
        exp1 += d * n1 / n
        var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if var <= 0:
        return float("nan")
    return float(chi2.sf((obs1 - exp1) ** 2 / var, 1))


def plot_learning_curves(histories, out_path):
    """Left: train/val loss for one representative model. Right: val C-index for all."""
    fig, (ax_loss, ax_c) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    ref_name = next(iter(histories))
    ref = histories[ref_name]
    ax_loss.plot(range(1, len(ref["train_loss"]) + 1), ref["train_loss"],
                 color="#0072B2", linewidth=1.8, label="train")
    if ref["val_loss"]:
        ax_loss.plot(range(1, len(ref["val_loss"]) + 1), ref["val_loss"],
                     color="#D55E00", linewidth=1.8, label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"Loss ({ref_name})")
    ax_loss.legend(frameon=False)
    _style(ax_loss)

    for name, h in histories.items():
        vc = h.get("val_cindex", [])
        if not vc:
            continue
        is_sngp = name == "sngp"
        ax_c.plot(range(1, len(vc) + 1), vc,
                  color="#009E73" if is_sngp else "0.6",
                  linewidth=2.0 if is_sngp else 1.3,
                  label="SNGP" if is_sngp else ("ensemble members" if name == ref_name else None),
                  zorder=3 if is_sngp else 2)
        if h.get("best_epoch"):
            ax_c.scatter([h["best_epoch"]], [h["best_cindex"]], s=25,
                         color="#009E73" if is_sngp else "0.4", zorder=4)
    ax_c.axhline(0.5, color="0.7", linestyle="--", linewidth=1.0, zorder=0)
    ax_c.set_xlabel("Epoch")
    ax_c.set_ylabel("Validation C-index")
    ax_c.set_title("Validation C-index (dots = selected epoch)")
    ax_c.legend(frameon=False)
    _style(ax_c)

    fig.suptitle("Training diagnostics", y=1.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_km_by_risk(pred_df, out_path, method_label):
    """Kaplan-Meier curves for patients split at the median predicted risk + log-rank p."""
    patients = _to_patient_level(pred_df)
    median = patients["risk_mean"].median()
    low = patients[patients["risk_mean"] <= median]
    high = patients[patients["risk_mean"] > median]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for grp, color, label in [
        (low, "#0072B2", f"low risk (n={len(low)})"),
        (high, "#D55E00", f"high risk (n={len(high)})"),
    ]:
        t, s = _km_curve(grp["time"], grp["event"])
        ax.step(t, s, where="post", color=color, linewidth=2.0, label=label)

    p = _logrank_p(low["time"], low["event"], high["time"], high["event"])
    if not np.isfinite(p):
        p_str = "p = n/a"
    else:
        p_str = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    ax.text(0.03, 0.06, f"log-rank {p_str}", transform=ax.transAxes, fontsize=10)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Kaplan-Meier by predicted risk ({method_label}, test)")
    ax.legend(frameon=False)
    _style(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_risk_distribution(pred_df, out_path, method_label):
    """Histogram of predicted risk by outcome (event vs censored)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    bins = np.histogram_bin_edges(pred_df["risk_mean"], bins=20)
    for ev, label in [(0, "censored"), (1, "event")]:
        vals = pred_df.loc[pred_df["event"] == ev, "risk_mean"]
        ax.hist(vals, bins=bins, color=EVENT_COLORS[ev], alpha=0.65,
                edgecolor="white", linewidth=0.4, label=f"{label} (n={len(vals)})")
    ax.set_xlabel("Predicted risk (risk mean)")
    ax.set_ylabel("Slides")
    ax.set_title(f"Risk-score distribution by outcome ({method_label}, test)")
    ax.legend(frameon=False)
    _style(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_method_agreement(predictions, out_path):
    """Pairwise agreement between methods for risk_mean (top row) and risk_std (bottom)."""
    from scipy.stats import pearsonr, spearmanr

    methods = list(predictions)
    pairs = [(methods[i], methods[j])
             for i in range(len(methods)) for j in range(i + 1, len(methods))]

    fig, axes = plt.subplots(2, len(pairs), figsize=(4.0 * len(pairs), 7.4))
    axes = np.atleast_2d(axes)
    for col, (a, b) in enumerate(pairs):
        merged = predictions[a].merge(predictions[b], on="slide_id", suffixes=("_a", "_b"))
        for row, field in enumerate(("risk_mean", "risk_std")):
            ax = axes[row][col]
            xa, yb = merged[f"{field}_a"], merged[f"{field}_b"]
            ax.scatter(xa, yb, s=20, color="0.35", alpha=0.6, linewidths=0)
            lims = [min(xa.min(), yb.min()), max(xa.max(), yb.max())]
            ax.plot(lims, lims, color="0.7", linestyle="--", linewidth=1.0)  # identity
            ax.text(0.05, 0.93,
                    f"r={pearsonr(xa, yb)[0]:.2f}\nρ={spearmanr(xa, yb)[0]:.2f}",
                    transform=ax.transAxes, va="top", fontsize=9)
            ax.set_xlabel(METHOD_LABELS[a])
            ax.set_ylabel(METHOD_LABELS[b])
            ax.set_title(field, fontsize=10)
            _style(ax, grid_axis="both")

    fig.suptitle("Method agreement: risk_mean (top) and risk_std (bottom)", y=1.01)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_vs_bagsize(predictions, bag_sizes, out_path):
    """Predictive uncertainty vs. number of patches per slide, per method."""
    from scipy.stats import spearmanr

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for m in predictions:
        df = predictions[m]
        n = df["slide_id"].map(bag_sizes).to_numpy(float)
        y = df["risk_std"].to_numpy(float)
        ok = np.isfinite(n) & np.isfinite(y)
        rho = spearmanr(n[ok], y[ok])[0] if ok.sum() > 2 else float("nan")
        ax.scatter(n, y, s=22, color=METHOD_COLORS[m], alpha=0.7, linewidths=0,
                   label=f"{METHOD_LABELS[m]} (ρ={rho:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("Patches per slide (log scale)")
    ax.set_ylabel("Predictive uncertainty (risk std)")
    ax.set_title("Uncertainty vs. bag size (test split)")
    ax.legend(frameon=False)
    _style(ax, grid_axis="both")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ood_uncertainty(predictions, ood_predictions, out_path):
    """
    Distance-awareness check: per-method uncertainty on in-distribution vs.
    feature-noised (OOD) inputs. A distance-aware method (SNGP especially) should
    shift its risk_std upward under perturbation; the x-tick shows the median
    OOD/ID ratio (> 1 = uncertainty grew).
    """
    methods = list(predictions)
    positions, data, colors, ratios = [], [], [], {}
    for i, m in enumerate(methods):
        id_std = predictions[m]["risk_std"].to_numpy()
        ood_std = ood_predictions[m]["risk_std"].to_numpy()
        positions += [i * 3 + 1, i * 3 + 2]
        data += [id_std, ood_std]
        colors += [METHOD_COLORS[m], METHOD_COLORS[m]]
        med_id = float(np.median(id_std))
        ratios[m] = float(np.median(ood_std) / med_id) if med_id > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                    showfliers=False, medianprops=dict(color="0.15", linewidth=1.2))
    for k, (patch, color) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.4 if k % 2 == 0 else 0.85)  # ID lighter, OOD darker
        patch.set_edgecolor("0.2")

    ax.set_xticks([i * 3 + 1.5 for i in range(len(methods))])
    ax.set_xticklabels([f"{METHOD_LABELS[m]}\n(×{ratios[m]:.2f})" for m in methods])
    ax.set_ylabel("Predictive uncertainty (risk std)")
    ax.set_title("Distance-awareness: ID (light) vs. OOD (dark) uncertainty")
    _style(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_plots(predictions, summary, histories, bag_sizes, ood_predictions, out_dir):
    """Render every comparison and diagnostic figure to a vector PDF in out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "pdf.fonttype": 42,  # embed editable TrueType text (Illustrator-friendly)
    })
    reference = predictions[REFERENCE_METHOD]
    ref_label = METHOD_LABELS[REFERENCE_METHOD]

    # Method comparison.
    plot_cindex_comparison(summary, out_dir / "cindex_comparison.pdf")
    plot_uncertainty_distributions(predictions, out_dir / "uncertainty_distributions.pdf")
    plot_selective_prediction(predictions, out_dir / "selective_prediction.pdf")
    plot_risk_vs_uncertainty(predictions, out_dir / "risk_vs_uncertainty.pdf")
    plot_method_agreement(predictions, out_dir / "method_agreement.pdf")
    # Uncertainty diagnostics.
    plot_uncertainty_vs_bagsize(predictions, bag_sizes, out_dir / "uncertainty_vs_bagsize.pdf")
    plot_ood_uncertainty(predictions, ood_predictions, out_dir / "ood_uncertainty.pdf")
    # Model / survival diagnostics.
    plot_learning_curves(histories, out_dir / "learning_curves.pdf")
    plot_km_by_risk(reference, out_dir / "km_by_risk.pdf", ref_label)
    plot_risk_distribution(reference, out_dir / "risk_distribution.pdf", ref_label)
    print(f"Saved figures to: {out_dir}")


# ----------------------------------------------------------------------
# Plotting.
# ----------------------------------------------------------------------


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    device = pick_device("auto")
    print(f"Device: {device}")

    # Load the split table
    print("Loading split table...")
    table = load_survival_table(SPLIT_CSV)
    print(f"Slides: {len(table)} | patients: {table['case_id'].nunique()}")
    print(f"Split sizes:\n{table['split'].value_counts()}")
    print("---")

    # Build one dataset per split
    print("Building per split datasets...")
    datasets = make_datasets(
        table,
        feature_key=FEATURE_KEY,
        max_patches=MAX_PATCHES,  # applies to the train split only
        seed=SEED,
    )
    for name, ds in datasets.items():
        print(f"\t{name}: {len(ds)} slides")
    print("---")

    # Wrap datasets in padded MIL DataLoaders
    print("Building padded (train, val, test) loaders")
    train_loader, val_loader, test_loader = make_dataloaders(
        datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        generator=make_generator(SEED),
        worker_init_fn=worker_init_fn,
    )
    eval_loader = test_loader if test_loader is not None else val_loader
    print(f"Train Batches:\t{len(train_loader) if train_loader else 0}")
    print(f"Val Batches:\t{len(val_loader) if val_loader else 0}")
    print(f"Test Batches:\t{len(test_loader) if test_loader else 0}")
    print("---")

    loss_fn = make_loss_fn()
    objective = (
        f"Cox + {LAMBDA_RNC} * SurvRNC (T={SURVRNC_TEMPERATURE})"
        if LAMBDA_RNC > 0 else "Cox partial-likelihood"
    )
    print(f"Objective: {objective}")
    print("---")

    # Deep Ensemble
    print(f"\nTraining {ENSEMBLE_SIZE} ensemble members...")
    members = []
    histories = {}  # per-model run history, for the learning-curve diagnostic
    for i in range(ENSEMBLE_SIZE):
        print(f"  member {i + 1}/{ENSEMBLE_SIZE}")
        seed_everything(SEED + i)  # different init -> ensemble diversity
        model = build_model(**MODEL_CONFIG)
        history = train(
            model, train_loader, val_loader,
            build_optimizer(model, OPTIMIZER, lr=LR, weight_decay=WEIGHT_DECAY),
            loss_fn=loss_fn, epochs=EPOCHS, device=device,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            checkpoint_dir=RUN_DIR / f"ensemble/member_{i}",
            model_config=MODEL_CONFIG, grad_clip=GRAD_CLIP,
        )
        print(f"\tBest Val C-Index {history['best_cindex']:.4f}")
        members.append(model)
        histories[f"member_{i}"] = history
    print("---")

    # SNGP
    print("\nTraining SNGP model...")
    seed_everything(SEED)
    sngp_model = build_sngp_model(**SNGP_CONFIG)
    history = train(
        sngp_model, train_loader, val_loader,
        build_optimizer(sngp_model, OPTIMIZER, lr=LR, weight_decay=WEIGHT_DECAY),
        loss_fn=loss_fn, epochs=EPOCHS, device=device,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        checkpoint_dir=RUN_DIR / "sngp",
        model_config=SNGP_CONFIG, grad_clip=GRAD_CLIP,
    )
    print(f"    best val C-index {history['best_cindex']:.4f}")
    histories["sngp"] = history
    print("    fitting GP covariance over the train split...")
    fit_sngp_covariance(sngp_model, train_loader, device)
    print("---")

    # Predict on the test split with each method (risk_mean, risk_std).
    print("\nPredicting with each uncertainty method (test split)...")
    predictions = {
        "mc_dropout": mc_dropout_predict(
            members[0], eval_loader, device, n_samples=MC_DROPOUT_SAMPLES
        ),
        "deep_ensemble": deep_ensemble_predict(members, eval_loader, device),
        "sngp": sngp_predict(sngp_model, eval_loader, device),
    }
    print("---")

    # Score and report side by side.
    print("\nUncertainty comparison (test split):")
    rows = []
    for name, pred_df in predictions.items():
        pred_df.to_csv(RUN_DIR / f"{name}_predictions.csv", index=False)
        result = score(pred_df)
        rows.append({"method": name, **result})
        print(
            f"  {name:<14} C-Index {result['c_index']:.4f} "
            f"(95% CI {result['ci_low']:.4f}-{result['ci_high']:.4f})  "
            f"Mean Risk STD {result['mean_risk_std']:.4f}"
        )
    print("---")

    summary = pd.DataFrame(rows)
    summary_path = RUN_DIR / "uncertainty_comparison.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")
    print("---")

    # Extra data the diagnostics need: real bag sizes + an OOD (feature-noise) pass.
    bag_sizes = bag_sizes_from_loader(eval_loader)
    print("\nOOD stress test (feature noise) for distance-awareness...")
    ood_loader = PerturbedLoader(eval_loader, OOD_NOISE_STD, seed=SEED)
    ood_predictions = {
        "mc_dropout": mc_dropout_predict(
            members[0], ood_loader, device, n_samples=MC_DROPOUT_SAMPLES
        ),
        "deep_ensemble": deep_ensemble_predict(members, ood_loader, device),
        "sngp": sngp_predict(sngp_model, ood_loader, device),
    }
    print("---")

    # Render every figure as a PDF into RESULT_DIR.
    print("\nGenerating plots...")
    generate_plots(predictions, summary, histories, bag_sizes, ood_predictions, RESULT_DIR)
    print("---")
    
    print("Complete")


if __name__ == "__main__":
    main()
