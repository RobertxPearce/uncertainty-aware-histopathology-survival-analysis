# Run a trained model over a split and report metrics.
#
# This closes the train -> test loop. The scoring primitives take an in-memory
# model and a prepared loader, so they compose right after train():
#   predict(model, loader)   -> per-slide risk DataFrame
#   evaluate(model, loader)  -> C-index + patient-level bootstrap CI
# evaluate_split() is a thin CLI convenience that wires the file/checkpoint
# plumbing around them: load a checkpoint saved by the trainer, rebuild the exact
# model from the config stored inside it, build a loader for one split, and score.
#
#   * The C-index is computed once over the whole split, not averaged per batch;
#     a ranking metric on tiny batches is biased and noisy.
#   * The confidence interval resamples patients (case_id), not slides. Slides
#     from one patient are correlated, so resampling slides would understate the
#     uncertainty. With only a few dozen test patients the interval is wide on
#     purpose, that width is the honest signal, not a bug to hide.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.dataset import collate_bags, make_datasets
from ..data.make_survival_metadata import load_survival_table
from ..models.abmil import build_model
from ..utils.device import pick_device
from ..utils.io import load_checkpoint
from .metrics import concordance_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _model_kwargs_from_config(config):
    """
    Pull build_model()'s kwargs out of a checkpoint's stored config.

    New checkpoints store exactly these keys; older ones stored a dump of CLI
    flags (with gated_attention instead of gated), so we read both and fall
    back to the model defaults for anything missing. This is what lets old
    checkpoints keep loading after the refactor.
    """
    config = config or {}
    return dict(
        input_dim=config.get("input_dim", 1024),
        embed_dim=config.get("embed_dim", 512),
        attention_dim=config.get("attention_dim", 256),
        dropout=config.get("dropout", 0.25),
        gated=config.get("gated", config.get("gated_attention", True)),
        hidden_dims=config.get("hidden_dims"),
        input_norm=config.get("input_norm", False),
        pool_norm=config.get("pool_norm", True),
        risk_hidden_dim=config.get("risk_hidden_dim"),
    )


def load_model(checkpoint_path, device="auto", map_location="cpu"):
    """Load a checkpoint, rebuild the model from its config, load weights, move to device."""
    device = pick_device(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    if "model_state" not in checkpoint:
        raise KeyError(
            f"{checkpoint_path} has no 'model_state'; is it a training checkpoint?"
        )
    model = build_model(**_model_kwargs_from_config(checkpoint.get("config")))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict(model, loader, device="auto"):
    """
    Run the model over a loader and gather one row per slide.

    Returns a DataFrame with columns: case_id, slide_id, risk, time, event.
    Risk is the raw Cox hazard score (higher = worse prognosis); no ordering or
    scoring happens here so the same frame feeds every downstream metric.
    """
    device = pick_device(device)
    model.to(device)
    model.eval()

    records = []
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)

        risk = model(features, mask=mask).detach().cpu().numpy().reshape(-1)

        time = batch["time"].numpy().reshape(-1)
        event = batch["event"].numpy().reshape(-1)
        for i in range(len(risk)):
            records.append(
                {
                    "case_id": batch["case_id"][i],
                    "slide_id": batch["slide_id"][i],
                    "risk": float(risk[i]),
                    "time": float(time[i]),
                    "event": int(event[i]),
                }
            )
    return pd.DataFrame.from_records(
        records, columns=["case_id", "slide_id", "risk", "time", "event"]
    )


# Backwards-compatible alias for the previous name.
collect_predictions = predict


def bootstrap_cindex(pred_df, n_boot=1000, seed=42, alpha=0.05, group_col="case_id"):
    """
    Patient-level bootstrap confidence interval for the C-index.

    Resamples whole patients with replacement (all of a patient's slides move
    together) n_boot times, recomputing the C-index on each resample. Returns
    (point, lo, hi, n_valid):
        point   : C-index on the observed data (nan if no comparable pairs)
        lo, hi  : the alpha/2 and 1-alpha/2 percentiles of the bootstrap scores
        n_valid : bootstrap resamples that yielded a defined C-index

    Resamples with no comparable pairs (e.g. every drawn patient censored) give
    nan and are dropped; lo/hi are nan if too few valid resamples remain.
    """
    point = concordance_index(pred_df["risk"], pred_df["time"], pred_df["event"])

    groups = pred_df[group_col].to_numpy()
    patients = np.unique(groups)
    # Precompute each patient's row indices so a resample is just a gather.
    rows_by_patient = {p: np.flatnonzero(groups == p) for p in patients}

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        drawn = rng.choice(patients, size=patients.size, replace=True)
        rows = np.concatenate([rows_by_patient[p] for p in drawn])
        sample = pred_df.iloc[rows]
        c = concordance_index(sample["risk"], sample["time"], sample["event"])
        if np.isfinite(c):
            scores.append(c)

    if len(scores) < 2:
        return point, float("nan"), float("nan"), len(scores)

    lo = float(np.percentile(scores, 100 * (alpha / 2)))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return point, lo, hi, len(scores)


def evaluate(model, loader, device="auto", n_boot=1000, seed=42):
    """
    Score a model on a prepared loader: C-index + patient-level bootstrap CI.

    Decoupled from files and checkpoints, pass an in-memory model and loader
    (for example right after train()). Returns:
        {predictions, c_index, ci_low, ci_high, n_boot_valid,
         n_slides, n_patients, n_events}
    """
    device = pick_device(device)
    pred_df = predict(model, loader, device)
    point, lo, hi, n_valid = bootstrap_cindex(pred_df, n_boot=n_boot, seed=seed)

    return {
        "predictions": pred_df,
        "c_index": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot_valid": n_valid,
        "n_slides": len(pred_df),
        "n_patients": int(pred_df["case_id"].nunique()) if len(pred_df) else 0,
        "n_events": int(pred_df["event"].sum()) if len(pred_df) else 0,
    }


def evaluate_split(
    split_csv,
    checkpoint_path,
    split="test",
    batch_size=32,
    feature_key="features",
    num_workers=0,
    device="auto",
    n_boot=1000,
    seed=42,
    project_root=PROJECT_ROOT,
    out_csv=None,
    verbose=True,
):
    """
    CLI-style convenience: evaluate one split of a CSV with a saved checkpoint.

    Wires the file/checkpoint plumbing around the reusable predict()/evaluate():
    loads the model, builds a loader for split, scores it, optionally writes
    per-slide predictions to out_csv, and returns the evaluate() dict plus
    "checkpoint" (and "out_csv" when written).
    """
    device = pick_device(device)
    model, checkpoint = load_model(checkpoint_path, device)

    table = load_survival_table(split_csv, project_root=project_root)
    datasets = make_datasets(table, feature_key=feature_key, splits=(split,))
    if split not in datasets:
        raise ValueError(f"{split_csv} has no '{split}' rows to evaluate.")
    loader = DataLoader(
        datasets[split],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_bags,
    )

    results = evaluate(model, loader, device, n_boot=n_boot, seed=seed)
    results["checkpoint"] = str(checkpoint_path)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results["predictions"].to_csv(out_csv, index=False)
        results["out_csv"] = str(out_csv)

    if verbose:
        _print_report(results, checkpoint, split, n_boot)

    return results


def _print_report(results, checkpoint, split, n_boot):
    """Print a compact human-readable summary of an evaluation run."""
    print(f"Checkpoint : {results['checkpoint']}")
    trained_epoch = checkpoint.get("epoch")
    val_cindex = checkpoint.get("val_cindex")
    if trained_epoch is not None:
        val_str = f"{val_cindex:.4f}" if isinstance(val_cindex, (int, float)) else "n/a"
        print(f"             epoch {trained_epoch}, val C-index {val_str}")
    print(
        f"Split      : {split}  "
        f"slides={results['n_slides']} patients={results['n_patients']} "
        f"events={results['n_events']}/{results['n_slides']}"
    )
    c = results["c_index"]
    lo, hi = results["ci_low"], results["ci_high"]
    if np.isfinite(c):
        if np.isfinite(lo) and np.isfinite(hi):
            print(
                f"C-index    : {c:.4f}   95% CI [{lo:.4f}, {hi:.4f}]  "
                f"({results['n_boot_valid']}/{n_boot} valid resamples)"
            )
        else:
            print(f"C-index    : {c:.4f}   (CI unavailable: too few comparable resamples)")
    else:
        print("C-index    : n/a (no comparable pairs in this split)")
    if "out_csv" in results:
        print(f"Predictions: {results['out_csv']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ABMIL + Cox checkpoint on a split (C-index + bootstrap CI)."
    )
    parser.add_argument("--split-csv", type=Path, required=True,
                        help="Split metadata CSV produced by make_splits.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Checkpoint to evaluate (e.g. runs/cox_baseline/best.pt).")
    parser.add_argument("--split", default="test",
                        help="Which split to evaluate (train | val | test).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional CSV path for per-slide predictions.")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-key", default="features",
                        help="Dataset key holding patch features inside each .h5 bag.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")

    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap resamples for the C-index CI (0 to skip).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the bootstrap resampling.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT,
                        help="Root used to resolve relative feature paths.")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_split(
        split_csv=args.split_csv,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        feature_key=args.feature_key,
        num_workers=args.num_workers,
        device=args.device,
        n_boot=args.n_boot,
        seed=args.seed,
        project_root=args.project_root,
        out_csv=args.out,
    )


if __name__ == "__main__":
    main()
