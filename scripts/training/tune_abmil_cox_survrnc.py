"""Tune a regularized ABMIL Cox + SurvRNC model on one frozen train/val split.

This script intentionally does only model tuning: no uncertainty estimators,
test-set scoring, or plots. It trains a compact grid, evaluates each selected
checkpoint at the patient level on validation data, and writes a ranked CSV.

Example:
    python scripts/training/tune_abmil_cox_survrnc.py
"""

import argparse
import json
import sys
from functools import partial
from itertools import product
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    build_model,
    build_optimizer,
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


DEFAULT_SPLIT_CSV = (
    PROJECT_ROOT / "data/processed/experiments/uni_v2_luad/splits.csv"
)
DEFAULT_OUT = PROJECT_ROOT / "runs/abmil_cox_survrnc_tuning_uni_v2_TCGA_LUAD"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune regularized ABMIL Cox + SurvRNC models on validation C-index."
    )
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--input-dim", type=int, default=1536)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.25, 0.40])
    parser.add_argument(
        "--lambda-rnc", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.10]
    )
    parser.add_argument("--temperature", type=float, default=2.0)

    parser.add_argument("--max-patches", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


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


def make_loaders(table, args, trial_seed):
    """Rebuild loaders per trial so shuffling and patch sampling are comparable."""
    datasets = make_datasets(
        table,
        feature_key="features",
        max_patches=args.max_patches,
        seed=trial_seed,
    )
    return make_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generator=make_generator(trial_seed),
        worker_init_fn=worker_init_fn,
    )


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    table = load_survival_table(args.split_csv, project_root=args.project_root)

    if not {"train", "val"}.issubset(set(table["split"])):
        raise ValueError("The split CSV must contain both train and val rows.")

    settings = vars(args).copy()
    settings.update(
        split_csv=str(args.split_csv),
        out=str(args.out),
        project_root=str(args.project_root),
        device=str(device),
    )
    with open(args.out / "tuning_config.json", "w") as handle:
        json.dump(settings, handle, indent=2, sort_keys=True)

    trials = list(product(args.dropouts, args.lambda_rnc))
    print(f"Device: {device}")
    print(f"Trials: {len(trials)} (test split will not be evaluated)")

    rows = []
    for trial_index, (dropout, lambda_rnc) in enumerate(trials):
        # Hold initialization, shuffling, and patch sampling constant so differences
        # between trials come from the hyperparameters rather than seed variance.
        trial_seed = args.seed
        seed_everything(trial_seed)
        trial_name = f"dropout_{dropout:g}__lambda_rnc_{lambda_rnc:g}"
        trial_dir = args.out / trial_name

        train_loader, val_loader, _ = make_loaders(table, args, trial_seed)
        model_config = {
            "input_dim": args.input_dim,
            "embed_dim": args.embed_dim,
            "attention_dim": args.attention_dim,
            "dropout": dropout,
            "gated": True,
        }
        model = build_model(**model_config)
        optimizer = build_optimizer(
            model,
            name="adamw",
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        loss_fn = (
            cox_loss_step
            if lambda_rnc == 0
            else partial(
                survrnc_cox_loss,
                lambda_rnc=lambda_rnc,
                temperature=args.temperature,
            )
        )

        print(f"\n[{trial_index + 1}/{len(trials)}] {trial_name}")
        history = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn=loss_fn,
            epochs=args.epochs,
            device=device,
            early_stopping_patience=args.patience,
            checkpoint_dir=trial_dir,
            model_config=model_config,
            grad_clip=args.grad_clip,
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
        ).to_csv(args.out / "tuning_summary.csv", index=False)
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
