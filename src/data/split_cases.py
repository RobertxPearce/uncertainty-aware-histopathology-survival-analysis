# Case-level train/val/test splits, stratified by event.
#
# Splits are assigned at the patient (case_id) level, not the slide level, so a
# patient with several slides never lands in two different splits (which would
# leak information and inflate the validation metric). Within each split we keep
# the event rate roughly constant by stratifying on the patient-level event
# label. The resulting split labels are written back onto the per-slide table so
# the exact partition is frozen on disk and reproducible from the seed.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_NAMES = ("train", "val", "test")


def assign_splits(
    metadata,
    val_fraction=0.20,
    test_fraction=0.10,
    seed=42,
    group_col="case_id",
    stratify_col="event",
    split_col="split",
):
    """
    Add a `split` column ("train"/"val"/"test") to a per-slide metadata table,
    partitioning at the patient level and stratifying by event.

    Every row belonging to the same `group_col` (patient) is assigned the same
    split. Fractions are applied per event class so each split holds a similar
    proportion of events. Set test_fraction=0 to produce only train/val.

    Returns a copy of `metadata` with the split column added.
    """
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("val_fraction and test_fraction must each be in [0, 1).")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1 (train needs a share).")
    for col in (group_col, stratify_col):
        if col not in metadata.columns:
            raise ValueError(f"metadata is missing required column: {col!r}")

    # Collapse to one row per patient. The stratify label is per-patient constant
    # in this pipeline, but take the max so a patient counts as an event if any
    # of their slides carries one.
    per_patient = (
        metadata.groupby(group_col)[stratify_col].max().reset_index()
    )

    rng = np.random.default_rng(seed)
    split_of_patient = {}
    for label in sorted(per_patient[stratify_col].unique()):
        patients = per_patient.loc[per_patient[stratify_col] == label, group_col].to_numpy()
        rng.shuffle(patients)

        n = len(patients)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        # Guard tiny classes: never spend the whole class on val/test.
        n_val = min(n_val, max(n - n_test - 1, 0))

        test_patients = patients[:n_test]
        val_patients = patients[n_test:n_test + n_val]
        train_patients = patients[n_test + n_val:]

        for p in train_patients:
            split_of_patient[p] = "train"
        for p in val_patients:
            split_of_patient[p] = "val"
        for p in test_patients:
            split_of_patient[p] = "test"

    out = metadata.copy()
    out[split_col] = out[group_col].map(split_of_patient)
    return out


def make_splits(
    metadata_csv,
    out_path,
    val_fraction=0.20,
    test_fraction=0.10,
    seed=42,
    group_col="case_id",
    stratify_col="event",
    verbose=True,
):
    """
    Read a per-slide metadata CSV, assign patient-level stratified splits, and
    write the table back out with a `split` column. Returns the DataFrame.
    """
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)

    split_df = assign_splits(
        metadata,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        group_col=group_col,
        stratify_col=stratify_col,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(out_path, index=False)

    if verbose:
        _print_split_summary(split_df, group_col=group_col, stratify_col=stratify_col)
        print(f"Saved to: {out_path}")

    return split_df


def _print_split_summary(split_df, group_col="case_id", stratify_col="event", split_col="split"):
    """Print per-split patient/slide counts and event rate for a quick sanity check."""
    print(f"{'split':<6} {'patients':>9} {'slides':>8} {'events':>8} {'event_rate':>11}")
    for name in SPLIT_NAMES:
        rows = split_df[split_df[split_col] == name]
        if rows.empty:
            continue
        per_patient = rows.groupby(group_col)[stratify_col].max()
        n_patients = len(per_patient)
        n_events = int(per_patient.sum())
        rate = per_patient.mean() if n_patients else float("nan")
        print(f"{name:<6} {n_patients:>9} {len(rows):>8} {n_events:>8} {rate:>11.3f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign patient-level, event-stratified train/val/test splits to a metadata CSV."
    )
    parser.add_argument("--metadata-csv", type=Path, required=True,
                        help="Per-slide survival metadata CSV (needs case_id and event columns).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output CSV path for the metadata table with a split column.")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of patients per event class held out for validation.")
    parser.add_argument("--test-fraction", type=float, default=0.15,
                        help="Fraction of patients per event class held out for test (0 to skip).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the shuffle.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the split summary.")
    return parser.parse_args()


def main():
    args = parse_args()
    make_splits(
        metadata_csv=args.metadata_csv,
        out_path=args.out,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
