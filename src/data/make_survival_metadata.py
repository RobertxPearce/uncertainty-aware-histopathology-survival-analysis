# Build the survival metadata table (case_id, time, event) from the GDC clinical files.
#
# Read the GDC sample sheet (file -> case mapping) and the clinical
# supplement TSV (per-case outcomes), derive the Cox (event, time)
# pair, and bag all slides per patient into one row.

import argparse
import ast
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# GDC TSVs use this token for missing values.
GDC_NULL = "'--"


def read_sample_sheet(sample_sheet_path, slide_suffix=".svs"):
    """
    Read the GDC sample sheet mapping slide files to patients.

    Returns a DataFrame with columns: file_id, file_name, submitter_id
    (submitter_id is the TCGA-xx-xxxx patient barcode). If slide_suffix is set,
    keeps only files with that extension.
    """
    sample_sheet_path = Path(sample_sheet_path)
    if not sample_sheet_path.is_file():
        raise FileNotFoundError(f"Sample sheet not found: {sample_sheet_path}")
    df = pd.read_csv(sample_sheet_path, sep="\t", dtype=str)

    rename = {"File ID": "file_id", "File Name": "file_name", "Case ID": "submitter_id"}
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise ValueError(f"Sample sheet {sample_sheet_path} missing columns: {missing}")

    df = df[list(rename)].rename(columns=rename)
    if slide_suffix:
        df = df[df["file_name"].str.endswith(slide_suffix)].copy()
    return df.reset_index(drop=True)


def read_clinical_tsv(clinical_tsv_path, null_token=GDC_NULL):
    """
    Read the GDC clinical supplement TSV and reduce it to one row per patient.

    The TSV repeats each patient across diagnosis/treatment rows, so we aggregate
    per submitter_id: take the max days_to_last_follow_up across diagnoses and the
    first of the (per-case constant) demographic fields. Returns a DataFrame with
    columns: submitter_id, case_id, vital_status, days_to_death, days_to_last_follow_up.
    """
    clinical_tsv_path = Path(clinical_tsv_path)
    if not clinical_tsv_path.is_file():
        raise FileNotFoundError(f"Clinical TSV not found: {clinical_tsv_path}")
    df = pd.read_csv(clinical_tsv_path, sep="\t", dtype=str, na_values=[null_token])

    rename = {
        "cases.submitter_id": "submitter_id",
        "cases.case_id": "case_id",
        "demographic.vital_status": "vital_status",
        "demographic.days_to_death": "days_to_death",
        "diagnoses.days_to_last_follow_up": "days_to_last_follow_up",
    }
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise ValueError(f"Clinical TSV {clinical_tsv_path} missing columns: {missing}")

    df = df[list(rename)].rename(columns=rename)
    df["days_to_death"] = pd.to_numeric(df["days_to_death"], errors="coerce")
    df["days_to_last_follow_up"] = pd.to_numeric(df["days_to_last_follow_up"], errors="coerce")

    per_case = (
        df.groupby("submitter_id")
        .agg(
            case_id=("case_id", "first"),
            vital_status=("vital_status", "first"),
            days_to_death=("days_to_death", "max"),
            days_to_last_follow_up=("days_to_last_follow_up", "max"),
        )
        .reset_index()
    )
    return per_case


def build_survival_table(file_case_df, clinical_df):
    """
    Derive the Cox (event, time) pair and bag slides into one row per patient.

    event = 1 if vital_status == "Dead" else 0.
    time  = days_to_death for deaths, else days_to_last_follow_up.
    Joins slides to clinical on submitter_id. Patients with no usable time are dropped.
    """
    clinical_df = clinical_df.copy()
    clinical_df["days_to_death"] = pd.to_numeric(clinical_df["days_to_death"], errors="coerce")
    clinical_df["days_to_last_follow_up"] = pd.to_numeric(
        clinical_df["days_to_last_follow_up"], errors="coerce"
    )

    clinical_df["event"] = (clinical_df["vital_status"] == "Dead").astype(int)
    clinical_df["time"] = clinical_df["days_to_death"].where(
        clinical_df["event"] == 1,
        clinical_df["days_to_last_follow_up"],
    )
    clinical_df["time"] = pd.to_numeric(clinical_df["time"], errors="coerce")

    merged = file_case_df.merge(clinical_df, on="submitter_id", how="left")

    survival_table = (
        merged.groupby("submitter_id")
        .agg(
            case_id=("case_id", "first"),
            submitter_id=("submitter_id", "first"),
            file_ids=("file_id", list),
            file_names=("file_name", list),
            n_slides=("file_id", "count"),
            vital_status=("vital_status", "first"),
            event=("event", "first"),
            time=("time", "first"),
        )
        .reset_index(drop=True)
    )

    # Drop patients with no follow-up time (unusable for survival modeling).
    survival_table = survival_table[survival_table["time"].notna()].copy()
    survival_table["event"] = survival_table["event"].astype(int)
    return survival_table


def make_survival_metadata(
    sample_sheet_path,
    clinical_tsv_path,
    out_path,
    slide_suffix=".svs",
    verbose=True,
):
    """Build the patient-level survival table from local TSVs. Writes and returns it."""
    file_case_df = read_sample_sheet(sample_sheet_path, slide_suffix=slide_suffix)
    if verbose:
        print(
            f"Sample sheet: {len(file_case_df)} slide files "
            f"| {file_case_df['submitter_id'].nunique()} patients"
        )

    clinical_df = read_clinical_tsv(clinical_tsv_path)
    if verbose:
        print(f"Clinical TSV: {len(clinical_df)} patients")

    survival_table = build_survival_table(file_case_df, clinical_df)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    survival_table.to_csv(out_path, index=False)

    if verbose:
        n_events = int(survival_table["event"].sum())
        n_censored = int((survival_table["event"] == 0).sum())
        print(f"\nFinal survival table: {len(survival_table)} patients")
        print(f"\t{n_events} events (deaths)")
        print(f"\t{n_censored} censored (alive)")
        print(f"\tEvent rate: {survival_table['event'].mean() * 100:.1f}%")
        print(f"Saved to: {out_path}")

    return survival_table


def _slide_stem_to_clinical(clinical_df):
    """
    Map each slide's file stem (the .h5/.svs name without extension) to its
    patient clinical row. The clinical `file_names` column stores a stringified
    list of .svs names, one entry per slide bagged for that patient.
    """
    stem_to_row = {}
    for _, row in clinical_df.iterrows():
        for svs_name in ast.literal_eval(row["file_names"]):
            stem_to_row[Path(svs_name).stem] = row
    return stem_to_row


def attach_feature_paths(
    clinical_csv,
    feature_dir,
    out_path,
    feature_suffix=".h5",
    project_root=PROJECT_ROOT,
    verbose=True,
):
    """
    Join the patient-level clinical table to the extracted feature bags on disk.

    Produces the per-slide table the dataset loads directly, with columns:
        case_id, slide_id, feature_path, time, event
    One row per feature file, matched to its patient by slide file stem.
    feature_path is stored relative to project_root so the CSV stays portable;
    use load_survival_table() to read it back with absolute paths.
    Returns the DataFrame and writes it to out_path.
    """
    clinical_csv = Path(clinical_csv)
    if not clinical_csv.is_file():
        raise FileNotFoundError(f"Clinical CSV not found: {clinical_csv}")
    clinical_df = pd.read_csv(clinical_csv)
    stem_to_row = _slide_stem_to_clinical(clinical_df)

    feature_dir = Path(feature_dir)
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
    feature_files = sorted(feature_dir.glob(f"*{feature_suffix}"))
    if not feature_files:
        raise RuntimeError(f"No *{feature_suffix} feature files found under {feature_dir}.")

    project_root = Path(project_root)
    records = []
    unmatched = []
    for feature_path in feature_files:
        stem = feature_path.stem
        row = stem_to_row.get(stem)
        if row is None:
            unmatched.append(feature_path.name)
            continue
        rel_path = os.path.relpath(feature_path.resolve(), project_root)
        records.append(
            {
                "case_id": row["submitter_id"],
                "slide_id": stem,
                "feature_path": Path(rel_path).as_posix(),
                "time": float(row["time"]),
                "event": int(row["event"]),
            }
        )

    metadata = pd.DataFrame(records)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(out_path, index=False)

    if verbose:
        print(f"Feature files: {len(feature_files)} | matched: {len(metadata)}")
        if unmatched:
            print(f"  WARNING: {len(unmatched)} unmatched, e.g. {unmatched[:3]}")
        if len(metadata):
            print(
                f"Events: {int(metadata['event'].sum())} / {len(metadata)} "
                f"| time range: {metadata['time'].min():.0f}-{metadata['time'].max():.0f} days"
            )
        print(f"Saved to: {out_path}")

    return metadata


def load_survival_table(csv_path, project_root=PROJECT_ROOT):
    """
    Read a per-slide survival table CSV, resolving relative feature paths to
    absolute against project_root. Returns a DataFrame ready for make_datasets.

    This is the one place feature-path resolution lives: downstream stages
    (make_datasets / make_dataloaders) receive a table with absolute paths and
    never touch project_root themselves.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Survival metadata CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    project_root = Path(project_root)
    df["feature_path"] = df["feature_path"].map(
        lambda p: str(Path(p) if Path(p).is_absolute() else project_root / p)
    )
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build survival metadata: patient-level clinical table, or per-slide feature join."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # clinical: build the patient-level survival table from local GDC TSVs.
    p_clin = subparsers.add_parser(
        "clinical", help="Build the patient-level clinical survival table from local GDC TSVs."
    )
    p_clin.add_argument("--sample-sheet", type=Path, required=True,
                        help="GDC sample sheet TSV (file -> case mapping).")
    p_clin.add_argument("--clinical-tsv", type=Path, required=True,
                        help="GDC clinical supplement TSV (per-case outcomes).")
    p_clin.add_argument("--slide-suffix", default=".svs",
                        help="Keep only sample-sheet files with this extension ('' to keep all).")
    p_clin.add_argument("--out", type=Path, required=True,
                        help="Output CSV path for the clinical survival table.")
    p_clin.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    # features: join the clinical table to feature bags on disk (per-slide table).
    p_feat = subparsers.add_parser(
        "features", help="Join the clinical table to extracted feature bags (per-slide table)."
    )
    p_feat.add_argument("--clinical-csv", type=Path, required=True,
                        help="Patient-level clinical CSV produced by the 'clinical' command.")
    p_feat.add_argument("--feature-dir", type=Path, required=True,
                        help="Directory of extracted feature bags.")
    p_feat.add_argument("--feature-suffix", default=".h5", help="Feature file extension.")
    p_feat.add_argument("--out", type=Path, required=True,
                        help="Output CSV path for the per-slide survival metadata.")
    p_feat.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "clinical":
        make_survival_metadata(
            sample_sheet_path=args.sample_sheet,
            clinical_tsv_path=args.clinical_tsv,
            out_path=args.out,
            slide_suffix=args.slide_suffix,
            verbose=not args.quiet,
        )
    elif args.command == "features":
        attach_feature_paths(
            clinical_csv=args.clinical_csv,
            feature_dir=args.feature_dir,
            out_path=args.out,
            feature_suffix=args.feature_suffix,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
