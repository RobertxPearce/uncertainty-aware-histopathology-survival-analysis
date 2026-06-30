"""
Build a patient-level survival table from the TCGA-LUAD GDC data.

Queries GDC REST API to get:
  - file_id -> case_id mapping for all SVS files on disk
  - clinical data (vital_status, days_to_death, days_to_last_follow_up) per case

Input:  manifests/gdc_manifest_full_luad_dx.txt + data/slides/
Output: data/interim/matched_clinical_slides.csv
"""

import time
from pathlib import Path

import pandas as pd
import requests

ROOT          = Path(__file__).parent.parent
MANIFEST_PATH = ROOT / "manifests" / "gdc_manifest_full_luad_dx.txt"
SLIDES_DIR    = ROOT / "data" / "slides"
OUT_PATH      = ROOT / "data" / "interim" / "matched_clinical_slides.csv"

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
BATCH         = 100  # max items per API request


def gdc_post(url, payload, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Retry {attempt+1}/{retries}: {e}")
            time.sleep(5)


# ── 1. Check slides on disk ───────────────────────────────────────────────────
manifest = pd.read_csv(MANIFEST_PATH, sep="\t")
all_file_ids = manifest["id"].tolist()
print(f"Manifest file count: {len(all_file_ids)}")

valid_file_ids = [
    fid for fid in all_file_ids
    if list((SLIDES_DIR / fid).glob("*.svs"))
]
print(f"Files with SVS on disk: {len(valid_file_ids)} / {len(all_file_ids)}")

# ── 2. Fetch file -> case mapping (GDC API) ───────────────────────────────────
print("\nFetching file-case mapping from GDC API...")
file_case_rows = []
for i in range(0, len(valid_file_ids), BATCH):
    batch = valid_file_ids[i : i + BATCH]
    data = gdc_post(GDC_FILES_URL, {
        "filters": {"op": "in", "content": {"field": "file_id", "value": batch}},
        "fields": "file_id,file_name,cases.case_id,cases.submitter_id",
        "size": BATCH,
    })
    for hit in data["data"]["hits"]:
        for case in hit.get("cases", []):
            file_case_rows.append({
                "file_id":      hit["file_id"],
                "file_name":    hit["file_name"],
                "case_id":      case["case_id"],
                "submitter_id": case["submitter_id"],
            })
    print(f"  {min(i+BATCH, len(valid_file_ids))}/{len(valid_file_ids)} done")

file_case_df = pd.DataFrame(file_case_rows)
print(f"\nFile-case rows: {len(file_case_df)}")
print(f"Unique cases:   {file_case_df['case_id'].nunique()}")

# ── 3. Fetch clinical data (GDC API) ─────────────────────────────────────────
print("\nFetching clinical data from GDC API...")
case_ids = file_case_df["case_id"].unique().tolist()
print(f"Unique cases: {len(case_ids)}")

clinical_rows = []
for i in range(0, len(case_ids), BATCH):
    batch = case_ids[i : i + BATCH]
    data = gdc_post(GDC_CASES_URL, {
        "filters": {"op": "in", "content": {"field": "case_id", "value": batch}},
        "fields": (
            "case_id,submitter_id,"
            "demographic.vital_status,"
            "demographic.days_to_death,"
            "diagnoses.days_to_last_follow_up"
        ),
        "size": BATCH,
        "expand": "demographic,diagnoses",
    })
    for hit in data["data"]["hits"]:
        demo  = hit.get("demographic") or {}
        diags = hit.get("diagnoses") or []
        lfu_vals = [d.get("days_to_last_follow_up") for d in diags
                    if d.get("days_to_last_follow_up") is not None]
        clinical_rows.append({
            "case_id":                hit["case_id"],
            "vital_status":           demo.get("vital_status"),
            "days_to_death":          demo.get("days_to_death"),
            "days_to_last_follow_up": max(lfu_vals) if lfu_vals else None,
        })
    print(f"  {min(i+BATCH, len(case_ids))}/{len(case_ids)} done")

clinical_df = pd.DataFrame(clinical_rows)
print(f"\nClinical data rows: {len(clinical_df)}")

# ── 4. Set event and time ─────────────────────────────────────────────────────
clinical_df["days_to_death"]          = pd.to_numeric(clinical_df["days_to_death"],          errors="coerce")
clinical_df["days_to_last_follow_up"] = pd.to_numeric(clinical_df["days_to_last_follow_up"], errors="coerce")

clinical_df["event"] = (clinical_df["vital_status"] == "Dead").astype(int)
print(f"\nDead   (event=1): {clinical_df['event'].sum()}")
print(f"Alive  (event=0): {(clinical_df['event']==0).sum()}")

clinical_df["time"] = clinical_df["days_to_death"].where(
    clinical_df["event"] == 1,
    clinical_df["days_to_last_follow_up"],
)
clinical_df["time"] = pd.to_numeric(clinical_df["time"], errors="coerce")
print(f"Missing time: {clinical_df['time'].isna().sum()} / {len(clinical_df)}")

# ── 5. Merge and group by patient ─────────────────────────────────────────────
merged = file_case_df.merge(clinical_df, on="case_id", how="left")

survival_table = (
    merged.groupby("submitter_id")
    .agg(
        case_id      =("case_id",      "first"),
        submitter_id =("submitter_id", "first"),
        file_ids     =("file_id",      list),
        file_names   =("file_name",    list),
        n_slides     =("file_id",      "count"),
        vital_status =("vital_status", "first"),
        event        =("event",        "first"),
        time         =("time",         "first"),
    )
    .reset_index(drop=True)
)

# Drop patients with no time data
survival_table = survival_table[survival_table["time"].notna()].copy()

print(f"\nFinal survival table: {len(survival_table)} patients")
print(f"\t{int(survival_table['event'].sum())} events (deaths)")
print(f"\t{int((survival_table['event']==0).sum())} censored (alive)")
print(f"\tEvent rate: {survival_table['event'].mean()*100:.1f}%")
print(f"\nSlides per patient:")
print(survival_table["n_slides"].value_counts().sort_index())

# ── 6. Save ───────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
survival_table.to_csv(OUT_PATH, index=False)
print(f"\nSaved to: {OUT_PATH}")

# ── 7. Validate ───────────────────────────────────────────────────────────────
print(f"\n=== Validation ===")
print(f"Manifest files:          {len(all_file_ids)}")
print(f"Files with SVS on disk:  {len(valid_file_ids)}")
print(f"Cases with slides:       {file_case_df['case_id'].nunique()}")
print(f"Final table:             {len(survival_table)} cases (slides + time available)")
print(f"\nDropped:")
print(f"\t{len(all_file_ids) - len(valid_file_ids)} files — SVS not on disk")
print(f"\t{file_case_df['case_id'].nunique() - len(survival_table)} cases — missing follow-up time")
