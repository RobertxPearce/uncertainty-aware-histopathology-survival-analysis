# Data Directory

## Datasets

- The Cancer Genome Atlas Program (TCGA) Lung Adenocarcinoma (LUAD)
    - Data Source: [National Cancer Institute GDC Data Portal](https://portal.gdc.cancer.gov/)
    - Manifest: [gdc_manifest_full_luad_dx.txt](../manifests/gdc_manifest_full_luad_dx.txt)
    - Result: 478 cases, 541 files (some cases have multiple diagnostic slides)

    | Filter                | Value            |
    |------------------------|------------------|
    | Program               | TCGA             |
    | Project               | TCGA-LUAD        |
    | Access                | Open             |
    | Data Format           | svs              |
    | Data Type             | Slide Image      |
    | Experimental Strategy | Diagnostic Slide |

- The Cancer Genome Atlas Program (TCGA) Glioma (GBMLGG: Glioblastoma + Lower-Grade Glioma)
    - Data Source: [National Cancer Institute GDC Data Portal](https://portal.gdc.cancer.gov/)
    - Manifest: [gdc_manifest_full_gbmlgg_dx.txt](../manifests/gdc_manifest_full_gbmlgg_dx.txt)
    - Result: 879 cases, 1,703 files (TCGA-GBM: 860 files, TCGA-LGG: 843 files; some cases have multiple diagnostic slides)

    | Filter                | Value                 |
    |------------------------|-----------------------|
    | Program               | TCGA                  |
    | Project               | TCGA-GBM, TCGA-LGG    |
    | Access                | Open                  |
    | Data Format           | svs                   |
    | Data Type             | Slide Image           |
    | Experimental Strategy | Diagnostic Slide      |

## Directory Structure

```
data/
├── raw/                      # As-downloaded GDC data, organized per cohort
│   ├── TCGA_LUAD/
│   │   ├── clinical/         # GDC clinical export (TSV)
│   │   │   ├── biospecimen/          # aliquot, analyte, portion, sample, slide
│   │   │   └── clinical_supplement/  # clinical, exposure, family_history, follow_up, pathology_detail
│   │   ├── gdc_sample_sheet.*.tsv    # File-to-case map for the downloaded cart
│   │   ├── metadata.cart.*.json      # Full GDC cart metadata
│   │   └── slides/                   # Whole-slide images (.svs) — gitignored
│   └── TCGA_GBMLGG/          # Same layout as TCGA_LUAD
│
├── interim/                  # Cleaned/merged intermediate tables (per cohort)
│   ├── survival_table_luad.csv             # Case-level survival labels (event, time)
│   └── matched_clinical_pilot100_luad.csv  # Clinical merge for the 100-slide pilot
│
└── processed/                # Model-ready artifacts
    ├── features/             # Extracted patch embeddings (.h5) — gitignored
    │   ├── uni_v1/
    │   └── uni_v2/
    ├── trident/              # Trident tiling output (thumbnails, contours) — gitignored
    └── experiments/          # Experiment definitions (tracked)
        └── <run>/
            ├── splits.csv            # Train/val/test fold assignments
            └── survival_metadata.csv # Per-case survival labels for the run
```

## Notes

- **Provenance.** Each cohort keeps its GDC `manifest`, `gdc_sample_sheet`, and
  `metadata.cart.json` so the exact download can be reproduced. Manifests live
  under the top-level [`manifests/`](../manifests/) directory.
