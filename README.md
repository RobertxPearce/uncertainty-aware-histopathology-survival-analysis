# Uncertainty-Aware Histopathology Survival Analysis

Comparing MC-Dropout, Deep Ensembles, and SNGP for uncertainty-aware survival prediction from TCGA-LUAD whole-slide pathology images.

## Pipeline

```mermaid
flowchart TD
    A["① Data Preprocessing · done\nsurvival table ↔ image ID mapping\ntrain / val / test split"]
    B["② Preprocessing · Trident + UNI\nsegmentation → patching → feature extraction\n256×256 px, 20× · 1024-dim · saved as .pt files"]
    C["③ MIL Aggregation · comparison\nABMIL + entropy reg. vs TransMIL → 512-dim\nselect by C-index"]
    D["④ Survival Training · SurvRNC\nSurvRNC contrastive loss + Cox loss\nsimilar OS → pull · different OS → push"]
    E["⑤ Uncertainty Estimation · innovation\nSNGP vs MC Dropout vs Deep Ensemble\nECE comparison · calibration quality"]
    F["⑥ Evaluation\nC-index · Kaplan-Meier · heatmap · uncertainty plot\nhigh / low risk split · ROI heatmap overlay"]

    A --> B --> C --> D --> E --> F

    style A fill:#f0f0f0,stroke:#999
    style B fill:#e8f5f5,stroke:#6bc
    style C fill:#ede8f5,stroke:#96c
    style D fill:#fdecea,stroke:#e88
    style E fill:#fef6e4,stroke:#e6a817
    style F fill:#eaf5ea,stroke:#6a6
```

## Datasets
- The Cancer Genome Atlas Program (TCGA) Lung Adenocarcinoma (LUAD)
    - Data Source: [National Cancer Institute GDC Data Portal](https://portal.gdc.cancer.gov/)
    - Manifest: [gdc_manifest_full_luad_dx.txt](manifests/gdc_manifest_full_luad_dx.txt)
    - Result: 478 cases, 541 files (some cases have multiple diagnostic slides)

    | Filter                | Value            |
    |------------------------|------------------|
    | Program               | TCGA             |
    | Project               | TCGA-LUAD        |
    | Access                | Open             |
    | Data Format           | svs              |
    | Data Type             | Slide Image      |
    | Experimental Strategy | Diagnostic Slide |

## Contributors
| Name                 | University                             |
|----------------------| -------------------------------------- |
| Robert Pearce        | University of Nevada, Las Vegas        |
| Sejun Park           | Gyeonggi Science Technology University |
| Hailey(Heejae Kwon)  | Sookmyung Womens University            |
| HyeonKyeong Lee      | Gyeongsang National University         |

## Acknowledgments
This project was completed during the International AI & Machine Learning Summer Camp hosted by the [University of Nevada, Las Vegas](https://www.unlv.edu/cs). Resources and guidance were provided by [Dr. Mingon Kang](https://kang.dataxlab.org/index.php).
