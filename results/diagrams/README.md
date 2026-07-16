# Mermaid Diagrams

## Study Overview
```mermaid
flowchart LR
    A["WSIs<br/><small>TCGA-GBMLGG · 877 patients</small>"]
    B["UNI v2 Features<br/><small>[N, 1536] per slide</small>"]
    C["ABMIL + Cox Backbone<br/><small>attention MIL → risk score</small>"]
    D["Uncertainty Methods<br/><small>MC-dropout · ensemble · SNGP</small>"]
    E["Risk + Uncertainty<br/><small>risk mean and std</small>"]
    F["Evaluation<br/><small>C-index · selective AUC · 5-fold CV</small>"]
    A --> B --> C --> D --> E --> F
    classDef io   fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A;
    classDef data fill:#E6F1FB,stroke:#185FA5,color:#042C53;
    classDef model fill:#E1F5EE,stroke:#0F6E56,color:#04342C;
    classDef uq   fill:#FAEEDA,stroke:#BA7517,color:#412402;
    classDef eval fill:#EEEDFE,stroke:#534AB7,color:#26215C;
    class A io;
    class B data;
    class C model;
    class D uq;
    class E io;
    class F eval;
```

## Cohort Flow GBMLGG
```mermaid
flowchart TD
    ALL["TCGA-GBMLGG Cohort<br/><small>877 patients · 877 slides · 1 slide / patient</small>"]
    ALL --> OUT["Outcomes<br/><small>445 events · 432 censored</small>"]
    ALL --> CV["5-Fold Event-Stratified CV"]
    CV --> TEST["Held-Out Test<br/><small>~175 patients / fold</small>"]
    CV --> DEV["Development ~702<br/><small>train + 25% val (early stopping)</small>"]
    classDef io   fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A;
    classDef ev   fill:#FBEAF0,stroke:#993556,color:#4B1528;
    classDef step fill:#E6F1FB,stroke:#185FA5,color:#042C53;
    class ALL io;
    class OUT ev;
    class CV,TEST,DEV step;
```

## Feature Extraction Pipeline
```mermaid
flowchart LR
    WSI["Whole-Slide Image<br/><small>TCGA .svs</small>"]
    SEG["Tissue Segmentation<br/><small>TRIDENT · 10×</small>"]
    COORD["Patch Coordinates<br/><small>256 px @ 20×</small>"]
    RESIZE["Resize Patches<br/><small>256 → 224 px</small>"]
    ENC["UNI v2 Encoder<br/><small>UNI2-h foundation model</small>"]
    BAG["Feature Bag<br/><small>[N, 1536] · one .h5 per slide</small>"]
    WSI --> SEG --> COORD --> RESIZE --> ENC --> BAG
    classDef io   fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A;
    classDef step fill:#E6F1FB,stroke:#185FA5,color:#042C53;
    classDef out  fill:#E1F5EE,stroke:#0F6E56,color:#04342C;
    class WSI io;
    class SEG,COORD,RESIZE,ENC step;
    class BAG out;
```

## ABMIL Cox Architecture Diagram
```mermaid
flowchart TB
 subgraph ATT["Gated Attention (Ilse et al.)"]
    direction TB
        GATE["tanh(V·h) ⊙ sigmoid(U·h)<br><small>V, U: Linear 128→128</small>"]
        SCORE["Linear 128→1 → softmax over N<br><small>padding masked → weights aₙ</small>"]
  end
 subgraph ENC["ABMIL Encoder"]
    direction TB
        NORM["Input LayerNorm<br><small>over 1536 dims</small>"]
        PROJ["Patch Projection<br><small>Linear 1536→128 · ReLU · Dropout 0.25</small>"]
        ATT
        POOL["Attention Pooling<br><small>Σₙ aₙ·hₙ → [B, 128]</small>"]
        PNORM["Pool LayerNorm<br><small>[B, 128]</small>"]
  end
    GATE --> SCORE
    IN["Patch Feature Bag<br><small>[B, N, 1536] · UNI v2 + mask</small>"] --> NORM
    NORM --> PROJ
    PROJ -- attention features h --> GATE
    PROJ -- values h --> POOL
    SCORE -- weights aₙ --> POOL
    POOL --> PNORM
    PNORM --> HEAD["Cox Risk Head<br><small>Linear 128→1 → risk [B]</small>"]
    HEAD --> LOSS["Cox Partial-Likelihood Loss<br><small>risk set = one batch of 96</small>"]
    PNORM -. SNGP model only .-> SNGP["SNGP Head (variant)<br><small>RFF 1024 → risk + GP variance</small>"]

     IN:::io
     NORM:::enc
     PROJ:::enc
     GATE:::att
     SCORE:::att
     POOL:::enc
     PNORM:::enc
     HEAD:::head
     LOSS:::loss
     SNGP:::sngp
    classDef io    fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef enc   fill:#E6F1FB,stroke:#185FA5,color:#042C53
    classDef att   fill:#FAEEDA,stroke:#BA7517,color:#412402
    classDef head  fill:#EAF3DE,stroke:#3B6D11,color:#173404
    classDef loss  fill:#EEEDFE,stroke:#534AB7,color:#26215C
    classDef sngp  fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-dasharray:4 3
    style ENC fill:transparent
```

## Uncertainty Quantification Methods Pipeline
```mermaid
flowchart TB
    MCH["<b>MC Dropout</b><br>1 model"] --> MC1["Train One Model<br><small>dropout layers on</small>"]
    MC1 --> MC2["100 Forward Passes<br><small>dropout kept on</small>"]
    MC2 --> MC3["Mean + std<br><small>across passes</small>"]
    DEH["<b>Deep Ensemble</b><br>N models"] --> DE1["Train N Models<br><small>independent runs</small>"]
    DE1 --> DE2["One Pass Each<br><small>deterministic</small>"]
    DE2 --> DE3["Mean + std<br><small>across members</small>"]
    SNH["<b>SNGP</b><br>1 model"] --> SN1["Train One Model<br><small>spectral norm + GP</small>"]
    SN1 --> SN2["Fit GP Covariance<br><small>over training set</small>"]
    SN2 --> SN3["One Pass<br><small>risk + variance</small>"]
    IN["UNI V2 Patch Features"] --> MCH & DEH & SNH
    MC3 --> OUT["<b>Risk Mean + Risk std</b>"]
    DE3 --> OUT
    SN3 --> OUT
    OUT --> EVAL["<b>Selective AUC</b><br><small>sort by risk standard deviation <br> C-index vs coverage</small>"]

     MCH:::blue
     MC1:::blue
     MC2:::blue
     MC3:::blue
     DEH:::amber
     DE1:::amber
     DE2:::amber
     DE3:::amber
     SNH:::green
     SN1:::green
     SN2:::green
     SN3:::green
     IN:::gray
     OUT:::gray
     EVAL:::purple
    classDef blue fill:#E6F1FB,stroke:#185FA5,color:#042C53
    classDef amber fill:#FAEEDA,stroke:#BA7517,color:#412402
    classDef green fill:#EAF3DE,stroke:#3B6D11,color:#173404
    classDef gray fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef purple fill:#EEEDFE,stroke:#534AB7,color:#26215C
```

## SNGP GP Head
```mermaid
flowchart TB
 subgraph FIT["Covariance Fit (once, after training)"]
    direction TB
        F1["Pass Over TRAIN Set"]
        F2["Accumulate Precision<br><small>Σ φφᵀ + ridge 1.0</small>"]
        F3["Invert → Covariance"]
  end
    EMB["Slide Embedding<br><small>[B, 128] · spectral-norm encoder (bound 0.95)</small>"] --> RFF["Random Fourier Features<br><small>1024 features</small>"]
    RFF --> GP["GP Output Layer<br><small>Laplace approximation</small>"]
    GP --> MEAN["Posterior Mean → Risk"] & VAR["Posterior Variance → Risk std"]
    F1 --> F2
    F2 --> F3
    F3 -. feeds variance .-> GP

     EMB:::enc
     RFF:::enc
     GP:::gp
     MEAN:::out
     VAR:::out
     F1:::fit
     F2:::fit
     F3:::fit
    classDef enc  fill:#E6F1FB,stroke:#185FA5,color:#042C53
    classDef gp   fill:#E1F5EE,stroke:#0F6E56,color:#04342C
    classDef out  fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef fit  fill:#FAEEDA,stroke:#BA7517,color:#412402
    style FIT fill:transparent
```

## Cross Validation Design
```mermaid
flowchart TB
 subgraph FOLD["Per Fold (identical splits for all methods)"]
    direction TB
        HELD["Held-Out Test Fold<br><small>~175 patients</small>"]
        DEV["Remaining ~702<br><small>25% carved for val / early stopping</small>"]
        BB["Train Shared Backbones<br><small>5 ensemble members + 1 SNGP</small>"]
        SC["Score 3 UQ Methods<br><small>MC-dropout · ensemble · SNGP</small>"]
  end
    COH["877 patients<br><small>event-stratified</small>"] --> FOLDS["5-fold split"]
    DEV --> BB
    BB --> SC
    HELD --> SC
    FOLDS --> FOLD
    FOLD --> POOL["Pool Predictions Over Folds<br><small>C-index CI + selective AUC per method</small>"]

     COH:::io
     FOLDS:::io
     HELD:::io
     DEV:::io
     BB:::step
     SC:::step
     POOL:::out
    classDef io   fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef step fill:#E6F1FB,stroke:#185FA5,color:#042C53
    classDef out  fill:#EEEDFE,stroke:#534AB7,color:#26215C
    style FOLD fill:transparent
```

## Selective Prediction Explainer
```mermaid
flowchart TD
    IN["Per-Patient Risk Mean + std"]
    SORT["Sort by Risk std<br/><small>most confident first</small>"]
    SWEEP["Sweep Coverage k/n<br/><small>keep k most-confident patients</small>"]
    CIDX["Recompute C-Index<br/><small>on the retained k</small>"]
    CURVE["Coverage vs C-Index Curve<br/><small>good UQ = curve falls as k grows</small>"]
    AUC["Selective AUC<br/><small>normalised area · higher = more useful UQ</small>"]
    IN --> SORT --> SWEEP --> CIDX --> CURVE --> AUC
    classDef io   fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A;
    classDef step fill:#E6F1FB,stroke:#185FA5,color:#042C53;
    classDef out  fill:#EEEDFE,stroke:#534AB7,color:#26215C;
    class IN io;
    class SORT,SWEEP,CIDX,CURVE step;
    class AUC out;
```
