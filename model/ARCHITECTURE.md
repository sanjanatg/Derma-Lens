# DermaLens — FairDermNet: Architecture & System Walkthrough

## Project Goal

A fairness-aware, explainable deep learning pipeline for skin cancer risk assessment. The system classifies dermoscopic lesion images into **8 skin condition classes** while maintaining equitable performance across all six Fitzpatrick skin tones (I–VI). The trained model is exported as a TFLite INT8 file for deployment in a Flutter mobile app.

Two primary design pillars:
1. **Clinical accuracy** — ≥96% sensitivity for malignant lesion detection (classes 0–3).
2. **Algorithmic fairness** — equalized odds gap <5% across Fitzpatrick skin tone groups.

---

## Repository Structure

```
model/                           ← project root (pip-installable as dermalens-fairdermnet)
│
├── notebooks/
│   └── dermalens_train.ipynb    ← end-to-end training notebook (15 cells)
│
├── model/                       ← core neural network package
│   ├── __init__.py
│   ├── fairdermet.py            ← FairDermNet model architecture
│   ├── fairness_attention.py    ← FairnessAttentionModule
│   ├── gem_pooling.py           ← Generalized Mean Pooling
│   ├── loss_functions.py        ← multi-task fairness-aware loss
│   └── class_taxonomy.py        ← 8-class label definitions & dataset mappings
│
├── preprocessing/
│   ├── __init__.py
│   ├── hair_removal.py          ← black top-hat morphological hair removal + inpainting
│   ├── lesion_segmentation.py   ← Otsu thresholding lesion segmentation
│   ├── skin_tone_estimation.py  ← ITA° → Fitzpatrick skin tone estimation
│   ├── adaptive_normalization.py← CLAHE contrast enhancement
│   └── augmentation_pipeline.py ← Albumentations train/val/TTA pipelines
│
├── training/
│   ├── __init__.py
│   ├── trainer.py               ← optimizer, scheduler, training step helpers
│   ├── fairness_metrics.py      ← equalized odds gap, demographic parity gap
│   └── callbacks.py             ← EarlyStopping, ModelCheckpoint
│
├── export/
│   ├── __init__.py
│   ├── onnx_export.py           ← PyTorch → ONNX (opset 17)
│   └── quantize_tflite.py       ← ONNX → TFLite INT8 quantization
│
├── explainability/
│   ├── __init__.py
│   └── gradcam_plus_plus.py     ← GradCAM++ heatmaps + ABCDE feature scores
│
├── configs/
│   └── train_config.yaml        ← all hyperparameters
│
├── requirements.txt
├── setup.py
├── SETUP.md                     ← environment setup, kernel fix, known TODOs
└── ARCHITECTURE.md              ← this file
```

---

## The 8-Class Taxonomy

Defined in `model/class_taxonomy.py`:

| Index | Condition | Short | Risk Tier |
|---|---|---|---|
| 0 | Melanoma | MEL | High (malignant) |
| 1 | Basal Cell Carcinoma | BCC | High (malignant) |
| 2 | Squamous Cell Carcinoma | SCC | High (malignant) |
| 3 | Actinic Keratosis | AK | Medium (malignant / pre-cancerous) |
| 4 | Melanocytic Nevi | NV | Low (benign, dominant class) |
| 5 | Benign Keratosis | BKL | Low (benign) |
| 6 | Dermatofibroma | DF | Low (benign) |
| 7 | Vascular Lesion | VASC | Low (benign) |

Malignant = classes 0–3. Benign = classes 4–7.

> **Note on Bowen's Disease (BOD):** BOD was excluded from the final taxonomy
> because it is absent from all four available datasets (HAM10000, ISIC2019,
> PAD-UFES-20, DDI). The model trains on 8 classes. BOD can be added in a
> future experiment using images sourced from the ISIC archive.

---

## FairDermNet Architecture

Defined in `model/fairdermet.py`. Built on a **MobileNetV3-Large** backbone (pretrained ImageNet via `timm`), extended with a custom **FairnessAttentionModule**.

```
Input: 224×224×3 RGB (ImageNet-normalised)
    ↓
MobileNetV3-Large backbone (timm, pretrained)
    → feature map [B, 960, 7, 7]
    ↓
FairnessAttentionModule  (model/fairness_attention.py)
    ├── Channel Attention   (SE-style squeeze-excite, reduction=16)
    ├── Spatial Attention   (7×7 conv on avg+max pool concat)
    ├── FitzpatrickAuxBranch → skin_tone_logits [B, 6]   ← training only
    └── Tone-conditioned cross-attention modulator
        → attended feature map [B, 960, 7, 7]
    ↓
GeMPooling  (model/gem_pooling.py)
    learnable exponent p (init=3.0); p≈3 outperforms avg/max for fine-grained textures
    → [B, 960]
    ↓
Neck: Dropout(0.3) → Linear(960→128) → BatchNorm1d → ReLU
    → [B, 128]
    ↓
Risk Head: Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→9)
    → risk_logits [B, 9]

Training output: (risk_logits, skin_tone_logits)
Inference output: risk_logits only
```

**Key design decision — dual-head, single-head inference:** The Fitzpatrick auxiliary branch is fully active during training (gradient signal via `L_fitzpatrick`, drives cross-attention). At inference time it is discarded — zero runtime cost. The model learns to be tone-invariant without paying for it at deployment.

> **Note:** Parameter count and quantized size are to be verified after first training run.
> Estimate: ~5.4M parameters, ~5.8 MB INT8. The FairnessAttentionModule adds on top of
> vanilla MobileNetV3-Large — re-check with `sum(p.numel() for p in model.parameters())`.

---

## Fairness-Aware Loss Function

Defined in `model/loss_functions.py` as `FairnessAwareLoss`.

```
Total Loss = α × L_class + β × L_fitzpatrick + γ × L_fairness

  L_class        = 0.5 × FocalCE + 0.5 × WeightedCE
                   (class-imbalance aware; NV has ~19,580 samples vs BOD ~363)

  L_fitzpatrick  = CrossEntropy(skin_tone_logits, skin_tone_labels)

  L_fairness     = max(sensitivity_i) − min(sensitivity_i)
                   across Fitzpatrick groups i ∈ {0..5}
                   sensitivity = soft-differentiable TPR for malignant classes (0–4)
```

Default weights (from `configs/train_config.yaml`): **α=1.0, β=0.15, γ=0.25**.

Class weights are computed as inverse-frequency, normalised so NV=1.0.

The fairness regularisation term directly minimises the equalized odds gap during backprop — this is a gradient-level intervention, not post-hoc evaluation.

---

## Datasets

Four datasets are merged into a unified `master_dataset.csv` with columns:
`image_path, class_idx, skin_tone, dataset, dx_raw`.

| Dataset | Classes | Size (approx) | Skin Tone | Role |
|---|---|---|---|---|
| **HAM10000** | MEL, BCC, AK, NV, BKL, DF, VASC | ~10,000 | ITA°-estimated | Training |
| **ISIC 2019** | MEL, BCC, SCC, AK, NV, BKL, DF, VASC | ~25,000 | ITA°-estimated | Training |
| **PAD-UFES-20** | MEL, BCC, SCC, AK, BOD, BKL | ~2,298 | Fitzpatrick metadata | Training; only source of Bowen's (class 4) |
| **DDI (Stanford)** | Binary (mal/benign) | ~656 | Expert-labelled (FST I–II, III–IV, V–VI) | Held-out fairness evaluation only |

Stratified 70/15/15 train/val/test split, stratified by `class_idx + dataset`.

HAM10000 and ISIC 2019 lack Fitzpatrick labels. Skin tone is estimated automatically from surrounding skin pixels using the ITA° formula (see Preprocessing below). Results are cached in `.tone_cache.csv` files.

---

## Preprocessing Pipeline

Each module is independent and composable:

| Module | What it does |
|---|---|
| `hair_removal.py` | Black top-hat morphological transform (17×17 kernel) + inpainting to erase hair artefacts |
| `lesion_segmentation.py` | Otsu thresholding on LAB L-channel + morphological open/close → binary lesion mask |
| `skin_tone_estimation.py` | Computes ITA° = arctan((L−50)/b) in CIELAB from surrounding skin pixels; maps to Fitzpatrick I–VI (index 0–5) |
| `adaptive_normalization.py` | CLAHE on LAB L-channel for skin-tone-adaptive contrast enhancement |
| `augmentation_pipeline.py` | Three Albumentations pipelines: **train** (heavy spatial + colour + dropout augmentation), **val** (resize + normalise only), **TTA** (8 deterministic flips/rotations for test-time augmentation ensemble) |

---

## Training Configuration

All hyperparameters live in `configs/train_config.yaml`.

| Setting | Value |
|---|---|
| Backbone | MobileNetV3-Large (pretrained) |
| Input size | 224×224 |
| Batch size | 32 (effective 64 with 2× gradient accumulation) |
| Epochs | 50 (config); 30 (notebook setting for RTX 3050 6 GB) |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10) |
| Precision | fp16 AMP |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=7–8 on val balanced accuracy |
| Loss weights | α=1.0, β=0.15, γ=0.25 |

Training utilities:
- `training/trainer.py` — `build_optimizer`, `build_scheduler`, `training_step`
- `training/callbacks.py` — `EarlyStopping`, `save_checkpoint` (saves epoch, state_dict, optimizer state, metric)
- `training/fairness_metrics.py` — `group_sensitivity`, `equalized_odds_gap`, `demographic_parity_gap`, `performance_parity_gap`

---

## Notebook Cell Map (`notebooks/dermalens_train.ipynb` — 15 cells)

| Cell | Purpose |
|---|---|
| 1 | GPU health check (targets RTX 3050 6 GB); enables TF32 and cuDNN benchmark |
| 2 | All configurable hyperparameters in one place |
| 3 | Imports all project modules |
| 4 | Dataset loading — `build_ham10000_df`, `build_isic2019_df`, `build_pad_ufes_df`, `build_ddi_df`; merges into `master_dataset.csv` |
| 5 | `SkinLesionDataset` (PyTorch Dataset), train/val/test split, DataLoader creation; estimates missing skin tones via ITA° |
| 6 | Model instantiation (8-class), class weight computation, `FairnessAwareLoss` setup, VRAM check |
| 7 | Full training loop: AMP, gradient accumulation, gradient clipping, EarlyStopping, ModelCheckpoint, per-epoch fairness gap logging |
| 8 | Training curve plots: loss, balanced accuracy, overall accuracy, equalized odds gap |
| 9 | Test set evaluation: 8-class classification report, confusion matrix, per-class accuracy bar chart, binary sensitivity/specificity |
| 9b | DDI held-out fairness evaluation: per-FST-group sensitivity/specificity with expert-labelled skin tones |
| 9c | Clinical threshold analysis: ROC curve, threshold sweep (0.20–0.75), confusion matrices at 0.50 vs optimal, saves `threshold_config.json` for Flutter |
| 10 | Fairness bar charts — sensitivity & specificity per Fitzpatrick group I–VI |
| 11 | GradCAM++ visualisations on 3 test samples (original, heatmap overlay, raw heatmap) |
| 12 | Export to ONNX (opset 17) + TFLite INT8 (<6 MB target) |
| 13 | Final summary printout of all key metrics |

> Cells 9b and 9c were added as part of the multiclass patch. Any reference to "13 cells" elsewhere is stale.

---

## Explainability

`explainability/gradcam_plus_plus.py` exposes two functions:

- `generate_gradcam(model, input_tensor, target_layer_name)` — wraps `pytorch-grad-cam`'s `GradCAMPlusPlus`. Returns a heatmap overlay image and the raw grayscale heatmap. Applied to the last convolutional block of MobileNetV3 (7×7 spatial resolution) via `model.get_gradcam_target_layer()`.
- `abcde_feature_scores(image, heatmap)` — heuristic ABCDE dermoscopy rule extraction from the GradCAM attention region: Asymmetry, Border irregularity, Colour variety, Diameter, Evolution (hardcoded 0.0 — temporal data unavailable).

---

## Model Export

```
PyTorch model (eval mode)
    → onnx_export.py   → model.onnx       (opset 17, dynamic batch axis)
    → quantize_tflite.py → model.tflite   (INT8, target <6 MB, <25ms on Snapdragon 720G)
    → threshold_config.json               (clinical threshold for Flutter inference logic)
```

The clinical threshold is chosen at the ROC point where malignant sensitivity first reaches ≥96%, reflecting the clinical priority that false negatives (missed cancers) are far more costly than false positives. This threshold is saved for use in the Flutter app as `const double clinicalThreshold = X.XXX`.

---

## End-to-End Data Flow

```
Raw dermoscopy images (HAM10000 + ISIC2019 + PAD-UFES-20)
    → Hair removal (top-hat + inpaint)
    → Lesion segmentation (Otsu LAB)
    → Skin tone estimation (ITA° → Fitzpatrick 0–5)
    → CLAHE adaptive contrast normalisation
    → Albumentations augmentation (train: heavy | val/test: resize+normalise)
    → SkinLesionDataset → DataLoader
    ↓
FairDermNet forward pass
    MobileNetV3 → FairnessAttentionModule → GeMPool → Neck → Risk Head
    ↓
FairnessAwareLoss (α×focal_CE + β×fitzpatrick_CE + γ×equalized_odds_gap)
    ↓
AdamW + CosineAnnealing + AMP fp16 + gradient accumulation
    ↓
EarlyStopping → best.pth (ModelCheckpoint)
    ↓
Test evaluation: 8-class metrics + binary sensitivity/specificity
    ↓
DDI held-out fairness evaluation (expert-labelled skin tones)
    ↓
GradCAM++ explainability + ABCDE scores
    ↓
ONNX export → TFLite INT8 → Flutter mobile app (on-device inference)
```
