# DermaLens — FairDermNet

**A fairness-aware, explainable deep-learning pipeline for on-device skin cancer risk assessment.**

DermaLens is a final-year engineering project with two components:

| Component | Directory | What it is |
|---|---|---|
| **Model** | [`model/`](model/) | Python deep-learning pipeline — FairDermNet (MobileNetV3-Large + fairness attention + GeM pooling), 8-class training notebook, fairness metrics, GradCAM++ explainability, and PyTorch → ONNX → TFLite export (quantized, ≈5.5 MB). |
| **Mobile app** | [`dermalens_flutter/`](dermalens_flutter/) | Flutter app that captures a lesion photo with the phone camera and runs fully **offline, on-device** inference with the exported TFLite model. |

The system classifies dermoscopic/smartphone lesion images into **8 skin-condition classes** while training explicitly to keep performance equitable across Fitzpatrick skin tones (I–VI). The trained model is quantized (int8 weights, float32 activations — dynamic-range quantization) for a ≈5.5 MB on-device footprint.

> ⚠️ **Not a medical device.** The app is a decision-support prototype for an academic project. It does not provide a diagnosis. Do not rely on it for medical decisions.

---

## 1. The 8-Class Taxonomy (locked)

The project is strictly **8-class**. This order is a hard invariant — it must match everywhere: Python taxonomy, ONNX/TFLite export, `labels.txt`, and the Flutter UI.

| Index | Class | Short | Risk tier |
|---|---|---|---|
| 0 | Melanoma | MEL | High (malignant) |
| 1 | Basal Cell Carcinoma | BCC | High (malignant) |
| 2 | Squamous Cell Carcinoma | SCC | High (malignant) |
| 3 | Actinic Keratosis | AK | Medium (pre-cancerous) |
| 4 | Melanocytic Nevi | NV | Low (benign) |
| 5 | Benign Keratosis | BKL | Low (benign) |
| 6 | Dermatofibroma | DF | Low (benign) |
| 7 | Vascular Lesion | VASC | Low (benign) |

**Binary clinical grouping:** malignant = indices `0–3`, benign = indices `4–7`.

Bowen's Disease was deliberately **excluded** — it is absent from all four datasets used (HAM10000, ISIC 2019, PAD-UFES-20, DDI). Do not re-introduce it as a standalone class.

- **Source of truth:** [`model/model/class_taxonomy.py`](model/model/class_taxonomy.py)
- **Design rationale & full walkthrough:** [`model/ARCHITECTURE.md`](model/ARCHITECTURE.md)
- **AI-agent operating instructions:** [`GEMINI.md`](GEMINI.md) (single source of truth for agents working in this repo)

---

## 2. Repository Layout

```
.
├── model/                          # Python deep-learning pipeline (pip-installable as dermalens-fairdermnet)
│   ├── model/                      #   FairDermNet architecture, fairness attention, GeM pooling,
│   │                               #   fairness-aware loss, 8-class taxonomy
│   ├── preprocessing/              #   Hair removal, lesion segmentation, skin-tone (ITA°) estimation,
│   │                               #   CLAHE normalization, Albumentations augmentation
│   ├── training/                   #   Trainer helpers, callbacks, fairness metrics
│   ├── export/                     #   onnx_export.py + quantize_tflite.py (PyTorch → ONNX → TFLite)
│   ├── explainability/             #   GradCAM++ heatmaps + ABCDE feature scores
│   ├── notebooks/
│   │   └── dermalens_train.ipynb   #   End-to-end training notebook (numbered cells 1–13 + 9b/9c)
│   ├── configs/train_config.yaml   #   All hyperparameters
│   ├── checkpoints/                #   best.pth, training_history.json, DDI fairness report
│   ├── datasets/                   #   Merged CSVs, train/val/test splits, tone caches, DDI eval
│   ├── export_pipeline.py          #   best.pth → ONNX → TFLite → Flutter assets (one script)
│   ├── validate_export.py          #   PyTorch vs ONNX vs TFLite output parity checks
│   ├── ARCHITECTURE.md             #   Full architecture & system walkthrough
│   └── SETUP.md                    #   Python 3.10 env setup, Jupyter kernel fix, known TODOs
│
├── dermalens_flutter/              # Flutter mobile app (Riverpod, tflite_flutter, camera)
│   ├── lib/
│   │   ├── main.dart               #   App entry — onboarding / home routing
│   │   ├── screens/                #   onboarding, home, scan, processing, results, history,
│   │   │                           #   symptom checker, lesion guide, dermatologist, profile, about
│   │   ├── services/
│   │   │   └── inference_service.dart  #   TFLite inference + preprocessing (NCHW, ImageNet norms)
│   │   ├── providers/              #   Riverpod state (analysis, history, navigation)
│   │   └── theme/                  #   "Tactile Apothecary" app theme
│   └── assets/
│       ├── labels.txt              #   8 class names, in index order
│       └── threshold_config.json   #   Clinical threshold + model metadata
│
├── ppt_files/                      # Academic deliverables (synopsis, reports, presentations)
└── flutter app stitch design/      # UI design artifacts & screen HTML mockups
```

---

## 3. Model — Setup & Training

> Requires **Python 3.10** specifically (PyTorch 2.x CUDA wheels and TensorFlow 2.13 tested on 3.10; newer Python versions have caused CUDA init failures). Full details in [`model/SETUP.md`](model/SETUP.md).

```bash
cd model
pip install -r requirements.txt
```

1. Open `notebooks/dermalens_train.ipynb` under the **DermaLens (Python 3.10)** kernel (see `SETUP.md` for the one-time kernel registration).
2. Run cells 1–7 to train (RTX 3050 6 GB optimised; ~30 epochs in the notebook).
3. Cells 8–10 plot curves and run test-set + DDI held-out fairness evaluation.
4. Cell 11 produces GradCAM++ explainability visualisations.
5. Cell 9c computes the clinical threshold (target: ≥96% sensitivity for malignant classes) and writes `threshold_config.json`.

**Key hyperparameters** (`model/configs/train_config.yaml`): MobileNetV3-Large backbone, 224×224 input, AdamW (lr 1e-4), cosine annealing with warm restarts, AMP fp16, gradient accumulation ×2, loss weights α=1.0 / β=0.15 / γ=0.25 (classification / Fitzpatrick aux / equalized-odds fairness gap).

### Export pipeline

```bash
cd model
python export_pipeline.py     # best.pth → fairdermnet.onnx → fairdermnet_int8.tflite + labels.txt + threshold_config.json
python validate_export.py     # PyTorch vs ONNX vs TFLite parity (top-1 match, L∞ drift, mal/ben decision match)
```

`export_pipeline.py` writes the artifacts to `model/export/` **and copies them into `dermalens_flutter/assets/`** so the Flutter app can pick them up directly.

**Quantization note (why dynamic-range, not full INT8).** The exported model uses int8 weights with float32 activations (≈5.5 MB). Full INT8 — quantizing activations too — was tried and validated as catastrophically inaccurate (3% top-1 match vs PyTorch): the GeM pooling power op (`x^p`, learned `p≈2.88`) produces intermediates spanning ~±1.7M, so its INT8 activation scale is ~13615 and all precision is destroyed. Dynamic-range quantization keeps the power op in float32, so validated real-image performance is statistically identical to PyTorch (sensitivity 89.1% vs 90.9% at the clinical threshold, 150 held-out test images).

---

## 4. Mobile App — Setup & Build

The Flutter app (`dermalens_flutter/`) is a complete Riverpod-based app with onboarding, camera capture, scan processing, results with risk-tier messaging, history, a lesion guide (ABCDE education), symptom checker, and profile screens.

```bash
cd dermalens_flutter
flutter pub get
flutter run        # Android/iOS device with a camera
```

**Preprocessing parity is critical** — the app must mirror validation-time preprocessing exactly:
1. Decode image → resize to 224×224 (RGB, no BGR swap)
2. Normalize: `(pixel/255 − mean) / std` with ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
3. Layout **NCHW** `[1, 3, 224, 224]` (as exported)
4. Softmax over the 8 logits; **malignant probability = sum of classes 0–3**, compared against the clinical threshold.

This is implemented in [`dermalens_flutter/lib/services/inference_service.dart`](dermalens_flutter/lib/services/inference_service.dart).

---

## 5. Datasets

Four public datasets are merged into a unified `master_dataset.csv` (`image_path, class_idx, skin_tone, dataset, dx_raw`) with a stratified 70/15/15 train/val/test split:

| Dataset | Role |
|---|---|
| HAM10000 (~10k) | Training (7 of 8 classes) |
| ISIC 2019 (~25k) | Training (8 classes) |
| PAD-UFES-20 (~2.3k, smartphone photos) | Training / domain adaptation |
| DDI / Stanford (~656) | **Held-out fairness evaluation only** (expert-labelled Fitzpatrick tones) |

HAM10000 and ISIC 2019 lack Fitzpatrick labels, so skin tone is **estimated automatically** via the ITA° formula (`preprocessing/skin_tone_estimation.py`) and cached in `tone_cache_*.csv`.

---

## 6. Status — Implemented vs Incomplete

This is a work-in-progress final-year project. The codebase is largely scaffolded and trained, but several pieces are **not yet complete**. Be aware of the following before relying on any part of the system:

### ✅ Implemented
- Full FairDermNet architecture (`model/model/`): MobileNetV3-Large backbone + FairnessAttentionModule + GeMPooling + dual-head (risk + Fitzpatrick aux) training.
- 8-class taxonomy locked across Python, export scripts, `labels.txt`, and Flutter inference service.
- Preprocessing modules (hair removal, lesion segmentation, skin-tone estimation, CLAHE, augmentation).
- Training utilities, fairness metrics (equalized odds gap, demographic parity), early stopping + checkpointing.
- GradCAM++ explainability + ABCDE feature scores (`model/explainability/`).
- Export pipeline run end-to-end: `model/export/fairdermnet.onnx` + `fairdermnet_int8.tflite` (≈5.5 MB, dynamic-range quantized) generated and copied to `dermalens_flutter/assets/`.
- Clinical threshold **0.24** calibrated (Cell 9c: sensitivity 96.0%, specificity 51.6%, AUC 0.9185) and shipped in `threshold_config.json`.
- Export validation run on 100 real test images: ONNX vs PyTorch **100%** top-1 / 100% decision match; TFLite vs PyTorch 94% top-1 / 99% decision match (all flips are low-confidence near-ties).
- Trained checkpoint `model/checkpoints/best.pth` — **30 epochs recorded**, best val balanced accuracy ≈ **67.8%** (epoch 29).
- Flutter app with full screen flow and a correct TFLite preprocessing/inference service.

### ⚠️ Incomplete / known gaps
1. **Model has not converged to target performance.** Best val balanced accuracy ≈ 67.8% (still improving at epoch 30). The DDI fairness report (`model/checkpoints/test_fairness_report.txt`) shows `equalized_odds_gap_pct: 75.0` and `fairness_target_met: false` (target <5%), with very small per-group sample sizes (e.g. Type IV n=10, Type V n=3) making those numbers unreliable. More training / data are needed.
2. **Stale 9-class artifact.** `model/figures/confusion_matrix_9class.png` is named/dated for a 9-class run; the taxonomy is locked at 8 classes. Regenerate figures after the next training run.
3. **Default Flutter README.** `dermalens_flutter/README.md` is still the untouched Flutter template ("A new Flutter project") and does not describe the app.
4. **Docs vs. implementation drift.** [`ppt_files/DermaLens-Project-Documentation.md`](ppt_files/DermaLens-Project-Documentation.md) is the original proposal and describes an aspirational design that differs from the shipped code (e.g. a binary 64→2 classification head, 92.5% expected accuracy, 100-user study, GitHub Actions CI). Treat it as a proposal, not a description of the current implementation. Notably, `.github/` contains only leftover tooling hooks — **no CI workflows exist**.
5. **Notebook cell count.** The training notebook contains 36 raw cells; the docs' "15 cells" refers to the numbered logical cells (1–13 + 9b + 9c). No functional issue, just keep it in mind when reading cell references.

---

## 7. Suggested Next Steps

1. **Re-export after any retraining.** `model/export_pipeline.py` regenerates the ONNX/TFLite artifacts and refreshes `dermalens_flutter/assets/`. Re-run it whenever `best.pth` changes (e.g. after the fairness/accuracy improvements below).
2. **Continue training to push val balanced accuracy beyond ~68%** and re-run the DDI fairness evaluation with adequate per-group samples, then re-export (see step 1).
3. Replace `dermalens_flutter/README.md` with a real app README.
4. Build and smoke-test the Flutter app on a device (`flutter run`) — the model asset, `labels.txt`, and calibrated `threshold_config.json` (threshold **0.24**) are now in place.

---

## 8. Documentation Index

- [`GEMINI.md`](GEMINI.md) — AI-agent operating context, taxonomy lock, invariants, audit procedures
- [`model/ARCHITECTURE.md`](model/ARCHITECTURE.md) — architecture walkthrough, notebook cell map, data flow
- [`model/SETUP.md`](model/SETUP.md) — Python 3.10 setup, Jupyter kernel fix, known TODOs
- [`model/datasets/README_datasets.md`](model/datasets/README_datasets.md) — dataset guide
- [`ppt_files/DermaLens-Project-Documentation.md`](ppt_files/DermaLens-Project-Documentation.md) — original proposal/planning document
