# GEMINI.md - Instructional Context for DermaLens / FairDermNet

This file contains foundational instructions, architectural guidelines, environment configurations, and development conventions for the **DermaLens** project. Treat this document as the single source of truth and primary instruction set for all future AI operations within this workspace.

---

## 1. Project Overview
This repository contains DermaLens / FairDermNet, a final-year project for fairness-aware skin lesion risk assessment. The system is divided into two primary components:
* **`model/`:** Python deep learning pipeline, training notebook, export scripts, fairness metrics, GradCAM++, and TFLite export flow.
* **`dermalens_flutter/`:** Flutter mobile app that runs on-device inference using the exported TFLite model.

---

## 2. Final Taxonomy Lock
The project is strictly **8-class**, not 9-class and not 7-class. Use this exact class order everywhere:
1. **Melanoma** (MEL)
2. **Basal Cell Carcinoma** (BCC)
3. **Squamous Cell Carcinoma** (SCC)
4. **Actinic Keratosis** (AK)
5. **Melanocytic Nevi** (NV)
6. **Benign Keratosis** (BKL)
7. **Dermatofibroma** (DF)
8. **Vascular Lesion** (VASC)

### Index Mapping
* `0 = MEL`
* `1 = BCC`
* `2 = SCC`
* `3 = AK`
* `4 = NV`
* `5 = BKL`
* `6 = DF`
* `7 = VASC`

---

## 3. Binary Clinical Grouping
Unless explicitly told otherwise, use this binary grouping:
* **Malignant / Clinically Positive Bucket:** Classes `0, 1, 2, 3`
* **Benign Bucket:** Classes `4, 5, 6, 7`

### Constants
* `MALIGNANT_CLASSES = [0, 1, 2, 3]`
* `N_MALIGNANT = 4`

*Note: If any file contradicts this, treat it as stale, report it, and fix it.*

---

## 4. Sources of Truth
When editing, treat these as the primary sources of truth:
* `model/model/class_taxonomy.py`
* `model/ARCHITECTURE.md`
* `model/configs/train_config.yaml`
* `model/export_pipeline.py`
* `model/validate_export.py`
* `dermalens_flutter/lib/services/inference_service.dart`
* `dermalens_flutter/assets/labels.txt`
* `dermalens_flutter/assets/threshold_config.json`

---

## 5. Critical Invariants
Do not break these invariants:
* **Taxonomy Order Compatibility:** Taxonomy order in Python, ONNX/TFLite export, `labels.txt`, and Flutter UI must match exactly.
* **Preprocessing Parity:** Flutter preprocessing must mirror validation-time model preprocessing.
* **Tensor Shapes:** Input shape is expected to remain `1 x 3 x 224 x 224` unless the export pipeline is intentionally changed. Output shape is expected to remain `1 x 8`.
* **Preprocessing Pipeline:** Any change in normalization, resizing, channel order, or tensor layout must be reflected in both Python validation and Flutter inference.
* **No Standalone Bowen's Disease:** Do not re-introduce Bowen's disease as a standalone class unless the full pipeline, docs, and labels are intentionally redesigned.

---

## 6. Current Implementation Status
* `num_classes` is `8`.
* Malignant classes set to `0..3`.
* Flutter utilizes `tflite_flutter` package.
* `export_pipeline.py` generates `.tflite`, `labels.txt`, and `threshold_config.json`.
* `validate_export.py` compares PyTorch / ONNX / TFLite outputs.
* **Assumption:** This is mostly implemented, but verify before trusting.

---

## 7. Default Behavior Guidelines

### Default Workflow
When asked to work on this repo, follow this order:
1. **Audit first** — identify inconsistencies before editing.
2. **Show exact files** affected.
3. **Explain assumptions** briefly.
4. **Make minimal, targeted changes**.
5. **Preserve taxonomy lock** and preprocessing parity.
6. **Provide a short verification checklist** after edits.

### Preferred Response Format
For codebase tasks, respond in this structure:
1. **Findings**
2. **Files to change**
3. **Proposed patch / code**
4. **Why this change is needed**
5. **How to verify**
6. **Open risks or assumptions**

### Core Task Types
* Auditing stale 7-class / 9-class assumptions.
* Fixing export pipeline errors.
* Making TFLite export reproducible on Windows or Colab.
* Debugging Flutter inference mismatch.
* Checking PyTorch vs TFLite output drift.
* Updating project documentation for the final report/viva.
* Preparing final deployment steps for the mobile app.

---

## 8. Specific TFLite, Flutter, Export, & Documentation Rules

### TFLite / Flutter Rules
When touching Flutter inference:
* Use `tflite_flutter` for on-device inference.
* Keep asset names synchronized with `pubspec.yaml`.
* Ensure labels are loaded in exact model output order.
* Preprocessing steps must be:
  1. Image decode
  2. Resize to `224 x 224`
  3. RGB order (no BGR swap)
  4. Divide by `255.0`
  5. Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
  6. Layout as **NCHW** (`[1, 3, 224, 224]`) if that is what the exported model expects.
* Softmax should be applied consistently if the model output is logits.
* Binary malignant probability should be derived from the sum of classes `0..3` unless explicitly changed.

### Export Rules
When touching model export:
* Prefer a reproducible script over ad-hoc notebook-only steps.
* Validate exported artifacts after generation.
* Keep output artifact names stable unless asked to rename them.
* If TensorFlow/TFLite conversion fails locally, suggest a clean Colab or Linux export path rather than forcing risky dependency changes.

### Documentation Rules
When updating docs:
* Remove outdated 9-class and 7-class references.
* Use the final 8-class taxonomy consistently.
* Clearly distinguish multiclass prediction from binary malignant/benign thresholding.
* Do not overclaim medical validity. Frame the innovation around fairness-aware training, explainability, and deployment, not class count alone.

### Things to Avoid
* **DO NOT** silently change taxonomy order.
* **DO NOT** assume class names from old docs are still valid.
* **DO NOT** add large refactors unless necessary.
* **DO NOT** change preprocessing in Flutter without matching Python-side validation.
* **DO NOT** claim conversion succeeded unless the generated artifacts and validation outputs are shown.
* **DO NOT** overwrite user work without summarizing planned changes first.

---

## 9. Specialized Audit Procedures
If the user asks for an audit, check at minimum:
1. Class count assumptions.
2. Malignant class assumptions.
3. Labels ordering.
4. Threshold logic.
5. Flutter asset declarations.
6. ONNX input/output tensor shapes.
7. TFLite interpreter input expectations.
8. Documentation consistency across `README` / `ARCHITECTURE` / report files.

---

## 10. Academic Report / Viva Guidelines
Use language suitable for a final-year engineering project:
* Practical and evidence-based.
* Modest claims.
* Clear distinction between implemented features and planned improvements.

---

## 11. Current Next Priorities
Unless the user specifies another goal, prioritize these tasks:
1. Verify `best.pth` is truly an 8-class checkpoint.
2. Run / debug `export_pipeline.py`.
3. Run / debug `validate_export.py`.
4. Confirm Flutter inference works with the exported `.tflite` model.
5. Clean up documentation for the final 8-class story.
