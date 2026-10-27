# Setup — DermaLens

Environment notes for building the model and the Flutter app.

## Python environment (model pipeline)

The export pipeline (`model/export_pipeline.py`, `model/validate_export.py`) was
developed and validated on **Python 3.10** with a specific, pinned stack. These
versions matter — the pipeline will silently produce garbage or crash on
different ones:

| Package | Version | Why pinned |
|---|---|---|
| `tensorflow-cpu` | **2.13.0** | TFLite converter + interpreter used by export/validate. Newer TF breaks the onnx-tf hop below. |
| `numpy` | **1.26.4** | TF 2.13 was built against numpy 1.x; numpy 2.x crashes the TF import (`bfloat16` ABI mismatch). |
| `onnx` | **1.22.x** (as installed by `onnxscript 0.7.1`) | Torch 2.13's ONNX exporter (legacy path) needs modern onnx + onnxscript. |
| `onnxscript` | **0.7.1** | Required by `torch.onnx.export` on this torch version. |
| `onnx-tf` | **1.10.0** | ONNX → TF SavedModel conversion. |
| `torch` | 2.13 (whatever `best.pth` was trained with) | — |
| `onnxruntime`, `Pillow`, `pandas` | latest | Validation + calibration loading. |

### Known environment quirks (do not "fix" casually)

1. **TF 2.13 + numpy 2.x are incompatible.** If the environment was ever
   upgraded, install exactly `tensorflow-cpu==2.13.0` and `numpy==1.26.4`.
   On Windows, `tensorflow` (2.21) must be **uninstalled** — it shadows
   `tensorflow-cpu` and ships without `tensorflow.python.client`, breaking
   `onnx_tf`.

2. **`onnx-tf 1.10.0` requires `onnx.mapping`, removed in onnx ≥ 1.17.**
   A small shim was installed into site-packages at
   `onnx_tf/common/mapping_shim.py` (with edits to `data_type.py`,
   `sequence_empty.py`, and `pb_wrapper.py` importing it). The shim rebuilds
   the removed `onnx.mapping` tables (`TENSOR_TYPE_TO_NP_TYPE`,
   `TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE`, ...) and injects `mapping` into the
   `onnx` module at `onnx_tf` import time. **This lives in the global
   site-packages, not in the repo** — a fresh machine must re-apply it.
   See the module docstring in `model/export/quantize_tflite.py` and the shim
   file itself for the minimal patch.

3. **Export opset is pinned to 11.** The legacy TorchScript exporter is used
   because torch ≥ 2.10 refuses opsets < 18, and onnx-tf only implements
   handlers up to ~v13 with Squeeze/Unsqueeze taking axes as *attributes*
   (opset ≤ 11) rather than inputs. Below opset 14 torch decomposes HardSwish
   into ops onnx-tf supports. See `model/export/onnx_export.py`.

4. **Full INT8 is deliberately disabled** (dynamic-range quantization is
   shipped instead) — see the quantization note in `model/export_pipeline.py`.

### Recreating the environment

```bash
pip install tensorflow-cpu==2.13.0 numpy==1.26.4 onnx-tf onnxruntime Pillow pandas
pip install onnxscript onnx     # let pip resolve current onnx/onnxscript pair
# then re-apply the onnx-tf mapping shim (see quirk 2)
```

Then verify:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"        # 2.13.0
python -c "import onnx_tf; print('onnx-tf OK')"
cd model && python export_pipeline.py && python validate_export.py
```

## Flutter app

```bash
cd dermalens_flutter
flutter pub get
flutter run     # Android/iOS device with a camera
```

The app bundles `assets/fairdermnet_int8.tflite` (≈5.5 MB, dynamic-range
quantized), `assets/labels.txt`, and `assets/threshold_config.json`
(clinical threshold **0.24**). Regenerate all three by re-running
`model/export_pipeline.py` — it copies them into the Flutter assets dir.

> ⚠️ **Not a medical device.** Prototype for an academic project only.
