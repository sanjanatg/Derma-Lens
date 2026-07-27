"""
quantize_tflite.py — ONNX → TFLite INT8 conversion for FairDermNet (8-class)
==============================================================================
Pipeline:
    ONNX (opset 17) → TF SavedModel (via onnx-tf) → TFLite INT8 (quantized)

Requirements:
    pip install onnx-tf tensorflow-cpu==2.13.0

Usage from Cell 12 of the training notebook:
    from export.quantize_tflite import convert_onnx_to_tflite, benchmark_tflite
    tflite_path = convert_onnx_to_tflite(onnx_path, output_path, num_calibration_samples=200)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_onnx_to_tflite(
    onnx_path: str | Path,
    output_path: str | Path,
    num_calibration_samples: int = 200,
    quantize_int8: bool = True,
    input_shape: tuple = (1, 3, 224, 224),
) -> Path:
    """
    Convert an ONNX model to TFLite INT8.

    Strategy:
      1. onnx-tf converts ONNX → TF SavedModel
      2. tf.lite.TFLiteConverter with INT8 quantization converts SavedModel → TFLite

    Args:
        onnx_path:                Path to the .onnx file.
        output_path:              Destination .tflite path.
        num_calibration_samples:  Samples used for INT8 calibration.
        quantize_int8:            If False, export float32 TFLite (larger but no calibration).
        input_shape:              Model input shape (N, C, H, W).

    Returns:
        Path to the .tflite file.
    """
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    onnx_path   = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_model_dir = output_path.parent / "_tmp_saved_model"

    # ── Step 1: ONNX → TF SavedModel ────────────────────────────────────────
    print(f"[Export] Loading ONNX from {onnx_path} ...")
    model = onnx.load(str(onnx_path))
    tf_rep = prepare(model)
    print(f"[Export] Exporting to TF SavedModel at {saved_model_dir} ...")
    tf_rep.export_graph(str(saved_model_dir))

    # ── Step 2: SavedModel → TFLite ──────────────────────────────────────────
    print("[Export] Converting SavedModel → TFLite ...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if quantize_int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.float32   # keep float input for convenience
        converter.inference_output_type = tf.float32   # keep float output

        # Representative dataset — synthetic ImageNet-normalised images
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        def _representative_dataset():
            rng = np.random.default_rng(42)
            for _ in range(num_calibration_samples):
                raw  = rng.random((1, 3, 224, 224), dtype=np.float32)
                norm = (raw - mean) / std
                yield [norm.astype(np.float32)]

        converter.representative_dataset = _representative_dataset

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    mb = len(tflite_model) / 1e6
    print(f"[Export] ✓ TFLite written → {output_path}  ({mb:.2f} MB)")
    if mb > 6.0:
        print(f"[Export] ⚠ Size {mb:.2f} MB exceeds 6 MB target.")

    return output_path


def benchmark_tflite(
    tflite_path: str | Path,
    input_shape: tuple = (1, 3, 224, 224),
    n_runs: int = 20,
    warmup: int = 3,
) -> dict:
    """
    Run latency benchmark on the TFLite model using TFLite interpreter.

    Returns a dict with: mean_ms, std_ms, min_ms, max_ms.
    """
    import tensorflow as tf

    tflite_path = Path(tflite_path)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy = np.random.randn(*input_shape).astype(np.float32)

    latencies = []
    for i in range(warmup + n_runs):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = latencies[warmup:]
    result = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms":  float(np.std(latencies)),
        "min_ms":  float(np.min(latencies)),
        "max_ms":  float(np.max(latencies)),
    }
    print(f"[Benchmark] TFLite latency ({n_runs} runs): "
          f"{result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms")
    return result


def save_export_manifest(
    output_dir: str | Path,
    class_names: list,
    malignant_classes: list,
    clinical_threshold: float,
    tflite_size_mb: float,
    benchmark_ms: Optional[float] = None,
) -> Path:
    """
    Save a JSON manifest alongside the TFLite model for use in the Flutter app.
    This is the threshold_config.json consumed by InferenceService.dart.
    """
    output_dir = Path(output_dir)
    manifest = {
        "model_version":     "FairDermNet-8class-v1",
        "num_classes":       len(class_names),
        "class_names":       class_names,
        "malignant_indices": malignant_classes,
        "benign_indices":    [i for i in range(len(class_names)) if i not in malignant_classes],
        "clinical_threshold": clinical_threshold,
        "input_size":        [224, 224, 3],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
        },
        "tflite_size_mb":    round(tflite_size_mb, 3),
        "benchmark_mean_ms": round(benchmark_ms, 1) if benchmark_ms else None,
    }
    path = output_dir / "threshold_config.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"[Export] ✓ Manifest written → {path}")
    return path
