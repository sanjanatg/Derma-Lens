"""
export_pipeline.py — End-to-end export: best.pth → ONNX → TFLite INT8
=======================================================================
Run this script AFTER training has completed and best.pth exists.

Usage:
    python export_pipeline.py

Outputs (written to model/export/):
    fairdermnet_int8.tflite   — INT8-quantized TFLite model (<6 MB target)
    labels.txt                — 8 class names (one per line, in class index order)
    threshold_config.json     — Clinical threshold + metadata for Flutter

Requirements:
    pip install onnx onnx-tf tensorflow-cpu==2.13.0
    (onnxsim optional: pip install onnxsim)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Project root on sys.path ─────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent          # model/ directory
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from model.class_taxonomy import (
    CLASS_NAMES,
    NUM_CLASSES,
    MALIGNANT_CLASSES,
)
from model.fairdermet import FairDermNet
from export.onnx_export import export_to_onnx
from export.quantize_tflite import (
    convert_onnx_to_tflite,
    benchmark_tflite,
    save_export_manifest,
)


# ── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best.pth"
EXPORT_DIR      = PROJECT_ROOT / "export"
THRESHOLD_PATH  = PROJECT_ROOT / "checkpoints" / "threshold_config.json"

ONNX_PATH       = EXPORT_DIR / "fairdermnet.onnx"
TFLITE_PATH     = EXPORT_DIR / "fairdermnet_int8.tflite"
LABELS_PATH     = EXPORT_DIR / "labels.txt"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_clinical_threshold(threshold_json: Path) -> float:
    """
    Read the clinical threshold from the JSON produced by Cell 9c.
    Falls back to 0.5 if the file does not exist (not yet computed).
    """
    if threshold_json.exists():
        data = json.loads(threshold_json.read_text())
        threshold = data.get("clinical_threshold", 0.5)
        print(f"[Pipeline] Clinical threshold loaded: {threshold:.4f}")
        return float(threshold)
    print("[Pipeline] ⚠ threshold_config.json not found — using default 0.5")
    print("           Run Cell 9c in the training notebook to compute the true threshold.")
    return 0.5


def load_checkpoint(ckpt_path: Path) -> FairDermNet:
    """Load FairDermNet from best.pth (8-class, eval mode)."""
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run the training notebook (cells 1–7) to generate best.pth."
        )
    print(f"[Pipeline] Loading checkpoint: {ckpt_path} ...")
    model = FairDermNet.from_checkpoint(ckpt_path, num_classes=NUM_CLASSES)
    model.eval()
    print(f"[Pipeline] FairDermNet loaded — {NUM_CLASSES} classes, eval mode")
    return model


def write_labels(labels_path: Path, class_names: list) -> None:
    """Write labels.txt for Flutter assets."""
    labels_path.write_text("\n".join(class_names))
    print(f"[Pipeline] ✓ labels.txt → {labels_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DermaLens FairDermNet Export Pipeline  (8-class)")
    print("=" * 60)
    print(f"  Classes  : {NUM_CLASSES}")
    print(f"  Malignant: {sorted(MALIGNANT_CLASSES)}  (indices 0–3: MEL, BCC, SCC, AK)")
    print(f"  Benign   : 4–7 (NV, BKL, DF, VASC)")
    print()

    # 1. Load model
    model = load_checkpoint(CHECKPOINT_PATH)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Pipeline] Parameters: {n_params / 1e6:.2f} M")

    # 2. Export ONNX
    export_to_onnx(model, ONNX_PATH, input_size=224, opset_version=17, simplify=True)

    # 3. Convert to TFLite INT8
    tflite_path = convert_onnx_to_tflite(
        onnx_path=ONNX_PATH,
        output_path=TFLITE_PATH,
        num_calibration_samples=200,
        quantize_int8=True,
    )
    tflite_mb = tflite_path.stat().st_size / 1e6

    # 4. Benchmark TFLite on host CPU
    bench = benchmark_tflite(tflite_path, n_runs=20)

    # 5. Load clinical threshold
    clinical_threshold = load_clinical_threshold(THRESHOLD_PATH)

    # 6. Write labels.txt
    write_labels(LABELS_PATH, CLASS_NAMES)

    # 7. Write threshold_config.json manifest
    save_export_manifest(
        output_dir=EXPORT_DIR,
        class_names=CLASS_NAMES,
        malignant_classes=sorted(MALIGNANT_CLASSES),
        clinical_threshold=clinical_threshold,
        tflite_size_mb=tflite_mb,
        benchmark_ms=bench["mean_ms"],
    )
    
    # Check for placeholder status in generated config
    config = json.loads((EXPORT_DIR / "threshold_config.json").read_text())
    if config.get("_status") == "placeholder":
        print("\n" + "!" * 60)
        print("  WARNING: Shipping placeholder clinical threshold (0.5).")
        print("  Run Cell 9c in the training notebook to calibrate.")
        print("!" * 60 + "\n")

    # 8. Copy artefacts to Flutter assets
    flutter_assets = PROJECT_ROOT.parent / "dermalens_flutter" / "assets"
    if flutter_assets.exists():
        import shutil
        shutil.copy(TFLITE_PATH, flutter_assets / TFLITE_PATH.name)
        shutil.copy(LABELS_PATH, flutter_assets / "labels.txt")
        shutil.copy(EXPORT_DIR / "threshold_config.json",
                    flutter_assets / "threshold_config.json")
        print(f"[Pipeline] ✓ Assets copied to Flutter: {flutter_assets}")
    else:
        print(f"[Pipeline] ⚠ Flutter assets dir not found at {flutter_assets}")
        print(f"           Copy manually: {EXPORT_DIR}/*.tflite, labels.txt, threshold_config.json → dermalens_flutter/assets/")

    print()
    print("=" * 60)
    print("  Export complete!")
    print(f"  TFLite size  : {tflite_mb:.2f} MB  ({'✓' if tflite_mb < 6 else '✗'} <6 MB target)")
    print(f"  Host latency : {bench['mean_ms']:.1f} ms (CPU — scale to mobile)")
    print(f"  Threshold    : {clinical_threshold:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
