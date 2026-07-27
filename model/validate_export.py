"""
validate_export.py — PyTorch vs TFLite output validation (8-class FairDermNet)
===============================================================================
Compares PyTorch and TFLite model outputs on synthetic and real test images.

Reports:
  - Top-1 class match rate
  - Max softmax probability drift (L∞ per sample)
  - Malignant/benign decision consistency

Usage:
    python validate_export.py
    python validate_export.py --n 50 --images path/to/test_images/

Requirements:
    pip install onnxruntime Pillow
    (tensorflow-cpu==2.13.0 for TFLite interpreter)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Project root ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.class_taxonomy import CLASS_NAMES, MALIGNANT_CLASSES, NUM_CLASSES
from model.fairdermet import FairDermNet


# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best.pth"
TFLITE_PATH     = PROJECT_ROOT / "export" / "fairdermnet_int8.tflite"
ONNX_PATH       = PROJECT_ROOT / "export" / "fairdermnet.onnx"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

MALIGNANT_IDX = sorted(MALIGNANT_CLASSES)   # [0, 1, 2, 3]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_numpy(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Resize + ImageNet normalise a (H, W, 3) uint8 image.
    Returns (1, 3, 224, 224) float32.
    """
    from PIL import Image
    pil = Image.fromarray(img_rgb_uint8).resize((224, 224))
    arr = np.array(pil, dtype=np.float32) / 255.0      # [224, 224, 3]
    arr = arr.transpose(2, 0, 1)[None]                  # [1, 3, 224, 224]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.astype(np.float32)


def make_synthetic_batch(n: int, seed: int = 42) -> np.ndarray:
    """Generate n synthetic ImageNet-normalised images (1, 3, 224, 224) each."""
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 3, 224, 224), dtype=np.float32)
    return (raw - IMAGENET_MEAN) / IMAGENET_STD


# ── PyTorch runner ────────────────────────────────────────────────────────────

def run_pytorch(model: FairDermNet, inputs: np.ndarray) -> np.ndarray:
    """Returns softmax probs [N, 8] as numpy array."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(inputs)
        logits = model(x)   # [N, 8] in eval mode
        probs  = F.softmax(logits, dim=-1).numpy()
    return probs.astype(np.float32)


# ── TFLite runner ─────────────────────────────────────────────────────────────

def run_tflite(tflite_path: Path, inputs: np.ndarray) -> np.ndarray:
    """Run TFLite interpreter on inputs [N, 3, 224, 224] → probs [N, 8]."""
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    except Exception as e:
        raise RuntimeError(
            f"TFLite interpreter failed to load: {e}\n"
            "Ensure tensorflow-cpu==2.13.0 is installed."
        )

    interpreter.allocate_tensors()
    in_details  = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    all_probs = []
    for i in range(inputs.shape[0]):
        interpreter.set_tensor(in_details[0]["index"], inputs[i:i+1])
        interpreter.invoke()
        out = interpreter.get_tensor(out_details[0]["index"])       # [1, 8]
        # Apply softmax if the model outputs raw logits
        exp = np.exp(out - out.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        all_probs.append(probs[0])

    return np.array(all_probs, dtype=np.float32)


# ── ONNX runner (alternative / fallback) ─────────────────────────────────────

def run_onnx(onnx_path: Path, inputs: np.ndarray) -> np.ndarray:
    """Run ONNX Runtime on inputs [N, 3, 224, 224] → probs [N, 8]."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"input": inputs})[0]       # [N, 8]
    exp    = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs  = exp / exp.sum(axis=-1, keepdims=True)
    return probs.astype(np.float32)


# ── Comparison metrics ────────────────────────────────────────────────────────

def compare(pt_probs: np.ndarray, other_probs: np.ndarray, other_name: str) -> dict:
    """
    Compare PyTorch and another model's softmax probabilities.

    Returns dict with:
      top1_match_pct        — % of samples where argmax agrees
      mean_linf_drift       — mean per-sample L∞ probability drift
      max_linf_drift        — worst-case L∞ drift
      malignant_decision_match_pct — % agreement on malignant/benign classification
    """
    assert pt_probs.shape == other_probs.shape, "Shape mismatch"
    n = pt_probs.shape[0]

    pt_top1    = pt_probs.argmax(axis=1)
    other_top1 = other_probs.argmax(axis=1)
    top1_match = (pt_top1 == other_top1).mean() * 100

    linf = np.abs(pt_probs - other_probs).max(axis=1)
    mean_linf = linf.mean()
    max_linf  = linf.max()

    def is_malignant(probs):
        mal_prob = probs[:, MALIGNANT_IDX].sum(axis=1)
        return mal_prob >= 0.5   # simple threshold for decision comparison

    pt_mal    = is_malignant(pt_probs)
    other_mal = is_malignant(other_probs)
    decision_match = (pt_mal == other_mal).mean() * 100

    result = {
        "comparison":                    f"PyTorch vs {other_name}",
        "n_samples":                     n,
        "top1_match_pct":               round(float(top1_match), 2),
        "mean_linf_drift":              round(float(mean_linf), 6),
        "max_linf_drift":               round(float(max_linf), 6),
        "malignant_decision_match_pct": round(float(decision_match), 2),
    }

    print(f"\n{'─'*50}")
    print(f"  {result['comparison']}  (n={n})")
    print(f"{'─'*50}")
    print(f"  Top-1 match          : {top1_match:.1f}%")
    print(f"  Prob drift (L∞ mean) : {mean_linf:.6f}")
    print(f"  Prob drift (L∞ max)  : {max_linf:.6f}")
    print(f"  Mal/Ben decision match: {decision_match:.1f}%")

    if top1_match < 95:
        print(f"  ⚠ Top-1 match below 95% — check INT8 quantization quality.")
    if max_linf > 0.05:
        print(f"  ⚠ Max probability drift > 0.05 — check calibration dataset size.")
    if decision_match < 99:
        print(f"  ⚠ Malignant decision differs in {100-decision_match:.1f}% of samples.")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate TFLite export vs PyTorch")
    parser.add_argument("--n",      type=int, default=100, help="Number of synthetic test samples")
    parser.add_argument("--images", type=str, default=None, help="Directory of real test images (optional)")
    parser.add_argument("--no-tflite", action="store_true", help="Skip TFLite comparison (use ONNX only)")
    args = parser.parse_args()

    print("=" * 60)
    print("  DermaLens — Export Validation (8-class)")
    print("=" * 60)
    print(f"  Classes    : {CLASS_NAMES}")
    print(f"  Malignant  : {MALIGNANT_IDX} (0–3: MEL, BCC, SCC, AK)")
    print(f"  Benign     : [4, 5, 6, 7] (NV, BKL, DF, VASC)")
    print()

    # ── Load PyTorch model ────────────────────────────────────────────────────
    print(f"[Validate] Loading PyTorch model from {CHECKPOINT_PATH} ...")
    pt_model = FairDermNet.from_checkpoint(CHECKPOINT_PATH, num_classes=NUM_CLASSES)
    pt_model.eval()

    # ── Synthetic inputs ──────────────────────────────────────────────────────
    print(f"[Validate] Generating {args.n} synthetic test inputs ...")
    synth_inputs = make_synthetic_batch(args.n).astype(np.float32)

    # ── Real images (optional) ────────────────────────────────────────────────
    real_inputs = None
    if args.images:
        from PIL import Image
        img_dir = Path(args.images)
        img_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if img_paths:
            print(f"[Validate] Found {len(img_paths)} real images in {img_dir}")
            real_inputs = np.concatenate([
                preprocess_numpy(np.array(Image.open(p).convert("RGB")))
                for p in img_paths[:50]
            ], axis=0)
        else:
            print(f"[Validate] ⚠ No images found in {img_dir}")

    # ── Run PyTorch ───────────────────────────────────────────────────────────
    print("[Validate] Running PyTorch inference ...")
    pt_probs_synth = run_pytorch(pt_model, synth_inputs)

    all_results = []

    # ── ONNX comparison ───────────────────────────────────────────────────────
    if ONNX_PATH.exists():
        print("[Validate] Running ONNX Runtime inference ...")
        onnx_probs = run_onnx(ONNX_PATH, synth_inputs)
        all_results.append(compare(pt_probs_synth, onnx_probs, "ONNX Runtime"))
    else:
        print(f"[Validate] ⚠ ONNX not found at {ONNX_PATH} — skipping ONNX comparison.")

    # ── TFLite comparison ─────────────────────────────────────────────────────
    if not args.no_tflite:
        if TFLITE_PATH.exists():
            print("[Validate] Running TFLite interpreter inference ...")
            try:
                tflite_probs = run_tflite(TFLITE_PATH, synth_inputs)
                all_results.append(compare(pt_probs_synth, tflite_probs, "TFLite INT8"))

                if real_inputs is not None:
                    pt_probs_real     = run_pytorch(pt_model, real_inputs)
                    tflite_probs_real = run_tflite(TFLITE_PATH, real_inputs)
                    all_results.append(compare(pt_probs_real, tflite_probs_real, "TFLite INT8 (real images)"))

            except RuntimeError as e:
                print(f"[Validate] ⚠ TFLite comparison skipped: {e}")
        else:
            print(f"[Validate] ⚠ TFLite not found at {TFLITE_PATH}")
            print("           Run export_pipeline.py first.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Validation Summary")
    print("=" * 60)
    for r in all_results:
        status = "✓" if r["top1_match_pct"] >= 95 and r["malignant_decision_match_pct"] >= 99 else "⚠"
        print(f"  {status} {r['comparison']}")
        print(f"      Top-1 match={r['top1_match_pct']}%  "
              f"L∞ drift={r['max_linf_drift']:.4f}  "
              f"Decision match={r['malignant_decision_match_pct']}%")

    if not all_results:
        print("  ⚠ No comparisons completed — run export_pipeline.py first.")


if __name__ == "__main__":
    main()
