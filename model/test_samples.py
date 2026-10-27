"""
test_samples.py — Run FairDermNet on a few real test-set examples
==================================================================
Picks one example per class from split_test.csv (8 examples total),
preprocesses with the exact validation pipeline (resize 224x224 +
ImageNet normalisation, NCHW), runs inference, and prints the full
8-class probability distribution, top-1 prediction, and the
malignant/benign clinical decision vs ground truth.

Usage:
    python test_samples.py [--n-per-class 1] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.fairdermet import FairDermNet
from model.class_taxonomy import CLASS_NAMES, NUM_CLASSES, MALIGNANT_CLASSES
from preprocessing import get_val_transforms

TEST_CSV = PROJECT_ROOT / "datasets" / "split_test.csv"
CKPT     = PROJECT_ROOT / "checkpoints" / "best.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=1,
                        help="examples to test per class (default 1)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    model = FairDermNet.from_checkpoint(CKPT, num_classes=NUM_CLASSES)
    model.eval().to(device)
    print(f"Model: FairDermNet — {NUM_CLASSES} classes, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params\n")

    # ── Load test split ─────────────────────────────────────────────────────
    test_df = pd.read_csv(TEST_CSV)
    test_df = test_df[test_df["class_idx"] >= 0].reset_index(drop=True)

    # ── Pick examples: one per class, seeded ────────────────────────────────
    rng = np.random.default_rng(args.seed)
    examples = []
    for cls in range(NUM_CLASSES):
        subset = test_df[test_df["class_idx"] == cls]
        if len(subset) == 0:
            print(f"  [skip] class {cls} ({CLASS_NAMES[cls]}): no test samples")
            continue
        pick = subset.sample(n=min(args.n_per_class, len(subset)),
                             random_state=rng).reset_index(drop=True)
        examples.append(pick)
    if not examples:
        print("No test examples found.")
        return
    pick_df = pd.concat(examples, ignore_index=True)

    # ── Val transform (mirrors training exactly) ────────────────────────────
    transform = get_val_transforms()

    print(f"Testing {len(pick_df)} examples from split_test.csv\n")
    print(f"{'true':<8} {'top-1':<8} {'p(top-1)':>9} {'malig?':>7}  "
          f"{'decision':<14} image")
    print("-" * 100)

    correct = 0
    malig_correct = 0
    results = []
    with torch.no_grad():
        for _, row in pick_df.iterrows():
            img = cv2.imread(row["image_path"])
            if img is None:
                print(f"  [missing image] {row['image_path']}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

            logits = model(x)                     # [1, 8] eval mode
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

            top1 = int(probs.argmax())
            true = int(row["class_idx"])
            mal_prob = float(probs[list(MALIGNANT_CLASSES)].sum())
            is_mal = mal_prob >= 0.5
            true_mal = true in MALIGNANT_CLASSES

            correct += top1 == true
            malig_correct += is_mal == true_mal

            decision = ("MALIGNANT" if is_mal else "benign") + \
                       (" ✓" if is_mal == true_mal else " ✗")
            name = Path(row["image_path"]).name
            print(f"{CLASS_NAMES[true]:<8} {CLASS_NAMES[top1]:<8} "
                  f"{probs[top1]:8.1%} {mal_prob:6.1%}  "
                  f"{decision:<14} {name}")
            results.append({
                "image": name, "true": CLASS_NAMES[true],
                "pred": CLASS_NAMES[top1], "top1_prob": float(probs[top1]),
                "malignant_prob": mal_prob,
                "true_malignant": bool(true_mal),
                "pred_malignant": bool(is_mal),
                "probs": probs,
            })

    print("-" * 100)
    n = len(results)
    if n:
        print(f"Top-1 accuracy on sampled examples : {correct}/{n} "
              f"({correct / n:.0%})")
        print(f"Malignant/benign decision accuracy : {malig_correct}/{n} "
              f"({malig_correct / n:.0%})")

        print("\nFull softmax distributions (rows = examples):")
        header = " ".join(f"{c[:4]:>5}" for c in CLASS_NAMES)
        print(f"{'true':<22} {header}")
        for r in results:
            dist = " ".join(f"{p:5.1%}" for p in r["probs"])
            print(f"{r['true']:<22} {dist}")


if __name__ == "__main__":
    main()
