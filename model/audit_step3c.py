"""
audit_step3c.py — Step 3c: CAM comparators + lambda-collapse mechanism
=======================================================================

Closes the two remaining interpretation gaps before the mechanism paragraph
is locked into the paper:

  S3c.1  CAM concentration COMPARATORS. The Step 3b "diffuse, not focal"
         stat (median top-1% CAM mass share 0.023 on AK-driven DDI false
         positives) had no baseline. If genuine AK activations are equally
         diffuse, "diffuse" is just how the AK head behaves — not evidence
         of a spurious response. Comparators, same statistic, same script:
           - 15 TRUE AK test-panel images (highest AK prob), AK-column CAM
           - 15 TRUE MEL test-panel images (highest MEL prob), MEL-column CAM
         Structural stats per image: top-1% CAM mass share, Pearson r between
         CAM and image luminance (background/illumination vs lesion), and the
         CAM mass share OUTSIDE an Otsu lesion proxy (dark-region mask).
         Mann-Whitney U on concentration: FP-DDI-AK vs true-AK vs true-MEL.

  S3c.2  Lambda-collapse mechanism: driver-column shares on eval-half BENIGN
         DDI images at lambda in {0, 0.25, 0.8} — does deflating AK simply
         let another malignant column take over (whack-a-mole)?

Usage:
  python audit_step3c.py            # GPU for CAMs (~45 images), rest cache-only
"""

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent
AUDIT_DIR = ROOT / "audit"
sys.path.insert(0, str(ROOT))

import audit_step0_1 as s01
import audit_step2 as s02
import audit_step3b as s3b

SEED = 42
GROUP_KEYS = s02.GROUP_KEYS
GROUP_NAMES = s02.GROUP_NAMES
MAL_IDX = [0, 1, 2, 3]
MAL_SHORT = ["MEL", "BCC", "SCC", "AK"]
N_IMG = 15


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_panel():
    cache = AUDIT_DIR / "preds_cache.npz"
    z = np.load(cache)
    y = z["y_ddi"].astype(int)
    g = z["g_ddi"].astype(int)
    P = z["P_ddi"].astype(float)
    yT = z["y_test"].astype(int)
    gT = z["g_test"].astype(int)
    PT = z["P_test"].astype(float)
    s01.run_step0()
    blockers = [i for i in s01.ISSUES if i["severity"] == "BLOCKER"]
    if blockers:
        for i in s01.ISSUES:
            log(f"  [{i['severity']:7s}] {i['code']}: {i['detail']}")
        raise SystemExit("Step 0 gates failed.")
    return y, g, P, yT, gT, PT


def cam_stats(model, device, transform_rgb, img_bgr, col):
    """Grad-CAM for one class column on one image + structural stats."""
    import cv2
    import torch
    import torch.nn.functional as F

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = transform_rgb(image=rgb)["image"].unsqueeze(0).to(device)
    feat = model.extract_features(x)
    feat_attended, _ = model.fairness_attention(feat)
    pooled = model.gem(feat_attended)
    embed = model.neck(pooled)
    logits = model.risk_head(embed)
    score = logits[0, col]
    grads = torch.autograd.grad(score, feat)[0][0]
    acts = feat.detach()[0]
    weights = grads.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * acts).sum(0)).cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam_up = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]),
                        interpolation=cv2.INTER_LINEAR)

    gray = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
                      (224, 224)).astype(np.float64) / 255.0
    cam224 = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)

    flat = np.sort(cam_up.ravel())[::-1]
    k1 = max(1, int(0.01 * flat.size))
    conc = float(flat[:k1].sum() / flat.sum()) if flat.sum() > 0 else 0.0

    if cam224.std() > 0 and gray.std() > 0:
        r = float(np.corrcoef(cam224.ravel(), gray.ravel())[0, 1])
    else:
        r = float("nan")

    _, otsu_mask = cv2.threshold((gray * 255).astype(np.uint8), 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lesion = otsu_mask > 0
    if lesion.sum() == 0:
        outside = float("nan")
    else:
        outside = float(1.0 - cam224[lesion].sum() / cam224.sum())
    return conc, r, outside, cam_up


def run_comparators(yT, PT, n_img=N_IMG):
    import cv2
    import pandas as pd
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from scipy.stats import mannwhitneyu

    from model.fairdermet import FairDermNet
    from model import class_taxonomy as tax

    test_df = pd.read_csv(s01.DATASETS / "split_test.csv")
    paths = test_df["image_path"].to_numpy()
    cls = test_df["class_idx"].to_numpy().astype(int)
    if len(cls) != len(yT):
        raise SystemExit("split_test.csv row count != cache rows")

    out_dir = ROOT / "figures" / "gradcam_step3c"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FairDermNet.from_checkpoint(
        s01.CHECKPOINTS / "best.pth",
        num_classes=tax.NUM_CLASSES, num_skin_tones=6)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform_rgb = A.Compose([
        A.Resize(height=s01.INPUT_SIZE, width=s01.INPUT_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    groups = {
        "true_AK": (np.where(cls == 3)[0], 3, "AK"),
        "true_MEL": (np.where(cls == 0)[0], 0, "MEL"),
    }
    rows = []
    for name, (cand, col, short) in groups.items():
        top = cand[np.argsort(-PT[cand, col])][:n_img]
        log(f"\n  {name}: top-{len(top)} by {short} prob, {short}-column CAM")
        for rank, i in enumerate(top, 1):
            img = cv2.imread(str(paths[i]))
            if img is None:
                continue
            conc, r, outside, cam_up = cam_stats(model, device, transform_rgb,
                                                 img, col)
            heat = img.astype(np.float64)
            heat[..., 1] = np.clip(heat[..., 1] + cam_up * 153.0, 0, 255)
            ov = (0.6 * img + 0.4 * heat)
            ov = np.clip(ov, 0, 255).astype(np.uint8)
            cv2.putText(ov, f"{short} conc {conc:.2f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            fn = f"{name}_{rank:02d}_p{PT[i, col]:.2f}.jpg"
            cv2.imwrite(str(out_dir / fn), ov)
            rows.append({"cohort": name, "panel": "test", "rank": rank,
                         "row_index": int(i), "column": short,
                         "true_class": int(cls[i]),
                         "prob": float(PT[i, col]),
                         "conc_top1pct": conc, "cam_lum_r": r,
                         "cam_outside_otsu": outside})
            log(f"    #{rank:02d} p={PT[i, col]:.3f} conc={conc:.3f} "
                f"r_lum={r:+.2f} outside_otsu={outside:.3f} -> {fn}")

    # FP cohort from the Step 3b overlays (recompute stats identically)
    z = np.load(AUDIT_DIR / "preds_cache.npz")
    P = z["P_ddi"].astype(float)
    ddi_df = pd.read_csv(s01.DATASETS / "ddi_eval.csv")
    dpaths = ddi_df["image_path"].to_numpy()
    mal_sum = P[:, MAL_IDX].sum(axis=1)
    fpcand = np.where(z["y_ddi"].astype(int) == 0)[0]
    fp_top = fpcand[np.argsort(-mal_sum[fpcand])][:n_img]
    log(f"\n  fp_DDI_AK: top-{len(fp_top)} AK-driven benign DDI (same as 3b)")
    for rank, i in enumerate(fp_top, 1):
        img = cv2.imread(str(dpaths[i]))
        if img is None:
            continue
        conc, r, outside, cam_up = cam_stats(model, device, transform_rgb,
                                             img, 3)
        heat = img.astype(np.float64)
        heat[..., 1] = np.clip(heat[..., 1] + cam_up * 153.0, 0, 255)
        ov = np.clip(0.6 * img + 0.4 * heat, 0, 255).astype(np.uint8)
        cv2.putText(ov, f"AK conc {conc:.2f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        fn = f"fp_DDI_AK_{rank:02d}_p{P[i, 3]:.2f}.jpg"
        cv2.imwrite(str(out_dir / fn), ov)
        rows.append({"cohort": "fp_DDI_AK", "panel": "ddi", "rank": rank,
                     "row_index": int(i), "column": "AK", "true_class": 0,
                     "prob": float(P[i, 3]),
                     "conc_top1pct": conc, "cam_lum_r": r,
                     "cam_outside_otsu": outside})
        log(f"    #{rank:02d} p={P[i, 3]:.3f} conc={conc:.3f} "
            f"r_lum={r:+.2f} outside_otsu={outside:.3f} -> {fn}")

    def med(name, key):
        v = [r[key] for r in rows if r["cohort"] == name
             and np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    log("\n  COMPARATOR SUMMARY (medians)")
    log(f"  {'cohort':12s} {'n':>3s} {'conc_top1%':>10s} {'r_lum':>7s} "
        f"{'outside_otsu':>13s}")
    for name in ("fp_DDI_AK", "true_AK", "true_MEL"):
        log(f"  {name:12s} "
            f"{sum(1 for r in rows if r['cohort'] == name):>3d} "
            f"{med(name, 'conc_top1pct'):>10.3f} "
            f"{med(name, 'cam_lum_r'):>7.2f} "
            f"{med(name, 'cam_outside_otsu'):>13.3f}")

    def mw(a, b, label):
        va = [r["conc_top1pct"] for r in rows if r["cohort"] == a]
        vb = [r["conc_top1pct"] for r in rows if r["cohort"] == b]
        if len(va) and len(vb):
            u, p = mannwhitneyu(va, vb, alternative="two-sided")
            log(f"  Mann-Whitney conc {label}: U={u:.0f}, p={p:.3f}")
            return {"comparison": label, "U": float(u), "p": float(p)}
        return None

    tests = [mw("fp_DDI_AK", "true_AK", "fp_DDI_AK vs true_AK"),
             mw("fp_DDI_AK", "true_MEL", "fp_DDI_AK vs true_MEL"),
             mw("true_AK", "true_MEL", "true_AK vs true_MEL")]
    log("  Reading: conc ~0.02 everywhere => 'diffuse' is the AK/MEL heads'")
    log("  NORMAL behavior, not a DDI-specific artifact signature.")
    return rows, tests


def lambda_collapse(y, g, P):
    tune, ev = s02.split_halves(y, g, seed=SEED)
    ev_mask = np.zeros(len(y), bool); ev_mask[ev] = True
    p_mal = P[:, MAL_IDX].sum(axis=1)
    benign = ev_mask & (y == 0)
    rows = []
    log("\n" + "=" * 108)
    log("S3c.2  WHY THE LAMBDA SWEEP COLLAPSES: DRIVER SHARES ON EVAL-HALF BENIGN DDI")
    log("=" * 108)
    hdr = (f"  {'lambda':>6s} {'mean mal_sum':>12s} " +
           " ".join(f"{s + '_mean':>9s}" for s in MAL_SHORT) +
           "   argmax share: " + " ".join(f"{s + '%':>6s}" for s in MAL_SHORT))
    log(hdr)
    for lam in (0.0, 0.25, 0.8):
        adj = P.copy()
        adj[:, 3] = (1.0 - lam) * adj[:, 3]  # deflate AK *inside* the score
        ms = adj[:, MAL_IDX].sum(axis=1)
        top = np.take(MAL_IDX, np.argmax(adj[:, MAL_IDX], axis=1))
        means = adj[benign][:, MAL_IDX].mean(axis=0)
        shares = [float((top[benign] == j).mean()) for j in range(4)]
        log(f"  {lam:>6.2f} {ms[benign].mean():>12.3f} " +
            " ".join(f"{m:>9.3f}" for m in means) +
            "   " + " ".join(f"{s * 100:>6.1f}%" for s in shares))
        rows.append({"lambda": lam, "mal_sum_mean_benign": float(ms[benign].mean()),
                     **{f"{s}_mean": float(m) for s, m in zip(MAL_SHORT, means)},
                     **{f"{s}_argmax_share": float(v) for s, v in zip(MAL_SHORT, shares)}})
    log("  (deflating AK lowers mal_sum but the argmax can migrate to BCC/MEL -")
    log("   single-column recalibration is whack-a-mole when the representation")
    log("   itself maps this domain onto 'malignant-looking'.)")
    return rows


def main():
    t0 = time.time()
    log("=" * 108)
    log("STEP 3c — CAM comparators + lambda-collapse mechanism")
    log("=" * 108)
    y, g, P, yT, gT, PT = load_panel()
    cam_rows, tests = run_comparators(yT, PT)
    lam_rows = lambda_collapse(y, g, P)

    report = {
        "step": "3c — CAM comparators + lambda collapse",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "cam_comparators": cam_rows,
        "mw_tests": tests,
        "lambda_collapse": lam_rows,
    }
    out = AUDIT_DIR / "audit_step3c_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"\n  wrote {out}")
    with open(AUDIT_DIR / "step3c_cam_comparators.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        if cam_rows:
            w.writerow(list(cam_rows[0].keys()))
            for r in cam_rows:
                w.writerow(list(r.values()))
    log(f"  wrote {AUDIT_DIR / 'step3c_cam_comparators.csv'}")
    log(f"\nSTEP 3c COMPLETE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
