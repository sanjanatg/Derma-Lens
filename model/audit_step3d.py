"""
audit_step3d.py — Step 3d: eyeball-pass support for the 45 Grad-CAM overlays
=============================================================================

The comparator experiment (Step 3c) showed CAM *concentration* does not
separate FP-DDI-AK from true-AK. Before the mechanism paragraph is finalized
a human must look at the images, and two questions need quantitative proxies:

  S3d.1  CAM PLACEMENT (needs the model, 45 images): where does the AK/MEL
         activation sit? edge-band mass share (outer 12.5% frame border),
         center-of-mass distance from frame center, and the luminance of the
         top-10% CAM pixels relative to the image median ("looks at darker
         pixels than the scene average?").

  S3d.2  IMAGE-LEVEL ARTIFACT SCAN (no model): corner-vs-center luminance
         ratio (vignetting proxy, classic DDI capture artifact), Otsu
         dark-mask area fraction, mean saturation. Flags candidate
         capture artifacts (dark vignette, color chart / ruler candidates)
         for the human to confirm or reject.

Outputs: audit/step3d_placement.csv, audit/audit_step3d_report.json,
figures/gradcam_step3d/sheet_{cohort}.png (original | overlay per case) and
review_gradcam_step3c.html (triads, base64-embedded, stats captions) for the
eyeball pass.

Usage:  python audit_step3d.py        # GPU for the 45 CAMs, rest CPU
"""

import base64
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent
AUDIT_DIR = ROOT / "audit"
FIG3C = ROOT / "figures" / "gradcam_step3c"
FIG3D = ROOT / "figures" / "gradcam_step3d"
sys.path.insert(0, str(ROOT))

import audit_step0_1 as s01
import audit_step3c as s3c

SEED = 42
COHORTS = ("fp_DDI_AK", "true_AK", "true_MEL")
CELL = 256  # contact-sheet cell size


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_cohort_meta():
    """cohort -> list of dicts(rank, row_index, column, prob) from the 3c CSV."""
    df = pd.read_csv(AUDIT_DIR / "step3c_cam_comparators.csv")
    meta = {}
    for c in COHORTS:
        sub = df[df["cohort"] == c].sort_values("rank")
        meta[c] = sub.to_dict("records")
    return meta


def image_paths_for(panel):
    if panel == "test":
        df = pd.read_csv(s01.DATASETS / "split_test.csv")
    else:
        df = pd.read_csv(s01.DATASETS / "ddi_eval.csv")
    return df["image_path"].to_numpy()


def image_artifact_scan(img_bgr):
    """No-model capture-artifact proxies on the ORIGINAL image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    h, w = gray.shape

    def box_mean(y0, y1, x0, x1):
        return float(gray[int(y0):int(y1), int(x0):int(x1)].mean())

    cw, ch = 0.15 * w, 0.15 * h
    corners = [box_mean(0, ch, 0, cw), box_mean(0, ch, w - cw, w),
               box_mean(h - ch, h, 0, cw), box_mean(h - ch, h, w - cw, w)]
    corner = float(np.mean(corners))
    center = box_mean(0.3 * h, 0.7 * h, 0.3 * w, 0.7 * w)
    vig = corner / center if center > 0 else float("nan")

    _, otsu = cv2.threshold((gray * 255).astype(np.uint8), 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_frac = float((otsu > 0).mean())

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = float(hsv[..., 1].mean() / 255.0)
    # saturated-color-patch candidates (color chart): share of pixels with
    # high saturation AND mid-high value in any of 6 coarse hue bins
    s_mask = (hsv[..., 1] > 120) & (hsv[..., 2] > 90)
    chart_frac = float(s_mask.mean())

    return {"corner_lum": corner, "center_lum": center,
            "vignette_ratio": vig, "otsu_area_frac": otsu_frac,
            "mean_saturation": sat, "sat_patch_frac": chart_frac}


def cam_placement(cam_up, img_bgr):
    """Where does the activation sit, and what does it look at?"""
    c = cam_up.astype(np.float64)
    tot = c.sum()
    if tot <= 0:
        return {k: float("nan") for k in
                ("edge_band_share", "com_dist", "hot_lum_ratio")}
    h, w = c.shape
    ys, xs = np.mgrid[0:h, 0:w]
    com_y, com_x = (c * ys).sum() / tot / h, (c * xs).sum() / tot / w
    com_dist = float(np.hypot(com_y - 0.5, com_x - 0.5))

    bx, by = int(0.125 * w), int(0.125 * h)
    band = np.ones_like(c, dtype=bool)
    band[by:h - by, bx:w - bx] = False
    edge_share = float(c[band].sum() / tot)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    k = max(1, int(0.10 * c.size))
    hot = np.argsort(c, axis=None)[::-1][:k]
    hot_lum = float(gray.ravel()[hot].mean())
    med_lum = float(np.median(gray))
    return {"edge_band_share": edge_share, "com_dist": com_dist,
            "hot_lum_ratio": hot_lum / med_lum if med_lum > 0 else float("nan")}


def contact_sheet(cohort, rows, out_png):
    """One sheet per cohort: original | overlay per case, labelled."""
    tiles = []
    for r in rows:
        ov = cv2.imread(str(r["overlay_path"]))
        orig = cv2.imread(str(r["src_path"]))
        if ov is None or orig is None:
            continue
        ov = cv2.resize(ov, (CELL, CELL))
        orig = cv2.resize(orig, (CELL, CELL))
        sep = np.full((CELL, 4, 3), 255, np.uint8)
        tile = np.hstack([orig, sep, ov])
        bar = np.full((34, tile.shape[1], 3), 255, np.uint8)
        label = (f"{cohort} #{r['rank']:02d}  p={r['prob']:.2f}  "
                 f"edge={r['edge_band_share']:.2f}  "
                 f"hotlum={r['hot_lum_ratio']:.2f}  "
                 f"vig={r['vignette_ratio']:.2f}")
        cv2.putText(bar, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile]))
    if tiles:
        sheet = np.vstack(tiles)
        cv2.imwrite(str(out_png), sheet)
        log(f"  wrote {out_png} ({sheet.shape[1]}x{sheet.shape[0]})")


def write_review_html(all_rows, out_html):
    """Self-contained triad review page (base64 images, stats captions)."""
    by = {c: [r for r in all_rows if r["cohort"] == c] for c in COHORTS}
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<style>body{font-family:Segoe UI,Arial;margin:14px;"
             "background:#fafafa}",
             "h2{margin:18px 0 6px}",
             "table{border-collapse:collapse}",
             "td{padding:2px 6px;vertical-align:top;text-align:center;"
             "font-size:11px}",
             "img{width:230px;height:230px;object-fit:cover;"
             "border:1px solid #999}",
             ".cap{color:#333}</style></head><body>",
             "<h1>Step 3c/3d Grad-CAM triad review — fp_DDI_AK | true_AK "
             "| true_MEL</h1>",
             "<p>Each triad: AK-driven DDI false positive (AK-column CAM) | "
             "true AK (AK-column CAM) | true melanoma (MEL-column CAM). "
             "<p>Each triad: AK-driven DDI false positive (AK-column CAM) | "
             "true AK (AK-column CAM) | true melanoma (MEL-column CAM). "
             "GREEN overlay intensity = Grad-CAM for the stated column "
             "(green channel boosted, 60/40 blend).</p>"]
    for i in range(15):
        parts.append("<table><tr>")
        for c in COHORTS:
            r = by[c][i]
            b64 = base64.b64encode(Path(r["overlay_path"]).read_bytes()).decode()
            parts.append(
                f"<td><img src='data:image/jpeg;base64,{b64}'><br>"
                f"<span class='cap'><b>{c} #{r['rank']:02d}</b> "
                f"p={r['prob']:.2f}<br>"
                f"edge={r['edge_band_share']:.2f} "
                f"hotlum={r['hot_lum_ratio']:.2f} "
                f"vig={r['vignette_ratio']:.2f}</span></td>")
        parts.append("</tr></table>")
    parts.append("</body></html>")
    Path(out_html).write_text("\n".join(parts), encoding="utf-8")
    log(f"  wrote {out_html} ({Path(out_html).stat().st_size / 1e6:.1f} MB)")


def main():
    t0 = time.time()
    log("=" * 104)
    log("STEP 3d — CAM placement + capture-artifact scan + triad sheets")
    log("=" * 104)
    s01.run_step0()
    blockers = [i for i in s01.ISSUES if i["severity"] == "BLOCKER"]
    if blockers:
        for i in s01.ISSUES:
            log(f"  [{i['severity']:7s}] {i['code']}: {i['detail']}")
        raise SystemExit("Step 0 gates failed.")

    meta = load_cohort_meta()
    paths_by_panel = {"test": image_paths_for("test"),
                      "ddi": image_paths_for("ddi")}

    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from model.fairdermet import FairDermNet
    from model import class_taxonomy as tax
    from scipy.stats import mannwhitneyu

    model = FairDermNet.from_checkpoint(
        s01.CHECKPOINTS / "best.pth",
        num_classes=tax.NUM_CLASSES, num_skin_tones=6)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log(f"  model on {device.type}")

    transform_rgb = A.Compose([
        A.Resize(height=s01.INPUT_SIZE, width=s01.INPUT_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    FIG3D.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for cohort in COHORTS:
        col = 3 if cohort != "true_MEL" else 0
        panel = "ddi" if cohort.startswith("fp_") else "test"
        srcs = paths_by_panel[panel]
        log(f"\n  {cohort}: {len(meta[cohort])} images, column {col}")
        for r in meta[cohort]:
            src = Path(str(srcs[r["row_index"]]))
            ov_path = FIG3C / f"{cohort}_{r['rank']:02d}_p{r['prob']:.2f}.jpg"
            img = cv2.imread(str(src))
            if img is None or not ov_path.exists():
                log(f"    #{r['rank']:02d} SKIP (missing {src.name})")
                continue
            art = image_artifact_scan(img)
            _, _, _, cam_up = s3c.cam_stats(model, device, transform_rgb,
                                            img, col)
            place = cam_placement(cam_up, img)
            row = {"cohort": cohort, "panel": panel, "rank": r["rank"],
                   "row_index": r["row_index"],
                   "src_path": str(src), "overlay_path": str(ov_path),
                   "prob": r["prob"], **art, **place}
            all_rows.append(row)
            log(f"    #{r['rank']:02d} p={row['prob']:.2f} "
                f"edge={place['edge_band_share']:.3f} "
                f"com={place['com_dist']:.3f} "
                f"hotlum={place['hot_lum_ratio']:.2f} "
                f"vig={art['vignette_ratio']:.2f} "
                f"otsu={art['otsu_area_frac']:.2f}")

    log("\n  COHORT SUMMARY (medians)")
    keys = ("edge_band_share", "com_dist", "hot_lum_ratio",
            "vignette_ratio", "otsu_area_frac")
    log(f"  {'cohort':12s} " + " ".join(f"{k[:14]:>15s}" for k in keys))
    summ = {}
    for c in COHORTS:
        vals = {k: float(np.median([r[k] for r in all_rows
                                    if r["cohort"] == c
                                    and np.isfinite(r[k])]))
                for k in keys}
        summ[c] = vals
        log(f"  {c:12s} " + " ".join(f"{vals[k]:>15.3f}" for k in keys))

    def mw(key, label):
        a = [r[key] for r in all_rows if r["cohort"] == "fp_DDI_AK"
             and np.isfinite(r[key])]
        b = [r[key] for r in all_rows if r["cohort"] == "true_AK"
             and np.isfinite(r[key])]
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        log(f"  Mann-Whitney {label}: U={u:.0f}, p={p:.3f}")
        return {"comparison": label, "stat": key, "U": float(u),
                "p": float(p)}

    tests = [mw("edge_band_share", "edge_band fp_DDI vs true_AK"),
             mw("hot_lum_ratio", "hot_lum fp_DDI vs true_AK"),
             mw("vignette_ratio", "vignette fp_DDI vs true_AK")]
    flags = [r for r in all_rows if r["vignette_ratio"] < 0.75]
    log(f"\n  vignette-flagged images (corner/center lum < 0.75): "
        f"{len(flags)}")
    for r in flags:
        log(f"    {r['cohort']} #{r['rank']:02d} vig={r['vignette_ratio']:.2f}"
            f"  ({Path(r['src_path']).name})")

    contact_sheet("fp_DDI_AK", [r for r in all_rows
                                if r["cohort"] == "fp_DDI_AK"],
                  FIG3D / "sheet_fp_DDI_AK.png")
    contact_sheet("true_AK", [r for r in all_rows if r["cohort"] == "true_AK"],
                  FIG3D / "sheet_true_AK.png")
    contact_sheet("true_MEL",
                  [r for r in all_rows if r["cohort"] == "true_MEL"],
                  FIG3D / "sheet_true_MEL.png")
    write_review_html(all_rows, ROOT.parent / "review_gradcam_step3c.html")

    report = {
        "step": "3d — CAM placement + capture-artifact scan",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "cohort_medians": summ,
        "mw_tests": tests,
        "vignette_flagged": [{"cohort": r["cohort"], "rank": r["rank"],
                              "vignette_ratio": r["vignette_ratio"],
                              "src": r["src_path"]} for r in flags],
        "rows": [{k: v for k, v in r.items()
                  if k not in ("src_path", "overlay_path")}
                 for r in all_rows],
    }
    (AUDIT_DIR / "audit_step3d_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    with open(AUDIT_DIR / "step3d_placement.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["cohort", "panel", "rank", "row_index", "prob",
                *keys, "corner_lum", "center_lum", "mean_saturation",
                "sat_patch_frac"]
        w.writerow(cols)
        for r in all_rows:
            w.writerow([r[c] for c in cols])
    log(f"\n  wrote {AUDIT_DIR / 'step3d_placement.csv'}")
    log(f"STEP 3d COMPLETE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
