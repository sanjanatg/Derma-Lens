"""
audit_step3e.py — Step 3e: capture-modality confound check on CAM comparators
=============================================================================

Raised in figure review: the true-MEL comparators are dermoscopic captures
(circular, black-surround) while both AK cohorts are clinical close-ups. If
"lesion-localized CAM" is partly a property of dermoscopic framing (little
background left to diffuse onto), the MEL-vs-AK localization contrast is
confounded with capture modality.

  S3e.1  Tag all 45 step3c comparator images by capture modality, two ways:
         (a) provenance — the `dataset` column of split_test.csv / ddi_eval.csv
             (HAM10000 & ISIC2019 ~ dermoscopic; PAD-UFES-20 ~ clinical photo;
             DDI ~ clinical photo);
         (b) quantitative proxy — dark-border fraction and minimum border
             luminance of the original image (black dermoscope surround).
  S3e.2  Cross-tabs: cohort x modality for the 45; and for the WHOLE test
         panel, class (MEL/AK) x dataset — the dataset-level modality mix the
         comparator selection drew from.
  S3e.3  Mann-Whitney on the existing concentration stats
         (conc_top1pct, cam_outside_otsu): MEL vs AK unadjusted, then
         modality-matched where cell sizes allow, plus the within-class
         modality effect (derm vs clinical MEL images).

Outputs:
  audit/step3e_modality_tags.csv   per-image tags + stats
  audit/audit_step3e_report.json   cross-tabs and test results
  console log

Usage:
  python audit_step3e.py            # cache/CSV-only, no model, no GPU
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent
AUDIT_DIR = ROOT / "audit"
sys.path.insert(0, str(ROOT))

import audit_step0_1 as s01

LOG_T0 = datetime.now()


def log(msg):
    print(f"[{(datetime.now() - LOG_T0).total_seconds():6.1f}s] {msg}", flush=True)


DERM_DATASETS = {"HAM10000", "ISIC2019"}
CLINICAL_DATASETS = {"PAD_UFES_20", "PAD-UFES-20", "DDI"}
DARK_BORDER_FRAC_THR = 0.50   # >50% of border pixels near-black
DARK_BORDER_LUM_THR = 0.20    # or min border-band median luminance < 0.20


def border_stats(img_path):
    """Dark-border fraction + min border-band median luminance (proxy for
    dermoscopic black surround). Purely geometric, no model."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    h, w = gray.shape
    bw = max(2, int(round(0.02 * min(h, w))))  # ~2% border band
    bands = [gray[:bw, :], gray[-bw:, :], gray[:, :bw], gray[:, -bw:]]
    border = np.concatenate([b.ravel() for b in bands])
    return {
        "dark_border_frac": float((border < 0.25).mean()),
        "min_border_band_lum": float(min(b.mean() for b in bands)),
        "border_pixels": int(border.size),
    }


def tag_modality(ds_name, bstats):
    if bstats is None:
        return "unknown", "unknown"
    prov = "derm" if ds_name in DERM_DATASETS else (
        "clinical" if ds_name in CLINICAL_DATASETS else "unknown")
    proxy = "derm" if (bstats["dark_border_frac"] > DARK_BORDER_FRAC_THR or
                       bstats["min_border_band_lum"] < DARK_BORDER_LUM_THR) \
        else "clinical"
    return prov, proxy


def mw(a, b):
    from scipy.stats import mannwhitneyu
    a = [v for v in a if np.isfinite(v)]
    b = [v for v in b if np.isfinite(v)]
    if len(a) < 3 or len(b) < 3:
        return {"n_a": len(a), "n_b": len(b), "p": None}
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "n_a": len(a), "n_b": len(b), "p": float(p)}


def main():
    comp = pd.read_csv(AUDIT_DIR / "step3c_cam_comparators.csv")
    test_df = pd.read_csv(s01.DATASETS / "split_test.csv")
    ddi_df = pd.read_csv(s01.DATASETS / "ddi_eval.csv")
    log(f"comparators: {len(comp)} rows | test panel: {len(test_df)} | "
        f"ddi: {len(ddi_df)}")

    # ---------- S3e.1 tag the 45 ----------
    rows = []
    for _, r in comp.iterrows():
        panel_df = test_df if r["panel"] == "test" else ddi_df
        if r["row_index"] >= len(panel_df):
            log(f"  !! row_index {r['row_index']} out of range for {r['panel']}")
            continue
        src = panel_df.iloc[int(r["row_index"])]["image_path"]
        ds = panel_df.iloc[int(r["row_index"])].get("dataset", "DDI")
        b = border_stats(src)
        prov, proxy = tag_modality(str(ds), b)
        rows.append({**r.to_dict(), "dataset": str(ds), "image_path": str(src),
                     "modality_prov": prov, "modality_proxy": proxy,
                     **(b or {})})
    tags = pd.DataFrame(rows)
    tags.to_csv(AUDIT_DIR / "step3e_modality_tags.csv", index=False)
    log(f"wrote step3e_modality_tags.csv ({len(tags)} rows)")

    # ---------- S3e.2 cross-tabs ----------
    ct_cohort = pd.crosstab(tags["cohort"], tags["modality_prov"])
    log("\n  cohort x modality (provenance):\n" + ct_cohort.to_string())
    agree = (tags["modality_prov"] == tags["modality_proxy"]).mean()
    log(f"  provenance vs dark-border proxy agreement: {agree:.0%}")

    test_df["is_derm"] = test_df["dataset"].isin(DERM_DATASETS)
    panel_ct = pd.crosstab(test_df["class_idx"], test_df["dataset"])
    log("\n  WHOLE test panel, class_idx x dataset:\n" + panel_ct.to_string())
    mel = test_df[test_df["class_idx"] == 0]
    ak = test_df[test_df["class_idx"] == 3]
    log(f"  MEL test images: {len(mel)} ({mel['is_derm'].mean():.0%} derm-provenance)")
    log(f"  AK  test images: {len(ak)} ({ak['is_derm'].mean():.0%} derm-provenance)")

    # ---------- S3e.3 tests on existing stats ----------
    res = {}
    for stat in ("conc_top1pct", "cam_outside_otsu"):
        res[f"{stat}__mel_vs_ak_unmatched"] = mw(
            tags.loc[tags.cohort == "true_MEL", stat],
            tags.loc[tags.cohort == "true_AK", stat])
        for mod in ("derm", "clinical"):
            m = tags.loc[(tags.cohort == "true_MEL") &
                         (tags.modality_prov == mod), stat]
            a = tags.loc[(tags.cohort == "true_AK") &
                         (tags.modality_prov == mod), stat]
            res[f"{stat}__mel_vs_ak__{mod}"] = mw(m, a)
        d = tags.loc[(tags.cohort == "true_MEL") &
                     (tags.modality_prov == "derm"), stat]
        c = tags.loc[(tags.cohort == "true_MEL") &
                     (tags.modality_prov == "clinical"), stat]
        res[f"{stat}__within_MEL_derm_vs_clinical"] = mw(d, c)
        d = tags.loc[(tags.cohort == "true_AK") &
                     (tags.modality_prov == "derm"), stat]
        c = tags.loc[(tags.cohort == "true_AK") &
                     (tags.modality_prov == "clinical"), stat]
        res[f"{stat}__within_AK_derm_vs_clinical"] = mw(d, c)
        d = tags.loc[(tags.cohort == "fp_DDI_AK") &
                     (tags.modality_prov == "derm"), stat]
        c = tags.loc[(tags.cohort == "fp_DDI_AK") &
                     (tags.modality_prov == "clinical"), stat]
        res[f"{stat}__within_FP_derm_vs_clinical"] = mw(d, c)

    log("\n  Mann-Whitney summary")
    for k, v in res.items():
        p = "n/a" if v["p"] is None else f"p={v['p']:.4f}"
        ma = v.get("median_a"); mb = v.get("median_b")
        med = "" if ma is None else f" medians {ma:.3f} vs {mb:.3f}"
        log(f"    {k:44s} n={v['n_a']}/{v['n_b']}{med} {p}")

    # ---------- S3e.4 modality-matched counterfactual CAM cohorts ----------
    cf = run_counterfactuals(test_df)
    if cf is not None:
        cf_tags = pd.concat([tags, cf], ignore_index=True)
        cf_res = {}
        for stat in ("conc_top1pct", "cam_outside_otsu", "cam_lum_r"):
            cf_res[f"{stat}__derm_MEL_vs_derm_AK"] = mw(
                cf_tags.loc[(cf_tags.cohort == "true_MEL") &
                            (cf_tags.modality_prov == "derm"), stat],
                cf_tags.loc[(cf_tags.cohort == "true_AK_derm"), stat])
            cf_res[f"{stat}__clin_MEL_vs_clin_AK"] = mw(
                cf_tags.loc[(cf_tags.cohort == "true_MEL_clinical"), stat],
                cf_tags.loc[(cf_tags.cohort == "true_AK"), stat])
            cf_res[f"{stat}__within_AK_derm_vs_clin"] = mw(
                cf_tags.loc[(cf_tags.cohort == "true_AK_derm"), stat],
                cf_tags.loc[(cf_tags.cohort == "true_AK"), stat])
            cf_res[f"{stat}__within_MEL_derm_vs_clin"] = mw(
                cf_tags.loc[(cf_tags.cohort == "true_MEL"), stat],
                cf_tags.loc[(cf_tags.cohort == "true_MEL_clinical"), stat])
        log("\n  MODALITY-MATCHED / COUNTERFACTUAL TESTS")
        for k, v in cf_res.items():
            p = "n/a" if v["p"] is None else f"p={v['p']:.4f}"
            ma = v.get("median_a"); mb = v.get("median_b")
            med = "" if ma is None else f" medians {ma:.3f} vs {mb:.3f}"
            log(f"    {k:44s} n={v['n_a']}/{v['n_b']}{med} {p}")
        res.update(cf_res)

    report = {
        "generated": LOG_T0.isoformat(),
        "cohort_x_modality_prov": {str(k): int(v) for k, v in
                                   ct_cohort.stack().items()},
        "prov_vs_proxy_agreement": float(agree),
        "test_panel_class_x_dataset": {str(k): int(v) for k, v in
                                       panel_ct.stack().items()},
        "test_panel_mel_derm_frac": float(mel["is_derm"].mean()),
        "test_panel_ak_derm_frac": float(ak["is_derm"].mean()),
        "tests": res,
    }
    (AUDIT_DIR / "audit_step3e_report.json").write_text(
        json.dumps(report, indent=2))
    log(f"\nwrote audit_step3e_report.json")


def run_counterfactuals(test_df):
    """S3e.4: CAM comparators with class and modality crossed.

    New cohorts (selection rule mirrors step3c: highest column prob within
    stratum):
      true_AK_derm       top-15 derm-provenance AK  (HAM10000/ISIC2019)
      true_MEL_clinical  all clinical-provenance MEL (PAD-UFES-20, n<=8)
    The original cohorts supply the other cells: true_AK (clinical AK,
    PAD), true_MEL (derm MEL). Overlays -> figures/gradcam_step3e/.
    """
    import cv2
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    from model.fairdermet import FairDermNet
    from model import class_taxonomy as tax
    from audit_step3c import cam_stats

    z = np.load(AUDIT_DIR / "preds_cache.npz")
    PT = z["P_test"].astype(float)
    cls = test_df["class_idx"].to_numpy().astype(int)
    ds = test_df["dataset"].astype(str).to_numpy()
    paths = test_df["image_path"].to_numpy()
    derm_mask = np.isin(ds, sorted(DERM_DATASETS))

    specs = [
        ("true_AK_derm", np.where((cls == 3) & derm_mask)[0], 3, "AK", 15),
        ("true_MEL_clinical", np.where((cls == 0) & ~derm_mask)[0], 0, "MEL", 15),
    ]
    out_dir = ROOT / "figures" / "gradcam_step3e"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FairDermNet.from_checkpoint(
        s01.CHECKPOINTS / "best.pth",
        num_classes=tax.NUM_CLASSES, num_skin_tones=6)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log(f"  S3e.4 counterfactual CAMs on {device}")

    transform_rgb = A.Compose([
        A.Resize(height=s01.INPUT_SIZE, width=s01.INPUT_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    rows = []
    for name, cand, col, short, n_take in specs:
        if len(cand) == 0:
            log(f"  !! {name}: no candidates found - skipped")
            continue
        top = cand[np.argsort(-PT[cand, col])][:n_take]
        log(f"  {name}: {len(top)} of {len(cand)} available "
            f"(top by {short} prob)")
        for rank, i in enumerate(top, 1):
            img = cv2.imread(str(paths[i]))
            if img is None:
                continue
            conc, r, outside, cam_up = cam_stats(model, device, transform_rgb,
                                                 img, col)
            heat = img.astype(np.float64)
            heat[..., 1] = np.clip(heat[..., 1] + cam_up * 153.0, 0, 255)
            ov = np.clip(0.6 * img + 0.4 * heat, 0, 255).astype(np.uint8)
            cv2.putText(ov, f"{short} conc {conc:.2f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            fn = f"{name}_{rank:02d}_p{PT[i, col]:.2f}.jpg"
            cv2.imwrite(str(out_dir / fn), ov)
            b = border_stats(paths[i])
            rows.append({"cohort": name, "panel": "test", "rank": rank,
                         "row_index": int(i), "column": short,
                         "true_class": int(cls[i]),
                         "prob": float(PT[i, col]),
                         "conc_top1pct": conc, "cam_lum_r": r,
                         "cam_outside_otsu": outside,
                         "dataset": str(ds[i]), "image_path": str(paths[i]),
                         "modality_prov": "derm" if derm_mask[i] else "clinical",
                         "modality_proxy": "derm" if b and (
                             b["dark_border_frac"] > DARK_BORDER_FRAC_THR or
                             b["min_border_band_lum"] < DARK_BORDER_LUM_THR)
                             else "clinical",
                         **(b or {})})
            log(f"    #{rank:02d} p={PT[i, col]:.3f} conc={conc:.3f} "
                f"outside={outside:.3f} -> {fn}")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(AUDIT_DIR / "step3e_counterfactual.csv", index=False)
    log(f"  wrote step3e_counterfactual.csv ({len(df)} rows), "
        f"overlays in figures/gradcam_step3e/")
    return df


if __name__ == "__main__":
    main()
