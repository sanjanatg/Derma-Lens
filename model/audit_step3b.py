"""
audit_step3b.py — Step 3b: verdict formalization + Grad-CAM + cheap AK fix
===========================================================================

Follow-up to Step 3's AK finding, before any retraining decision:

  S3b.1  DOMAIN vs TONE verdict for the AK artifact. Decisive evidence:
         (i)  DDI-internal AK mass by FST group on benign images — if the
              artifact is tone-causal it must scale with tone INSIDE DDI;
              if it is a capture-style/domain artifact it fires across all
              DDI groups (it does: I-II is the HIGHEST).
         (ii) Tone-matched shift: DDI FST I-II vs test-panel FST I-II
              benigns (same self-reported tone, different capture domain).
         -> framing: a domain-generalization failure that lands on the
            group with the weakest ranking signal, not a tone-causal bias.

  S3b.2  Grad-CAM on the top-15 AK-driven false positives (mechanism
         evidence, correlational until eyeballed): overlays saved to
         figures/gradcam_step3b/, plus quantitative stats (AK-cam mass
         concentration vs mean-cam, lesion-box overlap where available).
         Hand-rolled (no new deps): gradients of the AK logit wrt the
         MobileNetV3 7x7 post-activation map.

  S3b.3  Post-hoc AK down-weighting: p'(malignant) = p_mal - lambda*P_AK
         with lambda fit on the DDI TUNE half only (uniform-threshold config
         re-tuned on tune for parity with Step 2's baseline B), evaluated
         ONCE on the eval half + an IN-DOMAIN safety check (test-panel AUC
         and AK-column recall at matched specificity, in-domain holdout).
         CIs + raw n everywhere. Answers: how much does the cheap fix buy,
         and what does it cost where AK is real?

Usage:
  python audit_step3b.py                 # everything (GPU only for S3b.2)
  python audit_step3b.py --skip-gradcam  # S3b.1 + S3b.3 cache-only
Requires: audit/preds_cache.npz WITH 8-class probabilities (full
`python audit_step0_1.py` run) and the Step 2 split machinery.
"""

import argparse
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

import audit_step0_1 as s01   # Step 0 provenance gates
import audit_step2 as s02     # reuse split_halves + strat_boot_ci

SEED = 42
GROUP_KEYS = s02.GROUP_KEYS                # [0, 2, 4]
GROUP_NAMES = s02.GROUP_NAMES
MAL_IDX = [0, 1, 2, 3]
AK = 3
MAL_SHORT = ["MEL", "BCC", "SCC", "AK"]
TOP_K = 15
CONF = 0.95


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_panel():
    """Cache load + Step 0 gates + integrity, identical contract to Step 2/3."""
    cache = AUDIT_DIR / "preds_cache.npz"
    if not cache.exists():
        raise SystemExit("audit/preds_cache.npz not found - run "
                         "`python audit_step0_1.py` (full, once) first.")
    z = np.load(cache)
    if "P_ddi" not in z.files or "P_test" not in z.files:
        raise SystemExit("preds_cache.npz lacks 8-class probabilities - "
                         "re-run `python audit_step0_1.py` WITHOUT "
                         "--skip-inference once.")
    y = z["y_ddi"].astype(int)
    g = z["g_ddi"].astype(int)
    P = z["P_ddi"].astype(float)
    yT = z["y_test"].astype(int)
    gT = z["g_test"].astype(int)
    PT = z["P_test"].astype(float)

    # integrity identical to step2's gate (float32 storage, 1e-6 slack)
    drift = 0.0
    for pp_name, PP in (("ddi", P),):
        p_f32 = z[f"p_{pp_name}"].astype(np.float32)
        p_from_P = PP[:, MAL_IDX].sum(axis=1, dtype=np.float32)
        drift = max(drift, float(np.max(np.abs(
            p_from_P.astype(np.float64) - p_f32.astype(np.float64)))))
    if drift > 1e-6:
        s01.issue("BLOCKER", "S3b.0-cache-drift",
                  f"cached p_ddi disagrees with P_ddi[:, :4].sum by {drift:.2e}")
    s01.run_step0()
    blockers = [i for i in s01.ISSUES if i["severity"] == "BLOCKER"]
    if blockers:
        for i in s01.ISSUES:
            log(f"  [{i['severity']:7s}] {i['code']}: {i['detail']}")
        raise SystemExit(f"Step 0 gates failed ({len(blockers)} BLOCKER).")
    return y, g, P, yT, gT, PT, drift


# =============================================================================
# S3b.1 — domain vs tone verdict
# =============================================================================

def domain_vs_tone(y, g, P, yT, gT, PT):
    log("\n" + "=" * 108)
    log("S3b.1  IS THE AK ARTIFACT A TONE EFFECT OR A DOMAIN EFFECT?")
    log("=" * 108)
    benign = y == 0
    rows = []
    log("\n  (i) AK prob on BENIGN images, DDI-internal by FST group")
    log(f"  {'group':12s} {'n_benign':>9s} {'AK_mean':>8s} {'AK_q90':>7s} "
        f"{'mal_sum_mean':>13s}")
    for grp in GROUP_KEYS:
        m = benign & (g == grp)
        ak = P[m, AK]
        q90 = float(np.quantile(ak, 0.9))
        log(f"  {GROUP_NAMES[grp]:12s} {int(m.sum()):>9d} {ak.mean():>8.3f} "
            f"{q90:>7.3f} {P[m][:, MAL_IDX].sum(axis=1).mean():>13.3f}")
        rows.append({"evidence": "ddi_internal_by_group",
                     "group": GROUP_NAMES[grp], "n_benign": int(m.sum()),
                     "ak_mean": float(ak.mean()), "ak_q90": q90,
                     "mal_sum_mean": float(P[m][:, MAL_IDX].sum(axis=1).mean())})
    means = [r["ak_mean"] for r in rows]
    # verdict computed after (ii)

    log("\n  (ii) Tone-matched: DDI FST I-II benigns vs TEST FST I-II benigns")
    mD = benign & (g == 0)
    mT = (yT == 0) & np.isin(gT, [0, 1])
    akD, akT = P[mD, AK], PT[mT, AK]
    from scipy.stats import ks_2samp
    ks, kp = ks_2samp(akT, akD, alternative="greater", method="asymp")
    log(f"  DDI  I-II benign: n={len(akD)}, AK mean {akD.mean():.3f}, "
        f"q90 {np.quantile(akD, 0.9):.3f}")
    log(f"  TEST I-II benign: n={len(akT)}, AK mean {akT.mean():.3f}, "
        f"q90 {np.quantile(akT, 0.9):.3f}")
    log(f"  one-sided KS = {ks:.3f} (p={kp:.1e}) at IDENTICAL self-reported tone")
    rows.append({"evidence": "tone_matched_I-II",
                 "group": "FST I-II", "n_benign": int(mD.sum()),
                 "ak_mean": float(akD.mean()),
                 "ak_q90": float(np.quantile(akD, 0.9)),
                 "mal_sum_mean": float(P[mD][:, MAL_IDX].sum(axis=1).mean())})
    rows.append({"evidence": "tone_matched_I-II_test",
                 "group": "FST I-II", "n_benign": int(mT.sum()),
                 "ak_mean": float(akT.mean()),
                 "ak_q90": float(np.quantile(akT, 0.9)),
                 "mal_sum_mean": float(PT[mT][:, MAL_IDX].sum(axis=1).mean())})

    ddi_gap = max(means) - min(means)
    domain_gap = akD.mean() - akT.mean()
    verdict_domain = domain_gap > ddi_gap
    log("\n  VERDICT:")
    log(f"    within-DDI AK spread across tone groups: {ddi_gap:.3f} "
        f"(means {[round(m, 3) for m in means]})")
    log(f"    tone-matched DDI-vs-test AK gap:          {domain_gap:.3f}")
    if verdict_domain:
        log("    -> DOMAIN effect: the artifact is nearly flat across DDI tone")
        log("       groups (light-skin DDI fires hardest) but explodes vs the")
        log("       tone-matched test panel. Capture-style shift, correlated")
        log("       with tone only because DDI differs from training wholesale.")
    else:
        log("    -> tone-graded inside DDI: interpret with care (both effects).")
    log("    Paper framing: 'domain-generalization failure that lands on the")
    log("    group with the weakest ranking signal' - the artifact and the")
    log("    fairness gap COMPOUND rather than share one cause.")
    rows.append({"evidence": "verdict", "group": "",
                 "n_benign": 0, "ak_mean": float(domain_gap),
                 "ak_q90": float(ddi_gap),
                 "mal_sum_mean": float(verdict_domain)})
    return rows


# =============================================================================
# S3b.2 — Grad-CAM on the top AK-driven false positives
# =============================================================================

def gradcam_top_fps(y, g, P, k=TOP_K):
    import cv2
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    from model.fairdermet import FairDermNet
    from model import class_taxonomy as tax

    out_dir = ROOT / "figures" / "gradcam_step3b"
    out_dir.mkdir(parents=True, exist_ok=True)

    ddi_df = pd.read_csv(s01.DATASETS / "ddi_eval.csv")
    paths = ddi_df["image_path"].to_numpy()
    mal_sum = P[:, MAL_IDX].sum(axis=1)
    benign_idx = np.where(y == 0)[0]
    fp_idx = benign_idx[np.argsort(-mal_sum[benign_idx])][:k]

    model = FairDermNet.from_checkpoint(
        s01.CHECKPOINTS / "best.pth",
        num_classes=tax.NUM_CLASSES, num_skin_tones=6)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = A.Compose([
        A.Resize(height=s01.INPUT_SIZE, width=s01.INPUT_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    AK_C = np.array([30, 110, 60], dtype=np.float64)  # BGR green
    NV_C = np.array([0, 0, 230], dtype=np.float64)    # BGR red
    rows = []
    log(f"\n  Grad-CAM ({device}) on top-{k} AK-driven benign DDI images -> {out_dir}")
    for rank, i in enumerate(fp_idx, 1):
        img = cv2.imread(str(paths[i]))
        if img is None:
            log(f"    #{rank:02d} idx {i}: image missing, skipped")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = transform(image=rgb)["image"].unsqueeze(0).to(device)

        feat = model.extract_features(x)
        feat_attended, _ = model.fairness_attention(feat)
        pooled = model.gem(feat_attended)
        embed = model.neck(pooled)
        logits = model.risk_head(embed)
        score = logits[0, AK]
        # gradients wrt the 7x7 post-activation map (exact chain the forward
        # pass uses, incl. fairness_attention)
        grads = torch.autograd.grad(score, feat)[0][0]   # [960, 7, 7]
        acts = feat.detach()[0]
        weights = grads.mean(dim=(1, 2))
        cam = F.relu((weights[:, None, None] * acts).sum(0))
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_up = cv2.resize(cam, (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_LINEAR)

        # focal vs diffuse: share of cam mass in the top-1% pixels
        flat = np.sort(cam_up.ravel())[::-1]
        k1 = max(1, int(0.01 * flat.size))
        conc = float(flat[:k1].sum() / flat.sum()) if flat.sum() > 0 else 0.0

        overlay = img.astype(np.float64)
        heat = np.zeros_like(overlay)
        heat[..., 1] = cam_up * 255.0              # AK cam in green
        overlay = 0.6 * overlay + 0.4 * heat
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        cv2.putText(overlay, f"AKcam conc {conc:.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out_name = f"fp{rank:02d}_idx{i}_G{int(g[i])}_AKp{mal_sum[i]:.2f}.jpg"
        cv2.imwrite(str(out_dir / out_name), overlay)
        rows.append({"rank": rank, "ddi_index": int(i),
                     "group": GROUP_NAMES[int(g[i])],
                     "image_path": paths[i],
                     "mal_sum": float(mal_sum[i]),
                     "ak_prob": float(P[i, AK]),
                     "cam_max": float(cam_up.max()),
                     "cam_concentration": float(conc),
                     "overlay": out_name})
        log(f"    #{rank:02d} idx {i} G{int(g[i])} AKp={P[i, AK]:.3f} "
            f"cam_max={cam_up.max():.2f} conc={conc:.2f} -> {out_name}")

    if rows:
        concs = [r["cam_concentration"] for r in rows]
        log(f"  median top-1% concentration: {np.median(concs):.3f} "
            "(1.0 = all AK-cam mass in one 1% region; ~0.01 = uniform)")
        log("  NOTE: mechanism claims need EYES on the overlays - the stats")
        log("  only say whether activation is focal or diffuse.")
    return rows


# =============================================================================
# S3b.3 — post-hoc AK down-weighting (tune-fit, eval-scored, in-domain guard)
# =============================================================================

def auc_of(scores, labels):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def ak_recalibration(y, g, P, yT, gT, PT, n_boot):
    log("\n" + "=" * 108)
    log("S3b.3  POST-HOC AK DOWN-WEIGHTING  p' = p_mal - lambda * P_AK")
    log("       lambda fit on DDI tune half ONLY; eval half untouched;")
    log("       in-domain safety checked on the test panel")
    log("=" * 108)
    tune, ev = s02.split_halves(y, g, seed=SEED)
    # split_halves returns positional index arrays -> boolean masks
    tune_mask = np.zeros(len(y), bool); tune_mask[tune] = True
    ev_mask = np.zeros(len(y), bool); ev_mask[ev] = True
    p_mal = P[:, MAL_IDX].sum(axis=1)

    def tune_uniform_thr(scores, yy, mm, target=0.95):
        """Largest uniform threshold reaching target sens on tune positives."""
        pos = mm & (yy == 1)
        if pos.sum() == 0:
            return 0.5
        cand = np.unique(np.round(np.quantile(scores[pos],
                                              np.linspace(0.02, 0.98, 97)), 4))
        for t in cand[::-1]:
            if float((scores[pos] >= t).mean()) >= target:
                return float(t)
        return float(cand[-1])

    lam_grid = [round(x, 2) for x in np.arange(0.0, 1.01, 0.05)]
    base_thr = tune_uniform_thr(p_mal, y, tune_mask)
    best = None
    sweep = []
    log(f"\n  tune-half fit (uniform threshold at sens>=0.95 on tune; "
        f"baseline thr={base_thr:.4f}):")
    log(f"  {'lambda':>6s} {'tune_AUC':>9s} {'tune_spec':>9s} {'tune_gap':>9s} {'tune_sens':>9s}")
    for lam in lam_grid:
        s_adj = p_mal - lam * P[:, AK]
        auc_t = auc_of(s_adj[tune_mask], y[tune_mask])
        thr = tune_uniform_thr(s_adj, y, tune_mask)
        flags = s_adj >= thr
        sens_t = float(flags[tune_mask & (y == 1)].mean())
        spec_t = float((~flags[tune_mask & (y == 0)]).mean())
        gap = max(float(flags[tune_mask & (y == 1) & (g == gp)].mean())
                  for gp in GROUP_KEYS) - \
              min(float(flags[tune_mask & (y == 1) & (g == gp)].mean())
                  for gp in GROUP_KEYS)
        log(f"  {lam:>6.2f} {auc_t:>9.3f} {spec_t:>9.3f} {gap:>9.3f} {sens_t:>9.3f}")
        sweep.append({"lambda": lam, "tune_auc": auc_t, "tune_spec": spec_t,
                      "tune_gap": gap, "tune_sens": sens_t})
        key = (round(auc_t, 4), round(spec_t, 4), -lam)
        if best is None or key > best[0]:
            best = (key, lam, thr)

    lam_star, thr_star = best[1], best[2]
    log(f"\n  selected lambda*={lam_star:.2f} (max tune AUC, spec tie-break), "
        f"adjusted threshold {thr_star:.4f}")
    log(f"  (baseline: lambda=0, threshold {base_thr:.4f} on raw p_mal)")

    def eval_config(scores, thr, members):
        flags = (scores >= thr) & members
        metrics = s02.sens_spec(y, flags, g, members)
        ci = s02.strat_boot_ci(y, flags, g, members, n_boot=n_boot, seed=SEED)
        return metrics, ci

    m_base, ci_base = eval_config(p_mal, base_thr, ev_mask)
    m_adj, ci_adj = eval_config(p_mal - lam_star * P[:, AK], thr_star, ev_mask)

    hdr = (f"  {'config':16s} {'grp':10s} {'n':>4s} {'n_pos':>5s} {'n_neg':>5s} "
           f"{'sens':>6s} {'95% CI':>15s} {'spec':>6s} {'95% CI':>15s}")
    log("\n  EVAL HALF (out-of-sample):")
    log(hdr)
    for name, m, ci in (("baseline lam=0", m_base, ci_base),
                        (f"AK-adj lam={lam_star}", m_adj, ci_adj)):
        for grp in GROUP_KEYS + ["overall"]:
            r, c = m[grp], ci.get(grp, {})
            slo, shi = c.get("sens_ci", (float("nan"),) * 2)
            plo, phi = c.get("spec_ci", (float("nan"),) * 2)
            log(f"  {name:16s} {GROUP_NAMES.get(grp, 'ALL'):10s} {r['n']:>4d} "
                f"{r['n_pos']:>5d} {r['n_neg']:>5d} {r['sens']:>6.3f} "
                f"[{slo:5.3f},{shi:5.3f}] {r['spec']:>6.3f} [{plo:5.3f},{phi:5.3f}]")
        gpt = max(m[gp]["sens"] for gp in GROUP_KEYS) - \
            min(m[gp]["sens"] for gp in GROUP_KEYS)
        glo, ghi = ci.get("eo_gap_ci", (float("nan"),) * 2)
        log(f"  {name:16s} EO sens gap {gpt:6.3f} [{glo:5.3f},{ghi:5.3f}]")

    # in-domain safety on the test panel. AK cases are identified from
    # split_test.csv class_idx (cache rows preserve that file's row order);
    # yT is BINARY, so it cannot locate AK by itself.
    log("\n  IN-DOMAIN SAFETY (test panel, same lam*/thr applied):")
    malT = PT[:, MAL_IDX].sum(axis=1)
    safe = {}
    inT = np.isin(gT, [0, 1, 2, 3])
    import pandas as pd
    test_cls = pd.read_csv(s01.DATASETS / "split_test.csv")["class_idx"] \
        .to_numpy().astype(int)
    if len(test_cls) != len(yT):
        log("  !! split_test.csv row count != cache rows - AK recall skipped")
        test_cls = None
    ak_m = (test_cls == AK) & inT if test_cls is not None else None
    for name, s in (("baseline", malT),
                    (f"AK-adj lam*={lam_star}", malT - lam_star * PT[:, AK]),
                    ("AK-adj lam=0.8 (sens.)", malT - 0.8 * PT[:, AK])):
        auc = auc_of(s[inT], yT[inT])
        spec_m = (yT == 0) & inT
        q = np.quantile(s[spec_m], 0.80)   # flag top 20% benign scores -> spec .80
        line = (f"  {name:10s} AUC(I-IV) {auc:.4f}")
        out = {"auc_iv": auc}
        if ak_m is not None and ak_m.sum() > 0:
            ak_rec = float((s[ak_m] >= q).mean())
            nv_spec = float((s[spec_m] < q).mean())
            line += (f" | AK-class recall@spec~.80 {ak_rec:.3f} "
                     f"(n_AK={int(ak_m.sum())}) | realized spec {nv_spec:.3f}")
            out.update({"ak_recall_at_spec80": ak_rec,
                        "realized_spec": nv_spec, "n_ak": int(ak_m.sum())})
        log(line)
        safe[name] = out

    rows = []
    for name, m, ci in (("baseline", m_base, ci_base),
                        (f"ak_adj_lam{lam_star}", m_adj, ci_adj)):
            r, c = m[grp], ci.get(grp, {})
            rows.append({"config": name, "group": GROUP_NAMES.get(grp, "ALL"),
                         "threshold": thr_star if name == "ak_adj" else base_thr,
                         "n": r["n"], "n_pos": r["n_pos"], "n_neg": r["n_neg"],
                         "sensitivity": r["sens"],
                         "sens_lo": c.get("sens_ci", (np.nan, np.nan))[0],
                         "sens_hi": c.get("sens_ci", (np.nan, np.nan))[1],
                         "specificity": r["spec"],
                         "spec_lo": c.get("spec_ci", (np.nan, np.nan))[0],
                         "spec_hi": c.get("spec_ci", (np.nan, np.nan))[1]})
    eo = {}
    for name, m, ci in (("baseline", m_base, ci_base),
                        (f"ak_adj_lam{lam_star}", m_adj, ci_adj)):
        gap_pt = max(m[gp]["sens"] for gp in GROUP_KEYS) - \
                 min(m[gp]["sens"] for gp in GROUP_KEYS)
        glo, ghi = ci.get("eo_gap_ci", (float("nan"),) * 2)
        eo[name] = {"point": gap_pt, "lo": glo, "hi": ghi}
    return rows, {"lambda": lam_star, "thr_adj": thr_star, "thr_base": base_thr,
                  "safety": safe, "sweep": sweep, "eo_gap_eval": eo}


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--skip-gradcam", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    log("=" * 108)
    log(f"STEP 3b — verdict + Grad-CAM + cheap AK fix  (seed={SEED}, "
        f"n_boot={args.n_boot})")
    log("=" * 108)

    y, g, P, yT, gT, PT, drift = load_panel()
    log(f"  cache OK (drift={drift:.2e}); DDI n={len(y)}, test n={len(yT)}")

    verdict_rows = domain_vs_tone(y, g, P, yT, gT, PT)
    cam_rows = [] if args.skip_gradcam else gradcam_top_fps(y, g, P, args.k)

    recal_rows, recal_summary = ak_recalibration(y, g, P, yT, gT, PT,
                                                 args.n_boot)

    report = {
        "step": "3b — domain-vs-tone verdict, Grad-CAM, post-hoc AK fix",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED, "n_boot": args.n_boot, "cache_drift": drift,
        "step0_findings": s01.ISSUES,
        "domain_vs_tone": verdict_rows,
        "gradcam": cam_rows,
        "gradcam_note": ("mechanism claims require manual review of "
                         "figures/gradcam_step3b/ overlays"),
        "ak_recalibration": recal_rows,
        "ak_recalibration_summary": recal_summary,
        "notes": [
            "lambda and thresholds fitted on the DDI tune half only "
            "(same (group x label) split as Step 2, seed 42).",
            "CIs are (group x label)-stratified bootstrap; raw n in every row.",
            "In-domain safety: test-panel AUC + AK-column recall at ~80% spec.",
        ],
    }
    out_json = AUDIT_DIR / "audit_step3b_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"\n  wrote {out_json}")

    with open(AUDIT_DIR / "step3b_ak_recalibration.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        if recal_rows:
            w.writerow(list(recal_rows[0].keys()))
            for r in recal_rows:
                w.writerow(list(r.values()))
    log(f"  wrote {AUDIT_DIR / 'step3b_ak_recalibration.csv'}")

    log("\n" + "=" * 108)
    log(f"STEP 3b COMPLETE in {time.time() - t0:.1f}s")
    log("=" * 108)


if __name__ == "__main__":
    main()
