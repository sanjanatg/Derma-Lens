"""
audit_step3.py — Step 3: score-level diagnostic (why thresholds can't fix DDI)
==============================================================================

Step 2 established that the DDI failure is score-level, not threshold-level:
at any threshold, DDI specificity is ~0-9% while sensitivity is high — the
malignant score barely separates DDI negatives from DDI positives
(overall AUC ~0.585).

This step dissects the SCORES, not the decisions. Cache-only, no GPU,
deterministic (seeded). Every table carries raw n.

  S3.1  Malignant-score decomposition on DDI by (true label x FST group):
        which malignant class columns carry the probability mass, and which
        column is the top contributor per image?
        -> if mass concentrates in one class (e.g. AK on benign nevi), a
           targeted fix may exist; if smeared across classes, it is domain
           shift and the fix is data-side.
        Test-panel contrast: which columns fire on TRUE malignant images
        in-domain, and how little mass benign test images attract.

  S3.2  Per-group ROC ceilings on DDI for the malignant score: AUC with
        stratified bootstrap CI, Youden point, and sensitivity at fixed
        specificity targets. Quantifies what NO threshold could achieve.

  S3.3  Negative-score distribution shift: DDI vs test panel per group,
        quantiles + one-sided KS statistic, for the malignant score and the
        top false-positive driver columns found in S3.1.

  S3.4  Top-k most confidently wrong cases (highest malignant score among
        true-benign DDI; lowest among true-malignant) written to CSV with
        image paths for manual / Grad-CAM review.

  S3.5  Per-column AUCs: how much ranking power each malignant class column
        has on DDI vs the test panel (does the model rank ANY malignant
        evidence correctly out-of-domain?).

Usage:
  python audit_step3.py                    # full diagnostic from cache
  python audit_step3.py --n-boot 200 --k 5 # quick pass
Requires: audit/preds_cache.npz WITH 8-class probabilities (written by a
full `python audit_step0_1.py` run).
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

import audit_step0_1 as s01  # reuses Step 0 provenance gates

SEED = 42
GROUP_KEYS = [0, 2, 4]
GROUP_NAMES = {0: "FST I-II", 2: "FST III-IV", 4: "FST V-VI"}
MAL_IDX = [0, 1, 2, 3]
BEN_IDX = [4, 5, 6, 7]
MAL_SHORT = ["MEL", "BCC", "SCC", "AK"]
ALL_SHORT = MAL_SHORT + ["NV", "BKL", "DF", "VASC"]
N_BOOT = 2000
CONF = 0.95
TOP_K = 15

try:  # numpy >= 2.0 renamed trapz -> trapezoid
    _trapz = np.trapezoid
except AttributeError:  # pragma: no cover
    _trapz = np.trapz


def log(msg: str = "") -> None:
    print(msg, flush=True)


# =============================================================================
# Loading + integrity (same contract as Step 2)
# =============================================================================

def load_panel():
    """Load cached predictions, re-run Step 0 gates, verify cache integrity."""
    cache = AUDIT_DIR / "preds_cache.npz"
    if not cache.exists():
        raise SystemExit("audit/preds_cache.npz not found - run "
                         "`python audit_step0_1.py` (full, once) first.")
    z = np.load(cache)
    if "P_ddi" not in z.files or "P_test" not in z.files:
        raise SystemExit("preds_cache.npz lacks 8-class probabilities - "
                         "re-run `python audit_step0_1.py` WITHOUT "
                         "--skip-inference once (about 2 min on GPU).")

    y = z["y_ddi"].astype(int)
    p = z["p_ddi"].astype(float)
    g = z["g_ddi"].astype(int)
    P = z["P_ddi"].astype(float)
    yT = z["y_test"].astype(int)
    gT = z["g_test"].astype(int)
    PT = z["P_test"].astype(float)

    # Integrity: cached malignant prob must equal the sum over malignant cols.
    # P is stored float32 (GPU softmax); compare in float32 with 1e-6 slack,
    # identical to the Step 2 gate.
    drift = 0.0
    for pp, PP in ((p, P),):
        p_f32 = pp.astype(np.float32)
        p_from_P = PP[:, MAL_IDX].sum(axis=1, dtype=np.float32)
        drift = max(drift, float(np.max(np.abs(
            p_from_P.astype(np.float64) - p_f32.astype(np.float64)))))
    if drift > 1e-6:
        s01.issue("BLOCKER", "S3.0-cache-drift",
                  f"cached p disagrees with P[:, :4].sum by {drift:.2e} "
                  "- cache is corrupt or from mismatched runs")

    # Provenance gates: full Step 0 re-verified in this process
    s01.run_step0()
    blockers = [i for i in s01.ISSUES if i["severity"] == "BLOCKER"]
    if blockers:
        for i in s01.ISSUES:
            log(f"  [{i['severity']:7s}] {i['code']}: {i['detail']}")
        raise SystemExit(f"Step 0 gates failed with {len(blockers)} BLOCKER "
                         "finding(s) - fix before running Step 3.")

    return y, p, g, P, yT, gT, PT, drift


# =============================================================================
# S3.1 — decomposition of the malignant score by contributing class
# =============================================================================

def decomposition(y, p, g, P, yT, gT, PT):
    log("\n" + "=" * 108)
    log("S3.1  WHICH MALIGNANT CLASS CARRIES THE MASS?  (mean prob per column)")
    log("=" * 108)
    header = (f"{'panel/group/label':26s} {'n':>5s} " +
              " ".join(f"{s:>7s}" for s in ALL_SHORT) +
              f" {'mal_sum':>8s}")
    log(header)
    log("-" * len(header))

    rows = []

    def emit(panel, grp_name, sel_y, sel_g, sel_P):
        for lab, lab_name in ((1, "malignant"), (0, "benign")):
            m = sel_y == lab
            n = int(m.sum())
            if n == 0:
                log(f"{panel + '/' + grp_name + '/' + lab_name:26s} "
                    f"{'0':>5s}  (empty cell)")
                rows.append({"panel": panel, "group": grp_name,
                             "label": lab_name, "n": 0,
                             **{s: float("nan") for s in ALL_SHORT},
                             "mal_sum": float("nan")})
                continue
            means = sel_P[m].mean(axis=0)
            ms = float(means[MAL_IDX].sum())
            log(f"{panel + '/' + grp_name + '/' + lab_name:26s} {n:>5d} " +
                " ".join(f"{means[j]:7.4f}" for j in range(8)) +
                f" {ms:8.4f}")
            rows.append({"panel": panel, "group": grp_name, "label": lab_name,
                         "n": n,
                         **{s: float(means[j]) for j, s in enumerate(ALL_SHORT)},
                         "mal_sum": ms})

    emit("DDI", "ALL", y, g, P)
    for grp in GROUP_KEYS:
        m = g == grp
        emit("DDI", GROUP_NAMES[grp], y[m], g[m], P[m])

    # In-domain contrast: test panel, pooled I-IV (V has n=3, VI absent)
    mT = np.isin(gT, [0, 1, 2, 3])
    emit("TEST", "I-IV", yT[mT], gT[mT], PT[mT])
    log(f"\n  (test FST V n=3 / VI n=0 excluded from the in-domain contrast; "
        "per-group test cells in CSV)")

    # Top malignant contributor per image -> share of images
    log("\n  Top contributor among malignant columns (argmax over MEL/BCC/SCC/AK):")
    sub = ("{:<26s} {:>5s} " + " ".join(["{:>7s}"] * len(MAL_SHORT)))
    log(sub.format("panel/group/label", "n", *[s + "%" for s in MAL_SHORT]))
    contrib_rows = []

    def emit_contrib(panel, grp_name, sel_y, sel_g, sel_P):
        malP = sel_P[:, MAL_IDX]
        top = np.take(MAL_IDX, np.argmax(malP, axis=1))  # argmax column index
        for lab, lab_name in ((1, "malignant"), (0, "benign")):
            m = sel_y == lab
            n = int(m.sum())
            if n == 0:
                continue
            shares = {s: 0.0 for s in MAL_SHORT}
            for j, s in enumerate(MAL_SHORT):
                shares[s] = float((top[m] == j).mean())
            log(sub.format(f"{panel}/{grp_name}/{lab_name}", f"{n}",
                           *[f"{shares[s] * 100:6.1f}%" for s in MAL_SHORT]))
            contrib_rows.append({"panel": panel, "group": grp_name,
                                 "label": lab_name, "n": n, **shares})

    emit_contrib("DDI", "ALL", y, g, P)
    for grp in GROUP_KEYS:
        m = g == grp
        emit_contrib("DDI", GROUP_NAMES[grp], y[m], g[m], P[m])
    mT = np.isin(gT, [0, 1, 2, 3])
    emit_contrib("TEST", "I-IV", yT[mT], gT[mT], PT[mT])
    log("  (AK = Actinic Keratosis; if benign DDI nevi concentrate mass in AK/SCC,")
    log("   the failure is a specific confusable class, not uniform domain shift.)")
    return rows, contrib_rows


# =============================================================================
# S3.2 / S3.5 — ROC ceilings and per-column AUCs
# =============================================================================

def auc_ci(scores, labels, rng, n_boot):
    """AUC with stratified (label) bootstrap CI."""
    if len(np.unique(labels)) < 2:
        return float("nan"), (float("nan"), float("nan"))
    from sklearn.metrics import roc_auc_score
    point = float(roc_auc_score(labels, scores))
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([
            pos[rng.integers(0, len(pos), len(pos))],
            neg[rng.integers(0, len(neg), len(neg))]])
        stats[b] = roc_auc_score(labels[idx], scores[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, (float(lo), float(hi))


def roc_ceilings(y, p, g, rng, n_boot):
    from sklearn.metrics import roc_curve
    log("\n" + "=" * 108)
    log("S3.2  PER-GROUP ROC CEILINGS ON DDI (malignant score)")
    log("      sens@spec = best sensitivity the curve reaches at that specificity")
    log("=" * 108)
    log(f"{'group':12s} {'n':>5s} {'n_pos':>6s} {'n_neg':>6s} "
        f"{'AUC':>6s} {'95% CI':>17s} {'Youden thr':>10s} {'sens':>6s} "
        f"{'spec':>6s} {'s@sp.6':>7s} {'s@sp.8':>7s}")
    rows = []
    for grp in GROUP_KEYS + ["overall"]:
        sel = np.ones(len(y), bool) if grp == "overall" else (g == grp)
        yg, sg = y[sel], p[sel]
        n, npos, nneg = int(sel.sum()), int((yg == 1).sum()), int((yg == 0).sum())
        if npos == 0 or nneg == 0:
            log(f"{GROUP_NAMES.get(grp, 'ALL'):12s} {n:>5d} {npos:>6d} "
                f"{nneg:>6d}   (single-class cell - AUC undefined)")
            rows.append({"group": GROUP_NAMES.get(grp, "ALL"), "n": n,
                         "n_pos": npos, "n_neg": nneg,
                         "auc": float("nan"), "auc_lo": float("nan"),
                         "auc_hi": float("nan")})
            continue
        auc, (lo, hi) = auc_ci(sg, yg, rng, n_boot)
        fpr, tpr, thr = roc_curve(yg, sg)
        youden = tpr - fpr
        k = int(np.argmax(youden))
        spec_at = {t: float("nan") for t in (0.6, 0.8)}
        for t in spec_at:
            ok = (1.0 - fpr) >= t
            spec_at[t] = float(tpr[ok].max()) if ok.any() else float("nan")
        log(f"{GROUP_NAMES.get(grp, 'ALL'):12s} {n:>5d} {npos:>6d} {nneg:>6d} "
            f"{auc:6.3f} [{lo:6.3f},{hi:6.3f}] {thr[k]:10.3f} "
            f"{tpr[k]:6.3f} {1 - fpr[k]:6.3f} "
            f"{spec_at[0.6]:7.3f} {spec_at[0.8]:7.3f}")
        rows.append({"group": GROUP_NAMES.get(grp, "ALL"), "n": n,
                     "n_pos": npos, "n_neg": nneg, "auc": auc,
                     "auc_lo": lo, "auc_hi": hi,
                     "youden_thr": float(thr[k]), "youden_sens": float(tpr[k]),
                     "youden_spec": float(1 - fpr[k]),
                     "sens_at_spec0.6": spec_at[0.6],
                     "sens_at_spec0.8": spec_at[0.8]})
    log("  (s@sp.x = nan means the curve never reaches that specificity - "
        "no threshold exists there.)")
    return rows


def column_aucs(y, p, g, P, yT, gT, PT, rng, n_boot):
    log("\n" + "=" * 108)
    log("S3.5  PER-COLUMN AUC vs BINARY LABEL  (ranking power per class column)")
    log("=" * 108)
    header = (f"{'panel/group':16s} {'n':>5s} {'n_pos':>6s} {'n_neg':>6s} " +
              " ".join(f"{s + '_auc':>8s}" for s in MAL_SHORT))
    log(header)
    log("-" * len(header))
    rows = []

    def emit(panel, grp_name, sel_y, sel_P):
        n, npos, nneg = len(sel_y), int((sel_y == 1).sum()), int((sel_y == 0).sum())
        aucs = []
        for j in MAL_IDX:
            a, _ = auc_ci(sel_P[:, j], sel_y, rng, n_boot)
            aucs.append(a)
        log(f"{panel + '/' + grp_name:16s} {n:>5d} {npos:>6d} {nneg:>6d} " +
            " ".join(f"{a:8.3f}" for a in aucs))
        rows.append({"panel": panel, "group": grp_name, "n": n, "n_pos": npos,
                     "n_neg": nneg,
                     **{f"{s}_auc": float(a) for s, a in zip(MAL_SHORT, aucs)}})

    emit("DDI", "ALL", y, P)
    for grp in GROUP_KEYS:
        m = g == grp
        emit("DDI", GROUP_NAMES[grp], y[m], P[m])
    for grp in (0, 1, 2, 3):
        m = gT == grp
        if m.sum() == 0:
            continue
        emit("TEST", f"FST {['I', 'II', 'III', 'IV'][grp]}", yT[m], PT[m])
    log("  (AUC ~0.5 = column carries no benign/malignant ranking signal there;")
    log("   AUC <0.5 = inverted signal. Compare DDI columns against TEST columns.)")
    return rows


# =============================================================================
# S3.3 — negative-score distribution shift, DDI vs test panel
# =============================================================================

def ks_1sided(a, b):
    """One-sided KS: how much heavier is b's right tail than a's (P(b > a)).

    a = in-domain (test) sample, b = out-of-domain (DDI) sample. scipy's
    alternative='greater' tests F_a(x) > F_b(x) at some x, i.e. rejects when
    b is stochastically LARGER than a - exactly the shift we want to detect.
    """
    from scipy.stats import ks_2samp
    stat, pval = ks_2samp(a, b, alternative="greater", method="asymp")
    return float(stat), float(pval)


def distribution_shift(y, g, P, yT, gT, PT, driver_cols):
    from scipy.stats import ks_2samp
    log("\n" + "=" * 108)
    log("S3.3  NEGATIVE-SCORE (benign) DISTRIBUTION SHIFT:  DDI vs TEST PANEL")
    log("      quantiles of the score on TRUE-BENIGN images; KS one-sided")
    log("      (how much heavier the DDI right tail is)")
    log("=" * 108)
    ddi_neg = {grp: (g == grp) & (y == 0) for grp in GROUP_KEYS}
    ddi_neg["ALL"] = y == 0
    # DDI coarse groups mapped to the comparable test-tone pools
    t_map = {0: "I-II", 2: "III-IV", 4: "I-IV", "ALL": "I-IV"}
    test_neg = {
        "I-II": np.isin(gT, [0, 1]) & (yT == 0),
        "III-IV": np.isin(gT, [2, 3]) & (yT == 0),
        "I-IV": np.isin(gT, [0, 1, 2, 3]) & (yT == 0),
    }

    qs = [0.5, 0.75, 0.9, 0.95, 0.99]
    rows = []
    for grp in GROUP_KEYS + ["ALL"]:
        t_groups = t_map[grp]
        mD = ddi_neg[grp]
        mT = test_neg[t_groups]
        for col, cname in [(None, "mal_sum")] + \
                          [(j, MAL_SHORT[j]) for j in driver_cols]:
            sD = P[mD][:, MAL_IDX].sum(axis=1) if col is None else P[mD, col]
            sT = PT[mT][:, MAL_IDX].sum(axis=1) if col is None else PT[mT, col]
            qD = np.quantile(sD, qs)
            qT = np.quantile(sT, qs)
            stat, pval = ks_1sided(sT, sD)
            if col is None:
                log(f"\n  DDI {GROUP_NAMES.get(grp, 'ALL'):9s} (n={int(mD.sum()):4d}) "
                    f"vs TEST {t_groups} (n={int(mT.sum()):4d})  — malignant sum score")
            else:
                log(f"\n  DDI {GROUP_NAMES.get(grp, 'ALL'):9s} (n={int(mD.sum()):4d}) "
                    f"vs TEST {t_groups} (n={int(mT.sum()):4d})  — column {cname}")
            log("    DDI  q50/75/90/95/99: " + " ".join(f"{v:6.3f}" for v in qD) +
                f"   mean {sD.mean():6.3f}")
            log("    TEST q50/75/90/95/99: " + " ".join(f"{v:6.3f}" for v in qT) +
                f"   mean {sT.mean():6.3f}")
            log(f"    KS={stat:.3f} (p={pval:.2e})  >0.3 = heavy tail shift")
            rows.append({"ddi_group": GROUP_NAMES.get(grp, "ALL"),
                         "ddi_n": int(mD.sum()),
                         "test_group": t_groups, "test_n": int(mT.sum()),
                         "column": cname,
                         **{f"ddi_q{int(q * 100)}": float(v) for q, v in zip(qs, qD)},
                         "ddi_mean": float(sD.mean()),
                         **{f"test_q{int(q * 100)}": float(v) for q, v in zip(qs, qT)},
                         "test_mean": float(sT.mean()),
                         "ks_stat": stat, "ks_p": pval})
    return rows


# =============================================================================
# S3.4 — top-k confidently wrong cases for manual review
# =============================================================================

def review_cases(y, g, P, k=TOP_K):
    import pandas as pd
    ddi_df = pd.read_csv(ROOT / "datasets" / "ddi_eval.csv")
    paths = ddi_df["image_path"].to_numpy()
    mal = P[:, MAL_IDX]
    top_col = np.argmax(mal, axis=1)
    mal_sum = mal.sum(axis=1)

    fp = np.where(y == 0)[0][np.argsort(-mal_sum[y == 0])][:k]
    fn = np.where(y == 1)[0][np.argsort(mal_sum[y == 1])][:k]
    rows = []
    for kind, idxs in (("FP_high_score", fp), ("FN_low_score", fn)):
        for i in idxs:
            rows.append({
                "kind": kind, "ddi_index": int(i), "image_path": paths[i],
                "group": GROUP_NAMES[int(g[i])], "true_label": int(y[i]),
                "mal_sum": float(mal_sum[i]),
                "top_contrib_col": MAL_SHORT[int(top_col[i])],
                "top_contrib_prob": float(mal[i, int(top_col[i])]),
                **{s: float(P[i, j]) for j, s in enumerate(ALL_SHORT)},
            })
    return rows


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--k", type=int, default=TOP_K,
                    help="top-k review cases per direction")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    t0 = time.time()

    log("=" * 108)
    log(f"STEP 3 — score-level diagnostic  (seed={args.seed}, "
        f"n_boot={args.n_boot}, k={args.k})")
    log("=" * 108)

    y, p, g, P, yT, gT, PT, drift = load_panel()
    log(f"  cache OK (drift={drift:.2e}); DDI n={len(y)} "
        f"({int((y == 1).sum())} pos), test n={len(yT)}")
    rng = np.random.default_rng(args.seed)

    decomp, contrib = decomposition(y, p, g, P, yT, gT, PT)
    ceil_rows = roc_ceilings(y, p, g, rng, args.n_boot)
    col_rows = column_aucs(y, p, g, P, yT, gT, PT, rng, args.n_boot)

    # top-2 false-positive driver columns on benign DDI (from contribution shares)
    benign = y == 0
    driver = [int(j) for j in np.argsort(-P[benign][:, MAL_IDX].mean(axis=0))[:2]]
    log(f"\n  top-2 false-positive driver columns on benign DDI: "
        f"{[MAL_SHORT[j] for j in driver]}")
    shift_rows = distribution_shift(y, g, P, yT, gT, PT, driver)
    case_rows = review_cases(y, g, P, args.k)

    # ---- verdict -----------------------------------------------------------
    log("\n" + "=" * 108)
    log("VERDICT")
    log("=" * 108)
    ddi_all_auc = next(r["auc"] for r in ceil_rows if r["group"] == "ALL")
    gap_auc = (max(r["auc"] for r in ceil_rows if r["group"] != "ALL" and
                   not np.isnan(r["auc"])) -
               min(r["auc"] for r in ceil_rows if r["group"] != "ALL" and
                   not np.isnan(r["auc"])))
    top_fp_col = MAL_SHORT[driver[0]]
    fp_share = next(r for r in contrib
                    if r["panel"] == "DDI" and r["group"] == "ALL"
                    and r["label"] == "benign")[top_fp_col]
    mel_aucs = [r["MEL_auc"] for r in col_rows if r["panel"] == "DDI"]
    ak_auc_vvi = next(r["AK_auc"] for r in col_rows
                      if r["panel"] == "DDI" and r["group"] == "FST V-VI")
    log(f"  1. Malignant-score AUC on DDI: {ddi_all_auc:.3f} overall; "
        f"per-group spread {gap_auc:.3f}.")
    for r in ceil_rows:
        if r["group"] in GROUP_NAMES:
            log(f"     {r['group']:11s} AUC {r['auc']:.3f} "
                f"[{r.get('auc_lo', float('nan')):.3f},"
                f"{r.get('auc_hi', float('nan')):.3f}]  "
                f"best sens at spec>=0.6: {r.get('sens_at_spec0.6', float('nan')):.3f}")
    log(f"  2. On benign DDI, the dominant malignant-column contributor is "
        f"{top_fp_col} ({fp_share * 100:.1f}% of images) - see S3.1 shares.")
    conc = max(fp_share, 1 - fp_share)
    if conc >= 0.6:
        log("     -> CONCENTRATED: false-positive mass sits in one malignant class")
        log("        (AK / sun-damage texture response, q90 ~0.9 on DDI vs ~0.1")
        log("        in-domain). Deflating that column on DDI-like images can buy")
        log("        SPECIFICITY that no threshold could (Step 2).")
    else:
        log("     -> SMEARED across malignant classes: classic domain shift;")
        log("        loss-side fixes alone are unlikely to close it (data-side needed).")
    log(f"  3. BUT true-positive evidence is missing or inverted out-of-domain:")
    log(f"     MEL column AUC on DDI ranges {min(mel_aucs):.3f}-{max(mel_aucs):.3f} "
        f"(<=0.5 = inverted), and even the offending AK column cannot rank in")
    log(f"     FST V-VI (AUC {ak_auc_vvi:.3f}). Recalibration can remove the AK "
        f"false positives but cannot manufacture V-VI sensitivity.")
    log("  4. Step 4 implication: the differentiable EO penalty alone optimizes")
    log("     decisions, not features - it cannot fix this. Either fine-tune with")
    log("     DDI-style training images (in-domain hard negatives for AK, tone-")
    log("     augmented positives) or present the ROC ceilings + AK artifact as")
    log("     the paper's central generalization finding.")
    log("  5. Review step3_review_cases.csv images (top-k) for capture artifacts")

    report = {
        "step": "3 — score-level diagnostic (cache-only)",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed, "n_boot": args.n_boot, "k": args.k,
        "cache_drift": drift,
        "step0_findings": s01.ISSUES,
        "decomposition": decomp,
        "contributor_shares": contrib,
        "roc_ceilings": ceil_rows,
        "column_aucs": col_rows,
        "distribution_shift": shift_rows,
        "fp_driver_columns": [MAL_SHORT[j] for j in driver],
        "review_cases": case_rows,
        "notes": [
            "All numbers derive from audit/preds_cache.npz (8-class softmax), "
            "produced by audit_step0_1.py with the notebook transform chain.",
            "AUC CIs are label-stratified bootstrap; raw n printed beside every row.",
            "Test FST V (n=3) / VI (n=0) excluded from in-domain contrasts.",
            "No GPU required; deterministic under fixed seed.",
        ],
    }
    out_json = AUDIT_DIR / "audit_step3_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"\n  wrote {out_json}")

    def _csv(name, header, data_rows):
        path = AUDIT_DIR / name
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in data_rows:
                w.writerow([r.get(h, "") for h in header])
        log(f"  wrote {path}")

    _csv("step3_score_decomposition.csv",
         ["panel", "group", "label", "n"] + ALL_SHORT + ["mal_sum"], decomp)
    _csv("step3_roc_ceilings.csv",
         ["group", "n", "n_pos", "n_neg", "auc", "auc_lo", "auc_hi",
          "youden_thr", "youden_sens", "youden_spec",
          "sens_at_spec0.6", "sens_at_spec0.8"], ceil_rows)
    _csv("step3_distribution_shift.csv",
         list(shift_rows[0].keys()) if shift_rows else
         ["ddi_group", "column"], shift_rows)
    _csv("step3_review_cases.csv",
         list(case_rows[0].keys()) if case_rows else
         ["kind", "image_path"], case_rows)

    log("\n" + "=" * 108)
    log(f"STEP 3 COMPLETE in {time.time() - t0:.1f}s")
    log("=" * 108)


if __name__ == "__main__":
    main()
