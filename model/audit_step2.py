"""
audit_step2.py — Step 2: per-group threshold equalization (DDI panel)
=====================================================================

Problem: at the shipped single clinical threshold (0.25), DDI sensitivity
spans ~21 points across tone groups (FST V-VI worst). A single threshold is
calibrated on the (predominantly light-tone) test panel; on DDI the malignant
score distribution shifts by group.

Method (deliberately conservative — this goes in the paper):
  1. Re-check every Step 0 provenance gate by re-running the Step 0+1 audit's
     checks, then verify the prediction cache against the checkpoint state.
  2. Split the DDI panel into tune / eval halves, stratified by
     (tone group x true label) with seed 42. Thresholds are FITTED on the
     tune half only and evaluated ONCE on the eval half (out-of-sample).
  3. Candidate per-group thresholds: per-group score quantiles (2%..98%).
     Policy families:
       - "sens>=tau"  : smallest threshold reaching target sensitivity per
                        group (tau in {0.95, 0.96, 0.97})
       - "gap+target" : same, but the selected config minimizes the tune-half
                        equalized-odds gap with penalty lambda on the mean
                        |threshold - global-calibrated| distance
                        (lambda in {0, 0.25, 0.5, 1.0, 2.0})
  4. On the eval half, compare against two baselines:
       - baseline A: uniform 0.25 (the shipped threshold_config threshold)
       - baseline B: uniform threshold tuned on the tune half for the same
         target (isolates the *group-specific* benefit from plain re-tuning)
     with stratified bootstrap CIs for sensitivity, specificity and the
     equalized-odds gap. Raw n / n_pos / n_neg printed beside every number.

Usage:
  python audit_step2.py                 # full run (uses audit/preds_cache.npz)
  python audit_step2.py --step0-only    # provenance gates only
Requires: audit/preds_cache.npz written by audit_step0_1.py WITH 8-class
probabilities (run `python audit_step0_1.py` once without --skip-inference).
"""

import argparse
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

import audit_step0_1 as s01  # reuses Step 0 gates + bootstrap machinery

SEED = 42
N_BOOT = 2000
CONF = 0.95
GROUP_KEYS = [0, 2, 4]
GROUP_NAMES = {0: "FST I-II", 2: "FST III-IV", 4: "FST V-VI"}
MAL_IDX = list(range(4))       # contiguous 8-class: malignant = 0..3
BEN_IDX = list(range(4, 8))


def log(msg: str = "") -> None:
    print(msg, flush=True)


# =============================================================================
# Loading + integrity
# =============================================================================

def load_panel():
    """Load cached predictions, re-run Step 0 gates, verify cache integrity."""
    cache = AUDIT_DIR / "preds_cache.npz"
    if not cache.exists():
        raise SystemExit("audit/preds_cache.npz not found - run "
                         "`python audit_step0_1.py` (full, once) first.")
    z = np.load(cache)
    if "P_ddi" not in z.files:
        raise SystemExit("preds_cache.npz lacks 8-class probabilities - "
                         "re-run `python audit_step0_1.py` WITHOUT "
                         "--skip-inference once (about 2 min on GPU).")

    y = z["y_ddi"].astype(int)
    p = z["p_ddi"].astype(float)
    g = z["g_ddi"].astype(int)
    P = z["P_ddi"].astype(float)

    # Integrity: cached malignant prob must equal the sum over malignant cols.
    # P is stored float32 (GPU softmax) and p was accumulated from float32 sums,
    # so both sides are compared in float32; tolerance 1e-6 = several ulps of
    # float32 (~1e-7), enough to catch real corruption while ignoring rounding.
    p_f32 = p.astype(np.float32)
    p_from_P = P[:, MAL_IDX].sum(axis=1, dtype=np.float32)
    drift = float(np.max(np.abs(p_from_P.astype(np.float64)
                                - p_f32.astype(np.float64))))
    if drift > 1e-6:
        s01.issue("BLOCKER", "S2.0-cache-drift",
                  f"cached p_ddi disagrees with P_ddi[:, :4].sum by {drift:.2e} "
                  "- cache is corrupt or from mismatched runs")

    # Provenance gates: full Step 0 (checkpoint freshness, threshold config,
    # notebook guard, split-label reconciliation) re-verified in this process
    s01.run_step0()

    return y, p, g, P, drift


def split_halves(y, g, seed=SEED):
    """Stratified tune/eval split by (group, label). Returns index arrays."""
    rng = np.random.default_rng(seed)
    tune, ev = [], []
    for grp in GROUP_KEYS:
        for lab in (0, 1):
            idx = np.where((g == grp) & (y == lab))[0]
            idx = idx[rng.permutation(len(idx))]
            half = len(idx) // 2
            tune.extend(idx[:half])
            ev.extend(idx[half:])
    return np.sort(np.array(tune)), np.sort(np.array(ev))


# =============================================================================
# Metric helpers (all CIs stratified by group x label -> exact raw n)
# =============================================================================

def sens_spec(y, flags, g, members):
    """Sensitivity/specificity per group from binary flags (p >= threshold)."""
    out = {}
    for grp in GROUP_KEYS + ["overall"]:
        sel = np.ones(len(y), bool) if grp == "overall" else (g == grp)
        sel &= members
        yg, fg = y[sel], flags[sel]
        pos, neg = yg == 1, yg == 0
        sens = float(fg[pos].mean()) if pos.sum() else float("nan")
        spec = float((~fg[neg]).mean()) if neg.sum() else float("nan")
        out[grp] = {"n": int(sel.sum()), "n_pos": int(pos.sum()),
                    "n_neg": int(neg.sum()), "sens": sens, "spec": spec}
    return out


def eo_gap_from_flags(y, flags, g, idxs):
    """Max-min sensitivity gap over tone groups present in idxs."""
    sens = {}
    for grp in GROUP_KEYS:
        sel = (g == grp) & idxs
        pos = sel & (y == 1)
        if pos.sum() == 0:
            continue
        sens[grp] = float(flags[pos].mean())
    if len(sens) < 2:
        return float("nan"), sens
    return float(max(sens.values()) - min(sens.values())), sens


def strat_boot_ci(y, flags, g, members, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CIs for per-group sens/spec and the EO gap (eval half only).

    Resampling is stratified by (group x label), so every replicate keeps the
    raw n_pos / n_neg of each group exactly - the CI can never detach from n.
    """
    rng = np.random.default_rng(seed)
    strata = {}
    for grp in GROUP_KEYS:
        for lab in (0, 1):
            m = np.where(members & (g == grp) & (y == lab))[0]
            if len(m):
                strata[(grp, lab)] = m

    sens_b = {grp: [] for grp in GROUP_KEYS}
    spec_b = {grp: [] for grp in GROUP_KEYS}
    gap_b = []
    for _ in range(n_boot):
        picks = []
        for key, m in strata.items():
            picks.append(m[rng.integers(0, len(m), len(m))])
        picks = np.concatenate(picks)
        fg = flags[picks]
        yg = y[picks]
        gg = g[picks]
        for grp in GROUP_KEYS:
            gm = gg == grp
            pos = gm & (yg == 1)
            neg = gm & (yg == 0)
            if pos.sum():
                sens_b[grp].append(float(fg[pos].mean()))
            if neg.sum():
                spec_b[grp].append(float((~fg[neg]).mean()))
        gap, _ = eo_gap_from_flags(yg, fg, gg, np.ones(len(yg), bool))
        if np.isfinite(gap):
            gap_b.append(gap)

    def ci(vals):
        if not vals:
            return float("nan"), float("nan")
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    out = {grp: {"sens_ci": ci(sens_b[grp]), "spec_ci": ci(spec_b[grp])}
           for grp in GROUP_KEYS}
    out["eo_gap_ci"] = ci(gap_b)
    return out


# =============================================================================
# Threshold fitting (tune half ONLY)
# =============================================================================

def fit_thresholds(y, p, g, tune, tau_target, lam):
    """Fit one per-group threshold config on the tune half.

    For each group: smallest candidate threshold whose tune-half sensitivity
    reaches tau_target (falling back to the most-sensitive candidate).
    With lam > 0, sweep each group's threshold around its target-fit value to
    trade a little sensitivity for a lower tune-half EO gap / threshold drift.
    """
    cand = {}
    for grp in GROUP_KEYS:
        scores = p[(g == grp) & (y == 1) & tune]
        if len(scores) == 0:
            scores = p[(g == grp) & tune]
        qs = np.quantile(scores, np.linspace(0.02, 0.98, 97))
        cand[grp] = np.unique(np.round(qs, 4))

    # start: per-group threshold reaching tau_target. Candidates are ascending,
    # so scan DESCENDING and take the FIRST (i.e. largest) threshold that still
    # meets the target - maximizing specificity at fixed sensitivity. Scanning
    # ascending would pick a near-zero threshold (sens ~1, spec ~0) for every
    # group, a degenerate operating point.
    base = {}
    for grp in GROUP_KEYS:
        m = (g == grp) & (y == 1) & tune
        chosen = float(cand[grp][-1])
        for t in cand[grp][::-1]:
            if float((p[m] >= t).mean()) >= tau_target:
                chosen = float(t)
                break
        base[grp] = chosen

    if lam == 0:
        return base

    # gap+target family: perturb around base, score with EO gap + drift penalty
    tcal = json.loads((s01.CHECKPOINTS / "threshold_config.json").read_text())
    t_global = float(tcal.get("clinical_threshold", 0.25))
    best, best_score = None, np.inf
    for d1 in (-0.05, -0.025, 0.0, 0.025, 0.05):
        for d2 in (-0.05, -0.025, 0.0, 0.025, 0.05):
            for d3 in (-0.05, -0.025, 0.0, 0.025, 0.05):
                cfg = {0: base[0] + d1, 2: base[2] + d2, 4: base[4] + d3}
                flags = np.zeros(len(y), bool)
                for grp in GROUP_KEYS:
                    flags[(g == grp) & tune] = p[(g == grp) & tune] >= cfg[grp]
                gap, _ = eo_gap_from_flags(y, flags, g, tune)
                if not np.isfinite(gap):
                    continue
                drift = float(np.mean([abs(cfg[k] - t_global)
                                       for k in GROUP_KEYS]))
                score = gap + lam * drift
                if score < best_score:
                    best_score, best = score, cfg
    return best


def uniform_threshold(y, p, g, tune, tau_target):
    """Baseline B: one global threshold on the tune half reaching tau_target
    (largest such threshold -> specificity-maximizing; see fit_thresholds)."""
    m = tune & (y == 1)
    scores = p[m]
    qs = np.unique(np.round(np.quantile(scores, np.linspace(0.01, 0.99, 197)), 4))
    for t in qs[::-1]:
        if float((scores >= t).mean()) >= tau_target:
            return float(t)
    return float(qs[0])


# =============================================================================
# Reporting
# =============================================================================

def fmt(x, prec=1, scale=100.0):
    return f"{x * scale:.{prec}f}" if np.isfinite(x) else "n/a"


def evaluate_config(name, thr_by_group, y, p, g, ev, members, seed=SEED):
    """Evaluate a config on the eval half: table rows + bootstrap CIs."""
    flags = np.zeros(len(y), bool)
    for grp in GROUP_KEYS:
        flags[(g == grp) & members] = p[(g == grp) & members] >= thr_by_group[grp]
    m = members & ev
    mm = sens_spec(y, flags, g, m)
    ci = strat_boot_ci(y, flags, g, m, seed=seed)
    gap, _ = eo_gap_from_flags(y, flags, g, m)

    log(f"\n  {name}")
    log(f"  {'group':<12}{'threshold':>10}{'sens [95% CI]':>24}"
        f"{'spec [95% CI]':>24}{'n':>7}{'n_pos':>7}{'n_neg':>7}")
    for grp in GROUP_KEYS + ["overall"]:
        r, c = mm[grp], (ci[grp] if grp in ci else {})
        slo, shi = c.get("sens_ci", (float("nan"),) * 2)
        plo, phi = c.get("spec_ci", (float("nan"),) * 2)
        thr = thr_by_group.get(grp, thr_by_group.get("overall", float("nan")))
        log(f"  {GROUP_NAMES.get(grp, 'ALL'):<12}{thr:>10.4f}"
            f"{fmt(r['sens']):>9} [{fmt(slo)}, {fmt(shi)}]"
            f"{fmt(r['spec']):>14} [{fmt(plo)}, {fmt(phi)}]"
            f"{r['n']:>7,}{r['n_pos']:>7,}{r['n_neg']:>7,}")
    glo, ghi = ci["eo_gap_ci"]
    log(f"  EO gap (sensitivity, max-min):  {fmt(gap, 2)}%  "
        f"[{fmt(glo, 2)}, {fmt(ghi, 2)}]   target < 5%")
    return {"name": name, "thresholds": {str(k): v for k, v in thr_by_group.items()},
            "metrics": mm, "eo_gap": gap,
            "eo_gap_ci": [ci["eo_gap_ci"][0], ci["eo_gap_ci"][1]],
            "ci": {str(k): v for k, v in ci.items()}}


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="DermaLens Step 2: threshold equalization")
    ap.add_argument("--step0-only", action="store_true",
                    help="re-run provenance gates only (no fitting)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    n_boot, seed = args.n_boot, args.seed

    t0 = time.time()
    log("=" * 108)
    log("DermaLens - STEP 2: per-group threshold equalization (DDI panel)")
    log(f"  started: {datetime.now():%Y-%m-%d %H:%M:%S}  |  seed={seed}  "
        f"|  n_boot={n_boot}  |  conf={CONF}")
    log("=" * 108)

    y, p, g, P, drift = load_panel()
    log(f"  S2.0  cache integrity: max|p - P[:, :4].sum| = {drift:.2e} "
        f"(n={len(y):,}, pos={int(y.sum()):,})")

    blockers = [i for i in s01.ISSUES if i["severity"] == "BLOCKER"]
    warns = [i for i in s01.ISSUES if i["severity"] == "WARN"]
    if blockers or warns:
        for i in s01.ISSUES:
            if i["severity"] in ("BLOCKER", "WARN"):
                log(f"    [{i['severity']}] {i['code']}: {i['detail']}")
    log(f"  Provenance gates: {len(blockers)} BLOCKER, {len(warns)} WARN")
    if blockers:
        raise SystemExit("BLOCKER findings - refusing to fit thresholds.")

    if args.step0_only:
        return

    tc = json.loads((s01.CHECKPOINTS / "threshold_config.json").read_text())
    t_shipped = float(tc.get("clinical_threshold", 0.25))

    tune, ev = split_halves(y, g, seed=seed)
    tune_m = np.zeros(len(y), bool); tune_m[tune] = True
    ev_m = np.zeros(len(y), bool); ev_m[ev] = True
    n_t, n_e = int(tune_m.sum()), int(ev_m.sum())
    log(f"\n  S2.1  stratified tune/eval split (group x label, seed={seed}): "
        f"tune n={n_t} (pos={int(y[tune_m].sum())}), "
        f"eval n={n_e} (pos={int(y[ev_m].sum())})")
    log("        thresholds are fitted on TUNE only; eval half is touched once")

    # candidate grid
    log("\n  S2.2  fitting per-group thresholds on the tune half...")
    grid = []
    for tau in (0.95, 0.96, 0.97):
        for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
            cfg = fit_thresholds(y, p, g, tune_m, tau, lam)
            flags = np.zeros(len(y), bool)
            for grp in GROUP_KEYS:
                flags[(g == grp) & tune_m] = p[(g == grp) & tune_m] >= cfg[grp]
            gap, _ = eo_gap_from_flags(y, flags, g, tune_m)
            grid.append({"tau": tau, "lam": lam, "thr": cfg, "tune_gap": gap})
            log(f"    tau>={tau:.2f} lam={lam:<4} -> thr "
                f"{cfg[0]:.3f}/{cfg[2]:.3f}/{cfg[4]:.3f}  "
                f"tune EO gap {fmt(gap, 2)}%")

    # selection: lowest tune-half EO gap; tie-break = thresholds closest to the
    # shipped global calibration (keeps the app's operating point meaningful)
    def sel_key(row):
        drift_m = float(np.mean([abs(row["thr"][k] - t_shipped)
                                 for k in GROUP_KEYS]))
        return (round(row["tune_gap"], 6), round(drift_m, 6))

    best = sorted(grid, key=sel_key)[0]
    log(f"\n  S2.3  selected: tau>={best['tau']:.2f}, lam={best['lam']} "
        f"(tune EO gap {fmt(best['tune_gap'], 2)}%)")

    # ── out-of-sample evaluation on the eval half ──────────────────────────
    log("\n" + "=" * 108)
    log("S2.4  EVAL-HALF RESULTS (out-of-sample; thresholds fitted on tune half only)")
    log("      n / n_pos / n_neg are raw eval-half counts per group.")
    log("=" * 108)

    results = []
    results.append(evaluate_config(
        f"A. baseline - shipped uniform threshold {t_shipped:.4f} (threshold_config.json)",
        {k: t_shipped for k in GROUP_KEYS + ["overall"]},
        y, p, g, ev_m, np.ones(len(y), bool), seed=seed))
    t_uni = uniform_threshold(y, p, g, tune_m, best["tau"])
    results.append(evaluate_config(
        f"B. baseline - uniform threshold {t_uni:.4f} tuned on tune half "
        f"(target sens>={best['tau']:.2f})",
        {k: t_uni for k in GROUP_KEYS + ["overall"]},
        y, p, g, ev_m, np.ones(len(y), bool), seed=seed))
    results.append(evaluate_config(
        f"C. per-group thresholds (tau>={best['tau']:.2f}, lam={best['lam']})",
        {**best["thr"], "overall": t_uni},
        y, p, g, ev_m, np.ones(len(y), bool), seed=seed))

    # ── persist ─────────────────────────────────────────────────────────────
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seed": seed, "n_boot": n_boot, "conf": CONF,
        "cache_prob_drift": drift,
        "step0_findings": s01.ISSUES,
        "split": {"tune_n": n_t, "eval_n": n_e,
                  "tune_pos": int(y[tune_m].sum()), "eval_pos": int(y[ev_m].sum())},
        "shipped_threshold": t_shipped,
        "grid": [{"tau": r["tau"], "lam": r["lam"],
                  "thr": {str(k): v for k, v in r["thr"].items()},
                  "tune_gap": r["tune_gap"]} for r in grid],
        "selected": {"tau": best["tau"], "lam": best["lam"],
                     "thr": {str(k): v for k, v in best["thr"].items()}},
        "eval_results": results,
        "notes": {
            "design": ("thresholds fitted on a (group x label)-stratified tune "
                       "half of DDI and evaluated once on the held-out eval "
                       "half; baseline B (uniform re-tuned threshold) isolates "
                       "the group-specific benefit from plain re-tuning"),
            "n_is_raw": ("n, n_pos, n_neg are raw eval-half counts; bootstrap "
                         "is stratified by group x label so they are exact "
                         "in every replicate"),
            "small_n_warning": ("eval halves hold ~50 positives per group - "
                                "CIs are wide by construction; treat the "
                                "gap-reduction as indicative, not definitive"),
        },
    }
    out_json = AUDIT_DIR / "audit_step2_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"\n  wrote {out_json}")

    # flat CSV: one row per (config, group)
    import csv
    csv_path = AUDIT_DIR / "step2_threshold_equalization.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "group", "threshold", "n", "n_pos", "n_neg",
                    "sensitivity", "sens_ci_low", "sens_ci_high",
                    "specificity", "spec_ci_low", "spec_ci_high"])
        for res in results:
            for grp in GROUP_KEYS + ["overall"]:
                r = res["metrics"][grp]
                c = res["ci"].get(grp, {})
                slo, shi = c.get("sens_ci", (float("nan"),) * 2)
                plo, phi = c.get("spec_ci", (float("nan"),) * 2)
                w.writerow([res["name"], GROUP_NAMES.get(grp, "ALL"),
                            res["thresholds"].get(str(grp), ""),
                            r["n"], r["n_pos"], r["n_neg"],
                            r["sens"], slo, shi, r["spec"], plo, phi])
    log(f"  wrote {csv_path}")

    log("\n" + "=" * 108)
    log(f"STEP 2 COMPLETE in {time.time() - t0:.1f}s")
    log("  The paper should quote config C vs baseline A/B on the eval half,")
    log("  with raw n columns - not the tune-half fit numbers.")
    log("=" * 108)


if __name__ == "__main__":
    main()
