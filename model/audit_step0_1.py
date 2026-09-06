"""
audit_step0_1.py — DermaLens Step 0+1 audit
=============================================

STEP 0 — Non-model audit (no GPU): artifact/code consistency.
  S0.1  Checkpoint -> taxonomy compatibility (state_dict output width vs NUM_CLASSES=8)
  S0.2  Checkpoint freshness vs artifact timestamps
  S0.3  threshold_config.json internal consistency
  S0.4  Notebook code audit: 8-class assumption, stale remap cell,
        MALIGNANT_CLASSES drift, Cell 2 "9 classes" print
  S0.5  Label-consistency spot-check: split CSVs vs master_dataset.csv

STEP 1 — Statistical audit (loads checkpoint; ~1-2 min on RTX 3050): bootstrap CIs.
  S1.1  Test set (ITA-estimated tones), per FST group + overall:
        sensitivity, specificity, accuracy — each row carries raw n, n_pos, n_neg
  S1.2  DDI (expert-labelled tones), per FST bucket + overall:
        sensitivity, specificity — each row carries raw n, n_pos, n_neg
  S1.3  Group-pair sensitivity contrasts + EO-gap CI (test set & DDI)
  S1.4  Binary ROC-AUC CI (test set, DDI)

Method: 2,000 bootstrap resamples per metric, stratified by true binary label
within each group (positives and negatives resampled independently, so the
per-group n / n_pos / n_neg are EXACT in every resample — the printed n is not
an average). Percentile CIs; BCa optional via --bca. Deterministic seed (42).

Key design decision — raw n next to every CI:
  The per-group n (plus n_pos / n_neg) is printed inside the row label of every
  CI table and stored in every CSV/JSON row, so a number pulled into the paper
  cannot be quoted without its n.

Outputs (model/audit/):
  audit_step0_1_report.json   — machine-readable full report
  test_fairness_ci.csv        — S1.1 rows
  ddi_fairness_ci.csv         — S1.2 rows
  group_contrasts_ci.csv      — S1.3 rows (+ EO gap CI per panel)
  preds_cache.npz             — raw predictions (re-run CIs without GPU)

Usage (from model/):
  python audit_step0_1.py              # full run (Step 0 + Step 1)
  python audit_step0_1.py --step0      # Step 0 only, no GPU/model load
  python audit_step0_1.py --skip-inference   # re-run CIs from cached preds
  python audit_step0_1.py --bca        # BCa CIs (slower)

Point estimates use the same formulas as the notebook (tp/(tp+fn+1e-8) etc.),
so Step 1 numbers are directly comparable to Cells 9/9b/9c.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # never open GUI windows

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent           # model/
sys.path.insert(0, str(ROOT))       # so `from model.xxx` works like the notebook

DATASETS = ROOT / "datasets"
CHECKPOINTS = ROOT / "checkpoints"
NB_PATH = ROOT / "notebooks" / "dermalens_train.ipynb"
AUDIT_DIR = ROOT / "audit"

INPUT_SIZE = 224
BATCH_SIZE = 16
SEED = 42
N_BOOT = 2000
CONF = 0.95

# Anchor string the fairness module must contain: the malignant set must come
# from the shared taxonomy, never from a local hardcoded count of classes.
MALIGNANT_ANCHOR = "from model.class_taxonomy import MALIGNANT_CLASSES"

ISSUES: list[dict] = []


def issue(severity: str, code: str, detail: str) -> None:
    """Record a Step 0 finding. severity: BLOCKER | WARN | INFO."""
    ISSUES.append({"severity": severity, "code": code, "detail": detail})


# =============================================================================
# Bootstrap machinery
# =============================================================================

def boot_metric(y, p, g, g0, metric, n_boot=N_BOOT, bca=False, seed=SEED):
    """
    Bootstrap CI for `metric` computed on (y, p) restricted to group g == g0.

    Stratified by true binary label within the group: positives and negatives
    are resampled independently, so n_pos / n_neg are constant across resamples.
      sensitivity: positives only;  specificity: negatives only;
      accuracy / auc: stratified resample of the full group.
    Returns dict with raw n, n_pos, n_neg, point estimate and 95% CI.
    """
    if g0 == "overall":
        sel = np.ones(len(y), dtype=bool)
    else:
        sel = g == g0
    y_g, p_g = y[sel], p[sel]
    pos, neg = y_g == 1, y_g == 0
    n, n_pos, n_neg = int(sel.sum()), int(pos.sum()), int(neg.sum())
    out = {"n": n, "n_pos": n_pos, "n_neg": n_neg}

    # Point estimates — same formulas as the notebook
    if metric == "sensitivity":
        point = float((p_g[pos] >= 0.5).mean()) if n_pos else float("nan")
    elif metric == "specificity":
        point = float((p_g[neg] < 0.5).mean()) if n_neg else float("nan")
    elif metric == "accuracy":
        point = float(((p_g >= 0.5).astype(int) == y_g).mean()) if n else float("nan")
    elif metric == "auc":
        if n_pos and n_neg:
            from sklearn.metrics import roc_auc_score
            point = float(roc_auc_score(y_g, p_g))
        else:
            point = float("nan")
    else:
        raise ValueError(metric)
    out["point"] = point

    # Degenerate cases: CI honestly marked, never fabricated
    if n == 0 or n_pos == 0 or n_neg == 0:
        out.update(ci_low=None, ci_high=None, method="degenerate (empty class) — CI undefined")
        return out
    if n_pos == 1 or n_neg == 1:
        out.update(ci_low=0.0, ci_high=1.0, method="degenerate (n_class=1); CI [0,1] by convention")
        return out

    boot_rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(pos)
    idx_neg = np.flatnonzero(neg)
    draws_pos = boot_rng.choice(idx_pos, size=(n_boot, n_pos), replace=True)
    draws_neg = boot_rng.choice(idx_neg, size=(n_boot, n_neg), replace=True)

    vals = np.empty(n_boot, dtype=float)
    if metric == "sensitivity":
        for b in range(n_boot):
            vals[b] = float((p_g[draws_pos[b]] >= 0.5).mean())
    elif metric == "specificity":
        for b in range(n_boot):
            vals[b] = float((p_g[draws_neg[b]] < 0.5).mean())
    else:  # accuracy / auc — stratified resample of the full group
        for b in range(n_boot):
            yy, pp = y_g[draws_pos[b]], p_g[draws_pos[b]]
            yy2, pp2 = y_g[draws_neg[b]], p_g[draws_neg[b]]
            yb = np.concatenate([yy, yy2])
            pb = np.concatenate([pp, pp2])
            if metric == "accuracy":
                vals[b] = float(((pb >= 0.5).astype(int) == yb).mean())
            else:
                from sklearn.metrics import roc_auc_score
                vals[b] = float(roc_auc_score(yb, pb))

    alpha = (1.0 - CONF) / 2.0
    lo, hi = np.percentile(vals, [100 * alpha, 100 * (1 - alpha)])
    result = {"ci_low": float(lo), "ci_high": float(hi),
              "method": f"percentile bootstrap ({n_boot} reps, stratified by label)"}

    if bca:
        try:
            from scipy.stats import norm
            n_all = n_pos + n_neg
            all_idx = np.concatenate([idx_pos, idx_neg])
            jack = np.empty(n_all)
            for i in range(n_all):
                sub = np.delete(all_idx, i)
                yy, pp = y_g[sub], p_g[sub]
                if metric == "sensitivity":
                    keep = yy == 1
                    jack[i] = (pp[keep] >= 0.5).mean() if keep.any() else np.nan
                elif metric == "specificity":
                    keep = yy == 0
                    jack[i] = (pp[keep] < 0.5).mean() if keep.any() else np.nan
                elif metric == "accuracy":
                    jack[i] = ((pp >= 0.5).astype(int) == yy).mean()
                else:
                    from sklearn.metrics import roc_auc_score
                    jack[i] = roc_auc_score(yy, pp)
            jack = jack[np.isfinite(jack)]
            jmean = jack.mean()
            numer = np.sum((jmean - jack) ** 2)
            denom = np.sum((jmean - jack) ** 3)
            a = denom / (6.0 * numer ** 1.5) if numer > 0 else 0.0
            frac_lt = np.mean(vals < point)
            z0 = norm.ppf(np.clip(frac_lt, 1e-6, 1 - 1e-6))
            za, zb = norm.ppf(alpha), norm.ppf(1 - alpha)
            adj = np.array([
                norm.cdf(z0 + (z0 + za) / (1 - a * (z0 + za))),
                norm.cdf(z0 + (z0 + zb) / (1 - a * (z0 + zb))),
            ])
            adj = np.clip(adj, 0, 1)
            lo2, hi2 = np.percentile(vals, 100 * adj)
            result.update(ci_low=float(lo2), ci_high=float(hi2),
                          method=f"BCa bootstrap ({n_boot} reps, stratified by label)")
        except Exception as e:
            result["method"] += f" (BCa unavailable: {e})"

    out.update(result)
    return out


def gap_bootstrap(y, p, g, key_a, key_b, groups_present=None,
                  n_boot=N_BOOT, seed=SEED):
    """
    Sensitivity contrast (key_a - key_b) via stratified percentile bootstrap,
    plus the equalized-odds-gap replication distribution (max-min sensitivity
    across the groups present in `g` — same definition as the notebook's
    equalized_odds_gap on hard predictions). Positives-only resampling.
    """
    boot_rng = np.random.default_rng(seed)
    if groups_present is None:
        groups_present = sorted({int(x) for x in np.unique(g) if x >= 0})
    pos_sets = {k: np.flatnonzero((g == k) & (y == 1)) for k in groups_present}
    pos_sets = {k: v for k, v in pos_sets.items() if len(v) > 0}
    groups_present = sorted(pos_sets.keys())

    sens_draws = {k: np.empty(n_boot) for k in groups_present}
    for k in groups_present:
        dp = boot_rng.choice(pos_sets[k], size=(n_boot, len(pos_sets[k])), replace=True)
        for b in range(n_boot):
            sens_draws[k][b] = float((p[dp[b]] >= 0.5).mean())

    sens_point = {k: float((p[pos_sets[k]] >= 0.5).mean()) for k in groups_present}

    contrast = sens_draws[key_a] - sens_draws[key_b]
    lo, hi = np.percentile(contrast, [2.5, 97.5])
    gap_reps = np.vstack([sens_draws[k] for k in groups_present])
    gap_vals = gap_reps.max(axis=0) - gap_reps.min(axis=0)
    glo, ghi = np.percentile(gap_vals, [2.5, 97.5])
    gap_point = max(sens_point.values()) - min(sens_point.values())

    return {
        "contrast": f"{key_a} - {key_b}",
        "sens_a": sens_point[key_a], "sens_b": sens_point[key_b],
        "diff": sens_point[key_a] - sens_point[key_b],
        "ci_low": float(lo), "ci_high": float(hi),
        "n_a": int(len(pos_sets[key_a])), "n_b": int(len(pos_sets[key_b])),
        "eo_gap_point": float(gap_point),
        "eo_gap_ci_low": float(glo), "eo_gap_ci_high": float(ghi),
        "eo_groups": groups_present,
    }


# =============================================================================
# Model / data loading (Step 1)
# =============================================================================

def load_model_and_preds():
    """
    Loads the checkpoint, runs inference with the notebook's exact val
    transform chain, returns y (binary true), p (malignant prob), g (tone idx)
    for the test panel and the DDI panel.
    """
    import torch
    import torch.nn.functional as F
    import cv2
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    from model.fairdermet import FairDermNet
    from model.class_taxonomy import MALIGNANT_CLASSES, class_idx_to_binary
    from model import class_taxonomy as tax

    model = FairDermNet.from_checkpoint(
        CHECKPOINTS / "best.pth",
        num_classes=tax.NUM_CLASSES,          # 8 — GEMINI.md taxonomy lock
        num_skin_tones=6,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = A.Compose([
        A.Resize(height=INPUT_SIZE, width=INPUT_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    def infer(df: pd.DataFrame):
        """Notebook-identical loading (cv2.imread -> BGR2RGB -> val transform),
        malignant prob = sum of softmax over MALIGNANT_CLASSES (Cells 9b/9c).
        Also returns the full [N, num_classes] softmax matrix so downstream
        steps (report regeneration, threshold equalization) can re-derive any
        operating characteristic without a second GPU pass."""
        malignant_idx = sorted(MALIGNANT_CLASSES)
        p_list, n_missing, P_list = [], 0, []
        with torch.no_grad():
            for start in range(0, len(df), BATCH_SIZE):
                chunk = df.iloc[start:start + BATCH_SIZE]
                imgs = []
                for _, row in chunk.iterrows():
                    img = cv2.imread(str(row["image_path"]))
                    if img is None:
                        # notebook's SkinLesionDataset substitutes a zero image;
                        # mirror it, but count it so the report can flag it
                        img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
                        n_missing += 1
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imgs.append(transform(image=img_rgb)["image"])
                batch = torch.stack(imgs).to(device)
                logits = model(batch)
                probs = F.softmax(logits, dim=-1)
                mal_prob = probs[:, malignant_idx].sum(dim=-1).cpu().numpy()
                p_list.extend(mal_prob.tolist())
                P_list.append(probs.cpu().numpy().astype(np.float64))
        P = np.concatenate(P_list, axis=0) if P_list else np.zeros((0, 8))
        return np.asarray(p_list, dtype=float), n_missing, P

    # Panel A: test set (ITA-estimated tones)
    test_df = pd.read_csv(DATASETS / "split_test.csv")
    p_test, miss_test, P_test = infer(test_df)
    y_test = test_df["class_idx"].map(class_idx_to_binary).to_numpy().astype(int)
    g_test = test_df["skin_tone"].fillna(-1).astype(int).to_numpy()

    # Panel B: DDI (expert-labelled tones, coarse buckets 0/2/4)
    ddi_df = pd.read_csv(DATASETS / "ddi_eval.csv")
    p_ddi, miss_ddi, P_ddi = infer(ddi_df)
    y_ddi = ddi_df["ddi_binary"].to_numpy().astype(int)
    g_ddi = ddi_df["fitz_group"].map({"I-II": 0, "III-IV": 2, "V-VI": 4}).to_numpy()

    return {
        "test": {"y": y_test, "p": p_test, "g": g_test, "P": P_test, "n_missing": miss_test},
        "ddi": {"y": y_ddi, "p": p_ddi, "g": g_ddi, "P": P_ddi, "n_missing": miss_ddi},
    }


# =============================================================================
# Step 0 checks
# =============================================================================

def s0_5_label_consistency():
    master = pd.read_csv(DATASETS / "master_dataset.csv")
    m = master.set_index("image_path")
    for split in ["train", "val", "test"]:
        f = DATASETS / f"split_{split}.csv"
        if not f.exists():
            issue("BLOCKER", "S0.5-missing-split", f"{f.name} not found")
            continue
        s = pd.read_csv(f)
        merged = s.join(m[["class_idx", "skin_tone"]], on="image_path", rsuffix="_master")
        cmis = int((merged["class_idx"] != merged["class_idx_master"]).sum())
        tmis = int((merged["skin_tone"] != merged["skin_tone_master"]).sum())
        orphans = int((~s["image_path"].isin(m.index)).sum())
        if cmis or tmis or orphans:
            issue("BLOCKER", "S0.5-label-mismatch",
                  f"{f.name}: {cmis} class mismatches, {tmis} tone mismatches, "
                  f"{orphans} rows not in master_dataset.csv")
        else:
            print(f"  S0.5 OK  {f.name}: labels/tones match master_dataset.csv ({len(s):,} rows)")


def run_step0():
    # S0.1 checkpoint <-> taxonomy
    try:
        from model import class_taxonomy as tax
    except Exception as e:
        issue("BLOCKER", "S0.0-import", f"Failed to import project taxonomy: {e}")
        return

    ckpt = CHECKPOINTS / "best.pth"
    if not ckpt.exists():
        issue("BLOCKER", "S0.1-no-checkpoint", f"{ckpt} missing")
        return
    import torch
    state = torch.load(ckpt, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    final_key = next((k for k in state.keys()
                      if k.startswith("risk_head") and k.endswith("weight")
                      and state[k].ndim == 2 and state[k].shape[0] in (8, 9)),
                     None)
    if final_key is None:
        issue("WARN", "S0.1-head-shape",
              "could not locate final risk-head weight; head keys: "
              + str([k for k in state if k.startswith('risk_head')]))
    else:
        ckpt_classes = int(state[final_key].shape[0])
        if ckpt_classes != tax.NUM_CLASSES:
            issue("BLOCKER", "S0.1-class-mismatch",
                  f"checkpoint head ({final_key}) outputs {ckpt_classes} classes, "
                  f"taxonomy NUM_CLASSES={tax.NUM_CLASSES}")
        else:
            print(f"  S0.1 OK  checkpoint risk head outputs {ckpt_classes} classes "
                  f"== taxonomy NUM_CLASSES ({final_key})")
        if ckpt_classes == 9:
            issue("BLOCKER", "S0.1-legacy-9-class",
                  "checkpoint is a legacy 9-class model - retrain or remap before "
                  "quoting any number in the paper")

    # S0.2 artifact freshness
    art = {
        "best.pth": ckpt,
        "threshold_config.json": CHECKPOINTS / "threshold_config.json",
        "test_fairness_report.txt": CHECKPOINTS / "test_fairness_report.txt",
        "training_history.json": CHECKPOINTS / "training_history.json",
        "split_train.csv": DATASETS / "split_train.csv",
        "split_test.csv": DATASETS / "split_test.csv",
    }
    mt = {k: v.stat().st_mtime for k, v in art.items() if v.exists()}
    for k, v in art.items():
        if not v.exists():
            issue("WARN", "S0.2-missing-artifact", f"{k} not found")
    if "best.pth" in mt:
        older = [k for k, t in mt.items() if k != "best.pth" and t < mt["best.pth"]]
        if older:
            issue("WARN", "S0.2-stale-artifacts",
                  f"artifacts older than best.pth (may reflect a superseded run): "
                  f"{', '.join(older)}")
        print("  S0.2 artifact mtimes: "
              + ", ".join(f"{k}={datetime.fromtimestamp(t):%Y-%m-%d %H:%M}"
                          for k, t in sorted(mt.items())))
    # Known stale-run hazards: scratch logs that predate the data they describe.
    # files_claude/run_9b_9c.log (Aug 15) was retired as SUPERSEDED_2026-08-15
    # on Sep 6: its DDI numbers (AUC 0.6398) described a run predating the
    # Sep 5 ddi_eval.csv regeneration and were never reproducible. If a file
    # reappears at the old path, warn again.
    stale_log = ROOT / "files_claude" / "run_9b_9c.log"
    if stale_log.exists():
        issue("WARN", "S0.2-stale-ddi-log",
              "files_claude/run_9b_9c.log has reappeared - if it is the Aug 15 "
              "scratch log, its DDI numbers (e.g. AUC 0.6398) are SUPERSEDED and "
              "must not be quoted; current-state DDI metrics come from this "
              "audit's Step 1.")

    # S0.3 threshold_config internal consistency
    tc_path = CHECKPOINTS / "threshold_config.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        sens, thr = tc.get("sensitivity_at_clinical"), tc.get("clinical_threshold")
        if sens is not None and sens < 0.96:
            issue("WARN", "S0.3-sens-below-target",
                  f"threshold_config sensitivity_at_clinical={sens} < 0.96 despite "
                  f"method='{tc.get('method')}'")
        print(f"  S0.3     threshold_config: clinical_threshold={thr}, "
              f"sens={sens}, spec={tc.get('specificity_at_clinical')}, "
              f"auc={tc.get('roc_auc')}")

    # S0.4 notebook + project-module audit
    if not NB_PATH.exists():
        issue("BLOCKER", "S0.4-no-notebook", f"{NB_PATH} missing")
        return
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    whole = "\n".join(srcs)

    # The notebook imports compute_epoch_metrics from training/fairness_metrics;
    # drift there is invisible in the notebook source, so scan the module too.
    fm_path = ROOT / "training" / "fairness_metrics.py"
    fm_src = fm_path.read_text(encoding="utf-8") if fm_path.exists() else ""
    if not fm_src:
        issue("WARN", "S0.4-missing-fairness-module", f"{fm_path} missing/unreadable")

    _remap_flagged = False
    if "REMAP = {4: 5" in whole:
        _remap_flagged = True
        issue("BLOCKER", "S0.4-stale-remap",
              'notebook still contains "REMAP = {4: 5, 5: 6, 6: 7, 7: 8}" - re-running '
              "the remap cell on the already-remapped master_dataset.csv shifts "
              "NV->BKL->DF->VASC and silently corrupts labels. Delete/guard that cell "
              "before any Kernel > Restart & Run All.")
    if "Classes:              9" in whole:
        issue("WARN", "S0.4-cell2-9-class-print",
              'Cell 2 still prints "Classes: 9 (taxonomy loaded in Cell 3)" while the '
              "pipeline is 8-class")
    if "malignant_classes = {0, 1, 2}" in fm_src:
        issue("BLOCKER", "S0.4-3-malignant",
              f"{fm_path.name}: compute_epoch_metrics uses malignant_classes={{0,1,2}} "
              "- AK (class 3) is excluded from the per-epoch fairness signal; "
              "GEMINI.md locks malignant = {0,1,2,3}")
    if "all_preds[:, :3].sum(dim=1" in fm_src:
        issue("BLOCKER", "S0.4-3-class-prob",
              f"{fm_path.name}: compute_epoch_metrics derives malignant prob as "
              "all_preds[:, :3].sum - hardcodes 3 malignant classes; must sum over "
              "MALIGNANT_CLASSES (0..3)")

    # Guard against regression of the Sep 6 fixes. Kept permanently so a
    # future edit cannot silently reintroduce a 3-class malignant assumption.
    if MALIGNANT_ANCHOR not in fm_src:
        issue("BLOCKER", "S0.4-malignant-anchor-missing",
              f"{fm_path.name}: malignant set is no longer anchored to "
              "model.class_taxonomy.MALIGNANT_CLASSES - re-add the shared "
              "import so per-epoch and post-hoc fairness metrics stay in sync "
              "with GEMINI.md")
    if "Class distribution (verified 8-class):" in whole and not _remap_flagged:
        issue("INFO", "S0.4-remap-guard-present",
              "notebook remap cell is retired: legacy-indices guard in place "
              "(raises RuntimeError if master_dataset.csv is back in the "
              "pre-fix 9-class layout)")

    # S0.5
    s0_5_label_consistency()


# =============================================================================
# Step 1 reporting — raw n next to every CI
# =============================================================================

def fmt_ci(res, scale=100.0, prec=1, suffix="%"):
    if res.get("ci_low") is None:
        return f"{res['point'] * scale:.{prec}f}{suffix} [CI n/a]"
    return (f"{res['point'] * scale:.{prec}f}{suffix} "
            f"[{res['ci_low'] * scale:.{prec}f}, {res['ci_high'] * scale:.{prec}f}]")


def run_step1(panels, bca=False):
    rows_out = {"test": [], "ddi": []}
    contrast_rows = {"test": [], "ddi": []}
    s1_out = {"test": {}, "ddi": {}}

    for panel in ("test", "ddi"):
        y, p, g = panels[panel]["y"], panels[panel]["p"], panels[panel]["g"]
        n_miss = panels[panel].get("n_missing", 0)
        if n_miss:
            issue("WARN", f"S1-{panel}-missing-images",
                  f"{n_miss} images failed to load (zero-image substituted, exactly "
                  "as the notebook does); their rows stay inside the CIs.")

        if panel == "test":
            group_defs = [(0, "FST I"), (1, "FST II"), (2, "FST III"),
                          (3, "FST IV"), (4, "FST V"), (5, "FST VI"),
                          ("overall", "ALL (ITA-estimated tones)")]
        else:
            group_defs = [(0, "FST I-II"), (2, "FST III-IV"), (4, "FST V-VI"),
                          ("overall", "ALL (expert-labelled tones)")]

        n_tot = len(y)
        title = (f"S1.1  TEST SET  |  total n={n_tot:,}  |  ITA-estimated tones"
                 if panel == "test" else
                 f"S1.2  DDI HELD-OUT  |  total n={n_tot:,}  |  expert-labelled tones")
        print("\n" + "=" * 108)
        print(title)
        print("n / n_pos / n_neg are RAW per-group counts - quote them beside every CI.")
        print("=" * 108)
        header = (f"{'group':<44} | {'sensitivity [95% CI]':<28}"
                  f" | {'specificity [95% CI]':<28} | {'accuracy [95% CI]':<28}")
        print(header)
        print("-" * len(header))

        for g0, label in group_defs:
            sens = boot_metric(y, p, g, g0, "sensitivity", bca=bca)
            spec = boot_metric(y, p, g, g0, "specificity", bca=bca)
            acc = boot_metric(y, p, g, g0, "accuracy", bca=bca)
            label_n = (f"{label}  (n={sens['n']:,} | n_pos={sens['n_pos']:,} "
                       f"| n_neg={sens['n_neg']:,})")
            print(f"{label_n:<44} | {fmt_ci(sens):<28} | {fmt_ci(spec):<28} "
                  f"| {fmt_ci(acc):<28}")
            rows_out[panel].append({
                "panel": panel, "group": label, "group_key": str(g0),
                "n": sens["n"], "n_pos": sens["n_pos"], "n_neg": sens["n_neg"],
                "sensitivity": sens["point"],
                "sens_ci_low": sens["ci_low"], "sens_ci_high": sens["ci_high"],
                "specificity": spec["point"],
                "spec_ci_low": spec["ci_low"], "spec_ci_high": spec["ci_high"],
                "accuracy": acc["point"],
                "acc_ci_low": acc["ci_low"], "acc_ci_high": acc["ci_high"],
                "ci_method": sens.get("method", ""),
            })

        # Binary ROC-AUC with CI (overall)
        auc_res = boot_metric(y, p, g, "overall", "auc", bca=bca)
        s1_out[panel]["auc"] = auc_res
        print(f"\n  Binary ROC-AUC  (n={auc_res['n']:,} | n_pos={auc_res['n_pos']:,} "
              f"| n_neg={auc_res['n_neg']:,})   {fmt_ci(auc_res, scale=1, prec=4, suffix='')}")

        # Sensitivity contrasts + EO gap CI
        keys = [k for k, _ in group_defs if k != "overall"]
        pos_by_group = {k: int(((g == k) & (y == 1)).sum()) for k in keys}
        usable = sorted(k for k, npos in pos_by_group.items() if npos > 0)

        print("\n  Sensitivity contrasts (95% CI, stratified percentile bootstrap; "
              "n_pos = raw malignant counts per group):")
        print(f"  {'contrast':<28}{'diff':>9}{'95% CI':>22}{'n_pos_a':>9}{'n_pos_b':>9}")
        pairs = []
        if panel == "ddi":
            pairs = [(4, 0), (4, 2), (2, 0)]   # V-VI vs I-II, V-VI vs III-IV, III-IV vs I-II
        elif len(usable) >= 2:
            ref = usable[0]
            pairs = [(k, ref) for k in usable[1:]]
        ddi_names = {0: "FST I-II", 2: "FST III-IV", 4: "FST V-VI"}
        for a, b in pairs:
            if a not in usable or b not in usable:
                print(f"  {a} vs {b}: skipped (no malignant samples in one group)")
                continue
            c = gap_bootstrap(y, p, g, a, b, groups_present=usable)
            la = ddi_names.get(a, f"FST {a + 1}") if panel == "ddi" else f"FST {a + 1}"
            lb = ddi_names.get(b, f"FST {b + 1}") if panel == "ddi" else f"FST {b + 1}"
            print(f"  {la + ' - ' + lb:<28}{c['diff'] * 100:>8.1f}% "
                  f"[{c['ci_low'] * 100:>6.1f}, {c['ci_high'] * 100:>6.1f}]"
                  f"{c['n_a']:>9,}{c['n_b']:>9,}")
            c = {"panel": panel, **c}
            contrast_rows[panel].append(c)

        if contrast_rows[panel]:
            c0 = contrast_rows[panel][0]
            print(f"\n  Equalized-odds gap (sensitivity, max-min over groups "
                  f"{c0['eo_groups']}):  {c0['eo_gap_point'] * 100:.2f}%  "
                  f"[{c0['eo_gap_ci_low'] * 100:.2f}, {c0['eo_gap_ci_high'] * 100:.2f}]"
                  f"   target < 5%")
            s1_out[panel]["eo_gap"] = {
                "point": c0["eo_gap_point"],
                "ci_low": c0["eo_gap_ci_low"],
                "ci_high": c0["eo_gap_ci_high"],
            }

    return rows_out, contrast_rows, s1_out


# =============================================================================
# Outputs
# =============================================================================

def write_outputs(rows, contrasts, s1):
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "n_boot": N_BOOT,
        "conf": CONF,
        "step0_findings": ISSUES,
        "step1": {
            "test": {
                "groups": rows.get("test", []),
                "contrasts": contrasts.get("test", []),
                "auc": s1.get("test", {}).get("auc", {}),
                "eo_gap": s1.get("test", {}).get("eo_gap", {}),
            },
            "ddi": {
                "groups": rows.get("ddi", []),
                "contrasts": contrasts.get("ddi", []),
                "auc": s1.get("ddi", {}).get("auc", {}),
                "eo_gap": s1.get("ddi", {}).get("eo_gap", {}),
            },
        },
        "notes": {
            "tone_provenance": ("test-panel tones are ITA-estimated (not "
                                "expert-labelled); DDI tones are expert-labelled "
                                "but coarse (I-II / III-IV / V-VI)"),
            "n_is_raw": ("n, n_pos, n_neg are raw per-group counts; the bootstrap is "
                         "stratified by true label, so they are exact in every "
                         "resample"),
            "point_estimate_parity": ("same formulas as notebook Cells 9/9b/9c "
                                      "(tp/(tp+fn+1e-8) etc.)"),
        },
    }
    path = AUDIT_DIR / "audit_step0_1_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {path}")

    def _csv(name, header, data_rows):
        with open(AUDIT_DIR / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(data_rows)
        print(f"  wrote {AUDIT_DIR / name}")

    header = ["panel", "group", "group_key", "n", "n_pos", "n_neg",
              "sensitivity", "sens_ci_low", "sens_ci_high",
              "specificity", "spec_ci_low", "spec_ci_high",
              "accuracy", "acc_ci_low", "acc_ci_high", "ci_method"]
    for panel in ("test", "ddi"):
        data = [[r["panel"], r["group"], r["group_key"], r["n"], r["n_pos"], r["n_neg"],
                 r["sensitivity"], r["sens_ci_low"], r["sens_ci_high"],
                 r["specificity"], r["spec_ci_low"], r["spec_ci_high"],
                 r["accuracy"], r["acc_ci_low"], r["acc_ci_high"], r["ci_method"]]
                for r in rows.get(panel, [])]
        _csv("test_fairness_ci.csv" if panel == "test" else "ddi_fairness_ci.csv",
             header, data)

    header_c = ["panel", "contrast", "sens_a", "sens_b", "diff", "ci_low", "ci_high",
                "n_a", "n_b", "eo_gap_point", "eo_gap_ci_low", "eo_gap_ci_high"]
    data_c = [[c["panel"], c["contrast"], c["sens_a"], c["sens_b"], c["diff"],
               c["ci_low"], c["ci_high"], c["n_a"], c["n_b"],
               c["eo_gap_point"], c["eo_gap_ci_low"], c["eo_gap_ci_high"]]
              for panel in ("test", "ddi") for c in contrasts.get(panel, [])]
    _csv("group_contrasts_ci.csv", header_c, data_c)


# =============================================================================
# Main
# =============================================================================

def main():
    global N_BOOT, SEED
    t_start = time.time()
    parser = argparse.ArgumentParser(description="DermaLens Step 0+1 audit")
    parser.add_argument("--step0", action="store_true", help="Step 0 only (no GPU)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Reuse cached predictions in audit/preds_cache.npz")
    parser.add_argument("--bca", action="store_true", help="BCa CIs (slower)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    N_BOOT, SEED = args.n_boot, args.seed

    AUDIT_DIR.mkdir(exist_ok=True)

    print("=" * 108)
    print("DermaLens — STEP 0+1 AUDIT")
    print(f"  started: {datetime.now():%Y-%m-%d %H:%M:%S}  |  seed={SEED}  "
          f"|  n_boot={N_BOOT}  |  conf={CONF}")
    print("=" * 108)

    # ── STEP 0 ───────────────────────────────────────────────────────────────
    print("\nSTEP 0 — artifact / code audit (no model load)")
    print("-" * 108)
    run_step0()
    blockers = [i for i in ISSUES if i["severity"] == "BLOCKER"]
    warns = [i for i in ISSUES if i["severity"] == "WARN"]
    infos = [i for i in ISSUES if i["severity"] == "INFO"]

    if ISSUES:
        print("\n  Step 0 findings:")
        for sev, marker in (("BLOCKER", "!!"), ("WARN", " ~"), ("INFO", " .")):
            for i in ISSUES:
                if i["severity"] == sev:
                    print(f"    [{marker}] {i['code']}: {i['detail']}")
    print(f"\n  Step 0 summary: {len(blockers)} BLOCKER, {len(warns)} WARN, {len(infos)} INFO")

    if args.step0:
        write_outputs({}, {}, {})
        print(f"\nStep 0-only mode: report written to {AUDIT_DIR}/audit_step0_1_report.json")
        return

    # ── STEP 1 ───────────────────────────────────────────────────────────────
    print("\nSTEP 1 — bootstrap confidence intervals (checkpoint inference)")
    print("-" * 108)

    cache = AUDIT_DIR / "preds_cache.npz"
    if args.skip_inference and cache.exists():
        z = np.load(cache)
        panels = {
            "test": {"y": z["y_test"], "p": z["p_test"], "g": z["g_test"], "n_missing": 0},
            "ddi": {"y": z["y_ddi"], "p": z["p_ddi"], "g": z["g_ddi"], "n_missing": 0},
        }
        if "P_test" in z.files:
            panels["test"]["P"] = z["P_test"]
            panels["ddi"]["P"] = z["P_ddi"]
        else:
            issue("WARN", "S1-cache-without-probs",
                  "preds_cache.npz predates 8-class prob caching - run once without "
                  "--skip-inference to enable downstream steps (report regeneration, "
                  "threshold equalization)")
        print("  (using cached predictions)")
    else:
        panels = load_model_and_preds()
        np.savez_compressed(
            cache,
            y_test=panels["test"]["y"], p_test=panels["test"]["p"], g_test=panels["test"]["g"],
            P_test=panels["test"]["P"],
            y_ddi=panels["ddi"]["y"], p_ddi=panels["ddi"]["p"], g_ddi=panels["ddi"]["g"],
            P_ddi=panels["ddi"]["P"],
        )
        print(f"  cached predictions -> {cache}")

    rows, contrasts, s1 = run_step1(panels, bca=args.bca)
    write_outputs(rows, contrasts, s1)

    print("\n" + "=" * 108)
    print(f"AUDIT COMPLETE in {time.time() - t_start:.1f}s")
    print(f"  Outputs in {AUDIT_DIR}/: audit_step0_1_report.json, "
          f"test_fairness_ci.csv, ddi_fairness_ci.csv, group_contrasts_ci.csv")
    print(f"  Findings: {len(blockers)} BLOCKER / {len(warns)} WARN — "
          f"resolve before quoting any number in the paper")
    print("=" * 108)


if __name__ == "__main__":
    main()
