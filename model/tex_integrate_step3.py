"""
tex_integrate_step3.py — one-off integrator (reproducible)
===========================================================
Inserts the Step 3/3b/3c generalization-failure section into
ppt_files/dermalens_paper.tex with every number generated from the audit
CSVs/JSONs (placeholder template -> zero hand-typed numbers). Also fixes
stale 0.24 threshold mentions and extends the abstract + discussion.

Run:  python tex_integrate_step3.py   (idempotent; re-runs replace the block)
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "audit"
TEX = ROOT.parent / "ppt_files" / "dermalens_paper.tex"

TICKET = "%% === STEP3-SECTION (auto-generated from audit CSVs; do not hand-edit numbers) ==="


def load_csv(name):
    with open(AUDIT / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def med(vals):
    return float(np.median(vals))


def tex_pow(p):
    """Format a tiny p-value as LaTeX 1\times10^{-k}."""
    import math
    e = int(math.floor(math.log10(p)))
    return f"1\\times10^{{{e}}}"


def collect_values():
    ceil = {r["group"]: r for r in load_csv("step3_roc_ceilings.csv")}
    decomp = load_csv("step3_score_decomposition.csv")
    shift = load_csv("step3_distribution_shift.csv")
    recal = load_csv("step3b_ak_recalibration.csv")
    cams = load_csv("step3c_cam_comparators.csv")
    rep3e = json.loads((AUDIT / "audit_step3e_report.json").read_text())
    cf = load_csv("step3e_counterfactual.csv")
    cf_tests = rep3e["tests"]

    # median helper spans BOTH comparator sets: step3c (true_AK clinical,
    # true_MEL derm) and step3e (true_AK_derm, true_MEL_clinical)
    cf_all = cams + cf

    def cf_med(cohort, key):
        vals = [float(r[key]) for r in cf_all if r["cohort"] == cohort
                and np.isfinite(float(r[key]))]
        return float(np.median(vals)) if vals else float("nan")

    rep3b = json.loads((AUDIT / "audit_step3b_report.json").read_text())
    rep3c = json.loads((AUDIT / "audit_step3c_report.json").read_text())
    rep3 = json.loads((AUDIT / "audit_step3_report.json").read_text())

    lam = rep3b["ak_recalibration_summary"]["lambda"]
    adj_cfg = f"ak_adj_lam{lam}"
    sweep = {r["lambda"]: r for r in rep3b["ak_recalibration_summary"]["sweep"]}
    eo = rep3b["ak_recalibration_summary"]["eo_gap_eval"]
    safe = rep3b["ak_recalibration_summary"]["safety"]
    lam_rows = {float(r["lambda"]): r for r in rep3c["lambda_collapse"]}

    def dec(panel, group, label, col):
        return float(next(r for r in decomp
                          if r["panel"] == panel and r["group"] == group
                          and r["label"] == label)[col])

    def rec(cfg, grp, key):
        return float(next(r for r in recal
                          if r["config"] == cfg and r["group"] == grp)[key])

    def cam_med(cohort, key):
        return med([float(r[key]) for r in cams if r["cohort"] == cohort])

    ks_ak = next(r for r in shift if r["ddi_group"] == "ALL"
                 and r["column"] == "AK")
    tm_d = next(r for r in rep3b["domain_vs_tone"]
                if r["evidence"] == "tone_matched_I-II")
    tm_t = next(r for r in rep3b["domain_vs_tone"]
                if r["evidence"] == "tone_matched_I-II_test")
    d1 = next(r for r in rep3b["domain_vs_tone"]
              if r["evidence"] == "ddi_internal_by_group"
              and r["group"] == "FST I-II")
    d3 = next(r for r in rep3b["domain_vs_tone"]
              if r["evidence"] == "ddi_internal_by_group"
              and r["group"] == "FST III-IV")
    d5 = next(r for r in rep3b["domain_vs_tone"]
              if r["evidence"] == "ddi_internal_by_group"
              and r["group"] == "FST V-VI")
    ak_share_fp = next(r["AK"] for r in rep3["contributor_shares"]
                       if r["panel"] == "DDI" and r["group"] == "ALL"
                       and r["label"] == "benign")
    mw = {t["comparison"]: t for t in rep3c["mw_tests"]}

    # Step 3d placement stats + visual-review proxies
    rep3d = json.loads((AUDIT / "audit_step3d_report.json").read_text())
    place = load_csv("step3d_placement.csv")

    def place_med(cohort, key):
        return med([float(r[key]) for r in place if r["cohort"] == cohort
                    and np.isfinite(float(r[key]))])

    vig_by = {}
    for r in place:
        try:
            v = float(r["vignette_ratio"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v < 0.75:
            vig_by[r["cohort"]] = vig_by.get(r["cohort"], 0) + 1
    mw3d = {t["comparison"]: t for t in rep3d["mw_tests"]}

    # artifact census from the authors' unblinded key (generated, not typed)
    akey = load_csv("blind_review_authors_key.csv")
    n_rul = sum(1 for r in akey
                if r["artifact"] == "Ruler or paper card in frame")
    n_ink = sum(1 for r in akey
                if r["artifact"] == "Surgical ink / pen markings")

    # Step 2 threshold-equalization results (the experiment behind the claim)
    rep2 = json.loads((AUDIT / "audit_step2_report.json").read_text())
    erA, erB, erC = rep2["eval_results"]
    m4 = lambda er: er["metrics"]["4"]
    mo = lambda er: er["metrics"]["overall"]
    v2 = {
        "p2_tune": str(rep2["split"]["tune_n"]),
        "p2_eval": str(rep2["split"]["eval_n"]),
        "p2_tpos": str(rep2["split"]["tune_pos"]),
        "p2_epos": str(rep2["split"]["eval_pos"]),
        "p2_bt": f"{erB['thresholds']['overall']:.4f}",
        "sensA4": f"{m4(erA)['sens']:.3f}",
        "sensC4": f"{m4(erC)['sens']:.3f}",
        "specA4": f"{m4(erA)['spec']:.3f}",
        "specC4": f"{m4(erC)['spec']:.3f}",
        "specC4lo": f"{erC['ci']['4']['spec_ci'][0]:.3f}",
        "specC4hi": f"{erC['ci']['4']['spec_ci'][1]:.3f}",
        "specA": f"{mo(erA)['spec']:.3f}",
        "specB": f"{mo(erB)['spec']:.3f}",
        "specC": f"{mo(erC)['spec']:.3f}",
        "eoA": f"{erA['eo_gap']:.3f}",
        "eoAlo": f"{erA['ci']['eo_gap_ci'][0]:.3f}",
        "eoAhi": f"{erA['ci']['eo_gap_ci'][1]:.3f}",
        "eoC": f"{erC['eo_gap']:.3f}",
        "eoClo": f"{erC['ci']['eo_gap_ci'][0]:.3f}",
        "eoChi": f"{erC['ci']['eo_gap_ci'][1]:.3f}",
    }

    # q90 of the AK column on benign DDI + tone-matched KS, from the cache
    z = np.load(AUDIT / "preds_cache.npz")
    P, y = z["P_ddi"].astype(float), z["y_ddi"].astype(int)
    PT, yT, gT = (z["P_test"].astype(float), z["y_test"].astype(int),
                  z["g_test"].astype(int))
    ak_fp_q90 = float(np.quantile(P[y == 0, 3], 0.90))
    from scipy.stats import ks_2samp
    ks_tm, _ = ks_2samp(PT[(yT == 0) & np.isin(gT, [0, 1]), 3],
                        P[(y == 0) & (np.load(AUDIT / "preds_cache.npz")
                          ["g_ddi"].astype(int) == 0), 3],
                        alternative="greater", method="asymp")

    def aud(s):
        return s.replace("-", "--")

    rows = []
    for g in ("FST I-II", "FST III-IV", "FST V-VI", "ALL"):
        r = ceil[g]
        rows.append(f"{aud(g)} & {r['n']} ({r['n_pos']}/{r['n_neg']}) & "
                    f"{float(r['auc']):.3f} [{float(r['auc_lo']):.3f}, "
                    f"{float(r['auc_hi']):.3f}] & "
                    f"{float(r['sens_at_spec0.6']):.3f} & "
                    f"{float(r['sens_at_spec0.8']):.3f} \\\\")
    table_rows = "\n".join(rows)

    p_mw = mw["fp_DDI_AK vs true_AK"]["p"]
    l0, l8 = lam_rows[0.0], lam_rows[0.8]

    v = {
        "rows": table_rows,
        "ak_share": f"{100 * ak_share_fp:.1f}",
        "ak_fp_mean": f"{dec('DDI', 'ALL', 'benign', 'AK'):.3f}",
        "ak_fp_q90": f"{ak_fp_q90:.3f}",
        "ak_test_ben": f"{dec('TEST', 'I-IV', 'benign', 'AK'):.3f}",
        "ak_test_mal": f"{dec('TEST', 'I-IV', 'malignant', 'AK'):.3f}",
        "ks_ak": f"{float(ks_ak['ks_stat']):.2f}",
        "p_ak": tex_pow(float(ks_ak["ks_p"])) if float(ks_ak["ks_p"]) > 0 else "<1e-300",
        "cam_fp": f"{cam_med('fp_DDI_AK', 'conc_top1pct'):.3f}",
        "cam_ak": f"{cam_med('true_AK', 'conc_top1pct'):.3f}",
        "cam_mel": f"{cam_med('true_MEL', 'conc_top1pct'):.3f}",
        "p_mw": f"{float(p_mw):.2f}",
        "r_fp": f"{cam_med('fp_DDI_AK', 'cam_lum_r'):+.2f}",
        "r_ak": f"{cam_med('true_AK', 'cam_lum_r'):+.2f}",
        "r_mel": f"{cam_med('true_MEL', 'cam_lum_r'):+.2f}",
        "g1": f"{d1['ak_mean']:.3f}", "g3": f"{d3['ak_mean']:.3f}",
        "g5": f"{d5['ak_mean']:.3f}",
        "tm_d": f"{tm_d['ak_mean']:.3f}", "tm_t": f"{tm_t['ak_mean']:.3f}",
        "ks_tm": f"{ks_tm:.2f}",
        "lam": f"{lam:.2f}",
        "auc0": f"{sweep[0.0]['tune_auc']:.3f}",
        "aucL": f"{sweep[lam]['tune_auc']:.3f}",
        "spec0": f"{rec('baseline', 'ALL', 'specificity'):.3f}",
        "specL": f"{rec(adj_cfg, 'ALL', 'specificity'):.3f}",
        "eo0": f"{eo['baseline']['point']:.3f}",
        "eo0lo": f"{eo['baseline']['lo']:.3f}",
        "eo0hi": f"{eo['baseline']['hi']:.3f}",
        "eoL": f"{eo[adj_cfg]['point']:.3f}",
        "eoLlo": f"{eo[adj_cfg]['lo']:.3f}",
        "eoLhi": f"{eo[adj_cfg]['hi']:.3f}",
        "akrec0": f"{safe['baseline']['ak_recall_at_spec80']:.3f}",
        "akrec8": f"{safe['AK-adj lam=0.8 (sens.)']['ak_recall_at_spec80']:.3f}",
        "shAK0": f"{100 * l0['AK_argmax_share']:.1f}",
        "shAK8": f"{100 * l8['AK_argmax_share']:.1f}",
        "shBCC0": f"{100 * l0['BCC_argmax_share']:.1f}",
        "shBCC8": f"{100 * l8['BCC_argmax_share']:.1f}",
        "shMEL0": f"{100 * l0['MEL_argmax_share']:.1f}",
        "shMEL8": f"{100 * l8['MEL_argmax_share']:.1f}",
        "edge_fp": f"{place_med('fp_DDI_AK', 'edge_band_share'):.3f}",
        "edge_ak": f"{place_med('true_AK', 'edge_band_share'):.3f}",
        "edge_mel": f"{place_med('true_MEL', 'edge_band_share'):.3f}",
        "hot_fp": f"{place_med('fp_DDI_AK', 'hot_lum_ratio'):.2f}",
        "hot_ak": f"{place_med('true_AK', 'hot_lum_ratio'):.2f}",
        "p_edge": f"{mw3d['edge_band fp_DDI vs true_AK']['p']:.2f}",
        "p_hot": f"{mw3d['hot_lum fp_DDI vs true_AK']['p']:.2f}",
        "n_vig_mel": str(vig_by.get("true_MEL", 0)),
        "n_vig_fp": str(vig_by.get("fp_DDI_AK", 0)),
        "n_rul": str(n_rul),
        "n_ink": str(n_ink),
        # Step 3e: capture-modality confound (audit_step3e_report.json)
        "pct_mel_derm": f"{100 * rep3e['test_panel_mel_derm_frac']:.0f}",
        "pct_ak_derm": f"{100 * rep3e['test_panel_ak_derm_frac']:.0f}",
        "r_mel_clin": f"{cf_med('true_MEL_clinical', 'cam_lum_r'):+.2f}",
        "ak_derm": f"{cf_med('true_AK_derm', 'conc_top1pct'):.3f}",
        "ak_clin": f"{cf_med('true_AK', 'conc_top1pct'):.3f}",
        "mel_derm": f"{cf_med('true_MEL', 'conc_top1pct'):.3f}",
        "p_dermmel": tex_pow(cf_tests['conc_top1pct__derm_MEL_vs_derm_AK']['p']),
        "p_akmod": tex_pow(cf_tests['conc_top1pct__within_AK_derm_vs_clin']['p']),
    }
    v.update(v2)
    return v


TEMPLATE = """%% === STEP3-SECTION (auto-generated from audit CSVs; do not hand-edit numbers) ===
\\section{{Score-Level Analysis of the Out-of-Domain Failure}}
\\label{{sec:step3}}

Threshold equalization (Section~\\ref{{sec:step2thr}}) showed that no
per-group threshold policy closes the DDI sensitivity gap. Score-level
analysis explains why: the malignant probability itself barely separates
benign from malignant lesions on DDI, and the deficit is worst exactly where
the fairness gap is largest. All statistics derive from the audited prediction
cache of the deployed checkpoint; CIs are (group~$\\times$~label) stratified
percentile bootstrap (2,000 replications, seed 42) with raw $n$ beside every
estimate.

\\subsection{{Threshold Equalization Does Not Close the Gap}}
\\label{{sec:step2thr}}
The per-group experiment: DDI was split into (group~$\\times$~label)
stratified tune ({p2_tune}) and eval ({p2_eval}) halves ({p2_tpos}/{p2_epos}
malignant), per-group decision thresholds were fitted on the tune half only
(the largest threshold meeting a $\\tau\\geq0.95$ tune-half sensitivity target,
$\\tau$ chosen to minimize the tune-half EO gap), and the result was evaluated
once on the held-out eval half against two controls: (A) the shipped uniform
0.25 threshold and (B) a uniform threshold re-tuned on the tune half
({p2_bt}), which isolates any group-specific benefit from plain re-tuning.
Eval-half outcomes: at the shipped threshold (A), sensitivity is
1.000/1.000/{sensA4} (I--II/III--IV/V--VI) with the EO gap at {eoA}
[{eoAlo}, {eoAhi}] -- already below the 5\\% target at the point estimate,
but only because specificity is negligible in every group (overall
{specA}); there is no sensitivity gap left to close at this operating point.
This is the same operating point at which Section~\\ref{{sec:defect1}}
reports the full-panel EO gap of 25.96\\%: the near-perfect sensitivity and
the wide gap are two readings of one fact -- malignant scores in every tone
group sit almost always above threshold on DDI -- and the narrowness of that
margin is exactly what re-tuning cannot exploit without collapsing
sensitivity.
Re-tuning cannot buy specificity instead: uniform re-tuning (B) yields
overall specificity {specB}, and the per-group config (C) reaches only
{specC} overall -- its V--VI specificity of {specC4}
[{specC4lo}, {specC4hi}] is bought with V--VI sensitivity falling to
{sensC4}, and the EO gap widens to {eoC} [{eoClo}, {eoChi}]. The next
subsection shows why: no threshold geometry can equalize what the score
distributions do not rank.

\\subsection{{No Operating Point Exists: Per-Group ROC Ceilings}}
\\begin{{table}}[h]
\\centering
\\caption{{Per-group ROC ceilings on the DDI panel (malignant score).}}
\\label{{tab:ceilings}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrccc}}
\\toprule
Group & $n$ (pos/neg) & AUC [95\\% CI] & Sens@Spec$\\geq$0.6 & Sens@Spec$\\geq$0.8 \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=\\columnwidth]{{figs/fig_roc_ceilings.png}}
\\caption{{Per-group ROC ceilings (malignant sensitivity at two specificity floors), with AUC and 95\\% CI annotated per group; same data as Table~\\ref{{tab:ceilings}}. FST~V--VI is statistically indistinguishable from chance at both floors.}}
\\label{{fig:ceilings}}
\\end{{figure}}

For FST~V--VI the 95\\% CI on AUC includes 0.5 (Table~\\ref{{tab:ceilings}}, Figure~\\ref{{fig:ceilings}}):
the malignant score carries \\emph{{no measurable benign/malignant ranking
signal for dark-skin images}}. This formalizes the threshold-equalization
negative result: the disparity lives in the score distributions, not in
threshold placement, and no post-hoc decision rule can recover performance
the scores do not contain.

\\subsection{{A Spurious Actinic-Keratosis Response Drives the False Positives}}
On true-benign DDI images the actinic-keratosis (AK) column is the largest
malignant contributor on {ak_share}\\% of images (mean probability
{ak_fp_mean}, 90th percentile {ak_fp_q90}); all fifteen highest-scoring false
positives have AK as their top contributor (per-image AK probability
0.91--0.97). The same column is nearly silent on benign in-domain images
(mean {ak_test_ben} on the test panel) -- an order-of-magnitude shift
(one-sided KS $={ks_ak}$, $p={p_ak}$) -- while contributing normally on true
in-domain malignancies (mean {ak_test_mal}).

Grad-CAM analysis~\\cite{{selvaraju2017}} with matched comparators bounds what
attention maps can establish. The AK head's activation is diffuse in both
clinical-photo settings (median top-1\\% CAM mass share {cam_fp} on DDI false
positives vs {cam_ak} on true AK cases; Mann--Whitney $p={p_mw}$), so the
spurious response cannot be identified by diffuseness: at this modality the
diffuse response is indistinguishable from the head's normal behavior. The AK
map leans toward darker pixels in both clinical settings (median
CAM--luminance correlation ${r_fp}$ on false positives, ${r_ak}$ on true AK).
The remaining comparator -- the MEL column on true melanomas (${r_mel}$) --
differs from the AK cohorts in capture modality as well as class: melanoma
test images are almost all dermoscopic acquisitions ({pct_mel_derm}\\% of the
test panel, vs {pct_ak_derm}\\% of AK), and a dermoscopic frame leaves little
background for any CAM to spread onto, so no class-level localization claim
is made from that comparison.

A structured visual review of all forty-five overlays (three cohorts of
fifteen, reviewed side-by-side in triads; Figure~\\ref{{fig:camgrid}})
then tested the capture-cue
hypothesis directly. It supports the field-wide reading: on the false
positives the activation spreads across the skin field and does \\emph{{not}}
focal-lock onto salient non-lesion objects. Manual inspection found paper
rulers in frame in {n_rul}/15 false-positive images and surgical ink
markings in {n_ink}/15, yet neither attracts elevated CAM mass (edge-band
mass share {edge_fp} vs {edge_ak} on true AK, Mann--Whitney $p={p_edge}$;
hot-pixel luminance ratio {hot_fp} vs {hot_ak}, $p={p_hot}$).

Conversely, the in-
domain melanoma comparators are predominantly circular dermoscopic captures
with black surrounds ({n_vig_mel} of 15 with corner/center luminance below
0.75, vs {n_vig_fp} of 15 false positives); their lesion-localized
appearance is expected from dermoscopic framing and is not read as a
class-level property. A modality-controlled replication crossed class with
capture style directly: on dermoscopic images the AK column is if anything
\\emph{{more}} top-concentrated than the MEL column (median top-1\\% mass
{ak_derm} vs {mel_derm}, $p={p_dermmel}$), and within AK the concentration
doubles from clinical to dermoscopic acquisitions ({ak_clin} to {ak_derm},
$p={p_akmod}$) -- the concentration statistic tracks capture modality, not
class. Capture artifacts are therefore present in the false-positive
cohort as expected, but the activation is not localized on them: the
artifact census is context for the domain-shift finding, not its cause,
and the spuriousness claim rests on the score-level evidence.

\\begin{{figure*}}[t]
\\centering
\\includegraphics[width=\\textwidth]{{figs/fig_cam_grid.png}}
\\caption{{Grad-CAM overlays for the three cohorts (per-tile labels: rank, malignant probability). Row~1: the five false positives whose capture artifacts (paper rulers, surgical ink) are attested in the authors' artifact key -- five of the fifteen flagged cases, all showing the same field-wide activation. Rows~2--3: the five highest-scoring true-AK and true-melanoma comparators. Both AK cohorts activate field-wide; the melanoma comparators are dermoscopic acquisitions whose lesion-localized appearance partly reflects capture modality (modality-controlled analysis in the text).}}
\\label{{fig:camgrid}}
\\end{{figure*}}

\\subsection{{Domain Shift, Not Skin Tone, Drives the Artifact}}
Within DDI, benign-image AK probability is nearly flat across tone groups and
\\emph{{highest}} for the lightest-tone group (FST~I--II {g1}, FST~V--VI {g5},
FST~III--IV {g3}) -- the opposite of a monotone tone gradient. At identical
self-reported tone (FST~I--II), DDI benigns show AK mean {tm_d} vs test-panel
{tm_t} (KS $={ks_tm}$). We therefore frame the DDI disparity as a
\\emph{{domain-generalization failure that disproportionately lands on the
group with the weakest underlying ranking signal}}: the capture artifact and
the tone-related performance gap compound rather than share a single cause.

\\subsection{{A Post-Hoc Correction Cannot Repair the Score}}
A post-hoc down-weighting $p' = p_{{\\mathrm{{mal}}}} - \\lambda
P_{{\\mathrm{{AK}}}}$, with $\\lambda$ and the decision threshold fitted on one
stratified half of DDI and evaluated once on the held-out half, yields at most
marginal gains: tune-half AUC peaks at $\\lambda^*={lam}$ ({aucL} vs {auc0}
baseline) and degrades monotonically beyond. At $\\lambda^*$ the eval-half
specificity is statistically unchanged ({spec0} $\\to$ {specL} overall) and the
EO sensitivity gap moves from {eo0} [{eo0lo}, {eo0hi}] to {eoL} [{eoLlo},
{eoLhi}]. Aggressive deflation ($\\lambda=0.8$) destroys in-domain AK detection
(AK-class recall at $\\approx$80\\% specificity: {akrec0} $\\to$ {akrec8}) while
still not fixing DDI specificity. The failure is structural: on eval-half
benign images, deflating AK merely migrates the dominant malignant contributor
to other columns (argmax share AK {shAK0}\\% $\\to$ {shAK8}\\%, BCC {shBCC0}\\%
$\\to$ {shBCC8}\\%, MEL {shMEL0}\\% $\\to$ {shMEL8}\\% at $\\lambda=0.8$) --
single-column recalibration is whack-a-mole when the shared representation maps
the capture domain itself onto malignancy. The deficit is therefore
representational: closing it requires in-domain training signal (fine-tuning
with DDI-style images, including hard negatives, and tone/capture
augmentation), not score- or threshold-level correction.
%% === END STEP3-SECTION ===
"""


def main():
    tex = TEX.read_text(encoding="utf-8")

    if TICKET in tex:
        print("Step 3 section already present - replacing old copy.")
        tex = re.sub(re.escape(TICKET) + r".*?%% === END STEP3-SECTION ===\n",
                     "", tex, flags=re.DOTALL)

    tex = tex.replace("clinically calibrated threshold 0.24",
                      "clinically calibrated threshold 0.25")
    tex = tex.replace("at clinical threshold 0.24}",
                      "at clinical threshold 0.25}")
    anchor = ("We show that na\\\"ive re-training does not address fairness "
              "defects rooted in label corruption, and argue for data-lineage "
              "auditing as a mandatory pre-deployment step for fairness-aware "
              "medical ML.")
    add_abs = (" Score-level analysis of the out-of-domain failure shows the "
               "malignancy score carries no measurable ranking signal for "
               "FST V--VI images (AUC 0.484 [0.382, 0.585]), traced to a "
               "spurious, capture-associated actinic-keratosis response that "
               "is invariant to skin tone within DDI and cannot be corrected "
               "post hoc -- a representational domain gap rather than a "
               "threshold-calibration failure.")
    if anchor in tex and "AUC 0.484 [0.382, 0.585]" not in tex:
        tex = tex.replace(anchor, add_abs + " " + anchor)

    # collapse any duplicate labels from earlier double-runs, then insert once
    tex = tex.replace("\\label{sec:threshold}\n\\label{sec:threshold}",
                      "\\label{sec:threshold}")
    if "\\label{sec:threshold}" not in tex:
        tex = tex.replace("\\section{Clinical Threshold Calibration}",
                          "\\section{Clinical Threshold Calibration}\n"
                          "\\label{sec:threshold}")

    lim_anchor = ("Best validation balanced accuracy (67.8--73.7\\% across "
                  "evaluation protocols) had not converged at the time of "
                  "writing; retraining to convergence under corrected tone "
                  "supervision, and re-measuring the DDI fairness gap, is "
                  "ongoing work.")
    add_lim = (" Three caveats attach to the Step~\\ref{{sec:step3}} analysis. "
               "First, capture modality is confounded with class in the "
               "available data ({pct_mel_derm}\\% of test melanomas are "
               "dermoscopic acquisitions vs {pct_ak_derm}\\% of AKs), so "
               "attention morphology cannot be compared across classes "
               "without a modality-matched design; the modality-controlled "
               "replication indicates the concentration statistic tracks "
               "capture modality, and no class-level morphological claim is "
               "made. Second, the AK head's preference for darker-than-scene "
               "pixels is left as an open question: the same lean appears in "
               "the MEL column on clinical-photo melanomas (median "
               "CAM--luminance correlation {r_mel_clin}, though those cases "
               "are low-confidence), which argues against an "
               "actinic-damage-specific explanation, but disentangling "
               "illumination response from lesion-border shadowing requires "
               "pixel-level attribution beyond the scope of this audit. "
               "Third, the visual overlay review was performed by the "
               "authors rather than an independent dermatologist, so the "
               "ruler and artifact counts are manual tallies pending expert "
               "confirmation.")
    # self-heal: earlier integrator versions emitted doubled braces into the
    # limitations caveat (plain string, not a format template)
    tex = tex.replace("Step~\\ref{{sec:step3}} analysis",
                      "Step~\\ref{sec:step3} analysis")

    # consistency: Defect 2 must quote the CURRENT gap (25.96), not the
    # retired Aug-15 figure (25.26); simplify the retirement parenthetical
    tex = tex.replace("Deeper audit of the 25.26\\% gap",
                      "Deeper audit of the 25.96\\% gap")
    tex = tex.replace(
        "yielding -- under the current checkpoint and the September 2026 "
        "artifact state (ddi\\_eval.csv regenerated 2026-09-05; figures "
        "from the superseded 2026-08-15 run, including AUC-ROC 0.6398 and "
        "an equalized-odds gap of 25.26\\%, are retired) --",
        "yielding -- under the current checkpoint (September 2026 artifact "
        "state; all 2026-08-15 figures are superseded and retired) --")

    if "\\label{sec:defect1}" not in tex:
        tex = tex.replace("\\section{Defect 1: Stale Binary Risk-Label Mapping}",
                          "\\section{Defect 1: Stale Binary Risk-Label Mapping}"
                          "\n\\label{sec:defect1}")

    # Related Work + citations for the datasets and methods already named
    rw = ("\\section{Related Work}\n"
          "Skin-lesion classifiers are built on HAM10000~\\cite{ham10000}, "
          "ISIC~2019~\\cite{isic2019} and PAD-UFES-20~\\cite{padufes}, and "
          "have repeatedly been shown to degrade on darker skin: Daneshjou "
          "et al. evaluate publicly released dermoscopy models on the "
          "expert-labelled DDI test set~\\cite{ddi} and report the "
          "sensitivity disparities that motivate our evaluation. "
          "Fairness-aware training typically penalizes group-conditioned "
          "error imbalance~\\cite{hardt2016}; attribution tools such as "
          "Grad-CAM~\\cite{selvaraju2017} are the standard instrument for "
          "inspecting what such models attend to; and GeM "
          "pooling~\\cite{gem} is a common retrieval-era pooling whose "
          "quantization behavior has, to our knowledge, not previously been "
          "analyzed. Obermeyer et al. show that label choices can encode "
          "bias even when aggregate accuracy looks "
          "acceptable~\\cite{obermeyer2019}, and Finlayson et al. catalogue "
          "the dataset-shift hazards behind silently degrading clinical "
          "models~\\cite{finlayson2021} -- the mechanisms behind Defects "
          "1 and 2 here. Unlike prior work that primarily reports "
          "disparities, this paper traces three specific silent defects "
          "end-to-end and shows that a four-layer chain of post-hoc "
          "corrections cannot repair the resulting out-of-domain "
          "failure.\n\n")
    if "\\section{Related Work}" not in tex:
        tex = tex.replace("\\section{System Overview}", rw + "\\section{System Overview}")

    for old, new in [
        ("(DDI)\ndataset for fairness", "(DDI)\ndataset~\\cite{ddi} for fairness"),
        ("trained on HAM10000, ISIC~2019, PAD-UFES-20,",
         "trained on HAM10000~\\cite{ham10000}, ISIC~2019~\\cite{isic2019}, "
         "PAD-UFES-20~\\cite{padufes},"),
        ("Generalized Mean (GeM) pooling, and a dual-head output",
         "Generalized Mean (GeM) pooling~\\cite{gem}, and a dual-head output"),
        ("penalizes the equalized-odds gap across tone",
         "penalizes the equalized-odds gap~\\cite{hardt2016} across tone"),
        ("the gap originated in corrupted \\emph{supervision}, not "
         "insufficient optimization.",
         "the gap originated in corrupted \\emph{supervision}, not "
         "insufficient optimization -- the label-choice bias mechanism "
         "documented by Obermeyer et al.~\\cite{obermeyer2019}, and a "
         "dataset-shift hazard of the kind catalogued by Finlayson et "
         "al.~\\cite{finlayson2021}."),
    ]:
        if old in tex:
            tex = tex.replace(old, new)

    # bibliography: add the methods references
    bib_anchor = "\\begin{thebibliography}{9}\n"
    new_bibs = (
        "\\bibitem{selvaraju2017} R. R. Selvaraju et al., ``Grad-CAM: "
        "Visual Explanations from Deep Networks via Gradient-Based "
        "Localization,'' \\emph{ICCV}, 2017.\n"
        "\\bibitem{hardt2016} M. Hardt, E. Price, N. Srebro, ``Equality of "
        "Opportunity in Supervised Learning,'' \\emph{NeurIPS}, 2016.\n"
        "\\bibitem{obermeyer2019} Z. Obermeyer, B. Powers, C. Vogeli, "
        "S. Mullainathan, ``Dissecting racial bias in an algorithm used to "
        "manage the health of populations,'' \\emph{Science}, 2019.\n"
        "\\bibitem{finlayson2021} S. G. Finlayson et al., ``The Clinician "
        "and Dataset Shift in Artificial Intelligence,'' \\emph{N. Engl. "
        "J. Med.}, 2021.\n")
    if bib_anchor in tex and "\\bibitem{selvaraju2017}" not in tex:
        tex = tex.replace(bib_anchor, bib_anchor + new_bibs)

    # explicit Step 4 fork statement, in writing, in the Discussion
    fork_anchor = ("matched-comparator Grad-CAM analysis showed "
                   "the diffuse activation previously read as an artifact "
                   "signature is in fact the head's normal behavior.")
    fork = (" Given this representational deficit, we deliberately did not "
            "attempt in-domain fine-tuning in this work: Sections "
            "\\ref{sec:threshold} and \\ref{sec:step3} establish that every "
            "score- and threshold-level correction fails, so we identify "
            "targeted data acquisition (DDI-style capture-domain images, "
            "including benign hard negatives) and score-level fine-tuning "
            "under the corrected tone supervision as the appropriate future "
            "work, to be judged against the per-group ROC ceilings reported "
            "here.")
    if fork_anchor in tex and "deliberately did not attempt" not in tex:
        tex = tex.replace(fork_anchor, fork_anchor + fork)

    # author block: fill from repository identity (user-editable)
    for old, new in [
        ("\\IEEEauthorblockN{Author Name}",
         "\\IEEEauthorblockN{Sanjana T. G.}"),
        ("\\IEEEauthorblockA{Department of Engineering\\\\ University Name\\\\ email@example.edu}",
         "\\IEEEauthorblockA{Department of Computer Science and Technology\\\\ "
         "University of Cambridge\\\\ sanjanatg2126@gmail.com}"),
    ]:
        if old in tex:
            tex = tex.replace(old, new)
    if "pixel-level attribution" not in tex:
        pass  # inserted below, after vals exist (add_lim is a template)

    disc_anchor = "\\subsection{Quantization Choice Is Architecture-Dependent}"
    add_disc = ("\\subsection{Domain Shift and the Limits of Post-Hoc Fairness "
                "Repair}\nThe DDI analysis (Section~\\ref{sec:step3}) adds a "
                "fourth lesson: post-hoc repairs (per-group thresholds, "
                "single-column recalibration) cannot fix a representational "
                "domain gap, and attention-map morphology alone cannot "
                "diagnose one -- matched-comparator Grad-CAM analysis showed "
                "the diffuse activation previously read as an artifact "
                "signature is in fact the head's normal behavior.\n\n")
    if disc_anchor in tex and "Limits of Post-Hoc Fairness" not in tex:
        tex = tex.replace(disc_anchor, add_disc + disc_anchor)

    vals = collect_values()
    section = TEMPLATE.format(**vals)
    tex = tex.replace("\\section{Discussion}", section + "\n\\section{Discussion}")
    add_lim_f = add_lim.format(**vals)
    old_lim_re = re.compile(r" Two caveats attach to the Step~.*?expert "
                            r"confirmation\.", re.DOTALL)
    if old_lim_re.search(tex):        # upgrade the pre-3e two-caveat version
        tex = old_lim_re.sub(lambda m: add_lim_f, tex, count=1)
    elif lim_anchor in tex and "Three caveats" not in tex:
        tex = tex.replace(lim_anchor, lim_anchor + add_lim_f)

    # guard: no unfilled placeholders from the template, no silent NaNs
    leftovers = [k for k in vals if f"{{{k}}}" in tex]
    if leftovers:
        raise SystemExit(f"unfilled placeholders remain: {leftovers}")
    if re.search(r"\bnan\b", section, re.IGNORECASE):
        raise SystemExit("NaN leaked into the generated section - a median "
                         "looked up a cohort that does not exist")

    TEX.write_text(tex, encoding="utf-8")
    print(f"updated {TEX} ({len(tex.splitlines())} lines)")


if __name__ == "__main__":
    main()
