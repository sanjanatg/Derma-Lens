# Paper fact sheet — verified against artifacts (2026-09-06T22:18:24)

Every number below was recomputed at file-generation time from `model/audit/preds_cache.npz`, the audit CSV/JSON reports, and `checkpoints/threshold_config.json`. Provenance column names the source artifact. Do not hand-edit numbers; regenerate instead.

## 1. Final verified numbers

### 1.1 Test panel (in-domain)

| Metric | Value | Source / note |
|---|---|---|
| Shipped clinical threshold | 0.25 | threshold_config.json |
| threshold_config.default_threshold | 0.5 | threshold_config.json |
| threshold_config.clinical_threshold | 0.25 | threshold_config.json |
| threshold_config.sensitivity_at_clinical | 0.9603 | threshold_config.json |
| threshold_config.specificity_at_clinical | 0.5543 | threshold_config.json |
| threshold_config.fnr_at_clinical | 0.0397 | threshold_config.json |
| threshold_config.roc_auc | 0.9239 | threshold_config.json |
| Test malignant-score AUC | 0.9239 | fresh: cache |
| Test sens @ 0.25 | 0.9593 | fresh: cache |
| Test spec @ 0.25 | 0.5552 | fresh: cache |
| Test decision acc @ 0.25 | 0.6959 | fresh: cache |
| Test BalAcc (top-line) | 73.7% claimed | NOT recomputed in audit — training-log provenance (see §6.1) |
| n_test | 5647 (malignant 1967) | cache |

### 1.2 DDI panel (out-of-domain)

| Metric | Value | Source / note |
|---|---|---|
| DDI malignant-score AUC | 0.5849 | fresh: cache |
| DDI sens @ 0.5 | 0.8187 | fresh: cache |
| DDI spec @ 0.5 | 0.2412 | fresh: cache |
| n_DDI | 656 (malignant 171, benign 485) | cache |

**EO sensitivity gap (Step 1, BCa bootstrap):**

| Group | n | n_pos | n_neg | Sens [95% CI] | Source |
|---|---|---|---|---|---|
| FST I-II | 208 | 49 | 159 | 0.857 [0.714, 0.918] | ddi_fairness_ci.csv |
| FST III-IV | 241 | 74 | 167 | 0.905 [0.811, 0.946] | ddi_fairness_ci.csv |
| FST V-VI | 207 | 48 | 159 | 0.646 [0.479, 0.750] | ddi_fairness_ci.csv |
| ALL (expert-labelled tones) | 656 | 171 | 485 | 0.819 [0.749, 0.865] | ddi_fairness_ci.csv |

**Group contrasts (sensitivity gaps; ddi rows are the quoted fairness gaps):**

| Panel | Contrast | Sens_a | Sens_b | Diff [95% CI] | n_a | n_b | Panel EO gap [CI] |
|---|---|---|---|---|---|---|---|
| test | 1 - 0 | 0.812 | 1.000 | -0.188 [-0.209, -0.166] | 1275 | 20 | 0.333 [0.175, 1.000] |
| test | 2 - 0 | 0.847 | 1.000 | -0.153 [-0.181, -0.127] | 653 | 20 | 0.333 [0.175, 1.000] |
| test | 3 - 0 | 0.938 | 1.000 | -0.062 [-0.188, 0.000] | 16 | 20 | 0.333 [0.175, 1.000] |
| test | 4 - 0 | 0.667 | 1.000 | -0.333 [-1.000, 0.000] | 3 | 20 | 0.333 [0.175, 1.000] |
| ddi | 4 - 0 | 0.646 | 0.857 | -0.211 [-0.378, -0.047] | 48 | 49 | 0.260 [0.127, 0.418] |
| ddi | 4 - 2 | 0.646 | 0.905 | -0.260 [-0.412, -0.106] | 48 | 74 | 0.260 [0.127, 0.418] |
| ddi | 2 - 0 | 0.905 | 0.857 | 0.048 [-0.074, 0.164] | 74 | 49 | 0.260 [0.127, 0.418] |

### 1.3 Per-group ROC ceilings (Step 3)

| Group | n (pos/neg) | AUC [CI] | Sens@Spec≥0.6 | Sens@Spec≥0.8 |
|---|---|---|---|---|
| FST I-II | 208 (49/159) | 0.574 [0.486, 0.667] | 0.490 | 0.224 |
| FST III-IV | 241 (74/167) | 0.668 [0.592, 0.743] | 0.662 | 0.392 |
| FST V-VI | 207 (48/159) | 0.484 [0.382, 0.585] | 0.396 | 0.229 |
| ALL | 656 (171/485) | 0.585 [0.534, 0.636] | 0.503 | 0.357 |

### 1.4 AK-column behavior

| Stat | Benign DDI | Benign test (I–IV) | Source |
|---|---|---|---|
| AK mean | 0.375 | 0.046 | fresh: cache |
| AK q90 | 0.803 | 0.070 | fresh: cache |
| One-sided KS (test→DDI) | 0.76 (p = 1.3e-218) | | fresh: cache |

### 1.5 Step 2 threshold equalization (tune/eval halves)

Split: tune 326 (85 mal) / eval 330 (86 mal). Thresholds fitted on tune only; evaluated once on eval.

| Config | Overall sens | Overall spec | EO gap [CI] | V–VI sens | V–VI spec |
|---|---|---|---|---|---|
| A. baseline | 0.988 | 0.016 | 0.042 [0.000, 0.125] | 0.958 | 0.000 [0.000, 0.000] |
| B. baseline | 0.965 | 0.025 | 0.125 [0.000, 0.250] | 0.875 | 0.013 [0.000, 0.037] |
| C. per-group thresholds (tau>=0.95, lam=0.25) | 0.965 | 0.045 | 0.125 [0.000, 0.250] | 0.875 | 0.087 [0.025, 0.163] |

### 1.6 Step 3b AK down-weighting (λ sweep, tune/eval)

λ* = 0.25 (tune AUC peak 0.609 vs baseline 0.596); sweep tail λ=1.0 → 0.524.

| λ | tune AUC | eval EO gap [CI] |
|---|---|---|
| 0 | 0.596 | 0.125 [0.000, 0.250] |
| 0.25 | 0.609 | 0.083 [0.000, 0.208] |

In-domain safety: AK-class recall @≈80% spec 0.955 (baseline) → 0.444 (λ=0.8). Test AUC 0.9240 → 0.9234 at λ*.

### 1.7 CAM comparators + placement (Steps 3c/3d/3e)

**Capture-modality structure (Step 3e).** Test-panel melanoma is 99% derm-provenance vs AK 62%; the step3c comparator cohorts were fully modality-confounded (true-MEL 15/15 derm, true-AK 15/15 clinical, FP 15/15 clinical). Modality-matched counterfactuals: derm-AK top-1% concentration 0.039 vs derm-MEL 0.025 (MW p = 7.0e-03) — direction REVERSES vs the unmatched comparison; within-AK clinical to derm: 0.019 → 0.039 (p = 5.1e-06). Concentration tracks capture modality, not class; the FP-vs-true-AK spuriousness comparison was clinical-vs-clinical and is unaffected. Clinical-MEL cell (n = 8) is low-confidence (p 0.03–0.29), so the within-MEL contrast is selection-biased.

| Stat | fp_DDI_AK | true_AK | true_MEL | p (FP vs true-AK) |
|---|---|---|---|---|
| top-1% CAM mass share (med) | 0.023 | 0.019 | 0.025 | 0.11 (MW) |
| CAM–luminance r (med) | -0.29 | -0.21 | 0.11 | — |
| edge-band mass share (med) | 0.410 | 0.377 | 0.244 | 0.28 (MW) |
| hot-pixel luminance ratio (med) | 0.95 | 0.96 | — | 0.93 (MW) |

Vignette flags (corner/center lum < 0.75): FP 1/15, true_AK 0/15, true_MEL 9/15 (step3d_placement.csv).

**Artifact census (authors' unblinded key):** rulers in 8/15 FP targets, surgical ink in 6/15 (blind_review_authors_key.csv; blinded confirmation pending).

## 2. The four-step negative-result chain (plain skeleton)

1. **Label provenance.** We suspected the fairness objective had been optimized against the wrong malignant-class definition, which would have explained the disparity as an artifact. Verification of the training code, the epoch logging, and the git history showed the training-time penalty always used the full four-class malignant definition; the wrong definition existed only in a dead reporting path. The reported gap is therefore a property of the model, not of a metric bug.
2. **Threshold equalization.** If the gap were a threshold-placement problem, per-group decision thresholds fitted on a held-out tuning half should close it. Evaluated once on the untouched eval half against a uniformly re-tuned control, group-specific thresholds bought specificity only by giving up sensitivity in the same groups, leaving the gap unchanged within confidence intervals. The failure is in the score rankings, not where the threshold sits.
3. **Single-column recalibration.** Decomposition showed one malignant sub-score (the actinic-keratosis head) carries most false-positive mass out of domain, so deflating that column should help if it alone were spurious. With the deflation strength and threshold fitted on the tuning half only, ranking improved marginally at best and degraded monotonically beyond the peak, while the remaining columns took over the false positives and aggressive deflation destroyed genuine in-domain AK detection. The shared representation, not one column, maps this capture domain onto malignancy.
4. **Mechanism attribution.** If the response were triggered by visible capture artifacts (rulers, ink markings) or by skin tone itself, attention maps and within-dataset contrasts should show it. Matched-comparator Grad-CAM found the activation equally diffuse on genuine AK cases at matched clinical modality, field-wide rather than focal on the artifacts present in frame, and tone-invariant within the out-of-domain dataset — while an order-of-magnitude score shift versus in-domain images persists at matched tone. What remains is a representational domain gap that lands hardest on the tone group with the weakest underlying ranking signal.

## 3. Blinded expert review — NOT YET SCORED (package ready; scorer tested; awaiting reviewer answers)

Package: `blind_review_gradcam.html` (15 trials, balanced 8A/7B, seed 42), key in `audit/blind_review_key.csv`, protocol + invitation text in `audit/blind_review_protocol.md`, scorer `audit_score_blind_review.py` (pre-registered verdict: ≥12/15 correct ⇒ patterns distinguishable; else the 'visually indistinguishable' claim stands). Score with: `python audit_score_blind_review.py answers.json`.

## 4. Bibliography with load-bearing roles

| Key | Supports | Status |
|---|---|---|
| `ham10000` | Training dataset identity (main dermoscopy source). | load-bearing (dataset claim) |
| `isic2019` | Training dataset identity. | load-bearing (dataset) |
| `padufes` | Training dataset identity. | load-bearing (dataset) |
| `ddi` | External expert-labelled eval set; the disparity motivation for the whole audit. | load-bearing (central) |
| `gem` | GeM pooling method — the operation behind the INT8 collapse (Defect 3). | load-bearing (Defect 3) |
| `selvaraju2017` | Grad-CAM method used in §VIII.C comparators. | load-bearing (method) |
| `hardt2016` | Equalized-odds definition used by the loss and all gap measurements. | load-bearing (metric definition) |
| `obermeyer2019` | Label-choice bias mechanism behind Defect 2 framing. | contextual support |
| `finlayson2021` | Dataset-shift hazard framing for the generalization discussion. | contextual support |

## 5. Figure/table assets on disk right now

| Path | Caption/status |
|---|---|
| model/figures/gradcam_step3b/ (15 jpg) | Step 3b FP overlays, AK-column CAM (filenames carry row/group/prob). |
| model/figures/gradcam_step3c/ (45 jpg) | Comparator triads: 15 fp_DDI_AK + 15 true_AK + 15 true_MEL. |
| model/figures/gradcam_step3d/ (3 png) | Per-cohort contact sheets (original | overlay), stats-annotated — REVIEW strips, not paper figures. |
| model/figures/gradcam_step3e/ (23 jpg) | Modality-matched counterfactual overlays (derm-AK, clinical-MEL) from audit_step3e.py. |
| ppt_files/figs/fig_cam_grid.png | PAPER FIGURE, wired into the .tex via tex_integrate_step3.py; regenerate with model/make_paper_figures.py. |
| ppt_files/figs/fig_roc_ceilings.png | PAPER FIGURE, wired into the .tex via tex_integrate_step3.py; regenerate with model/make_paper_figures.py. |
| model/figures/confusion_matrix_9class.png | LEGACY figure — provenance likely the superseded 2026-08-15 run; regenerate before use (see §6.2). |
| model/figures/ddi_fairness_evaluation.png | LEGACY figure — provenance likely the superseded 2026-08-15 run; regenerate before use (see §6.2). |
| model/figures/fairness_by_skin_type.png | LEGACY figure — provenance likely the superseded 2026-08-15 run; regenerate before use (see §6.2). |
| model/figures/gradcam_examples.png | LEGACY figure — provenance likely the superseded 2026-08-15 run; regenerate before use (see §6.2). |
| model/figures/per_class_accuracy.png | LEGACY figure — provenance likely the superseded 2026-08-15 run; regenerate before use (see §6.2). |
| model/figures/threshold_analysis.png | LEGACY figure — provenance likely the superseded 2026-08-15 run; regenerate before use (see §6.2). |
| model/figures/training_curves.png | LEGACY figure — provenance likely the superseded 2026-08-15 run; regenerate before use (see §6.2). |
| ppt_files/dermalens_paper_compiled_check.pdf | Current compiled paper (5 pp, tectonic, clean). |
| ppt_files/DermaLens_Audit.pptx | 13-slide deck incl. Step 3 mechanism slide. |
| review_gradcam_step3c.html / blind_review_gradcam.html | Unblinded triad review page / blinded review form (regenerable, not committed). |

## 6. Shaky items — verify before presenting as settled

1. **73.7% top-line accuracy and Table II (quant) numbers** come from training/validation scripts, not the audited cache. If quoted, either re-derive from the cache or state the provenance.
2. **Legacy `model/figures/*.png`** (training curves, DDI fairness eval, threshold analysis, confusion/per-class) almost certainly predate the current checkpoint (superseded 2026-08-15 run). Do not build paper figures from them without regenerating.
3. **The 25.96% EO gap has CI [12.67%, 41.80%]** — wide, because only 171 malignant DDI images exist. Always quote with the CI; the point estimate alone overstates precision.
4. **Step 2/3b eval-half specificities sit at 0.0–0.09** with wide CIs; any 'specificity unchanged' statement must carry its interval.
5. **The blinded review has not run yet.** The sentence 'visually indistinguishable' is pre-registered to flip if the reviewer hits ≥12/15; do not let the draft harden before the answer lands.
5b. **Capture modality is confounded with class** (Step 3e, quantified): the true-MEL comparator cohort is dermoscopic while both AK cohorts are clinical photos, and the concentration statistic tracks modality, not class (direction reverses at matched modality). The paper carries this as a limitation; do NOT restore class-level 'AK diffuse / MEL focal' phrasing anywhere.
6. **Balanced accuracy 'not converged'** (67.8–73.7% across protocols): the paper already says retraining is ongoing — keep any final-performance claim tied to that caveat.
7. **Two different bases for AK shares**: 63.3% top-contributor is full-panel benign DDI; the whack-a-mole migration shares are eval-half benign only. Never mix them in one sentence.
8. **`files_claude/` dump files and the SUPERSEDED log** are retired; nothing in them may be quoted (the audit now fails the build if the old filename reappears).
9. **RESOLVED cross-check:** `threshold_config.json` (sens 0.9603, spec 0.5543) matches the audited cache at the same operating point within 0.1 pp (fresh: 0.9593 / 0.5552); the paper's "96.0% / 55.4%" figures are safe to quote. An apparent specificity discrepancy was a bug in an early draft of this fact sheet's own computation, not in the config.

*Generated by emit_factsheet_numbers.py in 1.2s.*