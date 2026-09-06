# Results chapter draft — Step 3/3b: generalization failure analysis

*All numbers from `audit/audit_step3_report.json`, `audit/audit_step3b_report.json`
and their CSVs. CIs are (group × label)-stratified percentile bootstrap, 2,000
replications, seed 42; raw n accompanies every estimate. Predictions from the
locked checkpoint (`best.pth`, 2026-08-20) via the audited cache
(`preds_cache.npz`, drift gate 1.19e-07); provenance gates re-verified at each run.*

## Proposed section X.Y — Score-level analysis of the out-of-domain failure

### X.Y.1 No operating point exists: per-group ROC ceilings

Threshold equalization (Section X.X) showed that no per-group threshold policy
closes the DDI sensitivity gap. Score-level analysis explains why: the malignant
probability itself barely separates benign from malignant lesions on DDI, and
the deficit is worst exactly where the fairness gap is largest.

| Group | n (pos/neg) | AUC [95% CI] | Best sens @ spec ≥ 0.6 | Best sens @ spec ≥ 0.8 |
|---|---|---|---|---|
| FST I–II   | 208 (49/159)  | 0.574 [0.486, 0.667] | 0.490 | 0.224 |
| FST III–IV | 241 (74/167)  | 0.668 [0.592, 0.743] | 0.662 | 0.392 |
| FST V–VI   | 207 (48/159)  | **0.484 [0.382, 0.585]** | 0.396 | 0.229 |
| All        | 656 (171/485) | 0.585 [0.548, 0.634] | 0.503 | 0.357 |

For FST V–VI the 95% CI on AUC includes 0.5: the model's malignant score carries
**no measurable benign/malignant ranking signal for dark-skin images**. This
formalizes the Step 2 negative result — the disparity lives in the score
distributions, not in threshold placement — and no post-hoc decision rule can
recover performance the scores do not contain.

### X.Y.2 The false-positive mechanism: a spurious actinic-keratosis response

Decomposing the malignant score into its four class components shows the
false-positive mass is **concentrated in a single column**: on true-benign DDI
images, the actinic-keratosis (AK) column is the largest malignant contributor
on 63.3% of images (mean probability 0.375, 90th percentile 0.803). All 15
highest-scoring false positives have AK as their top contributor (per-image AK
probability 0.91–0.97; `step3_review_cases.csv`).

The same AK column is nearly silent on benign in-domain images: mean 0.046
(90th percentile 0.070) on the test panel — an order-of-magnitude distribution
shift (one-sided KS = 0.77, p < 1e-200). Grad-CAM analysis with matched
comparators (`audit_step3c.py`; overlays in `figures/gradcam_step3b/` and
`figures/gradcam_step3c/`) bounds what attention maps can and cannot
establish. The AK head's activation is **diffuse in both settings** — median
top-1% CAM mass share 0.023 on the DDI false positives vs 0.019 on true AK
cases (Mann–Whitney p = 0.12; true-MEL comparator 0.025) — so diffuseness is
a property of the head, not a signature of the spurious response. The AK map
also leans toward darker pixels in both settings (median CAM–luminance
correlation −0.29 on DDI false positives, −0.21 on true AK, vs +0.11 for the
MEL column on true melanomas). We accordingly do **not** claim attention-map
morphology distinguishes the spurious DDI response from genuine AK detection:
the evidence for spuriousness is the score-level distribution shift above,
together with its tone-invariance (Section X.Y.3).

**Structured visual review (Step 3d, completed).** All 45 overlays were
reviewed side-by-side in triads (`review_gradcam_step3c.html`; placement
stats in `step3d_placement.csv`, Step 0 gates re-verified at run time). The
review supports the field-wide reading and rejects the capture-cue focal
hypothesis: on the false positives the activation spreads across the skin
field and does *not* focal-lock onto salient non-lesion objects. Paper
rulers are in frame in 8/15 false-positive images (most also carry surgical
ink markings), yet neither attracts elevated CAM mass (edge-band mass share
median 0.410 vs 0.377 on true AK, Mann–Whitney p = 0.28; hot-pixel luminance
ratio 0.95 vs 0.96, p = 0.93). Conversely, the in-domain melanoma
comparators carry prominent capture artifacts of their own (dermoscope
surrounds with black corners in 13/15; 9/15 with corner/center luminance
< 0.75, vs 1/15 false positives) while the MEL head remains lesion-localized
(edge-band share 0.244). Capture artifacts are therefore *present* in the
false-positive cohort as expected, but the activation is not localized on
them: the artifact census is context for the domain-shift finding, not its
cause, and the spuriousness claim rests on the score-level evidence.

### X.Y.3 Domain shift, not skin tone, drives the artifact

The AK artifact is a **domain effect, not a tone effect**:

- **Within DDI**, benign-image AK probability is nearly flat across tone groups
  and *highest* for the lightest-tone group (FST I–II 0.440, V–VI 0.371,
  III–IV 0.318) — the opposite of a monotone tone gradient.
- **At identical self-reported tone** (FST I–II), DDI benigns show AK mean 0.440
  vs test-panel 0.043 (KS = 0.78, p = 8×10⁻⁸¹).

We therefore frame the observed DDI disparity as a **domain-generalization
failure that disproportionately lands on the group with the weakest underlying
ranking signal**, with the capture-domain artifact and the tone-related
performance gap *compounding* rather than sharing a single cause.

### X.Y.4 A post-hoc correction cannot repair the score (negative result)

A post-hoc down-weighting of the AK column, p′ = p_mal − λ·P_AK with λ and the
decision threshold fitted on one stratified half of DDI and evaluated once on
the held-out half, yields at most marginal gains: tune-half AUC peaks at
λ = 0.25 (0.609 vs 0.596 baseline, +0.013) and *degrades monotonically beyond*
λ = 0.30 (0.524 at λ = 1.0). At λ* = 0.25 the eval-half specificity is
statistically unchanged (0.025 → 0.025 overall; per-group rows with CIs in
`step3b_ak_recalibration.csv`) and the EO sensitivity gap moves from
0.125 [0.000, 0.250] to 0.083 [0.000, 0.208]. Aggressive deflation (λ = 0.8)
destroys in-domain AK detection (AK-class recall at ~80% specificity:
0.955 → 0.444) while still not fixing DDI specificity. The failure is
structural: on eval-half benign images, deflating AK merely migrates the
dominant malignant contributor to other columns (argmax share AK 62.3% →
28.3%, BCC 15.6% → 29.9%, MEL 16.4% → 23.0% at λ = 0.8) — single-column
recalibration is whack-a-mole when the shared representation maps the capture
domain itself onto malignancy.

**Interpretation.** The AK column is *symptomatic* — deflating it cannot
restore ranking power that the shared representation never transferred. The
deficit is representational: closing it requires in-domain training signal
(e.g., fine-tuning with DDI-style images including hard negatives, and
tone/capture augmentation), not score- or threshold-level correction.

### X.Y.5 Limitations

DDI's coarse tone buckets and panel size (48–74 positives per group) yield wide
CIs; the test panel cannot support per-tone specificity for FST V (n = 3) or VI
(n = 0); Grad-CAM mechanism claims are correlational by nature, and the visual
overlay review was performed by the authors rather than an independent
dermatologist, so the ruler/artifact counts are manual tallies pending expert
confirmation; the AK-column decomposition reflects the model's learned
representation, not ground-truth dermatopathology.

**Open question (flagged, not resolved).** The AK head's activation leans
toward darker-than-scene pixels in *both* settings (median CAM–luminance
correlation −0.29 on DDI false positives, −0.21 on true AK cases, vs +0.11
for the MEL column on true melanomas). Two explanations are compatible with
current evidence: (i) a genuine (if crude) clinical signal — actinic damage
correlates with sun-exposure appearance in the training distribution — or
(ii) a confound with lesion-border darkness/shadowing unrelated to skin tone.
Disentangling them requires pixel-level attribution and is explicitly left
open; the darker-pixel observation should not be read as a bias claim about
the model's behavior across skin tones.

---

### Suggested one-line summary for the abstract

> On the out-of-domain DDI panel the model's malignancy score carries no
> measurable ranking signal for FST V–VI images (AUC 0.484 [0.382, 0.585]);
> the failure traces to a spurious, capture-associated actinic-keratosis
> response that is invariant to skin tone within DDI and cannot be corrected
> post hoc, indicating a representational domain gap rather than a
> threshold-calibration failure.

---

*Files backing this draft: `step3_score_decomposition.csv`,
`step3_roc_ceilings.csv`, `step3_distribution_shift.csv`,
`step3_review_cases.csv`, `step3b_ak_recalibration.csv`,
`step3c_cam_comparators.csv`, `step3d_placement.csv`,
`audit/audit_step3d_report.json`, `figures/gradcam_step3b/` (15 overlays),
`figures/gradcam_step3c/` (45 comparator overlays),
`figures/gradcam_step3d/` (per-cohort contact sheets),
`review_gradcam_step3c.html` (triad review page; base64-embedded).*
