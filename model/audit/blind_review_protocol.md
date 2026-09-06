# Expert review protocol — blinded Grad-CAM overlay comparison

**Purpose.** Independently verify (or refute) the Step 3d visual finding that
AK-driven false positives on DDI have activation patterns indistinguishable
from true-AK activations, and clearly different from MEL activations — and
tally capture artifacts on the FP cohort without cohort knowledge.

**Materials.** `blind_review_gradcam.html` (15 trials, images embedded, no
cohort labels). The reviewer must not open `model/audit/blind_review_key.csv`,
`review_gradcam_step3c.html`, or `figures/gradcam_step3c/`.

**Reviewer requirements.** One person with dermatology or dermatopathology
familiarity (or, minimum, a second author who did not perform the original
pass). ~15 minutes, one sitting.

**Tasks per trial.** (1) Forced choice: does the target's activation pattern
resemble candidate A or B? plus confidence 1–5. (2) Artifact checklist on the
target: ruler/paper card, surgical ink, printed text/device UI, dark
vignette/black surround, bubbles/streaks/glare, none.

**Analysis (study lead, after forms are collected).**
- Chance is 50%. Binomial test over 15 trials; ≥12/15 correct (p≈0.035) is
  the pre-declared threshold for "distinguishable."
- If reviewers *cannot* beat chance → the Step 3d claim stands as written.
- If they can → locate which features drive it (notes column) and soften the
  mechanism paragraph accordingly.
- Artifact tallies vs. key: agreement rates per artifact class; the "8/15
  rulers" figure becomes reviewer-confirmed only if a majority of reviewer
  ruler calls match the key.

**Pre-registration note.** This protocol and the trial key (seed 42) were
fixed before any review was conducted; do not regenerate the HTML before
collection, or the randomization changes.

---

## Message to send with the file (copy-paste)

> Subject: 15-minute blinded image review for a paper on skin-lesion AI
> fairness
>
> Hi — would you have ~15 minutes for a quick blinded review? It is part of
> a fairness audit of a skin-lesion classifier we are preparing for
> submission. The attached HTML file contains 15 short trials: for each you
> pick which of two candidate images has an attention pattern most like a
> target image, and tick any capture artifacts (rulers, ink markings, etc.)
> you can see. No medical judgment is needed — it is purely visual pattern
> matching, and it is fully blinded (neither of us knows which image is
> which until afterwards).
>
> Two small asks: (1) open it in a normal browser (Chrome/Edge/Firefox), it
> is a single self-contained file — nothing is uploaded anywhere; (2) do all
> 15 trials in one sitting without scrolling back to earlier ones. At the
> end there is a button that generates a text block of your answers — just
> copy-paste that back to me.
>
> Context you should NOT look up beforehand: the images come from two
> different dermatology photo cohorts; that is all I can say until after
> you submit. Happy to unblind and walk you through the full study
> afterwards — and you will be acknowledged in the paper.

---

*Based on: `figures/gradcam_step3c/` (45 overlays), `audit/step3d_placement.csv`,
`audit/audit_step3d_report.json`; original unblinded pass by the authors,
2026-09-06.*
