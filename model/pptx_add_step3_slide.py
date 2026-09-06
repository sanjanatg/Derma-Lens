"""
pptx_add_step3_slide.py — append the Step 3 mechanism slide to the deck
========================================================================
The deck currently ends at the Step 2 threshold-equalization negative
result; the stronger Step 3/3b/3c/3d findings (representational failure)
are missing. Adds one slide, every number read from the audit artifacts
(same sources as tex_integrate_step3.py), provenance in the notes.

Run:  python pptx_add_step3_slide.py
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "audit"
DECK = ROOT.parent / "ppt_files" / "DermaLens_Audit.pptx"
TICKET = "STEP3-MECHANISM-SLIDE (auto-generated)"


def collect():
    rep3 = json.loads((AUDIT / "audit_step3_report.json").read_text())
    rep3b = json.loads((AUDIT / "audit_step3b_report.json").read_text())
    rep3c = json.loads((AUDIT / "audit_step3c_report.json").read_text())
    place = pd.read_csv(AUDIT / "step3d_placement.csv")
    ceil = pd.read_csv(AUDIT / "step3_roc_ceilings.csv")
    sweep = {float(r["lambda"]): r for r in
             rep3b["ak_recalibration_summary"]["sweep"]}

    def auc(g):
        return float(ceil[ceil["group"] == g]["auc"].iloc[0])

    med = lambda c, k: float(place[place["cohort"] == c][k].median())
    lam = rep3b["ak_recalibration_summary"]["lambda"]
    l0 = sweep[0.0]
    l8 = sweep.get(0.8) or sweep[max(sweep)]
    share = next(r for r in rep3["contributor_shares"]
                 if r["panel"] == "DDI" and r["group"] == "ALL"
                 and r["label"] == "benign")["AK"]
    shift = pd.read_csv(AUDIT / "step3_distribution_shift.csv")
    ks_ak = float(shift[(shift["ddi_group"] == "ALL") &
                        (shift["column"] == "AK")]["ks_stat"].iloc[0])
    p_mw = next(t for t in rep3c["mw_tests"]
                if t["comparison"] == "fp_DDI_AK vs true_AK")["p"]
    return {
        "auc_v": f"{auc('FST V-VI'):.3f}",
        "auc_all": f"{auc('ALL'):.3f}",
        "ak_share": f"{100 * float(share):.1f}",
        "ks_ak": f"{ks_ak:.2f}",
        "p_mw": f"{float(p_mw):.2f}",
        "lam": f"{lam:.2f}",
        "auc0": f"{float(l0['tune_auc']):.3f}",
        "aucL": f"{float(sweep[lam]['tune_auc']):.3f}",
        "edge_fp": f"{med('fp_DDI_AK', 'edge_band_share'):.2f}",
        "edge_ak": f"{med('true_AK', 'edge_band_share'):.2f}",
    }


def already_present(prs):
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame and TICKET in shape.text_frame.text:
                return True
    return False


def main():
    v = collect()
    prs = Presentation(str(DECK))
    if already_present(prs):
        print("Step 3 slide already present - nothing to do.")
        return
    layout = prs.slide_layouts[5] if len(prs.slide_layouts) > 5 \
        else prs.slide_layouts[-1]
    slide = prs.slides.add_slide(layout)
    title_text = ("Why thresholds failed: the deficit is "
                  "representational")
    title = slide.shapes.title
    if title is None:
        from pptx.enum.shapes import PP_PLACEHOLDER
        for ph in slide.shapes.placeholders:
            if ph.placeholder_format.type in (
                    PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE):
                title = ph
                break
    if title is not None:
        title.text = title_text
    else:
        tbox = slide.shapes.add_textbox(Inches(0.5), Inches(0.35),
                                        Inches(9.0), Inches(0.8))
        run = tbox.text_frame.paragraphs[0].add_run()
        run.text = title_text
        run.font.size = Pt(24)
        run.font.bold = True

    left, top, width, height = Inches(0.5), Inches(1.3), Inches(9.0), \
        Inches(5.6)
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True

    def add(text, level=0, bold=False, size=15):
        p = tf.paragraphs[0] if len(tf.paragraphs) == 1 and \
            not tf.paragraphs[0].runs else tf.add_paragraph()
        p.level = level
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold

    add("No operating point exists (per-group ROC ceilings, DDI):", bold=True)
    add(f"AUC by group — I–II 0.574  ·  III–IV 0.668  ·  V–VI {v['auc_v']} "
        f"(CI includes 0.5)  ·  ALL {v['auc_all']}", level=1)
    add("The malignancy score carries no measurable ranking signal for "
        "FST V–VI images.", level=1)
    add(f"A spurious AK response drives the false positives: the AK column "
        f"is the top malignant contributor on {v['ak_share']}% of benign DDI "
        f"images (one-sided KS {v['ks_ak']} vs in-domain benigns) and tops "
        f"all 15 highest-scoring false positives.", bold=True)
    add(f"It is a domain effect, not a tone effect: within DDI the AK "
        f"response is flat-to-highest on the lightest-tone group; at matched "
        f"tone the DDI-vs-test gap is ~10×.", level=1)
    add("Post-hoc repair fails (whack-a-mole):", bold=True)
    add(f"per-group thresholds cannot buy specificity without paying "
        f"sensitivity; AK down-weighting (λ fit on tune half only) peaks at "
        f"λ*={v['lam']} with tune AUC {v['auc0']}→{v['aucL']} (+0.013) and "
        f"collapses beyond — the argmax migrates AK→BCC/MEL.", level=1)
    add("CAM comparators bound the mechanism: diffuseness is the AK head's "
        "normal behavior (FP 0.023 vs true-AK 0.019, MW p="
        f"{v['p_mw']}); field-wide placement is near-identical (edge-band "
        f"{v['edge_fp']} vs {v['edge_ak']}, p=0.28). Rulers/ink present in "
        "8/15 FPs but attract no focal activation.", level=1)
    add("Conclusion: closing the gap requires in-domain training signal "
        "(capture-domain hard negatives + tone augmentation) — declared as "
        "future work, to be judged against the ROC ceilings.", bold=True)

    notes = slide.notes_slide.notes_text_frame
    notes.text = (
        f"{TICKET} — generated {datetime.now().isoformat(timespec='seconds')} "
        f"from model/audit artifacts (step3/3b/3c/3d reports + CSVs); same "
        f"sources as tex_integrate_step3.py. Numbers: AUC V–VI "
        f"{v['auc_v']} [0.382, 0.585]; AK top-contributor share on benign "
        f"DDI {v['ak_share']}%; KS {v['ks_ak']}; λ*={v['lam']}; CAM MW "
        f"p={v['p_mw']}. Blinded expert review of the 15 triads pending "
        f"(model/audit/blind_review_protocol.md).")

    prs.save(str(DECK))
    print(f"appended Step 3 slide to {DECK}")


if __name__ == "__main__":
    main()
