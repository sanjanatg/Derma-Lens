"""
audit_make_blind_review.py — blinded review package for the overlay pass
=========================================================================
Builds a cohort-blind HTML experiment so an independent reviewer can test the
Step 3d claim that FP-DDI-AK overlays are visually indistinguishable from
true-AK overlays (and clearly different from MEL):

  15 trials x [Target | A | B]:
    Target  = an fp_DDI_AK overlay (unlabeled)
    {A, B}  = {one true_AK overlay, one true_MEL overlay}, order shuffled
  Task 1 (forced choice): which candidate matches the target's activation?
  Task 2 (artifact tally on the Target): ruler / surgical ink / device text /
          vignette / other — checkboxes, one free-text box.
  Answer key written separately (blind_review_key.csv) so the reviewer can
  be genuinely blind.

Outputs: ../blind_review_gradcam.html, audit/blind_review_key.csv
Usage:   python audit_make_blind_review.py
"""

import base64
import csv
import random
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "audit"
FIG3C = ROOT / "figures" / "gradcam_step3c"
OUT_HTML = ROOT.parent / "blind_review_gradcam.html"
SEED = 42
N_TRIALS = 15
COHORTS = ("fp_DDI_AK", "true_AK", "true_MEL")


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def overlay_path(cohort, rank, prob):
    return FIG3C / f"{cohort}_{rank:02d}_p{prob:.2f}.jpg"


def main():
    df = pd.read_csv(AUDIT / "step3c_cam_comparators.csv")
    by = {c: df[df["cohort"] == c].sort_values("rank").to_dict("records")
          for c in COHORTS}
    fp, ak, mel = by["fp_DDI_AK"], by["true_AK"], by["true_MEL"]
    rng = random.Random(SEED)

    trials, key = [], []
    # balance the correct side (guard against position-gaming):
    # 8x A, 7x B in random order, then place true_AK accordingly
    sides = ["A"] * 8 + ["B"] * 7
    rng.shuffle(sides)
    for i in range(N_TRIALS):
        f, a, m = fp[i], ak[i], mel[i]
        ak_side = sides[i]
        pair = [("true_AK", a, overlay_path("true_AK", a["rank"], a["prob"])),
                ("true_MEL", m, overlay_path("true_MEL", m["rank"], m["prob"]))]
        if ak_side == "B":
            pair.reverse()
        trials.append({
            "target_b64": b64(overlay_path("fp_DDI_AK", f["rank"], f["prob"])),
            "a_b64": b64(pair[0][2]), "b_b64": b64(pair[1][2]),
        })
        key.append({
            "trial": i + 1,
            "target_cohort": "fp_DDI_AK",
            "target_prob": f["prob"],
            "a_cohort": pair[0][0], "a_prob": pair[0][1]["prob"],
            "b_cohort": pair[1][0], "b_prob": pair[1][1]["prob"],
            "match_key": ak_side,
        })

    art = ("Ruler or paper card in frame", "Surgical ink / pen markings",
           "Printed text or device UI", "Dark vignette / black surround",
           "Bubbles, streaks or glare", "None of the above")

    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>Blinded overlay review</title><style>",
             "body{font-family:Segoe UI,Arial;margin:18px;background:#fafafa;"
             "color:#111;max-width:1050px}",
             "h1{font-size:22px} h2{font-size:16px;margin:22px 0 8px}",
             ".trial{border:1px solid #ccc;border-radius:8px;padding:12px;"
             "margin:14px 0;background:#fff}",
             "img{width:250px;height:250px;object-fit:cover;border:1px "
             "solid #888;border-radius:4px}",
             "table{border-collapse:collapse} td{padding:4px 10px;"
             "text-align:center;font-size:13px}",
             "label{margin-right:14px;white-space:nowrap}",
             ".art label{white-space:normal;margin-right:10px}",
             "textarea{width:95%;height:26px}",
             ".rules li{margin:4px 0}</style></head><body>",
             "<h1>Blinded Grad-CAM overlay review — 15 trials</h1>",
             "<div class='rules'><b>What you are looking at.</b> Each image "
             "is a skin-lesion photograph with a green attention overlay "
             "(Grad-CAM for one diagnostic head of the classifier; green "
             "intensity = activation).<ul>",
             "<li><b>Target</b>: one fixed image. <b>A</b> and <b>B</b>: two "
             "candidate images, one drawn from each of two cohorts.</li>",
             "<li><b>Task 1</b>: judge which candidate's activation pattern "
             "best matches the target's (forced choice, then confidence).</li>",
             "<li><b>Task 2</b>: for the <i>target</i> only, tick every "
             "capture artifact you can see.</li>",
             "<li>Please complete all 15 trials in one sitting; do not "
             "reload between trials. Time: ~15 minutes.</li></ul></div>"]
    for i, t in enumerate(trials, 1):
        opts = "".join(
            f"<label><input type='radio' name='t{i}match' value='{v}'>"
            f"{v}</label>" for v in ("A", "B"))
        conf = "".join(
            f"<label><input type='radio' name='t{i}conf' value='{v}'>"
            f"{v}</label>" for v in ("1", "2", "3", "4", "5"))
        checks = "".join(
            f"<label><input type='checkbox' name='t{i}art' value='{a}'>"
            f"{a}</label>" for a in art)
        parts.append(
            f"<div class='trial'><h2>Trial {i} of {N_TRIALS}</h2><table><tr>"
            f"<td><img src='data:image/jpeg;base64,{t['target_b64']}'>"
            f"<br><b>Target</b></td>"
            f"<td><img src='data:image/jpeg;base64,{t['a_b64']}'><br>A</td>"
            f"<td><img src='data:image/jpeg;base64,{t['b_b64']}'><br>B</td>"
            f"</tr></table><p><b>Task 1.</b> The target's activation pattern "
            f"most resembles: {opts} &nbsp; Confidence (1=guess, 5=certain): "
            f"{conf}</p>"
            f"<p class='art'><b>Task 2.</b> Visible in the <i>target</i> "
            f"(tick all): {checks}</p>"
            f"<p>Notes (optional): <textarea name='t{i}notes'></textarea></p>"
            f"</div>")
    parts.append("<p><i>Completed forms: save the page or export answers as "
                 "instructed by the study lead. The answer key is held "
                 "separately (model/audit/blind_review_key.csv).</i></p>")
    parts.append("</body></html>")
    OUT_HTML.write_text("\n".join(parts), encoding="utf-8")

    with open(AUDIT / "blind_review_key.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0].keys()))
        w.writeheader()
        w.writerows(key)
    print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {AUDIT / 'blind_review_key.csv'}")
    print(f"trial match keys: "
          f"{[k['match_key'] for k in key]} (seed {SEED})")


if __name__ == "__main__":
    main()
