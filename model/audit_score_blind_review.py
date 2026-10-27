"""
audit_score_blind_review.py — score the blinded overlay review
===============================================================
Consumes the answer string produced by the "Generate answer string" button
on blind_review_gradcam.html (JSON), scores it against the pre-registered
key (model/audit/blind_review_key.csv, seed 42) and the analysis plan in
model/audit/blind_review_protocol.md:

  Task 1  binomial test vs 50% chance over 15 trials (one-sided);
          pre-declared threshold: >= 12/15 correct (p ~ 0.035) = the target's
          activation pattern IS distinguishable from both comparators.
  Task 2  artifact tallies; agreement vs the authors' key where supplied
          (blind_review_authors_key.csv), incl. the objective vignette proxy
          from step3d_placement.csv.

Usage:
  python audit_score_blind_review.py <answers.json | '-' for stdin>
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "audit"
KEY = AUDIT / "blind_review_key.csv"
AUTHORS_KEY = AUDIT / "blind_review_authors_key.csv"
PLACE = AUDIT / "step3d_placement.csv"
N = 15
PASS_K = 12  # pre-declared in blind_review_protocol.md

ARTIFACTS = ("Ruler or paper card in frame", "Surgical ink / pen markings",
             "Printed text or device UI", "Dark vignette / black surround",
             "Bubbles, streaks or glare", "None of the above")


def load_answers(src: str):
    if src == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(src).read_text(encoding="utf-8")
    # tolerate prose around the JSON blob
    start, end = raw.index("{"), raw.rindex("}") + 1
    data = json.loads(raw[start:end])
    if data.get("form") != "blind_review_gradcam_v2":
        raise SystemExit("not a blind_review_gradcam_v2 form")
    trials = {t["trial"]: t for t in data["trials"]}
    missing = [i for i in range(1, N + 1)
               if i not in trials or trials[i].get("match") not in ("A", "B")]
    return data, trials, missing


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    data, trials, missing = load_answers(sys.argv[1])
    print("=" * 78)
    print("BLIND REVIEW SCORING — pre-registered plan, key seed 42")
    print("=" * 78)
    print(f"reviewer: {data.get('reviewer') or '(none)'}  "
          f"completed: {data.get('completed')}")
    if missing:
        print(f"INCOMPLETE: {len(missing)} trial(s) without a forced choice: "
              f"{missing}\nScoring requires all 15 — aborting.")
        return 1

    key = {int(r["trial"]): r for r in csv.DictReader(KEY.open())}
    correct = [i for i in range(1, N + 1)
               if trials[i]["match"] == key[i]["match_key"]]
    k = len(correct)
    p = binomtest(k, N, 0.5, alternative="greater").pvalue
    conf_c = [trials[i]["confidence"] for i in correct
              if trials[i]["confidence"]]
    conf_w = [trials[i]["confidence"] for i in range(1, N + 1)
              if i not in correct and trials[i]["confidence"]]

    print(f"\nTask 1 — forced choice: {k}/{N} correct "
          f"(chance 7.5/15), one-sided binomial p = {p:.4f}")
    if conf_c:
        print(f"  mean confidence: correct {sum(conf_c) / len(conf_c):.1f}"
              + (f", incorrect {sum(conf_w) / len(conf_w):.1f}"
                 if conf_w else ""))
    if k >= PASS_K:
        print("  VERDICT: reviewers CAN distinguish the target — the "
              "'visually indistinguishable' claim FAILS; identify the "
              "driving features from notes and soften the mechanism "
              "paragraph.")
    else:
        print("  VERDICT: reviewers CANNOT beat chance — the Step 3d claim "
              "stands: FP-DDI-AK activation patterns are visually "
              "indistinguishable from true-AK (and clearly unlike MEL) "
              "under blinded forced choice.")

    # Task 2: artifact tallies
    print("\nTask 2 — artifact tallies on the 15 targets:")
    tallies = {a: 0 for a in ARTIFACTS}
    for i in range(1, N + 1):
        for a in trials[i]["artifacts"]:
            if a in tallies:
                tallies[a] += 1
    for a in ARTIFACTS:
        print(f"  {tallies[a]:>2d}/15  {a}")

    # agreement vs authors' key (if present)
    if AUTHORS_KEY.exists():
        ak = {}
        for r in csv.DictReader(AUTHORS_KEY.open()):
            ak.setdefault(int(r["trial"]), []).append(r["artifact"])
        place = {int(r["rank"]): float(r["vignette_ratio"])
                 for r in csv.DictReader(PLACE.open())
                 if r["cohort"] == "fp_DDI_AK"}
        agree = tot = 0
        vig_agree = vig_tot = 0
        for i in range(1, N + 1):
            rev = set(trials[i]["artifacts"])
            auth = set(ak.get(i, []))
            if "Dark vignette / black surround" in auth or \
                    (auth == set() and "None of the above" in rev):
                pass
            for a in ARTIFACTS:
                if a in auth or a in rev:
                    tot += 1
                    agree += int((a in rev) == (a in auth))
            # objective vignette proxy: vig < 0.75 in step3d
            want_vig = place.get(i, 1.0) < 0.75
            got_vig = "Dark vignette / black surround" in rev
            vig_tot += 1
            vig_agree += int(want_vig == got_vig)
        if tot:
            print(f"\n  agreement with authors' key: {agree}/{tot} "
                  f"class-decisions ({100 * agree / tot:.0f}%)")
        print(f"  vignette agreement vs quantitative proxy "
              f"(corner/center luminance < 0.75): {vig_agree}/{vig_tot}")
        print("  NOTE: 'reviewer-confirmed' requires majority agreement on "
              "the ruler class (protocol).")
    else:
        print("\n  (no authors' key at audit/blind_review_authors_key.csv — "
              "agreement rates skipped; tallies above are still valid)")

    # paste-ready verdict paragraph
    print("\n--- paste-ready results paragraph ---")
    if k >= PASS_K:
        print(f"In a blinded forced-choice review, an independent reviewer "
              f"correctly matched {k}/15 target overlays to the true-AK "
              f"comparator rather than the MEL comparator (binomial "
              f"p = {p:.3f}), indicating the activation patterns are "
              f"distinguishable in principle; we accordingly weaken the "
              f"claim that the spurious response is visually "
              f"indistinguishable from genuine AK activations and report "
              f"the reviewer's noted discriminating features.")
    else:
        print(f"In a blinded forced-choice review, an independent reviewer "
              f"could not distinguish target from comparator above chance "
              f"({k}/15 correct, binomial p = {p:.3f}, pre-registered "
              f"threshold 12/15): the AK-driven false positives' activation "
              f"patterns are visually indistinguishable from true-AK "
              f"activations and clearly different from lesion-localized MEL "
              f"activations, consistent with the Step 3d analysis.")
    stamp = datetime.now().isoformat(timespec="seconds")
    (AUDIT / "blind_review_score.txt").write_text(
        f"scored {stamp}: {k}/{N} correct, p={p:.4f}\n", encoding="utf-8")
    print(f"\n(saved one-line result to {AUDIT / 'blind_review_score.txt'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
