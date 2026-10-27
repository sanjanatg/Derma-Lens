"""
pptx_update_step3.py — one-off slide-9 number refresh (reproducible)
=====================================================================
Updates superseded numbers in ppt_files/DermaLens_Audit.pptx:
  0.24 -> 0.25 (calibrated threshold)
  51.6% -> 55.4% (specificity at the calibrated threshold, test panel)
  0.9185 -> 0.9239 (test AUC; matches threshold_config.json)
  "79% reduction" -> computed from the cache (currently 77%)
  FNR parenthetical -> recomputed per-malignant FN rates at 0.5 / 0.25
Scans every slide for other stale numbers and reports what it finds.
Writes a backup next to the deck before modifying.
"""

import re
import shutil
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DECK = ROOT.parent / "ppt_files" / "DermaLens_Audit.pptx"
BACKUP = DECK.with_name("DermaLens_Audit_pre-step3_backup.pptx")

STALE_MAP = [
    ("0.9185", "0.9239"),
    ("51.6%", "55.4%"),
    (">0.24<", ">0.25<"),
    ("(0.24)", "(0.25)"),
    # slide 5: superseded Aug-15 DDI operating point -> current checkpoint
    ("AUC 0.640", "AUC 0.585"),
    ("Sens 77.2%", "Sens 81.9%"),
    ("Spec 40.4%", "Spec 24.1%"),
    ("25.26%", "25.96%"),
]


def fnr_sentences():
    z = np.load(ROOT / "audit" / "preds_cache.npz")
    p = z["p_test"].astype(float)
    y = z["y_test"].astype(int)
    pos = y == 1
    fn05 = (1.0 - float((p[pos] >= 0.5).mean())) * 100.0
    fn025 = (1.0 - float((p[pos] >= 0.25).mean())) * 100.0
    reduction = 100.0 * (1.0 - fn025 / fn05)
    return fn05, fn025, reduction


def main():
    if not BACKUP.exists():
        shutil.copy2(DECK, BACKUP)
        print(f"backup written: {BACKUP.name}")

    fn05, fn025, red = fnr_sentences()
    print(f"cache-recomputed: per-malignant FN rate {fn05:.2f}% (thr 0.5) -> "
          f"{fn025:.2f}% (thr 0.25) = {red:.0f}% reduction")

    zin = zipfile.ZipFile(DECK)
    entries = {n: zin.read(n) for n in zin.namelist()}
    zin.close()

    changed = {}
    for name in list(entries):
        if not re.match(r"ppt/slides/slide\d+\.xml$", name):
            continue
        xml = entries[name].decode("utf-8")
        orig = xml
        hits = []
        for old, new in STALE_MAP:
            if old in xml:
                xml = xml.replace(old, new)
                hits.append(f"{old}->{new}")
        # FNR parenthetical (unicode arrow may appear literally or escaped)
        pat = (r"false-negative rate [0-9.]+(?:\s*(?:&#8594;|→|\u2192)\s*)"
               r"[0-9.]+ per 1000")
        new_fnr = f"false-negative rate {fn05:.1f}% → {fn025:.2f}% per malignant case"
        xml, n = re.subn(pat, new_fnr, xml)
        if n:
            hits.append(f"FNR parenthetical x{n}")
        # stale '79% reduction' style claim
        xml, n = re.subn(r"(\d+)% reduction in missed malignant cases",
                         f"{red:.0f}% reduction in missed malignant cases", xml)
        if n:
            hits.append(f"reduction-claim -> {red:.0f}% x{n}")
        if xml != orig:
            entries[name] = xml.encode("utf-8")
            changed[name] = hits

    for name, hits in changed.items():
        print(f"  {name}: {', '.join(hits)}")
    if not changed:
        print("  nothing to change")

    # remaining stale-number scan across all slides
    print("\n  remaining stale-number scan (all slides):")
    for name, blob in entries.items():
        if not re.match(r"ppt/slides/slide\d+\.xml$", name):
            continue
        txt = re.sub(r"<[^>]+>", " ", blob.decode("utf-8"))
        for needle in ("0.6398", "0.640 ", "25.26", "0.9185", "51.6",
                       "39.7", "75.0", "0.24 ", "(0.24)", "77.2", "40.4"):
            if needle in txt:
                i = txt.find(needle)
                print(f"  !! {name} still contains '{needle}': "
                      f"...{txt[max(0, i - 60):i + 60].strip()}...")

    with zipfile.ZipFile(DECK, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, blob in entries.items():
            zout.writestr(name, blob)
    print(f"\nrewrote {DECK.name} ({len(changed)} slide(s) changed)")


if __name__ == "__main__":
    main()
