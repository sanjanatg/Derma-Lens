"""
make_paper_figures.py — build the two paper figures from audited artifacts
==========================================================================
Fig. 1  figs/fig_roc_ceilings.png : per-group ROC ceiling — sensitivity at
        specificity >= 0.6 and >= 0.8 — from audit/step3_roc_ceilings.csv
        (every number read from the CSV; AUC + CI annotated under each group).
Fig. 2  figs/fig_cam_grid.png     : 3x5 Grad-CAM overlay grid assembled from
        figures/gradcam_step3c/. Row 1 = the five DDI AK false positives that
        carry capture artifacts per the authors' key (rulers/ink: trials
        01/07/08/10/12 of 15; 8/15 overall carry rulers — the caption must
        state this selection). Rows 2-3 = the five highest-score true AK and
        true melanoma comparators.

Outputs land in ../ppt_files/figs/ next to dermalens_paper.tex, so
\\includegraphics{figs/<name>} resolves when compiling from ppt_files.

Run:  python make_paper_figures.py
"""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "audit"
STEP3C = ROOT / "figures" / "gradcam_step3c"
OUT = ROOT.parent / "ppt_files" / "figs"

GROUP_LABEL = {
    "FST I-II": "FST I\u2013II",
    "FST III-IV": "FST III\u2013IV",
    "FST V-VI": "FST V\u2013VI",
    "ALL": "All",
}


def _font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------
# Fig. 1 — ROC ceilings (grouped bars from the CSV)
# --------------------------------------------------------------------------
def fig_roc_ceilings(out_png):
    with open(AUDIT / "step3_roc_ceilings.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    groups = [GROUP_LABEL[r["group"]] for r in rows]
    s60 = [float(r["sens_at_spec0.6"]) for r in rows]
    s80 = [float(r["sens_at_spec0.8"]) for r in rows]
    aucs = [(float(r["auc"]), float(r["auc_lo"]), float(r["auc_hi"]))
            for r in rows]

    x = range(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 2.1), dpi=300)
    b1 = ax.bar([i - w / 2 for i in x], s60, w, color="#2c6fbb",
                label="sensitivity at specificity $\\geq$ 0.6")
    b2 = ax.bar([i + w / 2 for i in x], s80, w, color="#9dbfdd",
                label="sensitivity at specificity $\\geq$ 0.8")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{100 * b.get_height():.0f}%", ha="center", fontsize=6.5)

    tick = [f"{g}\nAUC {a:.2f} ({lo:.2f}\u2013{hi:.2f})"
            for g, (a, lo, hi) in zip(groups, aucs)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(tick, fontsize=7)
    ax.set_ylabel("Sensitivity (malignant)", fontsize=8)
    ax.set_ylim(0, 0.78)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper right",
              handlelength=1.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_png)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. 2 — Grad-CAM triad grid (overlay-only tiles from step3c)
# --------------------------------------------------------------------------
ROW1_IDS = ["01", "07", "08", "10", "12"]   # artifact-bearing FPs (authors' key)
ROW_LABELS = {
    "fp_DDI_AK": "Row 1 \u2014 AK-driven DDI false positives (AK-column CAM); "
                 "five artifact-bearing cases of 15",
    "true_AK": "Row 2 \u2014 True AK, in-domain test set (AK-column CAM); "
               "five highest-score of 15",
    "true_MEL": "Row 3 \u2014 True melanoma, in-domain test set (MEL-column CAM); "
                "five highest-score of 15",
}
TILE_W, TILE_H, CAP_H, ROWBAR_H, SEP, GUT = 460, 364, 22, 30, 4, 8


def _tile(path):
    p = Path(path)
    im = Image.open(p).convert("RGB").resize((TILE_W, TILE_H), Image.LANCZOS)
    m = re.match(r".*_(\d+)_p(0\.\d+)\.jpg$", p.name)
    label = f"#{m.group(1)}  p={m.group(2)}" if m else p.name
    bar = Image.new("RGB", (TILE_W, CAP_H), "white")
    d = ImageDraw.Draw(bar)
    d.text((4, 3), label, fill="black", font=_font(15))
    out = Image.new("RGB", (TILE_W, CAP_H + TILE_H), "white")
    out.paste(bar, (0, 0))
    out.paste(im, (0, CAP_H))
    return out


def _cohort_files(cohort, ids):
    files = []
    for i in ids:
        hits = sorted(STEP3C.glob(f"{cohort}_{i}_p*.jpg"))
        if not hits:
            raise FileNotFoundError(f"{cohort}_{i}_p*.jpg not found in {STEP3C}")
        files.append(hits[0])
    return files


def fig_cam_grid(out_png):
    rows = [("fp_DDI_AK", ROW1_IDS), ("true_AK", [f"{i:02d}" for i in range(1, 6)]),
            ("true_MEL", [f"{i:02d}" for i in range(1, 6)])]
    n_cols = len(ROW1_IDS)
    W = n_cols * TILE_W + (n_cols - 1) * SEP + 2 * GUT
    H = len(rows) * (ROWBAR_H + CAP_H + TILE_H) + (len(rows) - 1) * (SEP + 6) + GUT
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    y = GUT
    for cohort, ids in rows:
        draw.text((GUT, y + 6), ROW_LABELS[cohort], fill="black", font=_font(17))
        y += ROWBAR_H
        for j, f in enumerate(_cohort_files(cohort, ids)):
            sheet.paste(_tile(f), (GUT + j * (TILE_W + SEP), y))
        y += CAP_H + TILE_H + SEP + 6
    sheet.save(out_png)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_roc_ceilings(OUT / "fig_roc_ceilings.png")
    print(f"wrote {OUT / 'fig_roc_ceilings.png'}")
    fig_cam_grid(OUT / "fig_cam_grid.png")
    print(f"wrote {OUT / 'fig_cam_grid.png'}")
    from PIL import Image as I
    for p in sorted(OUT.glob("*.png")):
        w, h = I.open(p).size
        print(f"  {p.name}: {w}x{h} px ({w / h:.2f}:1), "
              f"{p.stat().st_size / 1e6:.2f} MB")
