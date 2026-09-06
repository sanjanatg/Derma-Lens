"""
run_9b_9c.py — execute the fixed Cells 9b (DDI fairness) and 9c (threshold)
against the existing checkpoints/best.pth, without re-running training.

Runs the notebook's setup chain, then Cells 9 / 9b / 9c in order:
    index 2  (Cell 1: GPU check / CUDA_AVAILABLE)
    index 4  (Cell 2: config params)
    index 6  (Cell 3: imports)
    index 12 (Cell 5: splits + SkinLesionDataset + loaders)
    index 22 (Cell 9: test-set evaluation -> true_cls, pred_probs)
    index 24 (Cell 9b: DDI fairness eval)
    index 26 (Cell 9c: clinical threshold)

The DDI table is loaded from datasets/ddi_eval.csv (the kernel-restart
fallback path the notebook itself documents). Matplotlib runs headless (Agg).
"""
import os
import sys
import time
import json
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "dermalens_train.ipynb"
CHECKPOINTS = ROOT / "checkpoints"

CELLS = [2, 4, 6, 12, 22, 24, 26]


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    ns["sys"] = sys
    sys.path.insert(0, str(ROOT))

    # Shim: Cell 5's __getitem__ uses class_idx_to_binary, which the notebook
    # imports in the Cell 4 dataset-build cell (skipped here).
    exec("from model.class_taxonomy import class_idx_to_binary", ns)

    for idx in CELLS:
        cell = nb["cells"][idx]
        assert cell["cell_type"] == "code", f"cell {idx} is not code"
        src = "".join(cell["source"])
        banner = f"===== executing notebook cell #{idx} ====="
        print("\n" + "=" * 70, flush=True)
        print(banner, flush=True)
        print("=" * 70, flush=True)
        t0 = time.time()
        try:
            exec(compile(src, f"<notebook cell {idx}>", "exec"), ns)
        except Exception:
            print(f"\n!!! FAILED in notebook cell #{idx} after {time.time()-t0:.1f}s !!!",
                  flush=True)
            traceback.print_exc()
            sys.exit(1)
        # Windows spawn can't pickle classes defined in an exec() namespace;
        # the DDI loader in Cell 9b inherits NUM_WORKERS from Cell 2. Run it
        # single-process — inference results are unaffected.
        if idx == 4:
            ns["NUM_WORKERS"] = 0
            print("[driver] NUM_WORKERS forced to 0 (spawn/pickle guard)", flush=True)
        print(f"--- cell #{idx} done in {time.time()-t0:.1f}s ---", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("RECAP — key numbers pulled from the run", flush=True)
    print("=" * 70, flush=True)

    ddi_overall = ns.get("ddi_overall")
    if ddi_overall:
        print(f"DDI overall: acc={ddi_overall['accuracy']*100:.2f}% "
              f"auc={ddi_overall['auc']:.4f} "
              f"sens={ddi_overall['sensitivity']*100:.2f}% "
              f"spec={ddi_overall['specificity']*100:.2f}%")
    group_sens = ns.get("group_sens") or {}
    if group_sens:
        for k, v in sorted(group_sens.items()):
            print(f"  DDI tone group {k}: sensitivity={v:.4f}")
    for var in ("optimal_threshold", "optimal_sens", "optimal_spec", "roc_auc", "method_used"):
        if var in ns:
            print(f"{var} = {ns[var]}")


if __name__ == "__main__":
    main()
