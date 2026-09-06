"""
run_notebook_smoke.py — execute the REAL notebook under a real ipykernel,
the way Jupyter would ("Restart & Run All"), to verify Cells 9b/9c don't break
on Windows (in particular the DDI loader's num_workers=NUM_WORKERS pickling).

Cells to neutralize: NONE — after the patch_ddi_cell.py fixes (resume guard in
the training cell, cell 28/24/32/34 fixes) the full notebook runs end-to-end
without retraining: the training cell sees checkpoints/best.pth and rebuilds
`history` from training_history.json instead. allow_errors=True so a failure in
one cell doesn't mask the state of the rest; all error outputs are reported.

Kernel: project-local kernelspec files_claude/jupyter/kernels/py310, found via
JUPYTER_PATH (no changes to the user's Jupyter config or Python install).
"""
import os
import sys
import time
import json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "dermalens_train.ipynb"
TRIMMED = ROOT / "files_claude" / "trimmed_test.ipynb"
OUT_NB = ROOT / "files_claude" / "trimmed_test_out.ipynb"
LOG = ROOT / "files_claude" / "smoke_run.log"

# Cells to neutralize (index into the original notebook's cell list)
NEUTRALIZE = {}

os.environ["JUPYTER_PATH"] = str(ROOT / "files_claude" / "jupyter")
os.environ["MPLBACKEND"] = "Agg"


def build_trimmed():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    for idx, reason in NEUTRALIZE.items():
        cell = nb["cells"][idx]
        assert cell["cell_type"] == "code", f"cell {idx} not code"
        cell["source"] = [f"# neutralized for smoke test: {reason}\n", "pass\n"]
        cell["outputs"] = []
        cell["execution_count"] = None
    TRIMMED.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    print(f"trimmed notebook written: {TRIMMED}")


def main():
    build_trimmed()

    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(TRIMMED), as_version=4)
    client = NotebookClient(
        nb,
        kernel_name="py310",
        timeout=1800,          # per-cell timeout (test eval takes ~2.5 min)
        allow_errors=True,     # run everything, report all errors at the end
        resources={"metadata": {"path": str(ROOT)}},
    )

    t0 = time.time()
    log_fh = open(LOG, "w", encoding="utf-8")
    log_fh.write(f"starting nbclient at {time.strftime('%H:%M:%S')}\n")
    log_fh.write(f"neutralized cells: {sorted(NEUTRALIZE) or 'none (full run)'}\n")
    log_fh.flush()

    try:
        client.execute()
        status = "OK"
    except Exception as exc:
        status = f"FAILED: {type(exc).__name__}: {exc}"
    finally:
        elapsed = time.time() - t0
        log_fh.write(f"finished at {time.strftime('%H:%M:%S')} ({elapsed:.0f}s) -> {status}\n")
        log_fh.close()

    nbformat.write(nb, str(OUT_NB))
    print(f"STATUS: {status}")
    print(f"elapsed: {elapsed:.0f}s")
    print(f"executed notebook: {OUT_NB}")

    # Surface any cell-level errors that didn't stop the run
    errs = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.output_type == "error":
                errs += 1
                print(f"  ERROR in cell: {''.join(cell['source'])[:60]!r}")
                print("   ", out.get("ename"), out.get("evalue"))
    if errs:
        print(f"{errs} error output(s) present in executed notebook")
    sys.exit(0 if status == "OK" and errs == 0 else 1)


if __name__ == "__main__":
    main()
