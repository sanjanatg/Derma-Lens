import json
from pathlib import Path

path = Path(r'c:\Users\sanja\Documents\sanjana\CAMBRIDGE\final year project\model\notebooks\dermalens_train.ipynb')
nb = json.loads(path.read_text())
updated = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('Cell 9b — DDI Fairness Evaluation' in line for line in cell['source']):
        src = cell['source']
        new_prefix = [
            "# Cell 9b — DDI Fairness Evaluation (held-out, expert-labelled)\n",
            "# Requires: Cell 9 (loads best_model) OR checkpoints/best.pth from prior run\n",
            "if 'ddi_df' not in globals():\n",
            "    ddi_df = None\n",
            "\n",
            "if ddi_df is None or len(ddi_df) == 0:\n",
            "    print(\"ddi_df not available in this kernel session.\")\n",
            "    ddi_csv = DATASETS_DIR / \"ddi_eval.csv\"\n",
            "    if ddi_csv.exists():\n",
            "        ddi_df = pd.read_csv(ddi_csv)\n",
            "        print(f\"Loaded existing DDI eval table from {ddi_csv}\")\n",
            "    else:\n",
            "        ddi_df = None\n",
            "\n",
        ]
        # Replace the first block up to the blank line before second if.
        for i, line in enumerate(src):
            if line == "\n" and i > 0 and src[i-1] == "        ddi_df = None\n":
                cell['source'] = new_prefix + src[i+1:]
                updated += 1
                break
        break
if updated == 0:
    raise RuntimeError('Patch point not found')
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print('patched', updated)
