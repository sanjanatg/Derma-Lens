"""Add missing import to Cell 5 in the notebook JSON."""
import json
from pathlib import Path

nb_path = Path(r'model\notebooks\dermalens_train.ipynb')
nb = json.load(open(nb_path, 'r', encoding='utf-8'))

# Find Cell 5 code cell
for cell in nb['cells']:
    if cell.get('cell_type') == 'code' and cell.get('source'):
        source = ''.join(cell['source'])
        if 'manifest_path = DATASETS_DIR' in source and 'fairdermnet_rebuilt_splits' in source:
            # Found Cell 5
            print("Found Cell 5, adding import...")
            
            # Check if import already exists
            if 'from model.model.class_taxonomy import class_idx_to_binary' in source:
                print("Import already exists!")
                break
            
            # Add import after "from pathlib import Path"
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if 'from pathlib import Path' in line:
                    # Add import on next line
                    new_source.append('from model.model.class_taxonomy import class_idx_to_binary, NUM_CLASSES, CLASS_NAMES\n')
            
            cell['source'] = new_source
            print(f"Added import. New source lines: {len(new_source)}")
            
            # Save notebook
            with open(nb_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            
            print(f"✓ Saved {nb_path}")
            break
else:
    print("ERROR: Could not find Cell 5")
