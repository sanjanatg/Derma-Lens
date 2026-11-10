"""Add Fitzpatrick17k to modality assignment in Cell 5."""
import json
from pathlib import Path

nb_path = Path(r'model\notebooks\dermalens_train.ipynb')
nb = json.load(open(nb_path, 'r', encoding='utf-8'))

# Find Cell 5 and fix modality assignment
for cell in nb['cells']:
    if cell.get('cell_type') == 'code' and cell.get('source'):
        source_text = ''.join(cell['source'])
        if 'Unknown image source for modality assignment' in source_text:
            print("Found Cell 5 with modality assignment...")
            
            # Replace the modality assignment logic
            new_source = []
            skip_next = False
            for i, line in enumerate(cell['source']):
                if skip_next:
                    skip_next = False
                    continue
                    
                if 'if source_dataset in {"HAM10000", "ISIC2019"}:' in line:
                    # Replace the entire if-elif-else block
                    new_source.append('        if source_dataset in {"HAM10000", "ISIC2019"}:\n')
                    new_source.append('            modality = "dermoscopic"\n')
                    new_source.append('        elif source_dataset in {"PAD_UFES_20", "DDI", "Fitzpatrick17k"}:\n')
                    new_source.append('            modality = "clinical"\n')
                    new_source.append('        else:\n')
                    new_source.append('            raise ValueError(f"Unknown image source for modality assignment: {source_dataset}")\n')
                    # Skip the old lines
                    for j in range(i+1, len(cell['source'])):
                        if 'Binary malignant/benign label' in cell['source'][j]:
                            # Found the next section, stop skipping
                            break
                    # Jump ahead
                    while i < len(cell['source']) - 1 and 'Binary malignant/benign label' not in cell['source'][i]:
                        i += 1
                    continue
                else:
                    new_source.append(line)
            
            # Simpler approach - just do string replacement
            source_text = ''.join(cell['source'])
            source_text = source_text.replace(
                'elif source_dataset in {"PAD_UFES_20", "DDI"}:\n            modality = "clinical"',
                'elif source_dataset in {"PAD_UFES_20", "DDI", "Fitzpatrick17k"}:\n            modality = "clinical"'
            )
            
            cell['source'] = source_text.split('\n')
            # Add back newlines
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                            for i, line in enumerate(cell['source'])]
            
            print(f"Fixed modality assignment to include Fitzpatrick17k")
            
            # Save notebook
            with open(nb_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            
            print(f"Saved {nb_path}")
            break
else:
    print("ERROR: Could not find Cell 5 with modality assignment")
