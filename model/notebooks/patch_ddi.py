import json

with open('dermalens_train.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb['cells']:
    if 'ddi_dataset =' in ''.join(c.get('source', [])):
        for i, line in enumerate(c['source']):
            if 'transform=get_val_transforms(INPUT_SIZE),' in line:
                c['source'].insert(i+1, '        exclude_class_minus1=False,\n')
                break

with open('dermalens_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
