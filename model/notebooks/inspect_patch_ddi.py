import json
from pathlib import Path
path = Path(r'c:\Users\sanja\Documents\sanjana\CAMBRIDGE\final year project\model\notebooks\dermalens_train.ipynb')
nb = json.loads(path.read_text())
for i, cell in enumerate(nb['cells'], 1):
    if cell['cell_type'] == 'code' and any('Cell 9b — DDI Fairness Evaluation' in line for line in cell['source']):
        print('CELL_INDEX', i, 'ID', cell.get('id'))
        print('SOURCE_START')
        print(''.join(cell['source']))
        print('SOURCE_END')
        break
