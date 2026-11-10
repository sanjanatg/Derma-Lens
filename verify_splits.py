import pandas as pd

m = pd.read_csv(r'model\datasets\datasets\fairdermnet_rebuilt_splits.csv')
print(f'✓ Splits manifest: {len(m):,} rows')
print(f'  Train: {(m["split"]=="train").sum():,}')
print(f'  Val: {(m["split"]=="val").sum():,}')
print(f'  Test: {(m["split"]=="test").sum():,}')
print(f'\n✓ Fitzpatrick17k images: {(m["dataset"]=="Fitzpatrick17k").sum():,}')
print(f'\nDataset totals:')
for ds in ['HAM10000', 'ISIC2019', 'PAD_UFES_20', 'Fitzpatrick17k']:
    print(f'  {ds}: {(m["dataset"]==ds).sum():,}')
