#!/usr/bin/env python3
"""
Data Validation and Cleaning Script
====================================
Validates and cleans the master dataset before training.
Fixes file paths, validates images, removes corrupted data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def find_project_root():
    """Find project root directory"""
    p = Path.cwd()
    while p != p.parent:
        if (p / 'dermalens_flutter').is_dir():
            return p
        p = p.parent
    return Path.cwd()

PROJECT_ROOT = find_project_root()
DATASETS_DIR = PROJECT_ROOT / 'model' / 'datasets' / 'datasets'
MASTER_CSV = DATASETS_DIR / 'master_dataset.csv'

print("=" * 60)
print("🔍 DATA VALIDATION & CLEANING")
print("=" * 60)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Datasets Dir: {DATASETS_DIR}")
print(f"Master CSV: {MASTER_CSV}")
print()

# Step 1: Load master dataset
print("📂 Step 1: Loading master_dataset.csv...")
if not MASTER_CSV.exists():
    print(f"❌ ERROR: {MASTER_CSV} not found!")
    exit(1)

df = pd.read_csv(MASTER_CSV)
print(f"✅ Loaded {len(df):,} rows")
print(f"   Columns: {list(df.columns)}")
print()

# Step 2: Analyze data structure
print("📊 Step 2: Analyzing data structure...")
print(f"   Classes: {df['class_idx'].nunique()} unique")
print(f"   Datasets: {df['dataset'].unique()}")
print(f"   Skin tones: {sorted(df['skin_tone'].unique())}")
print()

print("   Class distribution:")
for class_idx in sorted(df['class_idx'].unique()):
    count = (df['class_idx'] == class_idx).sum()
    print(f"      Class {class_idx}: {count:,} samples")
print()

# Step 3: Fix file paths
print("🔧 Step 3: Fixing file paths...")
initial_paths = df['image_path'].iloc[0]
print(f"   Current path format: {initial_paths[:80]}...")

def fix_path(old_path):
    """Convert old hardcoded path to current system path"""
    old_path_str = str(old_path)
    
    # Extract relative path after 'datasets'
    if 'datasets' in old_path_str:
        parts = old_path_str.split('datasets')
        if len(parts) > 1:
            relative = parts[-1].lstrip('\\').lstrip('/')
            new_path = DATASETS_DIR / relative
            return str(new_path)
    
    return old_path

df['image_path'] = df['image_path'].apply(fix_path)
print(f"✅ Fixed {len(df)} file paths")
print(f"   New path format: {df['image_path'].iloc[0][:80]}...")
print()

# Step 4: Validate image files exist
print("🖼️  Step 4: Validating image files...")
missing_files = []
valid_indices = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
    img_path = Path(row['image_path'])
    if img_path.exists():
        valid_indices.append(idx)
    else:
        missing_files.append(str(img_path))

print(f"✅ Valid images: {len(valid_indices):,}")
print(f"❌ Missing images: {len(missing_files):,}")

if len(missing_files) > 0:
    print(f"   First 5 missing:")
    for f in missing_files[:5]:
        print(f"      {f}")
    print()

# Step 5: Remove rows with missing images
if len(missing_files) > 0:
    print("🧹 Step 5: Removing rows with missing images...")
    df_clean = df.loc[valid_indices].copy()
    print(f"✅ Cleaned dataset: {len(df_clean):,} rows (removed {len(missing_files):,})")
else:
    df_clean = df.copy()
    print("✅ Step 5: All images found, no cleaning needed")
print()

# Step 6: Validate image integrity
print("🔍 Step 6: Validating image integrity (sample check)...")
sample_size = min(100, len(df_clean))
sample_indices = np.random.choice(df_clean.index, sample_size, replace=False)
corrupted = []

for idx in tqdm(sample_indices, desc="Checking integrity"):
    img_path = df_clean.loc[idx, 'image_path']
    try:
        img = Image.open(img_path)
        img.verify()
    except Exception as e:
        corrupted.append(img_path)

if len(corrupted) > 0:
    print(f"⚠️  Found {len(corrupted)} corrupted images in sample")
    # Remove all corrupted images from main dataset
    df_clean = df_clean[~df_clean['image_path'].isin(corrupted)]
else:
    print(f"✅ All sampled images are valid")
print()

# Step 7: Check for missing labels
print("📋 Step 7: Checking for missing labels...")
missing_class = df_clean['class_idx'].isna().sum()
missing_skin = df_clean['skin_tone'].isna().sum()
missing_dataset = df_clean['dataset'].isna().sum()

print(f"   Missing class_idx: {missing_class}")
print(f"   Missing skin_tone: {missing_skin}")
print(f"   Missing dataset: {missing_dataset}")

if missing_class > 0:
    print("   ⚠️  Removing rows with missing class_idx...")
    df_clean = df_clean.dropna(subset=['class_idx'])

if missing_dataset > 0:
    print("   ⚠️  Removing rows with missing dataset...")
    df_clean = df_clean.dropna(subset=['dataset'])

# For skin_tone, -1 is acceptable (unknown)
df_clean['skin_tone'] = df_clean['skin_tone'].fillna(-1).astype(int)
print(f"✅ Cleaned labels: {len(df_clean):,} rows")
print()

# Step 8: Final statistics
print("📊 Step 8: Final dataset statistics...")
print(f"   Total samples: {len(df_clean):,}")
print(f"   Datasets: {df_clean['dataset'].value_counts().to_dict()}")
print()
print("   Class distribution:")
for class_idx in sorted(df_clean['class_idx'].unique()):
    count = (df_clean['class_idx'] == class_idx).sum()
    pct = 100 * count / len(df_clean)
    print(f"      Class {class_idx}: {count:,} samples ({pct:.1f}%)")
print()
print("   Skin tone distribution:")
for tone in sorted(df_clean['skin_tone'].unique()):
    count = (df_clean['skin_tone'] == tone).sum()
    pct = 100 * count / len(df_clean)
    tone_label = f"Unknown" if tone == -1 else f"Fitz {tone}"
    print(f"      {tone_label}: {count:,} samples ({pct:.1f}%)")
print()

# Step 9: Save cleaned dataset
print("💾 Step 9: Saving cleaned dataset...")
backup_path = DATASETS_DIR / 'master_dataset_backup.csv'
if MASTER_CSV.exists():
    df.to_csv(backup_path, index=False)
    print(f"✅ Backup saved: {backup_path}")

df_clean.to_csv(MASTER_CSV, index=False)
print(f"✅ Cleaned dataset saved: {MASTER_CSV}")
print()

# Step 10: Validate train/val/test splits
print("📑 Step 10: Checking split files...")
for split_name in ['train', 'val', 'test']:
    split_path = DATASETS_DIR / f'split_{split_name}.csv'
    if split_path.exists():
        df_split = pd.read_csv(split_path)
        print(f"   ✅ split_{split_name}.csv: {len(df_split):,} rows")
        
        # Fix paths in splits too
        df_split['image_path'] = df_split['image_path'].apply(fix_path)
        df_split.to_csv(split_path, index=False)
    else:
        print(f"   ⚠️  split_{split_name}.csv: Not found")
print()

# Summary
print("=" * 60)
print("✅ DATA VALIDATION COMPLETE")
print("=" * 60)
print(f"Original samples: {len(df):,}")
print(f"Cleaned samples: {len(df_clean):,}")
print(f"Removed: {len(df) - len(df_clean):,} ({100*(len(df)-len(df_clean))/len(df):.2f}%)")
print()
print("🎯 Dataset is ready for training!")
print("=" * 60)
