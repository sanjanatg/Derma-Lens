#!/usr/bin/env python3
"""
Fitzpatrick17k Integration Script (Enhanced)
============================================
Integrates Fitzpatrick17k dataset into master_dataset.csv
with proper label mapping to the 8-class taxonomy.

Features:
- Maps Fitzpatrick17k labels to 0-7 class indices
- Validates and cleans data (removes unmapped/invalid entries)
- Checks image files exist
- Ensures skin tone diversity (Fitz 0-6 scale)
- Removes duplicates
- Generates detailed statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import warnings
import sys
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
FITZ17K_CSV = PROJECT_ROOT / 'fitzpatrick17k.csv'
FITZ17K_DIR = DATASETS_DIR / 'Fitzpatrick17k_CategorizedAbbrvs'
MASTER_CSV = DATASETS_DIR / 'master_dataset.csv'

print("=" * 70)
print("🎨 FITZPATRICK17K INTEGRATION")
print("=" * 70)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Fitzpatrick17k CSV: {FITZ17K_CSV}")
print(f"Fitzpatrick17k Images: {FITZ17K_DIR}")
print(f"Master CSV: {MASTER_CSV}")
print()

# ═══════════════════════════════════════════════════════════════
# LABEL MAPPING: Fitzpatrick17k → DermaLens 8-class taxonomy
# ═══════════════════════════════════════════════════════════════
# Our classes: 0=MEL, 1=BCC, 2=SCC, 3=AK, 4=NV, 5=BKL, 6=DF, 7=VASC
LABEL_MAPPING = {
    # ── Malignant Melanoma ──
    'melanoma': 0,
    'lentigo maligna melanoma': 0,
    'superficial spreading melanoma': 0,
    'nodular melanoma': 0,
    'acral lentiginous melanoma': 0,
    
    # ── Basal Cell Carcinoma ──
    'basal cell carcinoma': 1,
    'basal cell carcinoma morpheiform': 1,
    'basal cell carcinoma nodular': 1,
    'basal cell carcinoma superficial': 1,
    
    # ── Squamous Cell Carcinoma ──
    'squamous cell carcinoma': 2,
    'squamous cell carcinoma in situ': 2,
    'bowen disease': 2,  # SCC in situ
    'bowens disease': 2,
    
    # ── Actinic Keratosis (pre-cancerous) ──
    'actinic keratosis': 3,
    'solar keratosis': 3,
    
    # ── Melanocytic Nevi (benign nevi) ──
    'nevus': 4,
    'melanocytic nevus': 4,
    'congenital nevus': 4,
    'dysplastic nevus': 4,
    'junctional nevus': 4,
    'compound nevus': 4,
    'intradermal nevus': 4,
    'blue nevus': 4,
    'halo nevus': 4,
    'spitz nevus': 4,
    
    # ── Benign Keratosis ──
    'seborrheic keratosis': 5,
    'seborrhoeic keratosis': 5,
    'lentigo': 5,
    'solar lentigo': 5,
    'lichenoid keratosis': 5,
    'sebaceous hyperplasia': 5,
    'epidermoid cyst': 5,
    'pilar cyst': 5,
    'milia': 5,
    'syringoma': 5,
    'epidermal nevus': 5,
    
    # ── Dermatofibroma ──
    'dermatofibroma': 6,
    
    # ── Vascular Lesions ──
    'hemangioma': 7,
    'angioma': 7,
    'cherry angioma': 7,
    'angiokeratoma': 7,
    'pyogenic granuloma': 7,
    'kaposi sarcoma': 7,  # vascular malignancy
    'angiosarcoma': 7,
    'venous lake': 7,
    'telangiectasia': 7,
    
    # Note: Inflammatory conditions, eczema, psoriasis, etc. are NOT mapped
    # They will be excluded (class_idx = -1) during filtering
}

CLASS_NAMES = ['MEL', 'BCC', 'SCC', 'AK', 'NV', 'BKL', 'DF', 'VASC']

print("📂 Step 1: Loading Fitzpatrick17k CSV...")
if not FITZ17K_CSV.exists():
    print(f"❌ ERROR: {FITZ17K_CSV} not found!")
    exit(1)

df_fitz = pd.read_csv(FITZ17K_CSV)
print(f"✅ Loaded {len(df_fitz):,} rows")
print()

print("📊 Step 2: Analyzing Fitzpatrick17k dataset...")
print(f"   Total images: {len(df_fitz):,}")
print(f"   Unique labels: {df_fitz['label'].nunique()}")
print(f"   Skin tone distribution:")
for tone in sorted(df_fitz['fitzpatrick_scale'].unique()):
    if tone != -1:
        count = (df_fitz['fitzpatrick_scale'] == tone).sum()
        pct = 100 * count / len(df_fitz)
        print(f"      Fitz {tone}: {count:,} ({pct:.1f}%)")
print()

print("🗺️  Step 3: Mapping labels to 8-class taxonomy...")
def map_label(label_str):
    """Map Fitzpatrick17k label to our class index"""
    label_lower = str(label_str).lower().strip()
    return LABEL_MAPPING.get(label_lower, -1)

df_fitz['class_idx'] = df_fitz['label'].apply(map_label)

# Keep only mapped labels
df_fitz_mapped = df_fitz[df_fitz['class_idx'] != -1].copy()
print(f"✅ Mapped {len(df_fitz_mapped):,} / {len(df_fitz):,} images")
print(f"   Unmapped (skipped): {len(df_fitz) - len(df_fitz_mapped):,}")

# Show unmapped labels for verification
unmapped = df_fitz[df_fitz['class_idx'] == -1]['label'].unique()
if len(unmapped) > 0:
    print(f"\n   Top unmapped labels (inflammatory/irrelevant):")
    label_counts = df_fitz[df_fitz['class_idx'] == -1]['label'].value_counts().head(10)
    for label, count in label_counts.items():
        print(f"      {label}: {count}")
print()

print("   ✅ Mapped class distribution:")
for class_idx in sorted(df_fitz_mapped['class_idx'].unique()):
    count = (df_fitz_mapped['class_idx'] == class_idx).sum()
    pct = 100 * count / len(df_fitz_mapped)
    print(f"      Class {class_idx} ({CLASS_NAMES[class_idx]:>4}): {count:5,} ({pct:5.1f}%)")
print()

# ═══════════════════════════════════════════════════════════════
# Step 4: Build image path lookup and validate images
# ═══════════════════════════════════════════════════════════════
print("🖼️  Step 4: Building image path index...")

if not FITZ17K_DIR.exists():
    print(f"❌ ERROR: Fitzpatrick17k image directory not found: {FITZ17K_DIR}")
    print(f"   Please ensure images are downloaded and placed in the datasets folder.")
    print(f"   Download from: https://github.com/mattgroh/fitzpatrick17k")
    exit(1)

# Build a lookup dictionary: md5hash -> full image path
# Images are in nested subfolders with filenames like: me_f0_103_e276cd90.jpg
# where the last part after the final underscore is a truncated md5 hash
print(f"   Scanning directory structure...")
md5_to_path = {}
short_hash_to_path = {}
categorized_dir = FITZ17K_DIR / "Categorized_AbbrvName"

if categorized_dir.exists():
    # Recursively find all .jpg files
    for img_file in categorized_dir.rglob("*.jpg"):
        # Extract the hash part from filename
        # Format: classabbrev_f<fitz>_<number>_<shortened_hash>.jpg
        # Example: me_f0_103_e276cd90.jpg
        filename_parts = img_file.stem.split('_')
        if len(filename_parts) >= 4:
            short_hash = filename_parts[-1]  # Last part after last underscore
            short_hash_to_path[short_hash] = str(img_file)
        
        # Also try full filename as hash (in case some are stored differently)
        full_stem = img_file.stem
        md5_to_path[full_stem] = str(img_file)
    
    print(f"   ✅ Found {len(short_hash_to_path):,} images in nested subfolders")
    print(f"      (Using shortened hash matching from filenames)")
else:
    print(f"   ⚠️  'Categorized_AbbrvName' subfolder not found.")
    exit(1)

print()

# Map image paths using md5hash lookup (try shortened hash matching)
print("🔗 Step 4b: Matching CSV entries to image files...")

# Initialize image_path column  
df_fitz_mapped['image_path'] = None

matched = 0
for idx, row in tqdm(df_fitz_mapped.iterrows(), total=len(df_fitz_mapped), desc="   Matching"):
    md5hash = row['md5hash']
    
    # Try exact match first
    if md5hash in md5_to_path:
        df_fitz_mapped.at[idx, 'image_path'] = md5_to_path[md5hash]
        matched += 1
    else:
        # Try shortened hash (last 8 characters of md5)
        # The Fitzpatrick17k dataset uses shortened hashes in filenames
        short_hash = md5hash[:8]  # First 8 characters
        if short_hash in short_hash_to_path:
            df_fitz_mapped.at[idx, 'image_path'] = short_hash_to_path[short_hash]
            matched += 1

# Filter out rows without matched images
missing_count = df_fitz_mapped['image_path'].isna().sum()
df_fitz_mapped = df_fitz_mapped[df_fitz_mapped['image_path'].notna()].copy()

print(f"   ✅ Matched images: {len(df_fitz_mapped):,}")
if missing_count > 0:
    print(f"   ⚠️  Not found in filesystem: {missing_count:,}")
    if missing_count == len(df_fitz_mapped) + missing_count:
        print(f"   ⚠️  WARNING: NO images were matched! Check filename format.")
print()

# Clean skin tone values (convert to 0-indexed: Fitz 1-6 → 0-5)
print("🎨 Step 4c: Processing skin tone data...")
df_fitz_mapped['skin_tone'] = df_fitz_mapped['fitzpatrick_scale'].apply(
    lambda x: int(x) - 1 if pd.notna(x) and x > 0 and x <= 6 else -1
)

# Add dataset column
df_fitz_mapped['dataset'] = 'Fitzpatrick17k'

# Keep only needed columns to match master_dataset schema
df_fitz_final = df_fitz_mapped[['image_path', 'class_idx', 'skin_tone', 'dataset', 'label']].copy()
df_fitz_final = df_fitz_final.rename(columns={'label': 'dx_raw'})

print(f"   ✅ Prepared {len(df_fitz_final):,} entries for integration")
print()

print("📊 Step 4d: Fitzpatrick17k skin tone distribution:")
tone_counts = df_fitz_final['skin_tone'].value_counts().sort_index()
for tone in sorted(df_fitz_final['skin_tone'].unique()):
    count = tone_counts.get(tone, 0)
    pct = 100 * count / len(df_fitz_final)
    if tone == -1:
        tone_label = "Unknown"
    else:
        tone_label = f"Fitz {tone} (Type {tone+1})"
    print(f"   {tone_label:25}: {count:5,} ({pct:5.1f}%)")
print()

print("📑 Step 5: Loading existing master_dataset...")
if not MASTER_CSV.exists():
    print(f"⚠️  Master dataset not found: {MASTER_CSV}")
    print(f"   Creating new master dataset with Fitzpatrick17k only.")
    df_master = pd.DataFrame()
else:
    df_master = pd.read_csv(MASTER_CSV)
    print(f"✅ Current master dataset: {len(df_master):,} rows")
print()

# Check for duplicates
print("🔍 Step 6: Checking for duplicates...")
if len(df_master) > 0:
    existing_images = set(df_master['image_path'].values)
    new_images = set(df_fitz_final['image_path'].values)
    duplicates = existing_images.intersection(new_images)
    print(f"   Duplicates found: {len(duplicates)}")
    if len(duplicates) > 0:
        df_fitz_final = df_fitz_final[~df_fitz_final['image_path'].isin(duplicates)]
        print(f"   Removed duplicates, keeping: {len(df_fitz_final):,} new images")
else:
    print(f"   No existing dataset - skipping duplicate check")
print()

# Merge datasets
print("🔗 Step 7: Merging Fitzpatrick17k into master dataset...")
if len(df_master) > 0:
    df_master_new = pd.concat([df_master, df_fitz_final], ignore_index=True)
else:
    df_master_new = df_fitz_final.copy()
print(f"✅ New master dataset size: {len(df_master_new):,} rows")
print(f"   Added: {len(df_fitz_final):,} new images from Fitzpatrick17k")
print()

# Verify class distribution after merge
print("📊 Step 7b: Updated class distribution:")
for class_idx in range(8):
    count = (df_master_new['class_idx'] == class_idx).sum()
    if count > 0:
        pct = 100 * count / len(df_master_new)
        print(f"   Class {class_idx} ({CLASS_NAMES[class_idx]:>4}): {count:6,} ({pct:5.1f}%)")
print()

# Show improved skin tone distribution
print("📊 Step 8: Updated skin tone distribution...")
for tone in sorted(df_master_new['skin_tone'].unique()):
    count = (df_master_new['skin_tone'] == tone).sum()
    pct = 100 * count / len(df_master_new)
    if tone == -1:
        tone_label = "Unknown"
    else:
        tone_label = f"Fitz {tone} (Type {tone+1})"
    print(f"   {tone_label:25}: {count:6,} ({pct:5.1f}%)")
print()

# Calculate fairness improvement
if len(df_master) > 0:
    old_dark_count = (df_master['skin_tone'] >= 3).sum()
    old_dark_pct = 100 * old_dark_count / len(df_master) if len(df_master) > 0 else 0
    new_dark_count = (df_master_new['skin_tone'] >= 3).sum()
    new_dark_pct = 100 * new_dark_count / len(df_master_new)
    
    print(f"🎯 Fairness Improvement:")
    print(f"   Darker skin tones (Fitz 4-6, idx 3-5) BEFORE: {old_dark_count:,} ({old_dark_pct:.2f}%)")
    print(f"   Darker skin tones (Fitz 4-6, idx 3-5) AFTER:  {new_dark_count:,} ({new_dark_pct:.2f}%)")
    print(f"   Improvement: +{new_dark_pct - old_dark_pct:.2f} percentage points")
else:
    new_dark_count = (df_master_new['skin_tone'] >= 3).sum()
    new_dark_pct = 100 * new_dark_count / len(df_master_new)
    print(f"🎯 Skin Tone Diversity:")
    print(f"   Darker skin tones (Fitz 4-6, idx 3-5): {new_dark_count:,} ({new_dark_pct:.2f}%)")
print()

# Save updated master dataset
print("💾 Step 9: Saving updated master dataset...")
if len(df_master) > 0:
    backup_path = DATASETS_DIR / 'master_dataset_before_fitz17k.csv'
    df_master.to_csv(backup_path, index=False)
    print(f"✅ Backup saved: {backup_path}")

df_master_new.to_csv(MASTER_CSV, index=False)
print(f"✅ Updated master dataset saved: {MASTER_CSV}")
print()

# Generate statistics report
print("📊 Step 10: Generating statistics report...")
stats_report = []
stats_report.append("=" * 70)
stats_report.append("FITZPATRICK17K INTEGRATION STATISTICS")
stats_report.append("=" * 70)
stats_report.append(f"Date: {pd.Timestamp.now()}")
stats_report.append(f"Total images in master dataset: {len(df_master_new):,}")
stats_report.append(f"Images added from Fitzpatrick17k: {len(df_fitz_final):,}")
stats_report.append("")
stats_report.append("Dataset composition:")
for ds, count in df_master_new['dataset'].value_counts().items():
    pct = 100 * count / len(df_master_new)
    stats_report.append(f"  {ds:20}: {count:6,} ({pct:5.1f}%)")
stats_report.append("")
stats_report.append("Class distribution (8 classes):")
for i in range(8):
    count = (df_master_new['class_idx'] == i).sum()
    pct = 100 * count / len(df_master_new) if len(df_master_new) > 0 else 0
    stats_report.append(f"  [{i}] {CLASS_NAMES[i]:>4}: {count:6,} ({pct:5.1f}%)")
stats_report.append("")
stats_report.append("Skin tone distribution:")
for tone in sorted(df_master_new['skin_tone'].unique()):
    count = (df_master_new['skin_tone'] == tone).sum()
    pct = 100 * count / len(df_master_new)
    label = "Unknown" if tone == -1 else f"Fitz {tone} (Type {tone+1})"
    stats_report.append(f"  {label:25}: {count:6,} ({pct:5.1f}%)")
stats_report.append("=" * 70)

report_text = "\n".join(stats_report)
print(report_text)

stats_file = DATASETS_DIR / 'fitzpatrick17k_integration_stats.txt'
with open(stats_file, 'w') as f:
    f.write(report_text)
print(f"\n✅ Statistics saved: {stats_file}")
print()

# Summary
print("=" * 70)
print("✅ FITZPATRICK17K INTEGRATION COMPLETE")
print("=" * 70)
print(f"✓ Total images: {len(df_master_new):,}")
print(f"✓ Fitzpatrick17k contribution: {len(df_fitz_final):,} images")
print(f"✓ All 8 classes mapped: {list(CLASS_NAMES)}")
print()
print("📋 NEXT STEPS:")
print("   1. Run: python validate_and_clean_data.py")
print("   2. Regenerate train/val/test splits")
print("   3. Retrain model with enhanced dataset")
print("=" * 70)
