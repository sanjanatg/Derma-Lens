"""Test Cell 5 imports and data loading outside Jupyter to verify fixes."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")
from model.model.class_taxonomy import class_idx_to_binary, NUM_CLASSES, CLASS_NAMES
print(f"✓ class_idx_to_binary imported: {class_idx_to_binary(0)} (should be 1 for melanoma)")
print(f"✓ NUM_CLASSES: {NUM_CLASSES}")
print(f"✓ CLASS_NAMES: {CLASS_NAMES[:3]}")

print("\nTesting augmentation pipeline...")
import importlib
import model.preprocessing.augmentation_pipeline as aug_module
importlib.reload(aug_module)  # Force reload to get latest version
from model.preprocessing.augmentation_pipeline import get_train_transforms

transforms = get_train_transforms(224)
print("✓ Transforms loaded successfully")
print(f"  Transform count: {len(transforms.transforms)}")

print("\n✓ All imports work correctly")
