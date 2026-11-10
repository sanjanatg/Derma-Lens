#!/usr/bin/env python3
"""
DermaLens Training Readiness Verification Script
Run this before starting model retraining to verify environment.
"""

import sys
import os
from pathlib import Path
import json

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_status(check_name, passed, message=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {check_name}")
    if message:
        print(f"       {message}")

def main():
    print_header("DermaLens Training Readiness Check")
    
    all_checks_passed = True
    
    # 1. Python Version
    print_header("1. Python Environment")
    py_version = sys.version_info
    py_ok = py_version.major == 3 and py_version.minor >= 10
    print_status("Python Version", py_ok, 
                 f"Found: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if not py_ok:
        print("       Required: Python 3.10+")
    all_checks_passed &= py_ok
    
    # 2. PyTorch and CUDA
    print_header("2. PyTorch & GPU")
    try:
        import torch
        torch_ok = True
        print_status("PyTorch Import", True, f"Version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print_status("CUDA Available", cuda_available)
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"       GPU: {gpu_name}")
            print(f"       VRAM: {vram_gb:.1f} GB")
            
            vram_ok = vram_gb >= 4.0
            print_status("VRAM Sufficient", vram_ok, 
                        f"{'Sufficient' if vram_ok else 'May need batch size adjustment'}")
            all_checks_passed &= vram_ok
        else:
            print("       WARNING: Training will be extremely slow on CPU!")
            all_checks_passed = False
            
    except ImportError as e:
        print_status("PyTorch Import", False, f"Error: {e}")
        all_checks_passed = False
    
    # 3. Required Packages
    print_header("3. Required Packages")
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('torchvision', 'TorchVision'),
        ('tqdm', 'TQDM'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    for package_name, display_name in required_packages:
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            print_status(display_name, True, f"Version: {version}")
        except ImportError:
            print_status(display_name, False, "Not installed")
            all_checks_passed = False
    
    # 4. Dataset Verification
    print_header("4. Dataset Verification")
    project_root = Path(__file__).parent.parent
    master_csv = project_root / "model" / "datasets" / "datasets" / "master_dataset.csv"
    
    csv_exists = master_csv.exists()
    print_status("Master Dataset CSV", csv_exists, 
                 f"Path: {master_csv}")
    
    if csv_exists:
        try:
            import pandas as pd
            df = pd.read_csv(master_csv)
            row_count = len(df)
            print_status("Dataset Size", row_count > 30000, 
                        f"{row_count:,} images (Expected: ~40,182)")
            
            required_cols = ['image_path', 'class_idx', 'skin_tone', 'dataset']
            cols_ok = all(col in df.columns for col in required_cols)
            print_status("Required Columns", cols_ok, 
                        f"Columns: {list(df.columns)}")
            
            # Check class distribution
            class_counts = df['class_idx'].value_counts().sort_index()
            print("       Class distribution:")
            for idx, count in class_counts.items():
                print(f"         Class {idx}: {count:>6,} samples")
            
            # Check skin tone distribution
            tone_counts = df['skin_tone'].value_counts().sort_index()
            print("       Skin tone distribution:")
            for tone, count in tone_counts.items():
                print(f"         Fitzpatrick {tone}: {count:>6,} samples")
            
            all_checks_passed &= cols_ok and row_count > 30000
            
        except Exception as e:
            print_status("Dataset Parse", False, f"Error: {e}")
            all_checks_passed = False
    else:
        all_checks_passed = False
    
    # 5. Project Structure
    print_header("5. Project Structure")
    required_dirs = [
        ("model", "Model directory"),
        ("model/notebooks", "Training notebooks"),
        ("model/checkpoints", "Checkpoint directory"),
        ("model/figures", "Figure output directory"),
    ]
    
    for dir_path, description in required_dirs:
        full_path = project_root / dir_path
        exists = full_path.exists()
        print_status(description, exists, f"Path: {full_path}")
        if not exists and 'checkpoint' in dir_path:
            # Checkpoints dir can be created automatically
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"       Created directory: {full_path}")
            except:
                pass
    
    # 6. Disk Space
    print_header("6. Disk Space")
    try:
        import shutil
        total, used, free = shutil.disk_usage(project_root)
        free_gb = free / (1024**3)
        disk_ok = free_gb >= 10
        print_status("Free Disk Space", disk_ok, 
                    f"{free_gb:.1f} GB free (min 10 GB required)")
        all_checks_passed &= disk_ok
    except Exception as e:
        print(f"       Warning: Could not check disk space: {e}")
    
    # 7. Notebook Existence
    print_header("7. Training Notebook")
    notebook_path = project_root / "model" / "notebooks" / "dermalens_train.ipynb"
    notebook_ok = notebook_path.exists()
    print_status("Training Notebook", notebook_ok, 
                 f"Path: {notebook_path}")
    all_checks_passed &= notebook_ok
    
    # Final Summary
    print_header("OVERALL STATUS")
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - READY FOR TRAINING")
        print("\nNext steps:")
        print("  1. Review RETRAINING_CHECKLIST.md")
        print("  2. Backup current model (if exists)")
        print("  3. Update hyperparameters in notebook Cell 2")
        print("  4. Start training: jupyter notebook model/notebooks/dermalens_train.ipynb")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - PLEASE FIX ISSUES ABOVE")
        print("\nCommon fixes:")
        print("  - Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("  - Install other packages: pip install numpy pandas opencv-python pillow tqdm matplotlib")
        print("  - Verify dataset exists: check model/datasets/datasets/master_dataset.csv")
        return 1

if __name__ == "__main__":
    sys.exit(main())
