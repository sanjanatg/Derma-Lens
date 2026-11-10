# 🚀 DermaLens Model Retraining - Pre-Flight Checklist

**Status**: Ready for Training  
**Dataset**: 40,182 images loaded  
**Target**: 78-82% balanced accuracy, <15% fairness gap  
**Estimated Time**: 6-8 hours on RTX 2050 (4GB VRAM)

---

## ✅ Pre-Training Verification (Complete This First!)

### 1. Dataset Verification ✅ COMPLETE
- [x] Master dataset exists: `model/datasets/datasets/master_dataset.csv`
- [x] Dataset size: 40,182 images
- [x] All 8 classes present (MEL, BCC, SCC, AK, NV, BKL, DF, VASC)
- [x] Skin tone labels populated
- [x] Image paths valid

### 2. Environment Check (Run Now)
```cmd
cd c:\Users\vasud\Downloads\FINAL_YEAR\Derma-Lens1\model

REM Check Python version
python --version
REM Expected: Python 3.10.x

REM Check GPU availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
REM Expected: CUDA Available: True, GPU: RTX 3050

REM Check key packages
python -c "import torch, torchvision, numpy, pandas, cv2; print('✓ All packages available')"
```

**Action Required**: Run these commands and verify output before proceeding.

---

### 3. Backup Current Model
```cmd
cd c:\Users\vasud\Downloads\FINAL_YEAR\Derma-Lens1\model

REM Create backup directory
mkdir checkpoints_backup

REM Backup existing checkpoint if it exists
copy checkpoints\best.pth checkpoints_backup\best_before_retraining.pth
copy checkpoints\training_history.json checkpoints_backup\history_before_retraining.json

echo ✓ Backup complete
```

**Action Required**: Run this now to preserve your current model.

---

### 4. Training Configuration Review

**Current Configuration** (from notebook Cell 2):
```yaml
EPOCHS: 30
BATCH_SIZE: 16
GRAD_ACCUM: 4
USE_AMP: True
LR: 1e-4
ALPHA (classification loss): 1.0
BETA (Fitzpatrick aux): 0.15
GAMMA (fairness gap): 0.25
```

**Recommended Updates for Better Performance**:
```python
# In the training notebook, Cell 2, update these:
EPOCHS = 80              # Train longer (up from 30)
LR = 5e-5               # Lower learning rate for stability (down from 1e-4)
GAMMA = 0.5             # Stronger fairness constraint (up from 0.25)
PATIENCE = 15           # More patience for early stopping (up from 7)
```

**Action Required**: Open `model/notebooks/dermalens_train.ipynb` and update Cell 2.

---

### 5. Disk Space Check
```cmd
REM Check available space on C: drive
wmic logicaldisk get name,freespace,size | findstr C:
```

**Requirements**:
- Minimum: 10 GB free space
- Recommended: 20 GB free space (for checkpoints, logs, exports)

**Action Required**: Ensure sufficient disk space.

---

### 6. System Resource Check
```cmd
REM Check RAM
wmic ComputerSystem get TotalPhysicalMemory
REM Minimum: 8 GB RAM recommended

REM Check GPU memory
nvidia-smi
REM Expected: 6144 MB VRAM available
```

**Action Required**: Verify system resources.

---

## 🎯 Training Execution Plan

### Step 1: Open Jupyter Notebook
```cmd
cd c:\Users\vasud\Downloads\FINAL_YEAR\Derma-Lens1\model
jupyter notebook notebooks\dermalens_train.ipynb
```

### Step 2: Execute Cells in Order
1. **Cell 1**: GPU check + environment ✅
2. **Cell 2**: Configuration (update EPOCHS, LR, GAMMA) 🔧
3. **Cell 3**: Import project packages ✅
4. **Cell 4**: Dataset loading ✅ (already verified)
5. **Cell 5**: Create data loaders and grouped manifest
6. **Cell 6**: Build model
7. **Cell 7**: Training loop (6-8 hours)
8. **Cell 8**: Evaluation
9. **Cell 9**: Export to TFLite

### Step 3: Monitor Training
**Watch for these metrics** (printed every epoch):
```
Epoch XX/80 | Loss: X.XXX | Acc: X.XX% | Val Acc: X.XX% | Fairness Gap: X.XX%
```

**Good signs**:
- ✅ Loss decreasing steadily
- ✅ Val Acc increasing (aim for 78%+)
- ✅ Fairness Gap decreasing (aim for <15%)
- ✅ No "CUDA out of memory" errors

**Bad signs**:
- ❌ Loss = NaN or Inf → STOP, reduce learning rate
- ❌ Val Acc stuck or decreasing → Model may be overfitting
- ❌ Out of memory → Reduce BATCH_SIZE to 8

---

## 📊 Success Criteria (How to Know Training Worked)

### Minimum Acceptable Performance:
- [ ] **Balanced Accuracy**: ≥78% (currently ~70%)
- [ ] **Fairness Gap**: <15% (currently ~26%)
- [ ] **Test AUC**: ≥0.93 (currently ~0.92)
- [ ] **DDI Sensitivity**: ≥70% on FST V-VI (currently ~65%)
- [ ] **Rare Class Recall**: All classes >55% (DF, VASC, SCC)

### Training Convergence Indicators:
- [ ] Validation loss stopped decreasing for 15 epochs
- [ ] Training completed 80 epochs OR early stopping triggered
- [ ] Final checkpoint saved to `checkpoints/best.pth`
- [ ] Training history saved to `checkpoints/training_history.json`

---

## 🚨 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# In Cell 2, reduce batch size:
BATCH_SIZE = 8      # Down from 16
GRAD_ACCUM = 8      # Up from 4 (keeps effective batch = 64)
```

### Issue 2: Training Too Slow
**If training takes >10 hours**:
- Check GPU utilization: `nvidia-smi` (should be 95-100%)
- Close other GPU applications
- Reduce NUM_WORKERS to 2 if CPU bottlenecked

### Issue 3: Fairness Gap Not Improving
**If gap stays >20% after 40 epochs**:
```python
# In Cell 2, increase fairness weight:
GAMMA = 0.75        # Up from 0.5
```

### Issue 4: Model Overfitting
**If Train Acc >> Val Acc (gap >10%)**:
- Early stopping will handle this automatically
- Increase dropout in model architecture
- Add more data augmentation

### Issue 5: Jupyter Kernel Crashes
**If kernel dies during training**:
1. Save current epoch checkpoint (auto-saved every epoch)
2. Restart kernel
3. Load checkpoint and resume:
```python
# In training cell, add this before training loop:
checkpoint_path = CHECKPOINTS / "best.pth"
if checkpoint_path.exists():
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    # Then continue training from this point
```

---

## 📋 Post-Training Tasks

### 1. Generate Evaluation Report
After training completes, run evaluation cells to generate:
- Confusion matrix
- Per-class precision/recall/F1
- ROC curves for each class
- Fairness metrics across skin tones
- Clinical threshold optimization

### 2. Export to TFLite
```cmd
cd c:\Users\vasud\Downloads\FINAL_YEAR\Derma-Lens1\model

REM Export model
python export_pipeline.py

REM Validate export
python validate_export.py
```

**Expected outputs**:
- `checkpoints/fairdermnet.onnx`
- `checkpoints/fairdermnet_int8.tflite`
- `checkpoints/threshold_config.json`

### 3. Copy to Flutter App
```cmd
cd c:\Users\vasud\Downloads\FINAL_YEAR\Derma-Lens1\model

REM Copy model files to Flutter assets
copy checkpoints\fairdermnet_int8.tflite ..\dermalens_flutter\assets\
copy checkpoints\threshold_config.json ..\dermalens_flutter\assets\
copy assets\labels.txt ..\dermalens_flutter\assets\

echo ✓ Model deployed to Flutter app
```

### 4. Test Flutter App
```cmd
cd c:\Users\vasud\Downloads\FINAL_YEAR\Derma-Lens1\dermalens_flutter

REM Get dependencies
flutter pub get

REM Run app
flutter run
```

---

## 🎯 Quick Start Command Sequence

**Copy-paste this entire block to start training:**

```cmd
REM 1. Navigate to project
cd c:\Users\vasud\Downloads\FINAL_YEAR\Derma-Lens1\model

REM 2. Verify environment
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only!')"

REM 3. Backup current model
mkdir checkpoints_backup 2>nul
copy checkpoints\best.pth checkpoints_backup\best_before_retraining.pth 2>nul

REM 4. Start Jupyter
jupyter notebook notebooks\dermalens_train.ipynb
```

**Then in Jupyter**:
1. Update Cell 2 with new hyperparameters (EPOCHS=80, LR=5e-5, GAMMA=0.5)
2. Run all cells: Kernel → Restart & Run All
3. Wait 6-8 hours
4. Check results in Cell 8-9

---

## ✅ Final Pre-Flight Checklist

Before clicking "Run All Cells":

- [ ] Dataset verified (40,182 images)
- [ ] GPU available (CUDA test passed)
- [ ] Backup created
- [ ] Cell 2 hyperparameters updated
- [ ] Disk space >10 GB free
- [ ] No other heavy applications running
- [ ] Jupyter notebook open
- [ ] Coffee/tea ready ☕ (it's a long wait!)

---

## 📞 Support

**If anything goes wrong:**
1. Save the error message
2. Check the "Common Issues" section above
3. Check training logs in `checkpoints/training.log`
4. Review training history: `checkpoints/training_history.json`

**Critical files to preserve if training fails:**
- `checkpoints/best.pth` (best model so far)
- `checkpoints/training_history.json` (metrics)
- `checkpoints/training.log` (detailed logs)

---

## 🏆 Expected Timeline

| Time | Activity |
|------|----------|
| T+0min | Start Jupyter, run Cells 1-6 (setup) |
| T+5min | Training begins (Cell 7) |
| T+30min | First checkpoint saved (Epoch 5) |
| T+2h | Epoch 20 complete, fairness gap improving |
| T+4h | Epoch 40 complete, accuracy plateau check |
| T+6h | Epoch 60 complete, fine-tuning phase |
| T+8h | Training complete (80 epochs) |
| T+8h10min | Evaluation complete (Cell 8-9) |
| T+8h30min | Export to TFLite complete |
| T+8h45min | Flutter app updated and tested |

---

**Last Updated**: November 7, 2026  
**Status**: ✅ READY TO BEGIN TRAINING  
**Next Action**: Run environment check commands above, then start Jupyter notebook.

**Good luck! 🚀**
