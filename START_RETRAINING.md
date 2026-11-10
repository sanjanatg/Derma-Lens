# 🚀 START HERE: DermaLens Model Retraining

**Quick Start Guide - 5 Minutes to Begin Training**

---

## 📍 Current Status

✅ **Dataset Ready**: 37,644 images loaded with 8 disease classes  
✅ **Skin Tones**: Fitzpatrick labels populated  
✅ **Training Code**: Notebook configured and tested  
🎯 **Goal**: Improve accuracy from ~70% to 78-82%, reduce fairness gap from 26% to <15%

---

## ⚡ Quick Start (3 Steps)

### Step 1: Verify Environment (2 minutes)

Open Command Prompt and run:

```cmd
cd c:\Users\vasud\Downloads\FINAL_YEAR\Derma-Lens1

python model\verify_training_readiness.py
```

**Expected Output**: `✅ ALL CHECKS PASSED - READY FOR TRAINING`

**If checks fail**: Install missing packages or fix issues shown.

---

### Step 2: Backup Current Model (1 minute)

```cmd
cd c:\Users\vasud\Downloads\FINAL_YEAR\Derma-Lens1\model

mkdir checkpoints_backup

copy checkpoints\best.pth checkpoints_backup\best_before_retraining.pth
```

---

### Step 3: Start Training (2 minutes setup + 6-8 hours training)

```cmd
cd c:\Users\vasud\Downloads\FINAL_YEAR\Derma-Lens1\model

jupyter notebook notebooks\dermalens_train.ipynb
```

**In Jupyter Notebook:**

1. **Update Cell 2** with these improved hyperparameters:
   ```python
   # Find these lines in Cell 2 and update:
   EPOCHS = 80              # Change from 30 to 80
   LR = 5e-5               # Change from 1e-4 to 5e-5
   GAMMA = 0.5             # Change from 0.25 to 0.5
   PATIENCE = 15           # Change from 7 to 15
   ```

2. **Run All Cells**: Click `Kernel → Restart & Run All`

3. **Monitor Progress**: Training will show progress every epoch
   ```
   Epoch 1/80 | Loss: 0.45 | Val Acc: 0.68 | Fairness Gap: 0.26
   Epoch 2/80 | Loss: 0.42 | Val Acc: 0.71 | Fairness Gap: 0.24
   ...
   ```

4. **Wait 6-8 hours** for training to complete

---

## 📊 What to Expect

### Training Progress Indicators

**Good Signs** ✅:
- Loss steadily decreasing
- Validation accuracy climbing toward 78%+
- Fairness gap decreasing toward 15%
- No "out of memory" errors

**Warning Signs** ⚠️:
- Loss stuck or increasing → May need to adjust learning rate
- Accuracy not improving after 30 epochs → Check hyperparameters
- Out of memory error → Reduce batch size in Cell 2

### Final Results (After 80 Epochs)

**Target Metrics**:
- ✅ Balanced Accuracy: **78-82%** (up from ~70%)
- ✅ Fairness Gap: **<15%** (down from 26%)
- ✅ Test AUC: **≥0.93** (up from 0.92)
- ✅ All classes recall: **>55%** (including rare classes)

---

## 🔧 Troubleshooting

### Problem: "CUDA out of memory"

**Solution**: In notebook Cell 2, change:
```python
BATCH_SIZE = 8      # Reduce from 16
GRAD_ACCUM = 8      # Increase from 4
```

### Problem: "No module named 'torch'"

**Solution**: Install PyTorch with CUDA:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Problem: Training too slow (>10 hours)

**Check GPU usage**:
```cmd
nvidia-smi
```
Should show 95-100% GPU utilization. If not, close other GPU apps.

---

## 📝 After Training Completes

### 1. Review Results
The notebook will automatically show:
- Training curves (loss, accuracy, fairness)
- Confusion matrix
- Per-class performance
- Fairness evaluation on DDI dataset

### 2. Export Model
```cmd
cd c:\Users\vasud\Downloads\FINAL_YEAR\Derma-Lens1\model

python export_pipeline.py
```

This creates:
- `checkpoints/fairdermnet_int8.tflite` (optimized for mobile)
- `checkpoints/threshold_config.json` (clinical thresholds)

### 3. Deploy to Flutter App
```cmd
copy checkpoints\fairdermnet_int8.tflite ..\dermalens_flutter\assets\
copy checkpoints\threshold_config.json ..\dermalens_flutter\assets\
```

### 4. Test Flutter App
```cmd
cd ..\dermalens_flutter

flutter pub get
flutter run
```

---

## 📚 Documentation

- **Full Checklist**: See `RETRAINING_CHECKLIST.md` for detailed steps
- **Improvement Plan**: See `IMPROVEMENT_PLAN_PHASE1.md` for rationale
- **Training Notebook**: `model/notebooks/dermalens_train.ipynb`

---

## 🎯 Success Criteria

Training is successful when:
- [x] All 80 epochs completed (or early stopping triggered appropriately)
- [x] Balanced accuracy ≥78%
- [x] Fairness gap <15%
- [x] Model exports to TFLite without errors
- [x] Flutter app loads and runs the new model

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Environment verification | 2 min |
| Backup current model | 1 min |
| Update notebook config | 2 min |
| **Training (80 epochs)** | **6-8 hours** |
| Evaluation | 10 min |
| Export to TFLite | 10 min |
| Deploy to Flutter | 5 min |
| **Total hands-on time** | **~30 min** |
| **Total wall time** | **6-8 hours** |

---

## 🚀 Ready? Let's Go!

**Copy and paste these commands to start:**

```cmd
REM Step 1: Verify
cd c:\Users\vasud\Downloads\FINAL_YEAR\Derma-Lens1
python model\verify_training_readiness.py

REM Step 2: Backup
cd model
mkdir checkpoints_backup
copy checkpoints\best.pth checkpoints_backup\best_before_retraining.pth

REM Step 3: Start Jupyter
jupyter notebook notebooks\dermalens_train.ipynb
```

Then in Jupyter:
1. Update Cell 2 hyperparameters (EPOCHS=80, LR=5e-5, GAMMA=0.5)
2. Click: Kernel → Restart & Run All
3. Go grab coffee ☕ (6-8 hour wait)

---

**Questions or issues?** Check `RETRAINING_CHECKLIST.md` for detailed troubleshooting.

**Good luck! Your improved model awaits! 🎉**

---

**Last Updated**: November 7, 2026  
**Status**: ✅ Ready to Begin
