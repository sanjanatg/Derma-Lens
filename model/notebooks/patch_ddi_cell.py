"""Patch dermalens_train.ipynb — idempotent fix set (safe to re-run):

1. Cell 4  (config):        add FORCE_RETRAIN = False
2. Cell 18 (training loop): skip training when checkpoints/best.pth exists and
                            FORCE_RETRAIN is False; rebuild `history` from
                            training_history.json so cells 20/34 still work
                            after a kernel restart ("Restart & Run All" resume).
3. Cell 28 (fairness bars): read fairness_metrics["group_metrics"] instead of
                            the nonexistent report.per_group_* dict dot-access.
4. Cell 24 (DDI eval):      save TEST-SET tp/tn/fp/fn/sens/spec under test_*
                            names before the DDI block reuses those names.
5. Cell 32 (export):        fix output filename fairdermet_int8 -> fairdermnet_int8
                            (cell 34 and export/fairdermnet_int8.tflite use 'n').
6. Cell 34 (final summary): recompute tp/fp/fn/sens/spec/f1/AUC from the
                            TEST-SET variables instead of trusting the bare
                            names that Cell 9b clobbers.
"""
import json
import subprocess
from pathlib import Path

import nbformat

NB = Path(__file__).resolve().parent / "dermalens_train.ipynb"

CELL18 = '''# Resume guard: if a checkpoint from a prior run exists, skip the (expensive)
# training loop and rebuild `history` from the saved per-epoch log instead, so
# downstream cells (curves, final summary) still work after a kernel restart.
# Set FORCE_RETRAIN = True in the config cell to retrain from scratch.
import json

if not FORCE_RETRAIN and (CHECKPOINTS / "best.pth").exists():
    print(f"Checkpoint found: {CHECKPOINTS / 'best.pth'}")
    print("Skipping training loop (set FORCE_RETRAIN = True to retrain).")
    history = []
    _hist_path = CHECKPOINTS / "training_history.json"
    if _hist_path.exists():
        with open(_hist_path) as f:
            history = json.load(f)
        print(f"Rebuilt history from training_history.json ({len(history)} epochs).")
    else:
        print("No training_history.json — history left empty (curves cell will be blank).")
else:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    optimiser  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=10, T_mult=2, eta_min=1e-6)
    scaler     = GradScaler(growth_interval=200) if USE_AMP else None
    stopper    = EarlyStopping(patience=PATIENCE, mode="max")
    ckpt_saver = ModelCheckpoint(path=CHECKPOINTS / "best.pth", monitor="balanced_accuracy", mode="max")
    history    = []

    print(f"Starting training — {EPOCHS} epochs | effective batch = {BATCH_SIZE}×{GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM}")
    print(f"RTX 3050 AMP: {'ON (fp16)' if USE_AMP else 'OFF'} | 8-class mode\\n")

    try:
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            train_preds, train_labels_ep, train_tones = [], [], []
            optimiser.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [train]", leave=False, ncols=90)
            for step, batch in enumerate(pbar):
                imgs         = batch["image"].to(DEVICE, non_blocking=True)
                class_labels = batch["class_idx"].to(DEVICE, non_blocking=True)
                tones        = batch["skin_tone"].to(DEVICE, non_blocking=True)

                with autocast('cuda', enabled=USE_AMP):
                    risk_logits, fitz_logits = model(imgs)
                    loss, breakdown = criterion(risk_logits, fitz_logits, class_labels, tones)
                    loss = loss / GRAD_ACCUM

                if USE_AMP:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % GRAD_ACCUM == 0:
                    if USE_AMP:
                        scaler.unscale_(optimiser)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimiser)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimiser.step()
                    optimiser.zero_grad()
                    scheduler.step(epoch + step / len(train_loader))

                train_loss += breakdown["loss_total"]
                with torch.no_grad():
                    p = torch.argmax(risk_logits, dim=-1).cpu().numpy()  # predicted class idx
                train_preds.extend(p)
                train_labels_ep.extend(class_labels.cpu().numpy())
                train_tones.extend(tones.cpu().numpy())
                pbar.set_postfix(loss=f"{breakdown['loss_total']:.4f}")

            model.eval()
            val_loss = 0.0
            val_preds, val_labels_ep, val_tones = [], [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [val]", leave=False, ncols=90):
                    imgs         = batch["image"].to(DEVICE, non_blocking=True)
                    class_labels = batch["class_idx"].to(DEVICE, non_blocking=True)
                    tones        = batch["skin_tone"].to(DEVICE, non_blocking=True)
                    model.train()
                    with autocast('cuda', enabled=USE_AMP):
                        risk_logits, fitz_logits = model(imgs)
                        _, vbd = criterion(risk_logits, fitz_logits, class_labels, tones)
                    model.eval()
                    val_loss += vbd["loss_total"]
                    p = torch.argmax(risk_logits, dim=-1).cpu().numpy()
                    val_preds.extend(p)
                    val_labels_ep.extend(class_labels.cpu().numpy())
                    val_tones.extend(tones.cpu().numpy())

            n_train  = max(1, len(train_loader))
            n_val    = max(1, len(val_loader))
            t_labels = np.array(train_labels_ep)
            t_preds  = np.array(train_preds)
            v_labels = np.array(val_labels_ep)
            v_preds  = np.array(val_preds)
            v_tones  = np.array(val_tones)

            t_m = {
                "accuracy":          accuracy_score(t_labels, t_preds),
                "balanced_accuracy": balanced_accuracy_score(t_labels, t_preds),
            }
            v_m = {
                "accuracy":          accuracy_score(v_labels, v_preds),
                "balanced_accuracy": balanced_accuracy_score(v_labels, v_preds),
            }

            # Fairness gap: binary malignant (class<5) sensitivity per skin tone
            v_bin_true = (v_labels < 5).astype(int)
            v_bin_pred = (v_preds  < 5).astype(int)
            tone_sens  = {}
            for t_idx in range(6):
                mask = v_tones == t_idx
                if mask.sum() < 5:
                    continue
                tp = int(((v_bin_pred[mask]==1) & (v_bin_true[mask]==1)).sum())
                fn = int(((v_bin_pred[mask]==0) & (v_bin_true[mask]==1)).sum())
                if tp + fn > 0:
                    tone_sens[t_idx] = tp / (tp + fn)
            fairness_gap = (max(tone_sens.values()) - min(tone_sens.values())
                            if len(tone_sens) >= 2 else 0.0)

            lr_now    = optimiser.param_groups[0]["lr"]
            vram_used = torch.cuda.memory_reserved(0) / 1e9 if CUDA_AVAILABLE else 0.0

            log = {
                "epoch":                   epoch + 1,
                "train_loss":              train_loss / n_train,
                "val_loss":                val_loss / n_val,
                "train_accuracy":          t_m["accuracy"],
                "train_balanced_accuracy": t_m["balanced_accuracy"],
                "val_accuracy":            v_m["accuracy"],
                "val_balanced_accuracy":   v_m["balanced_accuracy"],
                "val_fairness_gap":        fairness_gap,
                "lr":                      lr_now,
                "vram_gb":                 round(vram_used, 2),
            }
            history.append(log)
            print(f"Ep {epoch+1:02d} | Loss {log['train_loss']:.4f}→{log['val_loss']:.4f} | "
                  f"BalAcc {log['val_balanced_accuracy']:.4f} | Acc {log['val_accuracy']:.4f} | "
                  f"Gap {log['val_fairness_gap']:.3f} | LR {lr_now:.1e} | VRAM {vram_used:.2f}GB")

            # Use correct callback API: step(model, optimizer, epoch, metrics)
            ckpt_saver.step(model, optimiser, epoch, v_m)
            if stopper.step(v_m["balanced_accuracy"]):
                print(f"\\n⏹ Early stopping at epoch {epoch+1}.")
                break

            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
            gc.collect()

            if vram_used > 5.8:
                print(f"\\n⚠ VRAM near limit ({vram_used:.2f} GB). Reduce BATCH_SIZE to 8 in Cell 2 and restart.")
                break

    except torch.cuda.OutOfMemoryError:
        print("\\n⚠ CUDA Out of Memory! Reduce BATCH_SIZE to 8 in Cell 2 and restart the notebook.")
        raise

    print(f"\\n✓ Training complete. Best checkpoint: {CHECKPOINTS}/best.pth")
'''

CELL28 = '''FITZ_SHORT = ["I\\n(fair)", "II", "III\\n(med)", "IV", "V", "VI\\n(dark)"]
FITZ_COLORS = ["#FFE4C4","#FFDEAD","#D2B48C","#C19A6B","#8B6914","#4A2511"]

# Cell 9's fairness breakdown lives in `fairness_metrics["group_metrics"]`
# (int key -> {accuracy/sensitivity/specificity as fractions}). The old
# `report.per_group_sensitivity` dot-access never existed — report is a plain
# dict with "per_group" rows in percent.
gm = fairness_metrics.get("group_metrics", {}) if "fairness_metrics" in globals() else {}

if not gm:
    print("No per-group fairness data available — run Cell 9 (Test set evaluation) first. Skipping chart.")
else:
    sens_vals = [gm.get(t, {}).get("sensitivity", 0) for t in range(6)]
    spec_vals = [gm.get(t, {}).get("specificity", 0) for t in range(6)]
    x = np.arange(6)
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, sens_vals, w, label="Sensitivity", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + w/2, spec_vals, w, label="Specificity", color="darkorange", alpha=0.85)
    ax.axhline(0.962, color="steelblue", linestyle="--", alpha=0.6, linewidth=1.2, label="Sens. target")
    ax.axhline(0.850, color="darkorange", linestyle="--", alpha=0.6, linewidth=1.2, label="Spec. target")
    for bar in [*bars1, *bars2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    eo_gap = fairness_metrics.get("equalized_odds_gap", 0)
    ax.annotate(f"Equalized Odds Gap = {eo_gap*100:.1f}%  {'✓ TARGET MET' if eo_gap < 0.05 else '✗ Above target'}", xy=(0.02, 0.05), xycoords="axes fraction", fontsize=9, color="green" if eo_gap < 0.05 else "red", bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.set_xticks(x)
    ax.set_xticklabels(FITZ_SHORT, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Sensitivity & Specificity per Fitzpatrick Skin Type", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fairness_by_skin_type.png", dpi=150)
    plt.show()
    print(f"Saved: {FIGURES_DIR}/fairness_by_skin_type.png")
'''

SAVE_TEST_BLOCK = '''# --- Save TEST-SET binary counts under test_* names before the DDI block
# below reuses tp/tn/fp/fn for its own per-group arithmetic (Cell 13 reads
# the test-set numbers, so they must not be clobbered). ---
test_tp, test_tn = tp, tn
test_fp, test_fn = fp, fn
test_sens, test_spec = sens, spec
test_true_binary = true_binary

'''

CELL34_NEW_HEAD = '''best_epoch = max(history, key=lambda h: h["val_balanced_accuracy"])
from sklearn.metrics import roc_auc_score

# Recompute summary metrics from the TEST-SET variables (Cell 9). Cell 9b
# reuses the bare names tp/tn/fp/fn for DDI counts, so never trust them here.
tp, fp, fn, tn = test_tp, test_fp, test_fn, test_tn
sens = tp / (tp + fn + 1e-8)
spec = tn / (tn + fp + 1e-8)
f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
try:
    auc = roc_auc_score(test_true_binary, pred_probs[:, list(MALIGNANT_CLASSES)].sum(axis=1))
except Exception:
    auc = float("nan")'''

CELL34_OLD_HEAD = '''best_epoch = max(history, key=lambda h: h["val_balanced_accuracy"])
from sklearn.metrics import roc_auc_score
try:
    auc = roc_auc_score(true_binary, pred_probs[:, :4].sum(axis=1))
except Exception:
    auc = float("nan")
f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)'''


def src(nb, i):
    return "".join(nb.cells[i].source)


def set_src(nb, i, s):
    nb.cells[i].source = s


nb = nbformat.read(NB, as_version=4)
applied, skipped = [], []

# 1. config cell — FORCE_RETRAIN
s4 = src(nb, 4)
assert "EPOCHS         = 30" in s4, "cell 4 doesn't look like the config cell"
if "FORCE_RETRAIN" not in s4:
    set_src(nb, 4, s4.replace(
        "EPOCHS         = 30",
        "EPOCHS         = 30\nFORCE_RETRAIN = False  # True = retrain from scratch even if checkpoints/best.pth exists",
    ))
    applied.append("cell 4: FORCE_RETRAIN = False added to config")
else:
    skipped.append("cell 4: already has FORCE_RETRAIN")

# 2. training loop — resume guard
s18 = src(nb, 18)
if "FORCE_RETRAIN" not in s18:
    assert "Starting training" in s18 and "history" in s18, "cell 18 doesn't look like the training cell"
    set_src(nb, 18, CELL18)
    applied.append("cell 18: resume guard added (skips training if checkpoints/best.pth exists)")
else:
    skipped.append("cell 18: resume guard already present")

# 3. fairness bar charts — group_metrics fix
s28 = src(nb, 28)
if "group_metrics" not in s28:
    assert "per_group_sensitivity" in s28, "cell 28 doesn't look like the fairness-chart cell"
    set_src(nb, 28, CELL28)
    applied.append("cell 28: reads fairness_metrics['group_metrics'] instead of report.per_group_*")
else:
    skipped.append("cell 28: group_metrics fix already present")

# 4. DDI eval — rebuild from the git-committed canonical cell, then (a) insert
#    the test_* save block with correct (4-space) indentation and (b) force
#    num_workers=0: Windows spawn re-imports __main__ in worker processes, and
#    SkinLesionDataset is defined inside the notebook (cell 12), so workers
#    can't unpickle it -> spawn crash -> DataLoader hangs forever.
ANCHOR24 = "\n    ddi_preds, ddi_labels, ddi_tones = [], [], []"
SAVE24 = (
    "\n    # --- Save TEST-SET binary counts under test_* names before the DDI block\n"
    "    # below reuses tp/tn/fp/fn for its own per-group arithmetic (Cell 13 reads\n"
    "    # the test-set numbers, so they must not be clobbered). ---\n"
    "    test_tp, test_tn = tp, tn\n"
    "    test_fp, test_fn = fp, fn\n"
    "    test_sens, test_spec = sens, spec\n"
    "    test_true_binary = true_binary\n"
    "\n    ddi_preds, ddi_labels, ddi_tones = [], [], []"
)

committed24 = json.loads(
    subprocess.run(
        ["git", "show", "HEAD:model/notebooks/dermalens_train.ipynb"],
        capture_output=True, check=True,
    ).stdout.decode("utf-8")
)["cells"][24]
canon24 = "".join(committed24["source"])
assert "DDI FAIRNESS EVALUATION" in canon24, "canonical cell 24 mismatch"
assert ANCHOR24 in canon24, "canonical cell 24: anchor not found"

s24_new = canon24.replace(ANCHOR24, SAVE24, 1)
if "num_workers=NUM_WORKERS," in s24_new:
    s24_new = s24_new.replace(
        "num_workers=NUM_WORKERS,",
        "num_workers=0,  # Windows spawn can't unpickle notebook-defined SkinLesionDataset",
        1,
    )
    ddi_note = ", num_workers fixed 4 -> 0"
else:
    ddi_note = ""
set_src(nb, 24, s24_new)
applied.append(f"cell 24: rebuilt from HEAD + test_* save block{ddi_note}")

# 5. export — TFLite filename typo
s32 = src(nb, 32)
if "fairdermet_int8.tflite" in s32:
    set_src(nb, 32, s32.replace("fairdermet_int8.tflite", "fairdermnet_int8.tflite"))
    applied.append("cell 32: TFLite filename fixed fairdermet_int8 -> fairdermnet_int8")
else:
    skipped.append("cell 32: filename already correct")

# 5b. GradCAM cell (Cell 11) — rebuild from HEAD canon with pytorch-grad-cam
#     modern API: targets=[ClassifierOutputTarget(k)] instead of the removed
#     target_class kwarg, show_cam_on_image instead of the nonexistent
#     BaseCAM.overlay, and release() instead of the removed remove_hooks().
import subprocess

committed30 = json.loads(
    subprocess.run(
        ["git", "show", "HEAD:model/notebooks/dermalens_train.ipynb"],
        capture_output=True, check=True,
    ).stdout.decode("utf-8")
)["cells"][30]
canon30 = "".join(committed30["source"])

REPL30 = [
    (
        "target_layer = best_model.get_gradcam_target_layer()",
        "import numpy as np\n"
        "from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget\n"
        "from pytorch_grad_cam.utils.image import show_cam_on_image\n"
        "target_layer = best_model.get_gradcam_target_layer()",
    ),
    (
        "heatmap = cam_gen(tensor, target_class=1)",
        "# squeeze batch dim: CAMs come back as (1, H, W); OpenCV colormap wants 2-D\n"
        "    heatmap = np.squeeze(cam_gen(tensor, targets=[ClassifierOutputTarget(1)]))",
    ),
    (
        "overlay = cam_gen.overlay(img_rgb_224, heatmap, alpha=0.45)",
        "hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)\n"
        "    overlay = show_cam_on_image(img_rgb_224 / 255.0, hm_norm.astype(\"float32\"), use_rgb=True, image_weight=0.55)",
    ),
    (
        "cam_gen.remove_hooks()",
        "cam_gen.activations_and_grads.release()",
    ),
]

s30 = src(nb, 30)
if "np.squeeze(cam_gen" not in s30:
    for old, new in REPL30:
        assert old in canon30, f"cell 30 canonical anchor missing: {old[:60]!r}"
        canon30 = canon30.replace(old, new, 1)
    set_src(nb, 30, canon30)
    applied.append("cell 30: GradCAM++ modernized (targets=, show_cam_on_image, release())")
else:
    skipped.append("cell 30: GradCAM modern API already present")

# 6. final summary — recompute from test-set variables
s34 = src(nb, 34)
if "test_tp" not in s34:
    assert CELL34_OLD_HEAD in s34, "cell 34 head block not found"
    set_src(nb, 34, s34.replace(CELL34_OLD_HEAD, CELL34_NEW_HEAD))
    applied.append("cell 34: summary metrics recomputed from test_* variables")
else:
    skipped.append("cell 34: test_* recompute already present")

# 6b. final summary — tolerate a failed TFLite conversion (tflite_mb is None
#     would crash the format string).
OLD_TFLITE_LINE = (
    '    print(f"  TFLite size:        {tflite_mb:.2f} MB  '
    "{'✓' if tflite_mb and tflite_mb < 6.0 else '✗'}\")\n"
)
NEW_TFLITE_LINE = (
    "    if tflite_mb is None:\n"
    "        print(\"  TFLite size:        n/a (conversion failed — see Cell 12 log)\")\n"
    "    else:\n"
    "        print(f\"  TFLite size:        {tflite_mb:.2f} MB  {'✓' if tflite_mb < 6.0 else '✗'}\")\n"
)
if "tflite_mb is None" not in s34:
    if OLD_TFLITE_LINE not in s34:
        raise SystemExit("ABORT: cell 34 TFLite print line not found")
    set_src(nb, 34, s34.replace(OLD_TFLITE_LINE, NEW_TFLITE_LINE, 1))
    applied.append("cell 34: summary survives tflite_mb=None")
else:
    skipped.append("cell 34: tflite_mb guard already present")

# 7. DDI loader — num_workers=0 (applied as part of fix 4 above)

# 8. export cell (Cell 12) — opset pin + ONNX filename. onnx-tf 1.10 cannot
#    convert opset-17 models (Unsqueeze v13 unimplemented); the shipped Aug 16
#    pipeline was traced at opset 11, which onnx_export.py documents as the pin.
s32b = src(nb, 32)
changed32 = []
if "opset_version=17" in s32b:
    s32b = s32b.replace("opset_version=17", "opset_version=11", 1)
    changed32.append("opset 17 -> 11 (onnx-tf compat)")
if 'EXPORT_DIR / "fairdermet.onnx"' in s32b:
    s32b = s32b.replace('EXPORT_DIR / "fairdermet.onnx"', 'EXPORT_DIR / "fairdermnet.onnx"', 1)
    changed32.append("ONNX filename fairdermet -> fairdermnet")
if changed32:
    set_src(nb, 32, s32b)
    applied.append("cell 32: " + "; ".join(changed32))
else:
    skipped.append("cell 32: opset/filename already correct")

# Final safety net: every code cell must compile before we write the notebook.
for idx, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    try:
        compile("".join(cell.source), f"cell{idx}", "exec")
    except SyntaxError as e:
        raise SystemExit(f"ABORT: cell {idx} does not compile: {e}")

nbformat.write(nb, NB)

print("applied:")
for a in applied:
    print("  ✓", a)
for s in skipped:
    print("  -", s)
