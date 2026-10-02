# FairDermNet Model Quantization Guide

## 📋 Overview

This guide explains how to quantize your trained FairDermNet model from FP32 (32-bit floating point) to INT8 (8-bit integer) **without losing significant accuracy**. The goal is to reduce model size (~75% smaller) and improve inference speed while maintaining clinical performance.

---

## 🎯 Why Quantize?

### **Benefits**
- **Size Reduction:** 20-25 MB → ~6 MB (4x smaller)
- **Speed Improvement:** 2-4x faster inference on mobile CPUs
- **Memory Efficiency:** Lower RAM usage during inference
- **Battery Life:** Reduced power consumption on mobile devices

### **Trade-offs**
- **Accuracy Loss:** Typically 0.5-2% drop in accuracy (acceptable for most use cases)
- **Precision:** INT8 has less numerical precision than FP32
- **Risk:** For medical applications, must validate clinical metrics (sensitivity/specificity)

---

## 🔬 Quantization Types

### **1. Post-Training Quantization (Recommended)**
Convert a trained FP32 model → INT8 without retraining.

#### **Dynamic Range Quantization**
- **Method:** Quantize weights only (activations remain FP32)
- **Pros:** No calibration data needed, easiest to implement
- **Cons:** Minimal speed improvement, moderate accuracy drop
- **Use case:** Quick prototype

#### **Full Integer Quantization (Recommended for FairDermNet)**
- **Method:** Quantize both weights AND activations to INT8
- **Pros:** Maximum speed/size benefits, best for mobile
- **Cons:** Requires representative calibration dataset
- **Use case:** Production deployment

### **2. Quantization-Aware Training (QAT)**
Train the model with quantization in mind from the start.

- **Method:** Simulate INT8 during training (fake quantization nodes)
- **Pros:** Best accuracy retention, can even improve over baseline
- **Cons:** Requires retraining, more complex pipeline
- **Use case:** When accuracy is critical (medical applications)

---

## 🛠️ Recommended Approach for FairDermNet

### **Phase 1: Post-Training Quantization (Start Here)**
Test if PTQ gives acceptable accuracy. If yes, done! If no, proceed to Phase 2.

### **Phase 2: Quantization-Aware Training**
Retrain with QAT if PTQ accuracy drop is >2% or clinical metrics degrade.

---

## 📝 Step-by-Step: Post-Training Quantization

### **Prerequisites**
```python
# Python environment
pip install tensorflow==2.x  # or tensorflow-gpu
pip install onnx
pip install onnx-tf  # or onnx2tf
```

Your trained model in one of these formats:
- PyTorch `.pth` checkpoint
- ONNX `.onnx` file
- TensorFlow SavedModel
- Keras `.h5` file

---

### **Step 1: Convert Model to TensorFlow Format**

#### **From PyTorch → TensorFlow**
```python
import torch
import onnx
from onnx_tf.backend import prepare

# Load PyTorch model
model = YourModelClass()
model.load_state_dict(torch.load('fairdermnet.pth'))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'fairdermnet.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=13
)

# Convert ONNX → TensorFlow
onnx_model = onnx.load('fairdermnet.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('fairdermnet_tf')
```

#### **From TensorFlow (if already in TF format)**
```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('fairdermnet_fp32.h5')
# Or load SavedModel
model = tf.saved_model.load('fairdermnet_savedmodel')
```

---

### **Step 2: Prepare Calibration Dataset**

**Critical for accuracy retention!** Use representative real-world data.

```python
import numpy as np
from PIL import Image
import glob

def load_calibration_data(image_dir, num_samples=500):
    """
    Load calibration images for quantization.
    
    Guidelines:
    - Use 500-1000 diverse images from your training set
    - Cover all 8 classes proportionally
    - Include varied skin tones, lighting, angles
    - Match exact preprocessing used during training
    """
    image_paths = glob.glob(f'{image_dir}/**/*.jpg', recursive=True)
    
    # Sample diverse subset
    np.random.seed(42)
    sampled_paths = np.random.choice(image_paths, num_samples, replace=False)
    
    calibration_data = []
    for path in sampled_paths:
        # Load and preprocess (MATCH YOUR TRAINING PIPELINE EXACTLY)
        img = Image.open(path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Convert to NCHW if model expects it
        img_array = np.transpose(img_array, (2, 0, 1))
        calibration_data.append(img_array)
    
    return np.array(calibration_data)

# Generate calibration dataset
calibration_images = load_calibration_data(
    'path/to/training/images',
    num_samples=500
)
print(f'Calibration data shape: {calibration_images.shape}')
```

**Key Points:**
- ✅ Use **training set** images (model has seen these distributions)
- ✅ Include **all 8 classes** proportionally
- ✅ Match **exact preprocessing** from training
- ✅ Use **500-1000 samples** (more is better, diminishing returns after 1000)
- ❌ Don't use test set (would leak information)
- ❌ Don't use synthetic/augmented-only data

---

### **Step 3: Perform Full Integer Quantization**

```python
import tensorflow as tf

def representative_dataset_gen():
    """Generator for TFLite calibration"""
    for image in calibration_images:
        # TFLite expects batch dimension
        yield [np.expand_dims(image, axis=0).astype(np.float32)]

# Load TensorFlow model
converter = tf.lite.TFLiteConverter.from_saved_model('fairdermnet_tf')

# Configure full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

# Force INT8 for both weights and activations
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.int8
converter.inference_output_type = tf.uint8

# Optional: Enable experimental quantization features
converter.experimental_new_quantizer = True

# Convert
try:
    tflite_quant_model = converter.convert()
    print("✓ Quantization successful!")
except Exception as e:
    print(f"✗ Quantization failed: {e}")
    # Fallback: try relaxed mode
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS  # Allow some FP32 ops
    ]
    tflite_quant_model = converter.convert()
    print("⚠ Quantization with fallback (some layers remain FP32)")

# Save quantized model
with open('fairdermnet_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print(f"Model size: {len(tflite_quant_model) / 1024 / 1024:.2f} MB")
```

---

### **Step 4: Validate Quantized Model**

**Critical:** Test both accuracy AND clinical metrics.

```python
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def load_tflite_model(model_path):
    """Load TFLite interpreter"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, image):
    """Run inference on single image"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def validate_quantized_model(tflite_path, test_loader):
    """
    Validate quantized model on test set.
    
    Args:
        tflite_path: Path to quantized .tflite file
        test_loader: PyTorch/TF DataLoader with (images, labels)
    
    Returns:
        dict: Validation metrics
    """
    interpreter = load_tflite_model(tflite_path)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in test_loader:
        for i in range(len(images)):
            image = images[i].numpy()
            label = labels[i].item()
            
            # Run inference
            logits = run_inference(interpreter, np.expand_dims(image, 0))
            
            # Apply softmax
            probs = np.exp(logits) / np.sum(np.exp(logits))
            pred = np.argmax(probs)
            
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(probs[0])
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Classification report
    class_names = [
        'Melanoma', 'BCC', 'SCC', 'AK', 
        'Nevi', 'BenignKeratosis', 'Dermatofibroma', 'Vascular'
    ]
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # Clinical metrics (binary: malignant vs benign)
    malignant_indices = {0, 1, 2, 3}
    binary_labels = [1 if l in malignant_indices else 0 for l in all_labels]
    
    # Malignant probability = sum of probs for classes 0-3
    malignant_probs = [sum(p[i] for i in malignant_indices) for p in all_probs]
    binary_preds = [1 if p >= 0.5 else 0 for p in malignant_probs]
    
    sensitivity = np.mean([p == 1 for p, l in zip(binary_preds, binary_labels) if l == 1])
    specificity = np.mean([p == 0 for p, l in zip(binary_preds, binary_labels) if l == 0])
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }

# Run validation
metrics = validate_quantized_model('fairdermnet_int8.tflite', test_loader)

print(f"Quantized Model Accuracy: {metrics['accuracy']:.4f}")
print(f"Sensitivity (malignant detection): {metrics['sensitivity']:.4f}")
print(f"Specificity (benign detection): {metrics['specificity']:.4f}")
print("\nClassification Report:")
print(metrics['classification_report'])
```

---

### **Step 5: Compare FP32 vs INT8**

```python
def compare_models(fp32_path, int8_path, test_loader):
    """Compare original and quantized models"""
    
    # Load both models
    fp32_model = tf.keras.models.load_model(fp32_path)
    int8_interp = load_tflite_model(int8_path)
    
    # Test on same data
    fp32_metrics = validate_model(fp32_model, test_loader)
    int8_metrics = validate_quantized_model(int8_path, test_loader)
    
    # Print comparison
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'FP32':<15} {'INT8':<15} {'Δ':<10}")
    print("-" * 60)
    
    for metric in ['accuracy', 'sensitivity', 'specificity']:
        fp32_val = fp32_metrics[metric]
        int8_val = int8_metrics[metric]
        delta = int8_val - fp32_val
        print(f"{metric:<25} {fp32_val:.4f}         {int8_val:.4f}         {delta:+.4f}")
    
    # Model size
    import os
    fp32_size = os.path.getsize(fp32_path) / 1024 / 1024
    int8_size = os.path.getsize(int8_path) / 1024 / 1024
    print(f"{'Model Size (MB)':<25} {fp32_size:.2f}          {int8_size:.2f}          {int8_size/fp32_size:.2%}")
    
    print("=" * 60)
    
    # Decision criteria
    accuracy_drop = fp32_metrics['accuracy'] - int8_metrics['accuracy']
    sensitivity_drop = fp32_metrics['sensitivity'] - int8_metrics['sensitivity']
    
    if accuracy_drop < 0.02 and sensitivity_drop < 0.02:
        print("✓ PASS: Quantization acceptable (<2% accuracy/sensitivity drop)")
        return True
    else:
        print("✗ FAIL: Quantization caused significant performance degradation")
        print("   → Consider Quantization-Aware Training (QAT)")
        return False

# Run comparison
is_acceptable = compare_models('fairdermnet_fp32.h5', 'fairdermnet_int8.tflite', test_loader)
```

---

## 🎓 Advanced: Quantization-Aware Training (QAT)

**Use if Post-Training Quantization accuracy drop is >2%**

### **Concept**
Train the model with "fake quantization" nodes that simulate INT8 precision during forward/backward passes. The model learns to compensate for quantization errors.

### **Implementation (TensorFlow)**

```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Load pre-trained FP32 model
base_model = tf.keras.models.load_model('fairdermnet_fp32.h5')

# Apply quantization-aware training
quantize_model = tfmot.quantization.keras.quantize_model

# Annotate model for QAT
qat_model = quantize_model(base_model)

# Compile with same optimizer/loss as original training
qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune for 5-10 epochs
qat_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)

# Convert to TFLite INT8
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_qat_model = converter.convert()

with open('fairdermnet_int8_qat.tflite', 'wb') as f:
    f.write(tflite_qat_model)
```

### **Implementation (PyTorch)**

```python
import torch
import torch.quantization as tq

# Load pre-trained model
model = YourModelClass()
model.load_state_dict(torch.load('fairdermnet.pth'))

# Prepare model for QAT
model.train()
model.qconfig = tq.get_default_qat_qconfig('fbgemm')  # x86 backend
tq.prepare_qat(model, inplace=True)

# Fine-tune with fake quantization
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validate
    val_accuracy = evaluate(model, val_loader)
    print(f"Epoch {epoch}: Val Accuracy = {val_accuracy:.4f}")

# Convert to quantized model
model.eval()
model_int8 = tq.convert(model)

# Export to ONNX → TFLite (use conversion script from Step 1)
torch.onnx.export(model_int8, dummy_input, 'fairdermnet_qat.onnx')
```

---

## 🧪 Calibration Dataset Best Practices

### **Size Guidelines**
- **Minimum:** 100 images (bare minimum, not recommended)
- **Good:** 500 images (standard practice)
- **Better:** 1000 images (diminishing returns beyond this)
- **Overkill:** >2000 images (no significant improvement)

### **Diversity Requirements**
```python
# Example: Balanced calibration sampling
class_distribution = {
    'Melanoma': 80,           # 16%
    'BCC': 70,                # 14%
    'SCC': 50,                # 10%
    'AK': 60,                 # 12%
    'Nevi': 80,               # 16%
    'BenignKeratosis': 70,    # 14%
    'Dermatofibroma': 50,     # 10%
    'Vascular': 40            # 8%
}
# Total: 500 images
```

### **Skin Tone Diversity (Critical for Fairness)**
Based on Fitzpatrick scale:
- Type I-II (Light): 20%
- Type III-IV (Medium): 50%
- Type V-VI (Dark): 30%

### **Image Quality Checks**
```python
def validate_calibration_image(image):
    """Check if image is suitable for calibration"""
    # Check resolution
    if image.shape[:2] != (224, 224):
        return False, "Wrong resolution"
    
    # Check brightness (avoid over/underexposed)
    mean_brightness = np.mean(image)
    if mean_brightness < 30 or mean_brightness > 225:
        return False, "Poor exposure"
    
    # Check for artifacts
    if np.std(image) < 10:
        return False, "Too uniform (possibly corrupted)"
    
    return True, "OK"
```

---

## 📊 Acceptance Criteria

### **Quantization Success Metrics**

| Metric | Threshold | Critical? |
|--------|-----------|-----------|
| **Accuracy Drop** | <2% | ✅ Yes |
| **Sensitivity Drop** | <2% | ✅✅ Critical (medical) |
| **Specificity Drop** | <3% | ⚠️ Acceptable |
| **Model Size** | <10 MB | ✅ Yes |
| **Inference Time** | <500ms | ⚠️ Target |

### **Example Decision Tree**
```
Accuracy Drop < 1% AND Sensitivity Drop < 1%
  → ✓ Deploy INT8 model

Accuracy Drop 1-2% AND Sensitivity Drop < 2%
  → ⚠️ Clinical review required, likely OK

Accuracy Drop > 2% OR Sensitivity Drop > 2%
  → ✗ Try QAT or use FP32 model
```

---

## 🔧 Troubleshooting

### **Problem: Accuracy drops >5%**
**Causes:**
- Calibration dataset not representative
- Preprocessing mismatch (model trained with different normalization)
- Model architecture has quantization-sensitive layers (e.g., LayerNorm)

**Solutions:**
1. Increase calibration dataset size to 1000+ images
2. Verify preprocessing matches training EXACTLY
3. Try per-channel quantization: `converter.experimental_new_quantizer = True`
4. Use QAT instead of PTQ

### **Problem: Inference outputs nonsense**
**Causes:**
- Input/output tensor types mismatched
- NCHW vs NHWC layout confusion
- Quantization scale/zero-point incorrect

**Solutions:**
1. Check input tensor dtype: `interpreter.get_input_details()[0]['dtype']`
2. Verify tensor shapes match expected layout
3. Use `interpreter.get_tensor_details()` to inspect quantization params

### **Problem: Model too large (>10 MB)**
**Causes:**
- Quantization didn't apply to all layers
- Some layers fell back to FP32

**Solutions:**
1. Check converter logs for FP32 fallback warnings
2. Use strict INT8 mode: `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`
3. Simplify model architecture (remove custom layers)

---

## 📁 Final Checklist

Before deploying quantized model to production:

- [ ] Quantized model passes accuracy threshold (<2% drop)
- [ ] Clinical metrics validated (sensitivity/specificity)
- [ ] Tested on diverse skin tones (Fitzpatrick I-VI)
- [ ] Inference time measured on target devices (<500ms)
- [ ] Model file size verified (<10 MB)
- [ ] `threshold_config.json` updated with calibrated threshold
- [ ] Edge cases tested (poor lighting, blurry images, etc.)
- [ ] Regulatory/compliance review if applicable

---

## 🚀 Deployment Script

Complete end-to-end script:

```bash
#!/bin/bash
# deploy_quantized_model.sh

echo "Starting FairDermNet quantization pipeline..."

# 1. Convert PyTorch → ONNX → TensorFlow
python scripts/convert_to_tf.py \
    --pytorch_model models/fairdermnet.pth \
    --output_dir models/tf

# 2. Prepare calibration data
python scripts/prepare_calibration.py \
    --data_dir data/train \
    --num_samples 500 \
    --output calibration_data.npy

# 3. Quantize model
python scripts/quantize_model.py \
    --tf_model models/tf \
    --calibration calibration_data.npy \
    --output models/fairdermnet_int8.tflite

# 4. Validate quantized model
python scripts/validate_quantized.py \
    --tflite_model models/fairdermnet_int8.tflite \
    --test_dir data/test \
    --output validation_report.json

# 5. Benchmark inference speed
python scripts/benchmark.py \
    --model models/fairdermnet_int8.tflite \
    --num_runs 100 \
    --output benchmark_results.json

# 6. Copy to Flutter assets
cp models/fairdermnet_int8.tflite ../dermalens_flutter/assets/
echo "✓ Quantized model deployed to Flutter app!"
```

---

## 📚 References

- [TensorFlow Lite Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [INT8 Quantization: Deep Dive](https://arxiv.org/abs/1806.08342)
- [Medical AI Validation Guidelines](https://www.fda.gov/medical-devices/software-medical-device-samd)

---

**Last Updated:** August 10, 2026  
**Author:** DermaLens Team  
**Status:** Production-Ready Guide
