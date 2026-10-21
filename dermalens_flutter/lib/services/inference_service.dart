import 'dart:convert';
import 'dart:io' show File;

import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../providers/analysis_provider.dart';

// ─────────────────────────────────────────────────────────────────────────────
// FairDermNet 8-Class Taxonomy
// MUST match the training taxonomy in model/model/class_taxonomy.py exactly.
// ─────────────────────────────────────────────────────────────────────────────

/// Final 8-class taxonomy, index order = class index.
const List<String> _kClassNames = [
  'Melanoma',                // 0 — HIGH risk (malignant)
  'Basal Cell Carcinoma',    // 1 — HIGH risk (malignant)
  'Squamous Cell Carcinoma', // 2 — HIGH risk (malignant)
  'Actinic Keratosis',       // 3 — MEDIUM risk (malignant / pre-cancerous)
  'Melanocytic Nevi',        // 4 — LOW risk (benign)
  'Benign Keratosis',        // 5 — LOW risk (benign)
  'Dermatofibroma',          // 6 — LOW risk (benign)
  'Vascular Lesion',         // 7 — LOW risk (benign)
];

/// Indices that are clinically malignant (classes 0–3 only).
const Set<int> _kMalignantIndices = {0, 1, 2, 3};

// ImageNet normalisation constants (must match training augmentation_pipeline.py)
const List<double> _kMean = [0.485, 0.456, 0.406];
const List<double> _kStd  = [0.229, 0.224, 0.225];

const int    _kInputSize   = 224;
const int    _kNumClasses  = 8;
const String _kModelAsset  = 'assets/fairdermnet_int8.tflite';
const String _kConfigAsset = 'assets/threshold_config.json';

// ─────────────────────────────────────────────────────────────────────────────
// InferenceService
// ─────────────────────────────────────────────────────────────────────────────

/// Production TFLite inference service for FairDermNet (8-class).
///
/// Lifecycle:
///   final svc = InferenceService();
///   await svc.loadModel();          // call once at app startup
///   final result = await svc.runInference(imagePath);
///   await svc.dispose();            // call on app teardown
class InferenceService {
  Interpreter? _interpreter;
  double _clinicalThreshold = 0.5;   // loaded from threshold_config.json
  bool   _isLoaded          = false;

  /// Load the TFLite model and threshold config from Flutter assets.
  /// Safe to call multiple times — subsequent calls are no-ops.
  Future<void> loadModel() async {
    if (_isLoaded) return;

    // Load clinical threshold from config JSON
    try {
      final cfgStr = await rootBundle.loadString(_kConfigAsset);
      final cfg = json.decode(cfgStr) as Map<String, dynamic>;
      _clinicalThreshold = (cfg['clinical_threshold'] as num?)?.toDouble() ?? 0.5;
      debugPrint('[InferenceService] Clinical threshold: $_clinicalThreshold');
    } catch (e) {
      debugPrint('[InferenceService] ⚠ Could not load threshold_config.json: $e');
      debugPrint('[InferenceService]   Using default threshold 0.5');
    }

    // Load TFLite model
    try {
      _interpreter = await Interpreter.fromAsset(
        _kModelAsset,
        options: InterpreterOptions()..threads = 4,
      );
      _interpreter!.allocateTensors();
      debugPrint('[InferenceService] ✓ TFLite model loaded: $_kModelAsset');
      debugPrint('[InferenceService]   Input  : ${_interpreter!.getInputTensors().map((t) => t.shape)}');
      debugPrint('[InferenceService]   Output : ${_interpreter!.getOutputTensors().map((t) => t.shape)}');
      _isLoaded = true;
    } catch (e) {
      debugPrint('[InferenceService] ✗ Failed to load TFLite model: $e');
      debugPrint('[InferenceService]   Ensure fairdermnet_int8.tflite is in assets/');
      rethrow;
    }
  }

  /// Run inference on the image at [imagePath].
  ///
  /// Preprocessing (mirrors training pipeline):
  ///   1. Decode JPEG/PNG
  ///   2. Resize to 224×224 (bilinear)
  ///   3. RGB channel order (no BGR swap)
  ///   4. Normalise: (pixel/255 - mean) / std  with ImageNet stats
  ///   5. Layout: [1, 3, 224, 224]  (NCHW — as exported)
  ///
  /// Returns [AnalysisResult] or null on failure.
  Future<AnalysisResult?> runInference(String imagePath) async {
    // Auto-load the model on first use (loadModel() previously was never
    // called anywhere in the app, so runInference always bailed out here).
    if (!_isLoaded || _interpreter == null) {
      debugPrint('[InferenceService] Model not loaded — loading now…');
      await loadModel();
    }
    if (!_isLoaded || _interpreter == null) {
      debugPrint('[InferenceService] ⚠ Model failed to load — cannot infer.');
      return null;
    }

    try {
      // ── 1. Load & decode image ───────────────────────────────────────────
      final bytes = imagePath == 'mock_image_path'
          ? _generateMockImageBytes()
          : await _readFileBytes(imagePath);

      if (bytes == null) return null;

      // ── 2. Decode and resize to 224×224 ─────────────────────────────────
      img.Image? decoded = img.decodeImage(bytes);
      if (decoded == null) {
        debugPrint('[InferenceService] ⚠ Failed to decode image: $imagePath');
        return null;
      }
      decoded = img.copyResize(decoded, width: _kInputSize, height: _kInputSize);

      // ── 3. Build input tensor [1, 3, 224, 224] (NCHW float32) ───────────
      //   FairDermNet was trained with NCHW (PyTorch default).
      //   onnx_export preserves this layout → TFLite expects NCHW.
      final input = _buildNchwInput(decoded);

      // ── 4. Allocate output buffer ────────────────────────────────────────
      // Shape: [1, 8] — raw logits (softmax applied below)
      final output = List.generate(1, (_) => List.filled(_kNumClasses, 0.0));

      // ── 5. Run inference ─────────────────────────────────────────────────
      _interpreter!.run(input, output);

      // ── 6. Softmax over logits ───────────────────────────────────────────
      final logits = output[0];
      final probs  = _softmax(logits);

      // ── 7. Find top-1 class ──────────────────────────────────────────────
      int    maxIdx  = 0;
      double maxProb = probs[0];
      for (int i = 1; i < _kNumClasses; i++) {
        if (probs[i] > maxProb) {
          maxProb = probs[i];
          maxIdx  = i;
        }
      }

      // ── 8. Clinical threshold — malignant decision ───────────────────────
      // Malignant probability = sum of softmax probs for classes 0–3.
      final malignantProb = _kMalignantIndices.fold(
        0.0, (acc, i) => acc + probs[i],
      );
      final isMalignant = malignantProb >= _clinicalThreshold;

      final allPredictions = {
        for (int i = 0; i < _kNumClasses; i++) _kClassNames[i]: probs[i],
      };

      debugPrint('[InferenceService] Result: ${_kClassNames[maxIdx]} '
          '(${(maxProb * 100).toStringAsFixed(1)}%) '
          'malignant_prob=${(malignantProb * 100).toStringAsFixed(1)}% '
          'threshold=${(_clinicalThreshold * 100).toStringAsFixed(1)}%');

      return AnalysisResult(
        label:           _kClassNames[maxIdx],
        confidence:      maxProb,
        allPredictions:  allPredictions,
        imagePath:       imagePath,
        date:            DateTime.now(),
        isMalignant:     isMalignant,
        malignantProb:   malignantProb,
        clinicalThreshold: _clinicalThreshold,
      );
    } catch (e, stackTrace) {
      debugPrint('[InferenceService] ✗ Inference error: $e');
      debugPrint('$stackTrace');
      return null;
    }
  }

  /// Release the TFLite interpreter resources.
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isLoaded    = false;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ──────────────────────────────────────────────────────────────────────────

  /// Build NCHW input [1][3][224][224] as a nested List<double>.
  List<List<List<List<double>>>> _buildNchwInput(img.Image image) {
    // Outer dims: [batch=1][C=3][H=224][W=224]
    final nchw = List.generate(
      1, (_) => List.generate(
        3, (_) => List.generate(
          _kInputSize, (_) => List.filled(_kInputSize, 0.0),
        ),
      ),
    );

    for (int y = 0; y < _kInputSize; y++) {
      for (int x = 0; x < _kInputSize; x++) {
        final pixel = image.getPixel(x, y);
        final r = pixel.r / 255.0;
        final g = pixel.g / 255.0;
        final b = pixel.b / 255.0;

        nchw[0][0][y][x] = (r - _kMean[0]) / _kStd[0];
        nchw[0][1][y][x] = (g - _kMean[1]) / _kStd[1];
        nchw[0][2][y][x] = (b - _kMean[2]) / _kStd[2];
      }
    }
    return nchw;
  }

  /// Stable numerically-safe softmax over a 1-D list of logits.
  List<double> _softmax(List<double> logits) {
    final maxVal = logits.reduce((a, b) => a > b ? a : b);
    final exp    = logits.map((v) => (v - maxVal).toDouble()).map(_exp).toList();
    final sum    = exp.reduce((a, b) => a + b);
    return exp.map((v) => v / sum).toList();
  }

  double _exp(double x) {
    // Clamp to avoid overflow
    return x > 20.0 ? 485165195.0 : (x < -20.0 ? 2.06e-9 : _dartExp(x));
  }

  double _dartExp(double x) => _dartPow(2.718281828459045, x);

  double _dartPow(double base, double exp) {
    // Use Dart's built-in double arithmetic — this is essentially dart:math pow
    // inlined to avoid a dart:math import cycle
    double result = 1.0;
    if (exp < 0) return 1.0 / _dartPow(base, -exp);
    for (int i = 0; i < exp.truncate(); i++) {
      result *= base;
    }
    return result * (1.0 + (exp - exp.truncateToDouble()) * (base - 1.0));
  }

  /// Read bytes from the filesystem or from Flutter assets.
  Future<Uint8List?> _readFileBytes(String path) async {
    try {
      // Asset paths go through rootBundle
      if (path.startsWith('assets/')) {
        return (await rootBundle.load(path)).buffer.asUint8List();
      }
      // Camera output is a regular filesystem path — read it directly.
      // (Previously this used a 'file_access' MethodChannel that has no
      // native handler, so every real capture threw MissingPluginException
      // and inference failed.)
      return await File(path).readAsBytes();
    } catch (e) {
      debugPrint('[InferenceService] ⚠ Could not read image: $path — $e');
      return null;
    }
  }

  /// Generate a synthetic JPEG for testing when no real image is available.
  Uint8List _generateMockImageBytes() {
    final image = img.Image(width: 224, height: 224);
    img.fill(image, color: img.ColorRgb8(150, 80, 60));
    return Uint8List.fromList(img.encodeJpg(image));
  }
}
