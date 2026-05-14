import 'dart:math';
import 'package:flutter/foundation.dart';
import '../providers/analysis_provider.dart';

/// A mock [InferenceService] that simulates model inference without any
/// actual TFLite dependency. Replace with the real implementation once
/// the model file and a compatible tflite_flutter version are available.
class InferenceService {
  static const List<String> _labels = [
    'Melanocytic Nevus',
    'Melanoma',
    'Benign Keratosis',
    'Basal Cell Carcinoma',
    'Actinic Keratosis',
    'Vascular Lesion',
    'Dermatofibroma',
  ];

  Future<void> loadModel() async {
    // No-op in mock mode.
    debugPrint('[MockInference] Model loading skipped (demo mode).');
  }

  Future<AnalysisResult?> runInference(String imagePath) async {
    // Simulate network / processing latency.
    await Future<void>.delayed(const Duration(milliseconds: 1800));

    final rng = Random();

    // Generate random-ish probabilities that sum to 1.
    final rawValues = List<double>.generate(_labels.length, (_) => rng.nextDouble());
    final total = rawValues.fold(0.0, (a, b) => a + b);
    final probs = rawValues.map((v) => v / total).toList();

    // Pick the winning label.
    int maxIndex = 0;
    double maxProb = -1;
    for (int i = 0; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIndex = i;
      }
    }

    final predictions = {
      for (int i = 0; i < _labels.length; i++) _labels[i]: probs[i],
    };

    debugPrint('[MockInference] Result: ${_labels[maxIndex]} (${(maxProb * 100).toStringAsFixed(1)}%)');

    return AnalysisResult(
      label: _labels[maxIndex],
      confidence: maxProb,
      allPredictions: predictions,
      imagePath: imagePath,
      date: DateTime.now(),
    );
  }
}
