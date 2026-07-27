import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/inference_service.dart';

// ─────────────────────────────────────────────────────────────────────────────
// AnalysisResult — data object returned by InferenceService
// ─────────────────────────────────────────────────────────────────────────────

/// Holds the full output of one FairDermNet inference pass.
///
/// Key fields for UI and clinical display:
///   [label]            — top-1 class name
///   [confidence]       — top-1 softmax probability
///   [isMalignant]      — true if malignant probability >= [clinicalThreshold]
///   [malignantProb]    — sum of softmax probs for malignant classes 0–3
///   [allPredictions]   — map of all 8 class names → probability
///   [clinicalThreshold]— threshold used for the binary decision
class AnalysisResult {
  final String              label;
  final double              confidence;
  final Map<String, double> allPredictions;
  final String              imagePath;
  final DateTime            date;

  // Clinical decision fields
  final bool   isMalignant;
  final double malignantProb;
  final double clinicalThreshold;

  AnalysisResult({
    required this.label,
    required this.confidence,
    required this.allPredictions,
    required this.imagePath,
    required this.date,
    this.isMalignant       = false,
    this.malignantProb     = 0.0,
    this.clinicalThreshold = 0.5,
  });

  /// Risk tier string for UI display.
  ///   High risk  — malignant classes 0–1 (MEL, BCC)
  ///   Medium risk — malignant class 3 (AK) or 2 (SCC)
  ///   Low risk   — benign classes 4–7
  String get riskTier {
    if (isMalignant) {
      final idx = _labelToIndex(label);
      if (idx == 0 || idx == 1) return 'High risk — consult a dermatologist';
      return 'Medium risk — monitor closely';
    }
    return 'Low risk';
  }

  int _labelToIndex(String name) {
    const names = [
      'Melanoma',
      'Basal Cell Carcinoma',
      'Squamous Cell Carcinoma',
      'Actinic Keratosis',
      'Melanocytic Nevi',
      'Benign Keratosis',
      'Dermatofibroma',
      'Vascular Lesion',
    ];
    return names.indexOf(name);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Providers
// ─────────────────────────────────────────────────────────────────────────────

final analysisResultProvider = StateProvider<AnalysisResult?>((ref) => null);

final inferenceServiceProvider = Provider<InferenceService>((ref) {
  final svc = InferenceService();
  // Dispose when the provider is torn down (e.g. hot-restart in dev)
  ref.onDispose(svc.dispose);
  return svc;
});
