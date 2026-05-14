import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/inference_service.dart';

class AnalysisResult {
  final String label;
  final double confidence;
  final Map<String, double> allPredictions;
  final String imagePath;
  final DateTime date;

  AnalysisResult({
    required this.label,
    required this.confidence,
    required this.allPredictions,
    required this.imagePath,
    required this.date,
  });
}

final analysisResultProvider = StateProvider<AnalysisResult?>((ref) => null);

final inferenceServiceProvider = Provider<InferenceService>((ref) {
  return InferenceService();
});
