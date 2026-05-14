import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'analysis_provider.dart';

class HistoryNotifier extends StateNotifier<List<AnalysisResult>> {
  HistoryNotifier() : super([]);

  void addScan(AnalysisResult result) {
    state = [result, ...state];
  }

  void clearHistory() {
    state = [];
  }
}

final historyProvider = StateNotifierProvider<HistoryNotifier, List<AnalysisResult>>((ref) {
  return HistoryNotifier();
});
