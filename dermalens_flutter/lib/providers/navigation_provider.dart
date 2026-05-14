import 'package:flutter_riverpod/flutter_riverpod.dart';

enum AppScreen {
  onboarding,
  home,
  symptomChecker,
  scan,
  processing,
  results,
  dermatologist,
  history,
  lesionGuide,
  profile,
  about,
}

// Which bottom-nav tab is active (for screens that share the shell)
enum NavTab { home, scan, learn, profile }

final navigationProvider = StateProvider<AppScreen>((ref) => AppScreen.onboarding);
final navTabProvider = StateProvider<NavTab>((ref) => NavTab.home);
