import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:camera/camera.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'theme/app_theme.dart';
import 'screens/onboarding_screen.dart';
import 'screens/home_screen.dart';
import 'screens/symptom_checker_screen.dart';
import 'screens/scan_screen.dart';
import 'screens/processing_screen.dart';
import 'screens/results_screen.dart';
import 'screens/dermatologist_screen.dart';
import 'screens/history_screen.dart';
import 'screens/lesion_guide_screen.dart';
import 'screens/profile_screen.dart';
import 'screens/about_screen.dart';
import 'screens/app_shell.dart';
import 'providers/navigation_provider.dart';

/// Global list populated at startup – available to [ScanScreen].
List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  bool isFirstLaunch = true;
  try {
    final prefs = await SharedPreferences.getInstance();
    isFirstLaunch = prefs.getBool('firstLaunch') ?? true;
    if (isFirstLaunch) {
      await prefs.setBool('firstLaunch', false);
    }
  } catch (e) {
    debugPrint('Prefs init error: $e');
  }

  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Camera init error (will request at runtime): $e');
  }
  
  runApp(ProviderScope(
    overrides: [
      navigationProvider.overrideWith((ref) => isFirstLaunch ? AppScreen.onboarding : AppScreen.home),
    ],
    child: const DermaLensApp(),
  ));
}

class DermaLensApp extends ConsumerWidget {
  const DermaLensApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final screen = ref.watch(navigationProvider);
    return MaterialApp(
      title: 'DermaLens',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      home: _resolveScreen(screen),
    );
  }

  Widget _resolveScreen(AppScreen screen) {
    // Screens that live inside the bottom-nav shell
    const shellScreens = {
      AppScreen.home, AppScreen.history, AppScreen.lesionGuide, AppScreen.profile,
    };

    if (shellScreens.contains(screen)) {
      return AppShell(body: _body(screen));
    }
    return _body(screen);
  }

  Widget _body(AppScreen screen) {
    switch (screen) {
      case AppScreen.onboarding:     return const OnboardingScreen();
      case AppScreen.home:           return const HomeScreen();
      case AppScreen.symptomChecker: return const SymptomCheckerScreen();
      case AppScreen.scan:           return const ScanScreen();
      case AppScreen.processing:     return const ProcessingScreen();
      case AppScreen.results:        return const ResultsScreen();
      case AppScreen.dermatologist:  return const DermatologistScreen();
      case AppScreen.history:        return const HistoryScreen();
      case AppScreen.lesionGuide:    return const LesionGuideScreen();
      case AppScreen.profile:        return const ProfileScreen();
      case AppScreen.about:          return const AboutScreen();
    }
  }
}
