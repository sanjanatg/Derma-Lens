import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/navigation_provider.dart';
import '../theme/app_theme.dart';

class ProcessingScreen extends ConsumerStatefulWidget {
  const ProcessingScreen({super.key});
  @override
  ConsumerState<ProcessingScreen> createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends ConsumerState<ProcessingScreen>
    with TickerProviderStateMixin {
  late AnimationController _progressController;
  late AnimationController _scanLineController;
  late Animation<double> _scanLine;
  int _textIndex = 0;
  Timer? _textTimer;

  static const List<String> _messages = [
    'Analysing lesion...',
    'Running fairness correction...',
    'Generating Grad-CAM heatmap...',
    'Checking Fitzpatrick calibration...',
    'Finalising results...',
  ];

  @override
  void initState() {
    super.initState();
    _progressController = AnimationController(
      vsync: this, duration: const Duration(seconds: 4))
      ..addStatusListener((s) {
        if (s == AnimationStatus.completed) {
          ref.read(navigationProvider.notifier).state = AppScreen.results;
        }
      })
      ..forward();

    _scanLineController = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 1200))
      ..repeat();
    _scanLine = Tween<double>(begin: -1, end: 1).animate(
        CurvedAnimation(parent: _scanLineController, curve: Curves.easeInOut));

    _textTimer = Timer.periodic(const Duration(milliseconds: 850), (_) {
      if (mounted) setState(() => _textIndex = (_textIndex + 1) % _messages.length);
    });
  }

  @override
  void dispose() {
    _progressController.dispose();
    _scanLineController.dispose();
    _textTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.walnut,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // ── IMAGE + SCAN RING ──────────────────────────────
              AnimatedBuilder(
                animation: _progressController,
                builder: (_, child) => Stack(
                  alignment: Alignment.center,
                  children: [
                    // Progress ring
                    SizedBox(
                      width: 220, height: 220,
                      child: CircularProgressIndicator(
                        value: _progressController.value,
                        strokeWidth: 4,
                        color: AppColors.sage,
                        backgroundColor: Colors.white.withValues(alpha: 0.1),
                        strokeCap: StrokeCap.round,
                      ),
                    ),
                    // Image placeholder with scan line
                    ClipOval(
                      child: SizedBox(
                        width: 196, height: 196,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            // Faded skin image placeholder
                            Container(
                              color: AppColors.chestnut.withValues(alpha: 0.6),
                              child: const Icon(Icons.center_focus_weak_rounded,
                                  size: 60, color: Colors.white24),
                            ),
                            // Scan line
                            AnimatedBuilder(
                              animation: _scanLine,
                              builder: (_, __) => Positioned(
                                top: (0.5 + _scanLine.value * 0.5) * 196,
                                left: 0, right: 0,
                                child: Container(
                                  height: 2,
                                  decoration: BoxDecoration(
                                    gradient: LinearGradient(
                                      colors: [
                                        Colors.transparent,
                                        AppColors.sage.withValues(alpha: 0.9),
                                        Colors.transparent,
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 48),

              // ── CYCLING TEXT ───────────────────────────────────
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 400),
                child: Text(
                  _messages[_textIndex],
                  key: ValueKey(_textIndex),
                  textAlign: TextAlign.center,
                  style: GoogleFonts.plusJakartaSans(
                    fontSize: 20, fontWeight: FontWeight.w700,
                    color: Colors.white, height: 1.3,
                  ),
                ),
              ),
              const SizedBox(height: 20),

              Text('Please wait. Your privacy is protected.\nNo data leaves your device.',
                  textAlign: TextAlign.center,
                  style: GoogleFonts.plusJakartaSans(
                    fontSize: 13, color: Colors.white38, height: 1.6)),
              const SizedBox(height: 40),

              // ── PROGRESS BAR ──────────────────────────────────
              AnimatedBuilder(
                animation: _progressController,
                builder: (_, __) => Column(
                  children: [
                    ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: _progressController.value,
                        backgroundColor: Colors.white.withValues(alpha: 0.1),
                        color: AppColors.sage,
                        minHeight: 4,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text('${(_progressController.value * 100).round()}%',
                        style: GoogleFonts.plusJakartaSans(
                          fontSize: 12, fontWeight: FontWeight.w700,
                          color: AppColors.sage)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
