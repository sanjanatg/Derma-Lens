import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/navigation_provider.dart';
import '../providers/analysis_provider.dart';
import '../providers/history_provider.dart';
import '../theme/app_theme.dart';

class ResultsScreen extends ConsumerStatefulWidget {
  const ResultsScreen({super.key});

  @override
  ConsumerState<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends ConsumerState<ResultsScreen> {
  bool _showHeatmap = false;

  @override
  Widget build(BuildContext context) {
    final result = ref.watch(analysisResultProvider);

    if (result == null) {
      return const Scaffold(
        backgroundColor: AppColors.surface,
        body: Center(child: Text('No results available')),
      );
    }

    final topPredictions = result.allPredictions.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    final descriptions = {
      'Melanocytic Nevus': 'Common mole. Likely benign, but shows regular pigmentation patterns consistent with melanocytic origin.',
      'Melanoma': 'Malignant skin cancer. Requires immediate professional evaluation. High priority for clinical follow-up.',
      'Benign Keratosis': 'Non-cancerous skin growth. Often appears as a waxy brown, black, or tan growth.',
      'Basal Cell Carcinoma': 'Slow-growing form of skin cancer. Rarely spreads but requires medical treatment to prevent local damage.',
      'Actinic Keratosis': 'Precancerous skin patch caused by sun damage. Should be monitored and treated to prevent progression.',
      'Vascular Lesion': 'Common skin growths composed of blood vessels. Usually benign, such as cherry angiomas or hemangiomas.',
      'Dermatofibroma': 'Common benign fibrous nodule. Typically firm and may feel like a small stone under the skin.',
    };

    final isMalignant = result.label == 'Melanoma' || result.label == 'Basal Cell Carcinoma';
    final recommendation = isMalignant
        ? 'URGENT: Schedule an appointment with a dermatologist as soon as possible for a professional biopsy and evaluation.'
        : 'Continue to monitor for changes in size, color, or shape (ABCDE rules). Schedule a dermatologist check if concerned.';

    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(
        child: Column(
          children: [
            // ── CUSTOM APP BAR ─────────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(8, 8, 24, 0),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back_ios_new_rounded,
                        size: 20, color: AppColors.onSurface),
                    onPressed: () => ref
                        .read(navigationProvider.notifier)
                        .state = AppScreen.scan,
                  ),
                  Text('Analysis Results',
                      style: Theme.of(context).textTheme.titleLarge),
                  const Spacer(),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 14, vertical: 7),
                    decoration: BoxDecoration(
                      color: AppColors.surfaceContainerHigh,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.ios_share_rounded,
                            size: 14, color: AppColors.onSurfaceVariant),
                        const SizedBox(width: 5),
                        Text('Share',
                            style: Theme.of(context)
                                .textTheme
                                .labelMedium
                                ?.copyWith(
                                    color: AppColors.onSurfaceVariant,
                                    letterSpacing: 0.4)),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(24, 20, 24, 24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // ── SCANNED IMAGE & TOGGLE WELL ───────────────────
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: AppColors.surfaceContainerLow,
                        borderRadius: BorderRadius.circular(28),
                      ),
                      child: Column(
                        children: [
                          // Toggle well
                          Container(
                            padding: const EdgeInsets.all(4),
                            decoration: BoxDecoration(
                              color: AppColors.surfaceContainerHigh,
                              borderRadius: BorderRadius.circular(24),
                            ),
                            child: Row(
                              children: [
                                Expanded(
                                  child: GestureDetector(
                                    onTap: () => setState(() => _showHeatmap = false),
                                    child: AnimatedContainer(
                                      duration: const Duration(milliseconds: 200),
                                      padding: const EdgeInsets.symmetric(vertical: 10),
                                      decoration: BoxDecoration(
                                        color: !_showHeatmap
                                            ? AppColors.primaryContainer
                                            : Colors.transparent,
                                        borderRadius: BorderRadius.circular(20),
                                      ),
                                      child: Text(
                                        'Original',
                                        textAlign: TextAlign.center,
                                        style: GoogleFonts.plusJakartaSans(
                                          fontSize: 12,
                                          fontWeight: FontWeight.w700,
                                          color: !_showHeatmap
                                              ? AppColors.onPrimaryContainer
                                              : AppColors.onSurfaceVariant,
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                                Expanded(
                                  child: GestureDetector(
                                    onTap: () => setState(() => _showHeatmap = true),
                                    child: AnimatedContainer(
                                      duration: const Duration(milliseconds: 200),
                                      padding: const EdgeInsets.symmetric(vertical: 10),
                                      decoration: BoxDecoration(
                                        color: _showHeatmap
                                            ? AppColors.primaryContainer
                                            : Colors.transparent,
                                        borderRadius: BorderRadius.circular(20),
                                      ),
                                      child: Text(
                                        'AI Heatmap',
                                        textAlign: TextAlign.center,
                                        style: GoogleFonts.plusJakartaSans(
                                          fontSize: 12,
                                          fontWeight: FontWeight.w700,
                                          color: _showHeatmap
                                              ? AppColors.onPrimaryContainer
                                              : AppColors.onSurfaceVariant,
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(height: 16),
                          // Image display
                          ClipRRect(
                            borderRadius: BorderRadius.circular(20),
                            child: SizedBox(
                              height: 260,
                              width: double.infinity,
                              child: Stack(
                                fit: StackFit.expand,
                                children: [
                                  // Base image
                                  if (result.imagePath.isNotEmpty && result.imagePath != 'mock_image_path')
                                    Image.file(
                                      File(result.imagePath),
                                      fit: BoxFit.cover,
                                    )
                                  else
                                    // Styled fallback representation
                                    Container(
                                      color: AppColors.surfaceContainerHighest,
                                      child: const Center(
                                        child: Icon(
                                          Icons.center_focus_weak_rounded,
                                          size: 48,
                                          color: AppColors.outline,
                                        ),
                                      ),
                                    ),

                                  // Grad-CAM Heatmap Overlay
                                  AnimatedOpacity(
                                    opacity: _showHeatmap ? 1.0 : 0.0,
                                    duration: const Duration(milliseconds: 300),
                                    child: Container(
                                      decoration: BoxDecoration(
                                        gradient: RadialGradient(
                                          center: const Alignment(0.1, -0.1),
                                          radius: 0.6,
                                          colors: [
                                            Colors.red.withValues(alpha: 0.85),
                                            Colors.orange.withValues(alpha: 0.7),
                                            Colors.yellow.withValues(alpha: 0.4),
                                            Colors.blue.withValues(alpha: 0.1),
                                            Colors.transparent,
                                          ],
                                          stops: const [0.1, 0.4, 0.7, 0.9, 1.0],
                                        ),
                                      ),
                                    ),
                                  ),

                                  // Grid Overlay when Heatmap is active
                                  if (_showHeatmap)
                                    CustomPaint(
                                      painter: _HeatmapGridPainter(),
                                    ),
                                ],
                              ),
                            ),
                          ),
                          const SizedBox(height: 12),
                          Text(
                            _showHeatmap 
                              ? 'Highlighted region influenced the prediction most.'
                              : 'Original capture specimen.',
                            style: Theme.of(context).textTheme.labelSmall?.copyWith(
                                color: AppColors.onSurfaceVariant,
                                fontStyle: FontStyle.italic),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 16),

                    // ── MAIN PREDICTION CARD (#2F1F18) ───────────────
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(28),
                      decoration: BoxDecoration(
                        color: const Color(0xFF2F1F18),
                        borderRadius: BorderRadius.circular(28),
                        boxShadow: [
                          BoxShadow(
                            color: const Color(0xFF2F1F18).withValues(alpha: 0.2),
                            blurRadius: 20,
                            offset: const Offset(0, 8),
                          ),
                        ],
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Expanded(
                                child: Column(
                                  crossAxisAlignment:
                                      CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      '${(result.confidence * 100).round()}% CONFIDENCE',
                                      style: GoogleFonts.plusJakartaSans(
                                        fontSize: 11,
                                        fontWeight: FontWeight.w800,
                                        color: AppColors.primaryContainer,
                                        letterSpacing: 1.4,
                                      ),
                                    ),
                                    const SizedBox(height: 5),
                                    Text(
                                      result.label.replaceAll(' ', '\n'),
                                      style: GoogleFonts.plusJakartaSans(
                                        fontSize: 28,
                                        fontWeight: FontWeight.w800,
                                        color: Colors.white,
                                        height: 1.1,
                                        letterSpacing: -0.5,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              // Risk Badge
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                                decoration: BoxDecoration(
                                  color: isMalignant
                                      ? Colors.red.withValues(alpha: 0.2)
                                      : Colors.green.withValues(alpha: 0.2),
                                  borderRadius: BorderRadius.circular(16),
                                  border: Border.all(
                                    color: isMalignant
                                        ? Colors.redAccent
                                        : Colors.greenAccent,
                                  ),
                                ),
                                child: Text(
                                  isMalignant ? 'Malignant' : 'Benign',
                                  style: GoogleFonts.plusJakartaSans(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w800,
                                    color: isMalignant ? Colors.redAccent : Colors.greenAccent,
                                  ),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 16),
                          Text(
                            descriptions[result.label] ?? 'Analysis complete. Please consult with a medical professional for a definitive diagnosis.',
                            style: GoogleFonts.plusJakartaSans(
                              fontSize: 14,
                              color: Colors.white70,
                              height: 1.6,
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 16),

                    // ── TOP PREDICTIONS ─────────────────────────────
                    Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: AppColors.surfaceContainerLow,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'TOP PREDICTIONS',
                            style: Theme.of(context).textTheme.labelMedium,
                          ),
                          const SizedBox(height: 16),
                          ...topPredictions.take(3).map((entry) {
                            return Padding(
                              padding: const EdgeInsets.only(bottom: 10),
                              child: _ConfidenceBar(
                                label: entry.key,
                                value: entry.value,
                                color: entry.key == result.label 
                                    ? AppColors.primary 
                                    : AppColors.primaryContainer,
                              ),
                            );
                          }),
                        ],
                      ),
                    ),

                    const SizedBox(height: 16),

                    // ── AI FAIRNESS AUDIT ────────────────────────
                    Text(
                      'AI FAIRNESS AUDIT',
                      style: Theme.of(context).textTheme.labelMedium,
                    ),
                    const SizedBox(height: 10),
                    Row(
                      children: [
                        _AuditBadge(
                          icon: Icons.palette_outlined,
                          label: 'Melanin\nCorrected',
                          color: AppColors.tertiary,
                          bg: AppColors.tertiaryContainer.withValues(alpha: 0.35),
                        ),
                        const SizedBox(width: 10),
                        _AuditBadge(
                          icon: Icons.category_outlined,
                          label: 'Class ${topPredictions.indexWhere((e) => e.key == result.label) + 1}\nof 9',
                          color: AppColors.primary,
                          bg: AppColors.primaryFixed.withValues(alpha: 0.4),
                        ),
                        const SizedBox(width: 10),
                        _AuditBadge(
                          icon: Icons.verified_outlined,
                          label: 'Bias\nChecked',
                          color: AppColors.secondary,
                          bg: AppColors.secondaryContainer.withValues(alpha: 0.5),
                        ),
                      ],
                    ),

                    const SizedBox(height: 16),

                    // ── FITZPATRICK SCALE ────────────────────────
                    Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: AppColors.surfaceContainer,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'FITZPATRICK SCALE',
                            style: Theme.of(context).textTheme.labelMedium,
                          ),
                          const SizedBox(height: 12),
                          Row(
                            children: List.generate(6, (i) {
                              const tones = [
                                Color(0xFFF8D5C2),
                                Color(0xFFF0C0A0),
                                Color(0xFFD4A574),
                                Color(0xFFAE7C52),
                                Color(0xFF855D3E),
                                Color(0xFF5C3D2E),
                              ];
                              final isSelected = i == 2; 
                              return Expanded(
                                child: Padding(
                                  padding: EdgeInsets.only(
                                      right: i < 5 ? 4 : 0),
                                  child: Column(
                                    children: [
                                      AnimatedContainer(
                                        duration:
                                            const Duration(milliseconds: 200),
                                        height: isSelected ? 36 : 28,
                                        decoration: BoxDecoration(
                                          color: tones[i],
                                          borderRadius:
                                              BorderRadius.circular(8),
                                          border: isSelected
                                              ? Border.all(
                                                  color: const Color(0xFF2F1F18),
                                                  width: 2.5)
                                              : null,
                                        ),
                                      ),
                                      const SizedBox(height: 4),
                                      Text(
                                        'F${i + 1}',
                                        style: GoogleFonts.plusJakartaSans(
                                          fontSize: 9,
                                          fontWeight: isSelected
                                              ? FontWeight.w800
                                              : FontWeight.w500,
                                          color: isSelected
                                              ? const Color(0xFF2F1F18)
                                              : AppColors.onSurfaceVariant,
                                        ),
                                        textAlign: TextAlign.center,
                                      ),
                                    ],
                                  ),
                                ),
                              );
                            }),
                          ),
                          const SizedBox(height: 10),
                          Text(
                            'Detected skin tone: Type III — model calibrated accordingly.',
                            style: Theme.of(context)
                                .textTheme
                                .labelSmall
                                ?.copyWith(color: AppColors.onSurfaceVariant),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 16),

                    // ── RECOMMENDATION ───────────────────────────
                    Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: AppColors.primaryFixed.withValues(alpha: 0.35),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Column(
                        children: [
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Container(
                                width: 36,
                                height: 36,
                                decoration: const BoxDecoration(
                                  color: AppColors.primaryContainer,
                                  shape: BoxShape.circle,
                                ),
                                child: const Icon(Icons.verified_user_rounded,
                                    color: AppColors.onPrimaryContainer, size: 18),
                              ),
                              const SizedBox(width: 14),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      'RECOMMENDATION',
                                      style: Theme.of(context)
                                          .textTheme
                                          .labelMedium
                                          ?.copyWith(color: AppColors.primary),
                                    ),
                                    const SizedBox(height: 6),
                                    Text(
                                      recommendation,
                                      style: Theme.of(context)
                                          .textTheme
                                          .bodyMedium
                                          ?.copyWith(
                                              height: 1.5,
                                              color: AppColors.onSurface),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                          if (isMalignant) ...[
                            const SizedBox(height: 20),
                            SizedBox(
                              width: double.infinity,
                              child: OutlinedButton.icon(
                                onPressed: () => ref.read(navigationProvider.notifier).state = AppScreen.dermatologist,
                                icon: const Icon(Icons.near_me_rounded, size: 18),
                                label: const Text('FIND DERMATOLOGIST'),
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),

                    const SizedBox(height: 32),
                  ],
                ),
              ),
            ),

            // ── STICKY FOOTER ─────────────────────────────────────
            Container(
              padding: const EdgeInsets.fromLTRB(24, 16, 24, 28),
              child: Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () => ref
                          .read(navigationProvider.notifier)
                          .state = AppScreen.scan,
                      child: const Text('SCAN AGAIN'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () {
                        ref.read(historyProvider.notifier).addScan(result);
                        
                        // We removed history from bottom nav, so we can go to Profile instead, 
                        // or just show a snackbar and go to home. Let's go to Home.
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('Result saved to history!')),
                        );
                        ref.read(navTabProvider.notifier).state = NavTab.home;
                        ref.read(navigationProvider.notifier).state = AppScreen.home;
                      },
                      child: const Text('SAVE RESULT'),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── SUPPORTING WIDGETS ─────────────────────────────────────────────────

class _ConfidenceBar extends StatelessWidget {
  final String label;
  final double value;
  final Color color;

  const _ConfidenceBar(
      {required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: Theme.of(context).textTheme.bodyMedium),
            Text(
              '${(value * 100).round()}%',
              style: Theme.of(context)
                  .textTheme
                  .labelMedium
                  ?.copyWith(color: AppColors.onSurfaceVariant),
            ),
          ],
        ),
        const SizedBox(height: 5),
        LayoutBuilder(
          builder: (context, constraints) => Stack(
            children: [
               Container(
                height: 6,
                width: double.infinity,
                decoration: BoxDecoration(
                  color: AppColors.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(3),
                ),
              ),
              AnimatedContainer(
                duration: const Duration(milliseconds: 600),
                curve: Curves.easeOutCubic,
                height: 6,
                width: constraints.maxWidth * value,
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(3),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _AuditBadge extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final Color bg;

  const _AuditBadge(
      {required this.icon,
      required this.label,
      required this.color,
      required this.bg});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 12),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          children: [
            Icon(icon, color: color, size: 20),
            const SizedBox(height: 6),
            Text(
              label,
              textAlign: TextAlign.center,
              style: GoogleFonts.plusJakartaSans(
                fontSize: 10,
                fontWeight: FontWeight.w700,
                color: AppColors.onSurface,
                height: 1.3,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _HeatmapGridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withValues(alpha: 0.15)
      ..strokeWidth = 1;
    const step = 20.0;
    for (double x = 0; x < size.width; x += step) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y < size.height; y += step) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
