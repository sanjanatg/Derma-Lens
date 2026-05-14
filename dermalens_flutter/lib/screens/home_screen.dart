import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/navigation_provider.dart';
import '../providers/history_provider.dart';
import '../theme/app_theme.dart';

class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final history = ref.watch(historyProvider);

    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(24, 20, 24, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ── TOP BAR ──────────────────────────────────────────
              Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Good morning 👋',
                            style: GoogleFonts.plusJakartaSans(
                              fontSize: 13, fontWeight: FontWeight.w600,
                              color: AppColors.onSurfaceVariant,
                            )),
                        const SizedBox(height: 2),
                        Text('DermaLens',
                            style: GoogleFonts.plusJakartaSans(
                              fontSize: 24, fontWeight: FontWeight.w800,
                              color: AppColors.walnut, letterSpacing: -0.5,
                            )),
                      ],
                    ),
                  ),
                  GestureDetector(
                    onTap: () => ref.read(navigationProvider.notifier).state = AppScreen.profile,
                    child: Container(
                      width: 46, height: 46,
                      decoration: BoxDecoration(
                        color: AppColors.primaryContainer,
                        shape: BoxShape.circle,
                        border: Border.all(color: AppColors.sage, width: 2),
                      ),
                      child: const Icon(Icons.person_rounded,
                          color: AppColors.onPrimaryContainer, size: 22),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 24),

              // ── QUICK STATS ROW ──────────────────────────────────
              Row(
                children: [
                  _StatCard(label: 'Total Scans', value: '${history.length}', icon: Icons.document_scanner_rounded),
                ],
              ),
              const SizedBox(height: 20),

              // ── HERO SCAN CTA CARD ───────────────────────────────
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(28),
                decoration: BoxDecoration(
                  color: AppColors.walnut,
                  borderRadius: BorderRadius.circular(28),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                      decoration: BoxDecoration(
                        color: AppColors.sage.withValues(alpha: 0.25),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text('AI ANALYSIS READY',
                          style: GoogleFonts.plusJakartaSans(
                            fontSize: 10, fontWeight: FontWeight.w800,
                            color: AppColors.sage, letterSpacing: 1.2,
                          )),
                    ),
                    const SizedBox(height: 14),
                    Text('Ready\nto Scan?',
                        style: GoogleFonts.plusJakartaSans(
                          fontSize: 30, fontWeight: FontWeight.w800,
                          color: Colors.white, height: 1.15, letterSpacing: -0.5,
                        )),
                    const SizedBox(height: 8),
                    Text('Capture a skin lesion and receive\nexplainable AI analysis in seconds.',
                        style: GoogleFonts.plusJakartaSans(
                          fontSize: 13, color: Colors.white60, height: 1.5,
                        )),
                    const SizedBox(height: 22),
                    GestureDetector(
                      onTap: () => ref.read(navigationProvider.notifier).state = AppScreen.symptomChecker,
                      child: Container(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        decoration: BoxDecoration(
                          color: AppColors.sage,
                          borderRadius: BorderRadius.circular(50),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.document_scanner_rounded,
                                color: AppColors.onPrimaryContainer, size: 20),
                            const SizedBox(width: 8),
                            Text('START SCAN',
                                style: GoogleFonts.plusJakartaSans(
                                  fontSize: 15, fontWeight: FontWeight.w800,
                                  color: AppColors.onPrimaryContainer, letterSpacing: 1.2,
                                )),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),

              // ── RECENT SCANS ─────────────────────────────────────
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('Recent Scans',
                      style: Theme.of(context).textTheme.titleMedium),
                  GestureDetector(
                    onTap: () => ref.read(navigationProvider.notifier).state = AppScreen.history,
                    child: Text('See all',
                        style: GoogleFonts.plusJakartaSans(
                          fontSize: 13, fontWeight: FontWeight.w700,
                          color: AppColors.primary,
                        )),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              if (history.isEmpty)
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: AppColors.surfaceContainerLow,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Center(
                    child: Text('No scans yet. Start your first scan!',
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                              color: AppColors.onSurfaceVariant,
                            )),
                  ),
                )
              else
                ...history.take(3).map((scan) {
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: _RecentScanCard(
                      label: scan.label,
                      date: '${scan.date.day}/${scan.date.month}/${scan.date.year}',
                      confidence: scan.confidence,
                      isMalignant: scan.label == 'Melanoma' || scan.label == 'Basal Cell Carcinoma',
                    ),
                  );
                }),
              const SizedBox(height: 24),

              // ── TIPS CARD ─────────────────────────────────────────
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: AppColors.primaryFixed.withValues(alpha: 0.4),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  children: [
                    Container(
                      width: 40, height: 40,
                      decoration: const BoxDecoration(
                        color: AppColors.primaryContainer, shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.lightbulb_rounded,
                          color: AppColors.onPrimaryContainer, size: 20),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('SKIN TIP',
                              style: Theme.of(context).textTheme.labelMedium
                                  ?.copyWith(color: AppColors.primary)),
                          const SizedBox(height: 4),
                          Text('Use the ABCDE rule: Asymmetry, Border, Colour, Diameter, Evolution.',
                              style: Theme.of(context).textTheme.bodyMedium
                                  ?.copyWith(height: 1.5)),
                        ],
                      ),
                    ),
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

class _StatCard extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  const _StatCard({required this.label, required this.value, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppColors.surfaceContainer,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 18, color: AppColors.primary),
            const SizedBox(height: 8),
            Text(value,
                style: GoogleFonts.plusJakartaSans(
                  fontSize: 20, fontWeight: FontWeight.w800, color: AppColors.onSurface,
                )),
            const SizedBox(height: 2),
            Text(label,
                style: GoogleFonts.plusJakartaSans(
                  fontSize: 10, fontWeight: FontWeight.w600, color: AppColors.onSurfaceVariant,
                )),
          ],
        ),
      ),
    );
  }
}

class _RecentScanCard extends StatelessWidget {
  final String label;
  final String date;
  final double confidence;
  final bool isMalignant;
  const _RecentScanCard({
    required this.label, required this.date,
    required this.confidence, required this.isMalignant,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surfaceContainerLow,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        children: [
          Container(
            width: 52, height: 52,
            decoration: BoxDecoration(
              color: AppColors.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(14),
            ),
            child: const Icon(Icons.image_rounded, color: AppColors.outline, size: 24),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label,
                    style: GoogleFonts.plusJakartaSans(
                      fontSize: 14, fontWeight: FontWeight.w700, color: AppColors.onSurface,
                    )),
                const SizedBox(height: 3),
                Text('$date  ·  ${(confidence * 100).round()}% confidence',
                    style: GoogleFonts.plusJakartaSans(
                      fontSize: 11, color: AppColors.onSurfaceVariant,
                    )),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: isMalignant ? AppColors.malignantBg : AppColors.benignBg,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              isMalignant ? 'Malignant' : 'Benign',
              style: GoogleFonts.plusJakartaSans(
                fontSize: 10, fontWeight: FontWeight.w800,
                color: isMalignant ? AppColors.malignant : AppColors.benign,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
