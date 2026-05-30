import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/navigation_provider.dart';
import '../theme/app_theme.dart';

class AboutScreen extends ConsumerWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(
        child: Column(
          children: [
            // ── CUSTOM APP BAR ─────────────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(8, 8, 24, 0),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back_ios_new_rounded,
                        size: 20, color: AppColors.onSurface),
                    onPressed: () => ref
                        .read(navigationProvider.notifier)
                        .state = AppScreen.onboarding,
                  ),
                  Text(
                    'About DermaLens',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                ],
              ),
            ),

            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(24, 24, 24, 0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // ── WHAT IS CARD ─────────────────────────────────
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(28),
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                          colors: [
                            Color(0xFF3B4325),
                            Color(0xFF5A6241),
                          ],
                        ),
                        borderRadius: BorderRadius.circular(28),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 10, vertical: 5),
                            decoration: BoxDecoration(
                              color: AppColors.primaryContainer.withValues(alpha: 0.3),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(
                              'CORE PURPOSE',
                              style: GoogleFonts.plusJakartaSans(
                                fontSize: 10,
                                fontWeight: FontWeight.w800,
                                color: AppColors.primaryFixedDim,
                                letterSpacing: 1.4,
                              ),
                            ),
                          ),
                          const SizedBox(height: 14),
                          Text(
                            'What is\nDermaLens?',
                            style: GoogleFonts.plusJakartaSans(
                              fontSize: 26,
                              fontWeight: FontWeight.w800,
                              color: Colors.white,
                              height: 1.15,
                              letterSpacing: -0.5,
                            ),
                          ),
                          const SizedBox(height: 14),
                          Text(
                            'An AI-powered skin lesion classifier built for fairness and explainability across all skin tones.',
                            style: GoogleFonts.plusJakartaSans(
                              fontSize: 15,
                              color: Colors.white.withValues(alpha: 0.82),
                              height: 1.6,
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 16),

                    // ── FEATURE GRID ─────────────────────────────────
                    Row(
                      children: [
                        Expanded(
                          child: _FeatureTile(
                            icon: Icons.balance_rounded,
                            label: 'Fairness\nAudit',
                            color: AppColors.tertiary,
                            bg: AppColors.tertiaryContainer.withValues(alpha: 0.35),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: _FeatureTile(
                            icon: Icons.auto_awesome_rounded,
                            label: 'AI\nPowered',
                            color: AppColors.primary,
                            bg: AppColors.primaryFixed.withValues(alpha: 0.45),
                          ),
                        ),
                        const SizedBox(width: 12),
                        const Expanded(
                          child: _FeatureTile(
                            icon: Icons.visibility_rounded,
                            label: 'Explain-\nable',
                            color: AppColors.secondary,
                            bg: AppColors.secondaryContainer,
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 16),

                    // ── HOW IT WORKS ─────────────────────────────────
                    Container(
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        color: AppColors.surfaceContainer,
                        borderRadius: BorderRadius.circular(24),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'HOW IT WORKS',
                            style:
                                Theme.of(context).textTheme.labelMedium,
                          ),
                          const SizedBox(height: 18),
                          const _Step(
                              num: '01',
                              title: 'Capture',
                              body:
                                  'Frame the skin lesion clearly in the scanning ring.'),
                          const SizedBox(height: 16),
                          const _Step(
                              num: '02',
                              title: 'Analyse',
                              body:
                                  'Our model classifies across 9 dermoscopic categories.'),
                          const SizedBox(height: 16),
                          const _Step(
                              num: '03',
                              title: 'Review',
                              body:
                                  'Results include an AI fairness audit and Fitzpatrick tone correction.'),
                        ],
                      ),
                    ),

                    const SizedBox(height: 16),

                    // ── DISCLAIMER ───────────────────────────────────
                    Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: AppColors.surfaceContainerLow,
                        borderRadius: BorderRadius.circular(20),
                        border: const Border(
                          left: BorderSide(
                              color: AppColors.primaryContainer, width: 4),
                        ),
                      ),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Icon(Icons.info_outline_rounded,
                              color: AppColors.primary, size: 20),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'MEDICAL DISCLAIMER',
                                  style: Theme.of(context)
                                      .textTheme
                                      .labelMedium
                                      ?.copyWith(
                                          color: AppColors.primary,
                                          letterSpacing: 1.2),
                                ),
                                const SizedBox(height: 6),
                                Text(
                                  'DermaLens is a screening tool only. Results are not a substitute for professional medical diagnosis.',
                                  style: Theme.of(context)
                                      .textTheme
                                      .bodyMedium
                                      ?.copyWith(
                                          color: AppColors.onSurfaceVariant,
                                          fontStyle: FontStyle.italic,
                                          height: 1.5),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 32),
                  ],
                ),
              ),
            ),

            // ── STICKY CTA ────────────────────────────────────────────
            Container(
              padding: const EdgeInsets.fromLTRB(24, 16, 24, 28),
              decoration: BoxDecoration(
                color: AppColors.surface,
                boxShadow: [
                  BoxShadow(
                    color: AppColors.surfaceDim.withValues(alpha: 0.8),
                    blurRadius: 20,
                    offset: const Offset(0, -8),
                  ),
                ],
              ),
              child: ElevatedButton(
                onPressed: () =>
                    ref.read(navigationProvider.notifier).state = AppScreen.scan,
                child: const Text('CONTINUE TO SCAN'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _FeatureTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final Color bg;

  const _FeatureTile({
    required this.icon,
    required this.label,
    required this.color,
    required this.bg,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 24),
          const SizedBox(height: 10),
          Text(
            label,
            style: GoogleFonts.plusJakartaSans(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              color: AppColors.onSurface,
              height: 1.3,
            ),
          ),
        ],
      ),
    );
  }
}

class _Step extends StatelessWidget {
  final String num;
  final String title;
  final String body;

  const _Step(
      {required this.num, required this.title, required this.body});

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 32,
          height: 32,
          decoration: const BoxDecoration(
            color: AppColors.primaryContainer,
            shape: BoxShape.circle,
          ),
          child: Center(
            child: Text(
              num,
              style: GoogleFonts.plusJakartaSans(
                fontSize: 11,
                fontWeight: FontWeight.w800,
                color: AppColors.onPrimaryContainer,
              ),
            ),
          ),
        ),
        const SizedBox(width: 14),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: GoogleFonts.plusJakartaSans(
                  fontWeight: FontWeight.w700,
                  fontSize: 15,
                  color: AppColors.onSurface,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                body,
                style: GoogleFonts.plusJakartaSans(
                  fontSize: 13,
                  color: AppColors.onSurfaceVariant,
                  height: 1.5,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
