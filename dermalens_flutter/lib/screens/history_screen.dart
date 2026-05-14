import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/navigation_provider.dart';
import '../theme/app_theme.dart';

class HistoryScreen extends ConsumerWidget {
  const HistoryScreen({super.key});

  static const List<Map<String, dynamic>> _scans = [
    {'label': 'Basal Cell Carcinoma', 'date': 'May 11, 2026', 'conf': 0.87, 'malignant': true},
    {'label': 'Melanocytic Nevus',    'date': 'May 9, 2026',  'conf': 0.74, 'malignant': false},
    {'label': 'Actinic Keratosis',    'date': 'May 5, 2026',  'conf': 0.61, 'malignant': false},
    {'label': 'Melanoma',             'date': 'Apr 28, 2026', 'conf': 0.55, 'malignant': true},
    {'label': 'Dermatofibroma',       'date': 'Apr 20, 2026', 'conf': 0.82, 'malignant': false},
  ];

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(24, 20, 24, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Your Skin Health',
                  style: GoogleFonts.plusJakartaSans(
                    fontSize: 26, fontWeight: FontWeight.w800,
                    color: AppColors.walnut, letterSpacing: -0.5)),
              const SizedBox(height: 20),

              // ── STREAK CARD ───────────────────────────────────
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: AppColors.sage,
                  borderRadius: BorderRadius.circular(24),
                ),
                child: Row(
                  children: [
                    const Text('🔥', style: TextStyle(fontSize: 36)),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('7 Day Streak',
                              style: GoogleFonts.plusJakartaSans(
                                fontSize: 20, fontWeight: FontWeight.w800,
                                color: AppColors.onPrimaryContainer)),
                          Text('Keep monitoring for early detection.',
                              style: GoogleFonts.plusJakartaSans(
                                fontSize: 12, color: AppColors.onPrimaryContainer.withValues(alpha: 0.75))),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // ── STAT PILLS ────────────────────────────────────
              Row(
                children: [
                  _StatPill(label: 'Total Scans', value: '12'),
                  const SizedBox(width: 10),
                  _StatPill(label: 'Malignant Flags', value: '2', danger: true),
                  const SizedBox(width: 10),
                  _StatPill(label: 'Benign Results', value: '10'),
                ],
              ),
              const SizedBox(height: 20),

              // ── CALENDAR HEATMAP ─────────────────────────────
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppColors.surfaceContainer,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('MAY 2026', style: Theme.of(context).textTheme.labelMedium),
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 5, runSpacing: 5,
                      children: List.generate(31, (i) {
                        final scanned = [5, 9, 11, 18, 21, 25, 28].contains(i + 1);
                        return Container(
                          width: 30, height: 30,
                          decoration: BoxDecoration(
                            color: scanned
                                ? AppColors.sage
                                : AppColors.surfaceContainerHigh,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          alignment: Alignment.center,
                          child: Text('${i + 1}',
                              style: GoogleFonts.plusJakartaSans(
                                fontSize: 10, fontWeight: FontWeight.w700,
                                color: scanned
                                    ? AppColors.onPrimaryContainer
                                    : AppColors.onSurfaceVariant,
                              )),
                        );
                      }),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // ── SCAN ACTIVITY CHART ───────────────────────────
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppColors.surfaceContainerLow,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('SCAN ACTIVITY', style: Theme.of(context).textTheme.labelMedium),
                    const SizedBox(height: 12),
                    SizedBox(
                      height: 80,
                      child: CustomPaint(
                        size: const Size(double.infinity, 80),
                        painter: _LineChartPainter(),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // ── HISTORY LIST ──────────────────────────────────
              Text('Past Scans', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 12),
              ..._scans.map((scan) => Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Container(
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
                            Text(scan['label'] as String,
                                style: GoogleFonts.plusJakartaSans(
                                  fontSize: 13, fontWeight: FontWeight.w700)),
                            const SizedBox(height: 3),
                            Text('${scan['date']}  ·  ${((scan['conf'] as double) * 100).round()}%',
                                style: GoogleFonts.plusJakartaSans(
                                  fontSize: 11, color: AppColors.onSurfaceVariant)),
                          ],
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: scan['malignant'] == true ? AppColors.malignantBg : AppColors.benignBg,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          scan['malignant'] == true ? 'Malignant' : 'Benign',
                          style: GoogleFonts.plusJakartaSans(
                            fontSize: 10, fontWeight: FontWeight.w800,
                            color: scan['malignant'] == true ? AppColors.malignant : AppColors.benign,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              )),
            ],
          ),
        ),
      ),
    );
  }
}

class _StatPill extends StatelessWidget {
  final String label;
  final String value;
  final bool danger;
  const _StatPill({required this.label, required this.value, this.danger = false});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 10),
        decoration: BoxDecoration(
          color: danger ? AppColors.malignantBg : AppColors.surfaceContainer,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          children: [
            Text(value,
                style: GoogleFonts.plusJakartaSans(
                  fontSize: 20, fontWeight: FontWeight.w800,
                  color: danger ? AppColors.malignant : AppColors.onSurface)),
            const SizedBox(height: 2),
            Text(label,
                textAlign: TextAlign.center,
                style: GoogleFonts.plusJakartaSans(
                  fontSize: 9, fontWeight: FontWeight.w600,
                  color: danger ? AppColors.malignant : AppColors.onSurfaceVariant)),
          ],
        ),
      ),
    );
  }
}

class _LineChartPainter extends CustomPainter {
  static const List<double> _values = [2, 3, 1, 4, 2, 5, 3, 4, 6, 3, 4, 5];

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = AppColors.primary
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final fillPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter, end: Alignment.bottomCenter,
        colors: [AppColors.sage.withValues(alpha: 0.4), Colors.transparent],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height))
      ..style = PaintingStyle.fill;

    final maxV = _values.reduce((a, b) => a > b ? a : b);
    final step = size.width / (_values.length - 1);

    final path = Path();
    final fillPath = Path();

    for (int i = 0; i < _values.length; i++) {
      final x = i * step;
      final y = size.height - (_values[i] / maxV) * size.height;
      if (i == 0) {
        path.moveTo(x, y);
        fillPath.moveTo(x, size.height);
        fillPath.lineTo(x, y);
      } else {
        path.lineTo(x, y);
        fillPath.lineTo(x, y);
      }
    }
    fillPath.lineTo(size.width, size.height);
    fillPath.close();

    canvas.drawPath(fillPath, fillPaint);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter _) => false;
}
