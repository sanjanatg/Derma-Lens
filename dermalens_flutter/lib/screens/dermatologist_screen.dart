import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/navigation_provider.dart';
import '../theme/app_theme.dart';

class DermatologistScreen extends ConsumerWidget {
  const DermatologistScreen({super.key});

  static const List<Map<String, dynamic>> _doctors = [
    {
      'name': 'Dr. Priya Menon',
      'specialty': 'Dermatology & Skin Oncology',
      'distance': '0.8 km',
      'rating': 4.9,
      'available': true,
    },
    {
      'name': 'Dr. James Okafor',
      'specialty': 'Consultant Dermatologist',
      'distance': '1.4 km',
      'rating': 4.7,
      'available': false,
    },
    {
      'name': 'Dr. Sarah Lim',
      'specialty': 'Dermatology & Cosmetics',
      'distance': '2.1 km',
      'rating': 4.8,
      'available': true,
    },
  ];

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(8, 8, 24, 0),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back_ios_new_rounded, size: 20, color: AppColors.onSurface),
                    onPressed: () => ref.read(navigationProvider.notifier).state = AppScreen.results,
                  ),
                  Text('Find a Dermatologist', style: Theme.of(context).textTheme.titleLarge),
                ],
              ),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(24, 16, 24, 24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Search bar
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                      decoration: BoxDecoration(
                        color: AppColors.surfaceContainerLow,
                        borderRadius: BorderRadius.circular(50),
                      ),
                      child: Row(
                        children: [
                          const Icon(Icons.location_pin, color: AppColors.earthyTan, size: 20),
                          const SizedBox(width: 10),
                          Expanded(
                            child: TextField(
                              decoration: InputDecoration(
                                hintText: 'Search by location or name…',
                                hintStyle: GoogleFonts.plusJakartaSans(
                                  fontSize: 13, color: AppColors.onSurfaceVariant),
                                border: InputBorder.none,
                                isDense: true,
                              ),
                            ),
                          ),
                          const Icon(Icons.search_rounded, color: AppColors.outline, size: 20),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),

                    // Map placeholder
                    Container(
                      height: 180,
                      decoration: BoxDecoration(
                        color: AppColors.surfaceContainer,
                        borderRadius: BorderRadius.circular(24),
                        border: Border.all(color: AppColors.outlineVariant),
                      ),
                      child: Stack(
                        alignment: Alignment.center,
                        children: [
                          // Earthy grid pseudo-map
                          CustomPaint(
                            size: const Size(double.infinity, 180),
                            painter: _MapGridPainter(),
                          ),
                          // Mock pins
                          ...[
                            const Offset(0.3, 0.4), const Offset(0.6, 0.6), const Offset(0.7, 0.3)
                          ].map((p) => Positioned(
                            left: p.dx * 300,
                            top: p.dy * 160,
                            child: const Icon(Icons.location_pin, color: AppColors.earthyTan, size: 30),
                          )),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                            decoration: BoxDecoration(
                              color: AppColors.walnut,
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Text('3 clinics nearby',
                                style: GoogleFonts.plusJakartaSans(
                                  fontSize: 12, fontWeight: FontWeight.w700, color: Colors.white)),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 20),

                    Text('Nearby Dermatologists',
                        style: Theme.of(context).textTheme.titleMedium),
                    const SizedBox(height: 12),

                    ..._doctors.map((doc) => Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: Container(
                        padding: const EdgeInsets.all(18),
                        decoration: BoxDecoration(
                          color: AppColors.surfaceContainerLow,
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Row(
                          children: [
                            Container(
                              width: 50, height: 50,
                              decoration: const BoxDecoration(
                                color: AppColors.primaryContainer,
                                shape: BoxShape.circle,
                              ),
                              child: const Icon(Icons.person_rounded,
                                  color: AppColors.onPrimaryContainer, size: 24),
                            ),
                            const SizedBox(width: 14),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(doc['name'] as String,
                                      style: GoogleFonts.plusJakartaSans(
                                        fontSize: 14, fontWeight: FontWeight.w800, color: AppColors.onSurface)),
                                  const SizedBox(height: 2),
                                  Text(doc['specialty'] as String,
                                      style: GoogleFonts.plusJakartaSans(
                                        fontSize: 11, color: AppColors.onSurfaceVariant)),
                                  const SizedBox(height: 6),
                                  Row(
                                    children: [
                                      const Icon(Icons.star_rounded, color: Colors.amber, size: 14),
                                      const SizedBox(width: 3),
                                      Text('${doc['rating']}',
                                          style: GoogleFonts.plusJakartaSans(
                                            fontSize: 11, fontWeight: FontWeight.w700)),
                                      const SizedBox(width: 8),
                                      const Icon(Icons.near_me_rounded,
                                          color: AppColors.outline, size: 12),
                                      const SizedBox(width: 3),
                                      Text(doc['distance'] as String,
                                          style: GoogleFonts.plusJakartaSans(
                                            fontSize: 11, color: AppColors.onSurfaceVariant)),
                                      const SizedBox(width: 8),
                                      Container(
                                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                                        decoration: BoxDecoration(
                                          color: doc['available'] == true
                                              ? AppColors.benignBg : AppColors.surfaceContainerHigh,
                                          borderRadius: BorderRadius.circular(10),
                                        ),
                                        child: Text(
                                          doc['available'] == true ? 'Available Today' : 'By Appointment',
                                          style: GoogleFonts.plusJakartaSans(
                                            fontSize: 9, fontWeight: FontWeight.w800,
                                            color: doc['available'] == true
                                                ? AppColors.benign : AppColors.onSurfaceVariant,
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(width: 10),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                              decoration: BoxDecoration(
                                color: AppColors.sage,
                                borderRadius: BorderRadius.circular(50),
                              ),
                              child: Text('Book',
                                  style: GoogleFonts.plusJakartaSans(
                                    fontSize: 12, fontWeight: FontWeight.w800,
                                    color: AppColors.onPrimaryContainer)),
                            ),
                          ],
                        ),
                      ),
                    )),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _MapGridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = AppColors.outlineVariant.withValues(alpha: 0.5)
      ..strokeWidth = 1;
    const step = 30.0;
    for (double x = 0; x < size.width; x += step) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y < size.height; y += step) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }
  @override
  bool shouldRepaint(covariant CustomPainter _) => false;
}
