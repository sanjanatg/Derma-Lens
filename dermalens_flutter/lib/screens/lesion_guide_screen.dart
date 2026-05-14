import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/navigation_provider.dart';
import '../theme/app_theme.dart';

class LesionGuideScreen extends ConsumerStatefulWidget {
  const LesionGuideScreen({super.key});
  @override
  ConsumerState<LesionGuideScreen> createState() => _LesionGuideScreenState();
}

class _LesionGuideScreenState extends ConsumerState<LesionGuideScreen> {
  String _filter = 'All';
  Map<String, dynamic>? _selected;

  static const List<Map<String, dynamic>> _lesions = [
    {'code': 'MEL', 'name': 'Melanoma', 'type': 'Malignant',
     'desc': 'The most dangerous form of skin cancer. Develops in melanin-producing cells.',
     'warning': 'Irregular borders, multiple colours, rapid growth.',
     'urgency': 'Immediate — consult a specialist within days.'},
    {'code': 'BCC', 'name': 'Basal Cell Carcinoma', 'type': 'Malignant',
     'desc': 'Slow-growing cancer arising in basal skin cells. Rarely spreads.',
     'warning': 'Pearly bump, flat scar-like lesion, bleeding sore.',
     'urgency': 'Urgent — see a dermatologist within weeks.'},
    {'code': 'SCC', 'name': 'Squamous Cell Carcinoma', 'type': 'Malignant',
     'desc': 'Cancer of the squamous cells. Can spread if untreated.',
     'warning': 'Firm red nodule, scaly patch, new sore.',
     'urgency': 'Urgent — see a dermatologist within weeks.'},
    {'code': 'AK', 'name': 'Actinic Keratosis', 'type': 'Precancerous',
     'desc': 'Rough, scaly patch from sun damage. May develop into cancer.',
     'warning': 'Rough patch, burning sensation, hard waxy surface.',
     'urgency': 'Schedule — see a dermatologist within a month.'},
    {'code': 'BOD', 'name': "Bowen's Disease", 'type': 'Precancerous',
     'desc': 'Early-stage SCC confined to the outer skin layer.',
     'warning': 'Red, scaly plaque that persists or grows slowly.',
     'urgency': 'Schedule — monitor and treat within weeks.'},
    {'code': 'NV', 'name': 'Melanocytic Nevi', 'type': 'Benign',
     'desc': 'Common mole — a cluster of pigmented cells. Usually harmless.',
     'warning': 'Watch for ABCDE changes.',
     'urgency': 'Routine — annual skin check recommended.'},
    {'code': 'BKL', 'name': 'Benign Keratosis', 'type': 'Benign',
     'desc': 'Non-cancerous growths that look like warts or moles.',
     'warning': 'Usually no concern; monitor for changes.',
     'urgency': 'Routine — no immediate action required.'},
    {'code': 'DF', 'name': 'Dermatofibroma', 'type': 'Benign',
     'desc': 'Harmless fibrous nodule, usually on legs.',
     'warning': 'Firm bump, may dimple when squeezed.',
     'urgency': 'Routine — treatment only if symptomatic.'},
    {'code': 'VASC', 'name': 'Vascular Lesion', 'type': 'Benign',
     'desc': 'Growths made of blood vessels. Usually harmless, like cherry angiomas.',
     'warning': 'Bright red, blue or purple, may bleed if injured.',
     'urgency': 'Routine — cosmetic treatment optional.'},
  ];

  Color _badgeColor(String type) {
    if (type == 'Malignant') return AppColors.malignant;
    if (type == 'Precancerous') return const Color(0xFFE65100);
    return AppColors.benign;
  }

  Color _badgeBg(String type) {
    if (type == 'Malignant') return AppColors.malignantBg;
    if (type == 'Precancerous') return const Color(0xFFFFF3E0);
    return AppColors.benignBg;
  }

  List<Map<String, dynamic>> get _filtered {
    if (_filter == 'All') return _lesions;
    if (_filter == 'Malignant') return _lesions.where((l) => l['type'] == 'Malignant' || l['type'] == 'Precancerous').toList();
    return _lesions.where((l) => l['type'] == 'Benign').toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 20, 24, 0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Lesion Guide',
                      style: GoogleFonts.plusJakartaSans(
                        fontSize: 26, fontWeight: FontWeight.w800, color: AppColors.walnut)),
                  const SizedBox(height: 2),
                  Text('Know what to look for.',
                      style: Theme.of(context).textTheme.bodyMedium
                          ?.copyWith(color: AppColors.onSurfaceVariant)),
                  const SizedBox(height: 14),
                  // Filter pills
                  Row(
                    children: ['All', 'Malignant', 'Benign'].map((f) {
                      final sel = _filter == f;
                      return Padding(
                        padding: const EdgeInsets.only(right: 8),
                        child: GestureDetector(
                          onTap: () => setState(() => _filter = f),
                          child: AnimatedContainer(
                            duration: const Duration(milliseconds: 180),
                            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                            decoration: BoxDecoration(
                              color: sel ? AppColors.walnut : AppColors.surfaceContainerHigh,
                              borderRadius: BorderRadius.circular(50),
                            ),
                            child: Text(f,
                                style: GoogleFonts.plusJakartaSans(
                                  fontSize: 12, fontWeight: FontWeight.w700,
                                  color: sel ? Colors.white : AppColors.onSurface)),
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Expanded(
              child: GridView.builder(
                padding: const EdgeInsets.fromLTRB(24, 0, 24, 24),
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 3, crossAxisSpacing: 10, mainAxisSpacing: 10,
                  childAspectRatio: 0.85,
                ),
                itemCount: _filtered.length,
                itemBuilder: (context, i) {
                  final l = _filtered[i];
                  return GestureDetector(
                    onTap: () => setState(() => _selected = l),
                    child: Container(
                      decoration: BoxDecoration(
                        color: AppColors.surfaceContainerLow,
                        borderRadius: BorderRadius.circular(18),
                        border: Border.all(color: AppColors.outlineVariant.withValues(alpha: 0.5)),
                      ),
                      clipBehavior: Clip.antiAlias,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          // Placeholder Image Area
                          Expanded(
                            child: Container(
                              color: AppColors.surfaceContainerHighest,
                              child: const Icon(
                                Icons.image_outlined,
                                color: AppColors.outline,
                                size: 28,
                              ),
                            ),
                          ),
                          // Text content
                          Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: Column(
                              children: [
                                Text(l['name'] as String,
                                    textAlign: TextAlign.center,
                                    maxLines: 2, overflow: TextOverflow.ellipsis,
                                    style: GoogleFonts.plusJakartaSans(
                                      fontSize: 10, fontWeight: FontWeight.w700, color: AppColors.onSurface)),
                                const SizedBox(height: 4),
                                Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                  decoration: BoxDecoration(
                                    color: _badgeBg(l['type'] as String),
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Text(l['type'] as String,
                                      style: GoogleFonts.plusJakartaSans(
                                        fontSize: 8, fontWeight: FontWeight.w800,
                                        color: _badgeColor(l['type'] as String))),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                },
              ),
            ),
            // ── DETAIL BOTTOM SHEET ───────────────────────────────
            if (_selected != null)
              GestureDetector(
                onTap: () => setState(() => _selected = null),
                child: Container(
                  color: Colors.black.withValues(alpha: 0.4),
                  child: DraggableScrollableSheet(
                    initialChildSize: 0.5, minChildSize: 0.3, maxChildSize: 0.85,
                    builder: (_, controller) => Container(
                      padding: const EdgeInsets.fromLTRB(24, 12, 24, 24),
                      decoration: BoxDecoration(
                        color: AppColors.surface,
                        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
                      ),
                      child: ListView(
                        controller: controller,
                        children: [
                          Center(
                            child: Container(
                              width: 40, height: 4,
                              decoration: BoxDecoration(
                                color: AppColors.outlineVariant,
                                borderRadius: BorderRadius.circular(2)),
                            ),
                          ),
                          const SizedBox(height: 16),
                          Row(
                            children: [
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                                decoration: BoxDecoration(
                                  color: _badgeBg(_selected!['type'] as String),
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                child: Text(_selected!['code'] as String,
                                    style: GoogleFonts.plusJakartaSans(
                                      fontSize: 12, fontWeight: FontWeight.w900,
                                      color: _badgeColor(_selected!['type'] as String))),
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(_selected!['name'] as String,
                                    style: GoogleFonts.plusJakartaSans(
                                      fontSize: 20, fontWeight: FontWeight.w800, color: AppColors.walnut)),
                              ),
                            ],
                          ),
                          const SizedBox(height: 16),
                          _DetailRow(label: 'Description', body: _selected!['desc'] as String),
                          const SizedBox(height: 10),
                          _DetailRow(label: 'Warning Signs', body: _selected!['warning'] as String),
                          const SizedBox(height: 10),
                          _DetailRow(label: 'Urgency', body: _selected!['urgency'] as String, highlight: true),
                          const SizedBox(height: 20),
                          GestureDetector(
                            onTap: () => ref.read(navigationProvider.notifier).state = AppScreen.dermatologist,
                            child: Container(
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              decoration: BoxDecoration(
                                color: AppColors.sage,
                                borderRadius: BorderRadius.circular(50),
                              ),
                              child: Text('Find a Dermatologist',
                                  textAlign: TextAlign.center,
                                  style: GoogleFonts.plusJakartaSans(
                                    fontSize: 14, fontWeight: FontWeight.w800,
                                    color: AppColors.onPrimaryContainer)),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _DetailRow extends StatelessWidget {
  final String label;
  final String body;
  final bool highlight;
  const _DetailRow({required this.label, required this.body, this.highlight = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: highlight
            ? AppColors.primaryFixed.withValues(alpha: 0.4)
            : AppColors.surfaceContainerLow,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label.toUpperCase(),
              style: Theme.of(context).textTheme.labelMedium
                  ?.copyWith(color: highlight ? AppColors.primary : AppColors.onSurfaceVariant)),
          const SizedBox(height: 6),
          Text(body,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(height: 1.5)),
        ],
      ),
    );
  }
}
