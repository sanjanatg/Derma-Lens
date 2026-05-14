import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/navigation_provider.dart';
import '../theme/app_theme.dart';

class SymptomCheckerScreen extends ConsumerStatefulWidget {
  const SymptomCheckerScreen({super.key});
  @override
  ConsumerState<SymptomCheckerScreen> createState() => _SymptomCheckerScreenState();
}

class _SymptomCheckerScreenState extends ConsumerState<SymptomCheckerScreen> {
  String? _duration;
  final Set<String> _symptoms = {};
  int? _tappedZone;

  final List<String> _durations = ['< 1 week', '1–4 weeks', '1+ months'];
  final List<String> _symptomOptions = ['Growing', 'Bleeding', 'Itching', 'Colour change', 'None'];

  @override
  Widget build(BuildContext context) {
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
                    onPressed: () => ref.read(navigationProvider.notifier).state = AppScreen.home,
                  ),
                  Text('Symptom Checker', style: Theme.of(context).textTheme.titleLarge),
                ],
              ),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(24, 16, 24, 24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Where is the lesion?', style: Theme.of(context).textTheme.titleMedium),
                    const SizedBox(height: 4),
                    Text('Tap the area on the body map to mark it.',
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: AppColors.onSurfaceVariant)),
                    const SizedBox(height: 16),
                    // Body map
                    Container(
                      height: 220,
                      decoration: BoxDecoration(color: AppColors.surfaceContainer, borderRadius: BorderRadius.circular(24)),
                      child: Stack(
                        alignment: Alignment.center,
                        children: [
                          const Icon(Icons.accessibility_new_rounded, size: 140, color: AppColors.outlineVariant),
                          Positioned.fill(
                            child: GridView.count(
                              crossAxisCount: 3,
                              physics: const NeverScrollableScrollPhysics(),
                              children: List.generate(9, (i) => GestureDetector(
                                onTap: () => setState(() => _tappedZone = _tappedZone == i ? null : i),
                                child: Container(
                                  margin: const EdgeInsets.all(4),
                                  decoration: BoxDecoration(
                                    color: _tappedZone == i ? AppColors.sage.withValues(alpha: 0.35) : Colors.transparent,
                                    borderRadius: BorderRadius.circular(12),
                                    border: _tappedZone == i ? Border.all(color: AppColors.sage, width: 1.5) : null,
                                  ),
                                  child: _tappedZone == i ? const Icon(Icons.place_rounded, color: AppColors.primary, size: 18) : null,
                                ),
                              )),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 24),
                    Text('How long have you noticed it?', style: Theme.of(context).textTheme.titleMedium),
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 10,
                      children: _durations.map((d) {
                        final selected = _duration == d;
                        return GestureDetector(
                          onTap: () => setState(() => _duration = selected ? null : d),
                          child: AnimatedContainer(
                            duration: const Duration(milliseconds: 180),
                            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
                            decoration: BoxDecoration(
                              color: selected ? AppColors.walnut : AppColors.surfaceContainerHigh,
                              borderRadius: BorderRadius.circular(50),
                            ),
                            child: Text(d,
                                style: GoogleFonts.plusJakartaSans(
                                  fontSize: 13, fontWeight: FontWeight.w700,
                                  color: selected ? Colors.white : AppColors.onSurface,
                                )),
                          ),
                        );
                      }).toList(),
                    ),
                    const SizedBox(height: 24),
                    Text('Any changes recently?', style: Theme.of(context).textTheme.titleMedium),
                    const SizedBox(height: 4),
                    Text('Select all that apply.',
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: AppColors.onSurfaceVariant)),
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 10, runSpacing: 10,
                      children: _symptomOptions.map((s) {
                        final selected = _symptoms.contains(s);
                        return GestureDetector(
                          onTap: () => setState(() {
                            if (s == 'None') { _symptoms.clear(); _symptoms.add('None'); }
                            else { _symptoms.remove('None'); selected ? _symptoms.remove(s) : _symptoms.add(s); }
                          }),
                          child: AnimatedContainer(
                            duration: const Duration(milliseconds: 180),
                            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
                            decoration: BoxDecoration(
                              color: selected ? AppColors.primaryContainer : AppColors.surfaceContainerHigh,
                              borderRadius: BorderRadius.circular(50),
                              border: selected ? Border.all(color: AppColors.primary, width: 1.5) : null,
                            ),
                            child: Text(s,
                                style: GoogleFonts.plusJakartaSans(
                                  fontSize: 13, fontWeight: FontWeight.w700,
                                  color: selected ? AppColors.onPrimaryContainer : AppColors.onSurface,
                                )),
                          ),
                        );
                      }).toList(),
                    ),
                    const SizedBox(height: 32),
                    GestureDetector(
                      onTap: () => ref.read(navigationProvider.notifier).state = AppScreen.scan,
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.symmetric(vertical: 18),
                        decoration: BoxDecoration(color: AppColors.walnut, borderRadius: BorderRadius.circular(50)),
                        child: Text('CONTINUE TO SCAN',
                            textAlign: TextAlign.center,
                            style: GoogleFonts.plusJakartaSans(
                              fontSize: 15, fontWeight: FontWeight.w800, color: Colors.white, letterSpacing: 1.2,
                            )),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Center(
                      child: GestureDetector(
                        onTap: () => ref.read(navigationProvider.notifier).state = AppScreen.scan,
                        child: Text('Skip to direct scan',
                            style: GoogleFonts.plusJakartaSans(
                              fontSize: 13, fontWeight: FontWeight.w600,
                              color: AppColors.onSurfaceVariant, decoration: TextDecoration.underline,
                            )),
                      ),
                    ),
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
