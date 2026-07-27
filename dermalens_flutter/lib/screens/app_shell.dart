import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/navigation_provider.dart';
import '../theme/app_theme.dart';

/// Wraps the 4 bottom-nav screens with the shared nav bar.
class AppShell extends ConsumerWidget {
  final Widget body;
  const AppShell({super.key, required this.body});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final tab = ref.watch(navTabProvider);

    return Scaffold(
      backgroundColor: AppColors.background,
      body: body,
      bottomNavigationBar: _BottomNav(
        currentTab: tab,
        onTap: (t) {
          ref.read(navTabProvider.notifier).state = t;
          switch (t) {
            case NavTab.home:
              ref.read(navigationProvider.notifier).state = AppScreen.home;
            case NavTab.scan:
              ref.read(navigationProvider.notifier).state = AppScreen.symptomChecker;
            case NavTab.learn:
              ref.read(navigationProvider.notifier).state = AppScreen.lesionGuide;
            case NavTab.profile:
              ref.read(navigationProvider.notifier).state = AppScreen.profile;
          }
        },
      ),
    );
  }
}

class _BottomNav extends StatelessWidget {
  final NavTab currentTab;
  final void Function(NavTab) onTap;
  const _BottomNav({required this.currentTab, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final items = [
      (NavTab.home,    Icons.home_rounded,               'Home'),
      (NavTab.scan,    Icons.document_scanner_rounded,   'Scan'),
      (NavTab.learn,   Icons.menu_book_rounded,          'Learn'),
      (NavTab.profile, Icons.person_rounded,             'Profile'),
    ];

    return Container(
      decoration: BoxDecoration(
        color: AppColors.surfaceContainerLowest,
        border: const Border(top: BorderSide(color: AppColors.outlineVariant, width: 0.5)),
        boxShadow: [
          BoxShadow(
            color: AppColors.onSurface.withValues(alpha: 0.04),
            blurRadius: 16, offset: const Offset(0, -4),
          ),
        ],
      ),
      child: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: items.map(((NavTab, IconData, String) item) {
              final (tab, icon, label) = item;
              final active = currentTab == tab;
              final isScan = tab == NavTab.scan;

              if (isScan) {
                // Elevated scan FAB in the middle
                return GestureDetector(
                  onTap: () => onTap(tab),
                  child: Container(
                    width: 60, height: 60,
                    decoration: BoxDecoration(
                      color: AppColors.walnut,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: AppColors.walnut.withValues(alpha: 0.3),
                          blurRadius: 16, offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: Icon(icon, color: Colors.white, size: 28),
                  ),
                );
              }

              return GestureDetector(
                onTap: () => onTap(tab),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: active
                        ? AppColors.primaryFixed.withValues(alpha: 0.5)
                        : Colors.transparent,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(icon,
                          size: 22,
                          color: active ? AppColors.primary : AppColors.onSurfaceVariant),
                      const SizedBox(height: 3),
                      Text(label,
                          style: GoogleFonts.plusJakartaSans(
                            fontSize: 10, fontWeight: FontWeight.w700,
                            color: active ? AppColors.primary : AppColors.onSurfaceVariant,
                          )),
                    ],
                  ),
                ),
              );
            }).toList(),
          ),
        ),
      ),
    );
  }
}
