import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/navigation_provider.dart';
import '../theme/app_theme.dart';

class ProfileScreen extends ConsumerStatefulWidget {
  const ProfileScreen({super.key});
  @override
  ConsumerState<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends ConsumerState<ProfileScreen> {
  bool _darkMode = false;
  bool _notifications = true;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(24, 20, 24, 32),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Profile',
                  style: GoogleFonts.plusJakartaSans(
                    fontSize: 26, fontWeight: FontWeight.w800,
                    color: AppColors.walnut, letterSpacing: -0.5)),
              const SizedBox(height: 24),

              // ── AVATAR + NAME ────────────────────────────────────
              Center(
                child: Column(
                  children: [
                    Container(
                      width: 88, height: 88,
                      decoration: BoxDecoration(
                        color: AppColors.primaryContainer,
                        shape: BoxShape.circle,
                        border: Border.all(color: AppColors.sage, width: 3),
                      ),
                      child: const Icon(Icons.person_rounded,
                          size: 44, color: AppColors.onPrimaryContainer),
                    ),
                    const SizedBox(height: 12),
                    Text('Sanjana',
                        style: GoogleFonts.plusJakartaSans(
                          fontSize: 20, fontWeight: FontWeight.w800, color: AppColors.onSurface)),
                    Text('DermaLens User',
                        style: GoogleFonts.plusJakartaSans(
                          fontSize: 13, color: AppColors.onSurfaceVariant)),
                  ],
                ),
              ),
              const SizedBox(height: 28),

              // ── SETTINGS ROWS ────────────────────────────────────
              _SectionLabel('Settings'),
              const SizedBox(height: 10),
              Container(
                decoration: BoxDecoration(
                  color: AppColors.surfaceContainerLow,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Column(
                  children: [
                    _SettingsRow(icon: Icons.notifications_rounded, label: 'Notifications',
                        trailing: Switch(
                          value: _notifications,
                          activeColor: AppColors.sage,
                          onChanged: (v) => setState(() => _notifications = v),
                        )),
                    _Divider(),
                    _SettingsRow(
                        icon: Icons.bar_chart_rounded, 
                        label: 'Scan History',
                        trailing: const Icon(Icons.chevron_right_rounded, color: AppColors.outline),
                        onTap: () => ref.read(navigationProvider.notifier).state = AppScreen.history,
                    ),
                    _Divider(),
                    _SettingsRow(
                        icon: Icons.near_me_rounded, 
                        label: 'Find Nearest Dermatologist',
                        trailing: const Icon(Icons.chevron_right_rounded, color: AppColors.outline),
                        onTap: () => ref.read(navigationProvider.notifier).state = AppScreen.dermatologist,
                    ),
                    _Divider(),
                    _SettingsRow(icon: Icons.lock_rounded, label: 'Privacy & Data',
                        trailing: const Icon(Icons.chevron_right_rounded, color: AppColors.outline)),
                    _Divider(),
                    _SettingsRow(icon: Icons.language_rounded, label: 'Language',
                        trailing: const Icon(Icons.chevron_right_rounded, color: AppColors.outline)),
                    _Divider(),
                    _SettingsRow(icon: Icons.dark_mode_rounded, label: 'Dark Mode',
                        trailing: Switch(
                          value: _darkMode,
                          activeColor: AppColors.sage,
                          onChanged: (v) => setState(() => _darkMode = v),
                        )),
                    _Divider(),
                    _SettingsRow(icon: Icons.help_rounded, label: 'Help & Support',
                        trailing: const Icon(Icons.chevron_right_rounded, color: AppColors.outline)),
                  ],
                ),
              ),
              const SizedBox(height: 24),

              // ── ABOUT SECTION ────────────────────────────────────
              _SectionLabel('About DermaLens'),
              const SizedBox(height: 10),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(22),
                decoration: BoxDecoration(
                  color: AppColors.walnut,
                  borderRadius: BorderRadius.circular(24),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('What is DermaLens?',
                        style: GoogleFonts.plusJakartaSans(
                          fontSize: 16, fontWeight: FontWeight.w800, color: Colors.white)),
                    const SizedBox(height: 8),
                    Text(
                      'An AI-powered skin lesion classifier built for fairness and explainability across all skin tones. Not a medical diagnosis tool.',
                      style: GoogleFonts.plusJakartaSans(
                        fontSize: 13, color: Colors.white70, height: 1.5)),
                    const SizedBox(height: 16),
                    Wrap(
                      spacing: 8, runSpacing: 8,
                      children: ['9 Lesion Classes', 'Fairness-Aware AI', 'On-Device Processing']
                          .map((b) => Container(
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                            decoration: BoxDecoration(
                              color: AppColors.sage.withValues(alpha: 0.25),
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(color: AppColors.sage.withValues(alpha: 0.5)),
                            ),
                            child: Text(b,
                                style: GoogleFonts.plusJakartaSans(
                                  fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.sage)),
                          ))
                          .toList(),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: AppColors.malignantBg,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: AppColors.malignant.withValues(alpha: 0.3)),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.warning_rounded, color: AppColors.malignant, size: 22),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Medical Disclaimer: DermaLens does not provide a formal medical diagnosis. Always consult a qualified dermatologist.',
                        style: GoogleFonts.plusJakartaSans(
                          fontSize: 12, color: AppColors.malignant, height: 1.5)),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 28),

              // ── LOGOUT ───────────────────────────────────────────
              GestureDetector(
                onTap: () => ref.read(navigationProvider.notifier).state = AppScreen.onboarding,
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  decoration: BoxDecoration(
                    color: AppColors.chestnut,
                    borderRadius: BorderRadius.circular(50),
                  ),
                  child: Text('Log Out',
                      textAlign: TextAlign.center,
                      style: GoogleFonts.plusJakartaSans(
                        fontSize: 15, fontWeight: FontWeight.w800, color: Colors.white, letterSpacing: 0.5)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  final String text;
  const _SectionLabel(this.text);
  @override
  Widget build(BuildContext context) => Text(text.toUpperCase(),
      style: Theme.of(context).textTheme.labelMedium?.copyWith(color: AppColors.onSurfaceVariant));
}

class _Divider extends StatelessWidget {
  @override
  Widget build(BuildContext context) =>
      const Divider(height: 1, thickness: 0.5, color: AppColors.outlineVariant, indent: 52);
}

class _SettingsRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final Widget trailing;
  final VoidCallback? onTap;
  const _SettingsRow({required this.icon, required this.label, required this.trailing, this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(10),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Row(
          children: [
            Container(
              width: 36, height: 36,
              decoration: BoxDecoration(
                color: AppColors.primaryFixed.withValues(alpha: 0.5),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, size: 18, color: AppColors.primary),
            ),
            const SizedBox(width: 14),
            Expanded(child: Text(label, style: Theme.of(context).textTheme.bodyMedium
                ?.copyWith(fontWeight: FontWeight.w600))),
            trailing,
          ],
        ),
      ),
    );
  }
}
