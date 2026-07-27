import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

/// Stitch "Tactile Apothecary" design tokens
class AppColors {
  // ── Stitch primary palette – Muted Sage ──────────────────────────
  static const Color primary = Color(0xFF5A6241);
  static const Color onPrimary = Color(0xFFFFFFFF);
  static const Color primaryContainer = Color(0xFFA7B08A); // CTA Sage
  static const Color onPrimaryContainer = Color(0xFF3B4325);
  static const Color primaryFixed = Color(0xFFDEE7BE);
  static const Color primaryFixedDim = Color(0xFFC2CBA3);

  // ── Design spec named colours ────────────────────────────────────
  static const Color walnut = Color(0xFF2F1F18);       // darkest cards / text
  static const Color chestnut = Color(0xFF5B3A2A);     // secondary cards
  static const Color earthyTan = Color(0xFF8D6B4A);    // accent / borders
  static const Color sage = Color(0xFFA7B08A);         // CTA buttons

  // ── Secondary palette ────────────────────────────────────────────
  static const Color secondary = Color(0xFF6F5A51);
  static const Color onSecondary = Color(0xFFFFFFFF);
  static const Color secondaryContainer = Color(0xFFFADCD1);
  static const Color onSecondaryContainer = Color(0xFF756057);

  // ── Tertiary palette ─────────────────────────────────────────────
  static const Color tertiary = Color(0xFF70566C);
  static const Color tertiaryContainer = Color(0xFFC2A3BB);
  static const Color onTertiaryContainer = Color(0xFF50394D);

  // ── Surface hierarchy – Linen White system ───────────────────────
  static const Color surface = Color(0xFFFCF9F2);
  static const Color background = Color(0xFFF2EFE8); // spec background
  static const Color surfaceContainerLowest = Color(0xFFFFFFFF);
  static const Color surfaceContainerLow = Color(0xFFF6F3EC);
  static const Color surfaceContainer = Color(0xFFF1EEE7);
  static const Color surfaceContainerHigh = Color(0xFFEBE8E1);
  static const Color surfaceContainerHighest = Color(0xFFE5E2DB);
  static const Color surfaceDim = Color(0xFFDCDAD3);

  // ── On-surface ───────────────────────────────────────────────────
  static const Color onSurface = Color(0xFF1C1C18);
  static const Color onSurfaceVariant = Color(0xFF46483E);

  // ── Outline ──────────────────────────────────────────────────────
  static const Color outline = Color(0xFF77786D);
  static const Color outlineVariant = Color(0xFFC7C7BB);

  // ── Error / Risk ─────────────────────────────────────────────────
  static const Color error = Color(0xFFBA1A1A);
  static const Color errorContainer = Color(0xFFFFDAD6);
  static const Color malignant = Color(0xFFD32F2F);
  static const Color malignantBg = Color(0xFFFFEBEE);
  static const Color benign = Color(0xFF2E7D32);
  static const Color benignBg = Color(0xFFE8F5E9);

  // ── Inverse ──────────────────────────────────────────────────────
  static const Color inverseSurface = Color(0xFF31312C);
  static const Color inverseOnSurface = Color(0xFFF3F0E9);
  static const Color inversePrimary = Color(0xFFC2CBA3);
}

class AppTheme {
  static ThemeData get lightTheme {
    const colorScheme = ColorScheme(
      brightness: Brightness.light,
      primary: AppColors.primary,
      onPrimary: AppColors.onPrimary,
      primaryContainer: AppColors.primaryContainer,
      onPrimaryContainer: AppColors.onPrimaryContainer,
      secondary: AppColors.secondary,
      onSecondary: AppColors.onSecondary,
      secondaryContainer: AppColors.secondaryContainer,
      onSecondaryContainer: AppColors.onSecondaryContainer,
      tertiary: AppColors.tertiary,
      onTertiary: AppColors.onPrimary,
      tertiaryContainer: AppColors.tertiaryContainer,
      onTertiaryContainer: AppColors.onTertiaryContainer,
      error: AppColors.error,
      onError: AppColors.onPrimary,
      errorContainer: AppColors.errorContainer,
      onErrorContainer: Color(0xFF93000A),
      surface: AppColors.surface,
      onSurface: AppColors.onSurface,
      onSurfaceVariant: AppColors.onSurfaceVariant,
      outline: AppColors.outline,
      outlineVariant: AppColors.outlineVariant,
      inverseSurface: AppColors.inverseSurface,
      onInverseSurface: AppColors.inverseOnSurface,
      inversePrimary: AppColors.inversePrimary,
      surfaceTint: AppColors.primary,
    );

    final textTheme = GoogleFonts.plusJakartaSansTextTheme().copyWith(
      displayLarge: GoogleFonts.plusJakartaSans(
        fontSize: 52, fontWeight: FontWeight.w800,
        color: AppColors.onSurface, letterSpacing: -1.5, height: 1.1,
      ),
      displayMedium: GoogleFonts.plusJakartaSans(
        fontSize: 40, fontWeight: FontWeight.w700,
        color: AppColors.onSurface, letterSpacing: -1.0, height: 1.15,
      ),
      headlineLarge: GoogleFonts.plusJakartaSans(
        fontSize: 30, fontWeight: FontWeight.w700,
        color: AppColors.onSurface, letterSpacing: -0.5,
      ),
      headlineMedium: GoogleFonts.plusJakartaSans(
        fontSize: 24, fontWeight: FontWeight.w700,
        color: AppColors.onSurface, letterSpacing: -0.3,
      ),
      titleLarge: GoogleFonts.plusJakartaSans(
        fontSize: 18, fontWeight: FontWeight.w600, color: AppColors.onSurface,
      ),
      titleMedium: GoogleFonts.plusJakartaSans(
        fontSize: 15, fontWeight: FontWeight.w600,
        color: AppColors.onSurface, letterSpacing: 0.15,
      ),
      bodyLarge: GoogleFonts.plusJakartaSans(
        fontSize: 16, height: 1.6,
        color: AppColors.onSurface, fontWeight: FontWeight.w400,
      ),
      bodyMedium: GoogleFonts.plusJakartaSans(
        fontSize: 14, height: 1.5, color: AppColors.onSurface,
      ),
      labelLarge: GoogleFonts.plusJakartaSans(
        fontSize: 13, fontWeight: FontWeight.w700,
        letterSpacing: 1.4, color: AppColors.onSurface,
      ),
      labelMedium: GoogleFonts.plusJakartaSans(
        fontSize: 11, fontWeight: FontWeight.w700,
        letterSpacing: 1.2, color: AppColors.onSurfaceVariant,
      ),
      labelSmall: GoogleFonts.plusJakartaSans(
        fontSize: 10, fontWeight: FontWeight.w500,
        letterSpacing: 1.0, color: AppColors.onSurfaceVariant,
      ),
    );

    return ThemeData(
      useMaterial3: true,
      colorScheme: colorScheme,
      textTheme: textTheme,
      scaffoldBackgroundColor: AppColors.background,
      appBarTheme: AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        scrolledUnderElevation: 0,
        systemOverlayStyle: SystemUiOverlayStyle.dark,
        titleTextStyle: GoogleFonts.plusJakartaSans(
          fontSize: 17, fontWeight: FontWeight.w700,
          color: AppColors.onSurface, letterSpacing: -0.2,
        ),
        iconTheme: const IconThemeData(color: AppColors.onSurface),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.primaryContainer,
          foregroundColor: AppColors.onPrimaryContainer,
          minimumSize: const Size(double.infinity, 58),
          shape: const StadiumBorder(),
          elevation: 0,
          shadowColor: Colors.transparent,
          textStyle: GoogleFonts.plusJakartaSans(
            fontWeight: FontWeight.w800, fontSize: 15, letterSpacing: 1.4,
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: AppColors.onSurface,
          minimumSize: const Size(double.infinity, 58),
          shape: const StadiumBorder(),
          side: const BorderSide(color: AppColors.outlineVariant),
          textStyle: GoogleFonts.plusJakartaSans(
            fontWeight: FontWeight.w700, fontSize: 15, letterSpacing: 1.2,
          ),
        ),
      ),
      cardTheme: CardThemeData(
        color: AppColors.surfaceContainer,
        elevation: 0,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
        margin: EdgeInsets.zero,
      ),
    );
  }
}
