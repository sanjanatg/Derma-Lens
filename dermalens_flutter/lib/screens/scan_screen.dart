import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import '../providers/navigation_provider.dart';
import '../providers/analysis_provider.dart';
import '../theme/app_theme.dart';
import '../main.dart';

class ScanScreen extends ConsumerStatefulWidget {
  const ScanScreen({super.key});

  @override
  ConsumerState<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends ConsumerState<ScanScreen>
    with TickerProviderStateMixin {
  // ── Camera ──────────────────────────────────────────────────────────────
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  _CameraState _cameraState = _CameraState.requesting;

  // ── Animations ──────────────────────────────────────────────────────────
  late AnimationController _ringController;
  late AnimationController _shimmerController;
  late Animation<double> _ringScale;
  late Animation<double> _shimmerSlide;
  bool _isCapturing = false;
  bool _isFlashOn = false;

  // ──────────────────────────────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    _setupAnimations();
    _requestPermissionsAndInit();
  }

  void _setupAnimations() {
    _ringController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
    _ringScale = Tween<double>(begin: 0.92, end: 1.0).animate(
      CurvedAnimation(parent: _ringController, curve: Curves.easeInOut),
    );

    _shimmerController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat();
    _shimmerSlide = Tween<double>(begin: -1.0, end: 2.0).animate(
      CurvedAnimation(parent: _shimmerController, curve: Curves.easeInOut),
    );
  }

  // ── Permission + camera init ─────────────────────────────────────────
  Future<void> _requestPermissionsAndInit() async {
    // Request camera (+ storage so saves work on older Android).
    final statuses = await [
      Permission.camera,
      Permission.photos,        // iOS photo library / Android READ_MEDIA_IMAGES
      Permission.storage,       // Android ≤12 legacy storage
    ].request();

    final cameraGranted =
        statuses[Permission.camera] == PermissionStatus.granted;

    if (!cameraGranted) {
      if (mounted) setState(() => _cameraState = _CameraState.denied);
      return;
    }

    // Ensure cameras list is populated (may have been empty on startup).
    List<CameraDescription> availCams = cameras;
    if (availCams.isEmpty) {
      try {
        availCams = await availableCameras();
        cameras = availCams;
      } catch (_) {}
    }

    if (availCams.isEmpty) {
      if (mounted) setState(() => _cameraState = _CameraState.unavailable);
      return;
    }

    _controller = CameraController(
      availCams[0],
      ResolutionPreset.high,
      enableAudio: false,
    );
    _initializeControllerFuture = _controller!.initialize();

    try {
      await _initializeControllerFuture;
      if (mounted) setState(() => _cameraState = _CameraState.ready);
    } catch (e) {
      debugPrint('Camera initialize error: $e');
      if (mounted) setState(() => _cameraState = _CameraState.unavailable);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _ringController.dispose();
    _shimmerController.dispose();
    super.dispose();
  }

  // ── Capture + navigate to processing ─────────────────────────────────
  Future<void> _captureAndAnalyse() async {
    setState(() => _isCapturing = true);

    try {
      String imagePath = 'mock_image_path';

      if (_cameraState == _CameraState.ready &&
          _controller != null &&
          _controller!.value.isInitialized) {
        final image = await _controller!.takePicture();
        imagePath = image.path;
      }

      // Kick off inference in the background (mock: just store imagePath early)
      final inferenceService = ref.read(inferenceServiceProvider);
      final result = await inferenceService.runInference(imagePath);

      if (mounted) {
        if (result != null) {
          ref.read(analysisResultProvider.notifier).state = result;
          // Navigate to processing screen — it will auto-advance to results
          ref.read(navigationProvider.notifier).state = AppScreen.processing;
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Analysis failed. Please try again.')),
          );
          setState(() => _isCapturing = false);
        }
      }
    } catch (e) {
      debugPrint('Capture error: $e');
      if (mounted) setState(() => _isCapturing = false);
    }
  }

  Future<void> _toggleFlash() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    try {
      if (_isFlashOn) {
        await _controller!.setFlashMode(FlashMode.off);
        setState(() => _isFlashOn = false);
      } else {
        await _controller!.setFlashMode(FlashMode.torch);
        setState(() => _isFlashOn = true);
      }
    } catch (e) {
      debugPrint('Flash error: $e');
    }
  }

  // ──────────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // ── CAMERA / FALLBACK BACKGROUND ───────────────────────────
          Positioned.fill(child: _buildBackground()),

          // Top gradient fade
          Positioned(
            top: 0, left: 0, right: 0, height: 200,
            child: DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.black.withValues(alpha: 0.65),
                    Colors.transparent,
                  ],
                ),
              ),
            ),
          ),

          // Bottom gradient fade
          Positioned(
            bottom: 0, left: 0, right: 0, height: 280,
            child: DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.bottomCenter,
                  end: Alignment.topCenter,
                  colors: [
                    Colors.black.withValues(alpha: 0.75),
                    Colors.transparent,
                  ],
                ),
              ),
            ),
          ),

          // ── TOP BAR ─────────────────────────────────────────────
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: Row(
                children: [
                  GestureDetector(
                    onTap: () {
                      ref.read(navTabProvider.notifier).state = NavTab.home;
                      ref.read(navigationProvider.notifier).state = AppScreen.home;
                    },
                    child: Container(
                      width: 40, height: 40,
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.15),
                        shape: BoxShape.circle,
                        border: Border.all(
                            color: Colors.white.withValues(alpha: 0.2), width: 1),
                      ),
                      child: const Icon(Icons.close_rounded,
                          color: Colors.white, size: 20),
                    ),
                  ),
                  const SizedBox(width: 14),
                  Text(
                    'ANALYSIS MODE',
                    style: GoogleFonts.plusJakartaSans(
                      color: Colors.white.withValues(alpha: 0.8),
                      letterSpacing: 2.0,
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const Spacer(),
                  // Flash toggle button
                  GestureDetector(
                    onTap: _toggleFlash,
                    child: Container(
                      width: 40, height: 40,
                      decoration: BoxDecoration(
                        color: _isFlashOn ? Colors.white : Colors.white.withValues(alpha: 0.15),
                        shape: BoxShape.circle,
                        border: Border.all(
                            color: Colors.white.withValues(alpha: 0.2), width: 1),
                      ),
                      child: Icon(
                        _isFlashOn ? Icons.flash_on_rounded : Icons.flash_off_rounded,
                        color: _isFlashOn ? Colors.black : Colors.white,
                        size: 20,
                      ),
                    ),
                  ),
                  const SizedBox(width: 14),
                  // Live indicator dot
                  Container(
                    width: 8, height: 8,
                    decoration: BoxDecoration(
                      color: _cameraState == _CameraState.ready
                          ? AppColors.primaryContainer
                          : Colors.orange,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: (_cameraState == _CameraState.ready
                                  ? AppColors.primaryContainer
                                  : Colors.orange)
                              .withValues(alpha: 0.6),
                          blurRadius: 8,
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),

          // ── PERMISSION DENIED / NO CAMERA overlays ───────────────
          if (_cameraState == _CameraState.denied)
            _buildOverlay(
              icon: Icons.no_photography_outlined,
              title: 'Camera Access Required',
              body:
                  'DermaLens needs the camera to capture skin lesions.\nPlease grant access in Settings.',
              actionLabel: 'Open Settings',
              onAction: () async {
                await openAppSettings();
                // Re-try init when user returns.
                setState(() => _cameraState = _CameraState.requesting);
                _requestPermissionsAndInit();
              },
            )
          else if (_cameraState == _CameraState.unavailable)
            _buildOverlay(
              icon: Icons.videocam_off_outlined,
              title: 'Camera Unavailable',
              body:
                  'No camera was found on this device.\nYou can still run a demo analysis.',
              actionLabel: 'Run Demo',
              onAction: _captureAndAnalyse,
            )
          else if (_cameraState == _CameraState.requesting)
            const Center(
              child: CircularProgressIndicator(
                  color: AppColors.primaryContainer),
            )
          else ...[
            // ── SCANNING RING ───────────────────────────────────
            Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  AnimatedBuilder(
                    animation: _ringScale,
                    builder: (_, __) => Transform.scale(
                      scale: _ringScale.value,
                      child: SizedBox(
                        width: 280, height: 280,
                        child: Stack(
                          alignment: Alignment.center,
                          children: [
                            Container(
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: AppColors.primaryContainer
                                    .withValues(alpha: 0.12),
                                border: Border.all(
                                  color: AppColors.primaryContainer
                                      .withValues(alpha: 0.45),
                                  width: 1.5,
                                ),
                              ),
                            ),
                            Padding(
                              padding: const EdgeInsets.all(16),
                              child: Container(
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  border: Border.all(
                                    color: Colors.white.withValues(alpha: 0.25),
                                    width: 1,
                                  ),
                                ),
                              ),
                            ),
                            // Shimmer scan line
                            ClipOval(
                              child: AnimatedBuilder(
                                animation: _shimmerSlide,
                                builder: (_, __) => Transform.translate(
                                  offset: Offset(
                                      0, _shimmerSlide.value * 140 - 140),
                                  child: Container(
                                    height: 2,
                                    decoration: BoxDecoration(
                                      gradient: LinearGradient(
                                        colors: [
                                          Colors.transparent,
                                          AppColors.primaryContainer
                                              .withValues(alpha: 0.8),
                                          Colors.transparent,
                                        ],
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.4),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      _isCapturing
                          ? 'Analysing specimen…'
                          : 'Frame the lesion within the circle',
                      style: GoogleFonts.plusJakartaSans(
                        color: Colors.white.withValues(alpha: 0.85),
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // ── SHUTTER BUTTON ──────────────────────────────────
            Positioned(
              bottom: 52, left: 0, right: 0,
              child: Column(
                children: [
                  GestureDetector(
                    onTap: _isCapturing ? null : _captureAndAnalyse,
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      width: _isCapturing ? 70 : 80,
                      height: _isCapturing ? 70 : 80,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.white,
                        boxShadow: [
                          BoxShadow(
                            color: AppColors.primaryContainer
                                .withValues(alpha: 0.4),
                            blurRadius: 24,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(5),
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 200),
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: _isCapturing
                                ? AppColors.primaryContainer
                                : Colors.transparent,
                            border: Border.all(
                              color: _isCapturing
                                  ? AppColors.primaryContainer
                                  : Colors.black,
                              width: 2.5,
                            ),
                          ),
                          child: _isCapturing
                              ? const Center(
                                  child: SizedBox(
                                    width: 24, height: 24,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2.5,
                                      color: AppColors.onPrimaryContainer,
                                    ),
                                  ),
                                )
                              : null,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'TAP TO ANALYSE',
                    style: GoogleFonts.plusJakartaSans(
                      color: Colors.white.withValues(alpha: 0.6),
                      fontSize: 10,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1.8,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  // ── Helpers ─────────────────────────────────────────────────────────
  Widget _buildBackground() {
    if (_cameraState == _CameraState.ready &&
        _controller != null &&
        _controller!.value.isInitialized) {
      return CameraPreview(_controller!);
    }
    // Dark fallback viewfinder
    return Container(
      decoration: BoxDecoration(
        gradient: RadialGradient(
          center: Alignment.center,
          radius: 0.9,
          colors: [const Color(0xFF1a1a2e), Colors.black],
        ),
      ),
      child: CustomPaint(painter: _GridPainter()),
    );
  }

  Widget _buildOverlay({
    required IconData icon,
    required String title,
    required String body,
    required String actionLabel,
    required VoidCallback onAction,
  }) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 40),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: Colors.white54, size: 64),
            const SizedBox(height: 20),
            Text(
              title,
              textAlign: TextAlign.center,
              style: GoogleFonts.plusJakartaSans(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              body,
              textAlign: TextAlign.center,
              style: GoogleFonts.plusJakartaSans(
                color: Colors.white60,
                fontSize: 14,
                height: 1.5,
              ),
            ),
            const SizedBox(height: 32),
            GestureDetector(
              onTap: onAction,
              child: Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 28, vertical: 14),
                decoration: BoxDecoration(
                  color: AppColors.primaryContainer,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Text(
                  actionLabel,
                  style: GoogleFonts.plusJakartaSans(
                    color: AppColors.onPrimaryContainer,
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
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

// ── Camera state enum ─────────────────────────────────────────────────────
enum _CameraState { requesting, denied, unavailable, ready }

// ── Subtle grid background painter ───────────────────────────────────────
class _GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withValues(alpha: 0.03)
      ..strokeWidth = 1;
    const step = 40.0;
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
