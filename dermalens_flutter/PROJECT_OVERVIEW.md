# DermaLens - Project Overview

## 📋 Executive Summary

**DermaLens** is a premium mobile health application for AI-powered skin lesion analysis and classification. It uses on-device machine learning to provide instant skin health assessments with a focus on early detection of malignant conditions like melanoma and carcinoma.

**Version:** 1.0.0+1  
**Design Philosophy:** Tactile Apothecary aesthetic - combining clinical precision with warm, natural design  
**Target Platforms:** Android, iOS, Windows, Linux, macOS

---

## 🎯 Core Features

### 1. **AI-Powered Skin Analysis**
- Real-time skin lesion classification using TensorFlow Lite
- 8-class dermatological taxonomy (FairDermNet model)
- Clinical risk assessment (malignant vs. benign)
- Confidence scoring and probability distribution
- On-device inference (privacy-first, no cloud dependency)

### 2. **Image Capture & Processing**
- Native camera integration
- 224×224 image preprocessing pipeline
- ImageNet normalization (mean/std standardization)
- NCHW tensor layout for optimal performance

### 3. **Clinical Classification System**
**Malignant Conditions (High Risk):**
- Melanoma
- Basal Cell Carcinoma
- Squamous Cell Carcinoma
- Actinic Keratosis (pre-cancerous)

**Benign Conditions (Low Risk):**
- Melanocytic Nevi
- Benign Keratosis
- Dermatofibroma
- Vascular Lesion

### 4. **User Experience Features**
- **Onboarding Flow:** First-time user education
- **Scan Interface:** Camera capture with guidance
- **Processing Animation:** Real-time inference feedback
- **Results Dashboard:** Detailed analysis with visualizations
- **History Tracking:** Past scans and trends
- **Lesion Guide:** Educational reference library
- **Profile Management:** User settings and preferences
- **Dermatologist Referral:** Professional consultation pathways
- **Symptom Checker:** Pre-scan questionnaire

### 5. **Risk Assessment**
- Clinical threshold-based decision making (default: 0.5)
- Malignant probability aggregation (classes 0-3)
- Color-coded risk indicators (red for malignant, green for benign)
- Calibratable threshold system via configuration

---

## 🏗️ Architecture

### **Design Pattern**
- **Architecture:** MVVM with Provider pattern
- **State Management:** Riverpod (reactive, type-safe)
- **Navigation:** Declarative screen management
- **Dependency Injection:** Provider overrides

### **Project Structure**
```
lib/
├── main.dart                    # App entry point, camera init
├── theme/
│   └── app_theme.dart          # Tactile Apothecary design tokens
├── screens/                     # UI screens (12 total)
│   ├── onboarding_screen.dart
│   ├── home_screen.dart
│   ├── scan_screen.dart
│   ├── processing_screen.dart
│   ├── results_screen.dart
│   ├── history_screen.dart
│   ├── lesion_guide_screen.dart
│   ├── profile_screen.dart
│   ├── symptom_checker_screen.dart
│   ├── dermatologist_screen.dart
│   ├── about_screen.dart
│   └── app_shell.dart          # Bottom navigation wrapper
├── providers/                   # State management
│   ├── navigation_provider.dart
│   ├── analysis_provider.dart
│   └── history_provider.dart
└── services/
    └── inference_service.dart   # TFLite ML inference engine

assets/
├── labels.txt                   # 8-class taxonomy labels
├── threshold_config.json        # Clinical threshold configuration
└── fairdermnet_int8.tflite     # INT8-quantized model (missing)
```

---

## 🛠️ Technology Stack

### **Frontend Framework**
- **Flutter SDK:** 3.0.0+ (Dart 3.12.2)
- **UI Framework:** Material Design 3
- **Platform Support:** Android, iOS, Windows, macOS, Linux, Web*

*Web support limited due to TFLite FFI dependency

### **State Management & Architecture**
| Package | Version | Purpose |
|---------|---------|---------|
| `flutter_riverpod` | ^2.4.9 | Reactive state management, dependency injection |

### **Machine Learning & Image Processing**
| Package | Version | Purpose |
|---------|---------|---------|
| `tflite_flutter` | ^0.10.4 | On-device TensorFlow Lite inference (FFI-based) |
| `image` | ^4.1.7 | Image decoding, resizing, pixel manipulation |

**ML Model Specifications:**
- **Model:** FairDermNet-8class-v1
- **Format:** TFLite INT8 quantized
- **Size:** <6 MB (estimated)
- **Input Shape:** [1, 3, 224, 224] (NCHW)
- **Output Shape:** [1, 8] (raw logits)
- **Preprocessing:** ImageNet normalization
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### **Device Integration**
| Package | Version | Purpose |
|---------|---------|---------|
| `camera` | ^0.11.0 | Native camera access (Android/iOS) |
| `permission_handler` | ^11.3.0 | Runtime permissions (camera, storage) |
| `shared_preferences` | ^2.5.5 | Local key-value storage (onboarding state) |

### **UI & Design**
| Package | Version | Purpose |
|---------|---------|---------|
| `google_fonts` | ^6.1.0 | Plus Jakarta Sans typography |
| `flutter_svg` | ^2.0.9 | Scalable vector graphics support |

### **Development Tools**
| Package | Version | Purpose |
|---------|---------|---------|
| `flutter_lints` | ^3.0.0 | Dart/Flutter code analysis |
| `flutter_test` | SDK | Unit and widget testing |

---

## 🎨 Design System

### **Design Philosophy: Tactile Apothecary**
Warm, natural, and clinically trustworthy - blending traditional apothecary aesthetics with modern medical precision.

### **Color Palette**
**Primary (Muted Sage):**
- Primary: `#5A6241`
- Primary Container (CTA): `#A7B08A`
- Primary Fixed: `#DEE7BE`

**Named Design Tokens:**
- Walnut: `#2F1F18` (darkest text/cards)
- Chestnut: `#5B3A2A` (secondary cards)
- Earthy Tan: `#8D6B4A` (accents/borders)
- Sage: `#A7B08A` (CTA buttons)

**Surface Hierarchy (Linen White System):**
- Background: `#F2EFE8`
- Surface: `#FCF9F2`
- Surface Container: `#F1EEE7`
- Surface Dim: `#DCDAD3`

**Clinical Risk Colors:**
- Malignant: `#D32F2F` (red) on `#FFEBEE` (light red)
- Benign: `#2E7D32` (green) on `#E8F5E9` (light green)

### **Typography**
**Font Family:** Plus Jakarta Sans (Google Fonts)

| Style | Size | Weight | Use Case |
|-------|------|--------|----------|
| Display Large | 52px | 800 | Hero headlines |
| Display Medium | 40px | 700 | Section titles |
| Headline Large | 30px | 700 | Screen headers |
| Headline Medium | 24px | 700 | Card titles |
| Title Large | 18px | 600 | List headers |
| Body Large | 16px | 400 | Primary content |
| Label Large | 13px | 700 | Button text (1.4 letter-spacing) |

### **Component Styling**
- **Buttons:** Stadium shape (full rounded), 58px height
- **Cards:** 28px border radius, elevation: 0
- **AppBar:** Transparent, no elevation
- **Spacing:** Material 3 spacing system

---

## 🔬 Machine Learning Pipeline

### **Inference Flow**
```
1. Image Capture (Camera/Gallery)
   ↓
2. Image Decode (JPEG/PNG → RGB)
   ↓
3. Resize (Bilinear → 224×224)
   ↓
4. Normalize (ImageNet mean/std)
   ↓
5. Layout Transform (NHWC → NCHW)
   ↓
6. TFLite Inference (INT8 quantized)
   ↓
7. Softmax Activation (logits → probabilities)
   ↓
8. Clinical Threshold (malignant probability)
   ↓
9. Result Display (label + confidence + risk)
```

### **Model Training Alignment**
The inference pipeline **exactly mirrors** the training preprocessing:
- Same resize method (bilinear)
- Same normalization constants
- Same channel order (RGB, no BGR swap)
- Same tensor layout (NCHW)
- Same augmentation pipeline reference

### **Clinical Decision Logic**
```dart
malignant_probability = sum(probabilities[0:3])  // Classes 0-3
is_malignant = malignant_probability >= clinical_threshold

if is_malignant:
    show_red_alert()
    recommend_dermatologist()
else:
    show_green_status()
    provide_educational_content()
```

---

## 📱 Platform Support Matrix

| Platform | Status | Requirements |
|----------|--------|--------------|
| **Android** | ✅ Full Support | API 21+ (Android 5.0+) |
| **iOS** | ✅ Full Support | iOS 12.0+ |
| **Windows** | ✅ Full Support | Visual Studio 2022 + C++ workload |
| **macOS** | ✅ Full Support | Xcode 13+ |
| **Linux** | ✅ Full Support | GTK 3.0+, Clang |
| **Web** | ❌ Limited | No TFLite support (dart:ffi unavailable) |

**Note:** Web builds can run UI/navigation but ML inference will fail due to FFI dependency.

---

## 🔐 Privacy & Security

### **Privacy-First Design**
- **100% On-Device Processing:** No cloud uploads, no external API calls
- **Local Storage Only:** Scan history stored in SharedPreferences
- **No User Authentication:** No personal data collection
- **Camera Permissions:** Explicitly requested, user-controlled
- **Offline Capable:** Full functionality without internet

### **Data Handling**
- Images processed in memory only
- No automatic backup or sync
- User can delete history at any time
- No analytics or tracking

---

## 🚀 Running the Application

### **Prerequisites**
1. Flutter SDK 3.0.0+
2. Platform-specific tools:
   - Android: Android Studio + SDK
   - iOS: Xcode + CocoaPods
   - Windows: Visual Studio 2022 + Desktop C++ workload
   - Linux: GTK dev libraries
   - macOS: Xcode Command Line Tools

### **Installation**
```bash
# Clone repository
cd dermalens_flutter

# Install dependencies
flutter pub get

# Run on connected device
flutter run

# Run on specific platform
flutter run -d windows
flutter run -d chrome  # Limited functionality
flutter run -d <android-device-id>
```

### **Build Commands**
```bash
# Android APK
flutter build apk --release

# Android App Bundle
flutter build appbundle --release

# iOS
flutter build ios --release

# Windows
flutter build windows --release

# macOS
flutter build macos --release
```

---

## ⚠️ Current Limitations

### **Missing Assets**
- **fairdermnet_int8.tflite:** ML model file not included
  - Expected location: `assets/fairdermnet_int8.tflite`
  - Size: <6 MB
  - Format: TFLite INT8 quantized
  - Status: Must be added before inference will work

### **Clinical Validation**
- **Threshold not calibrated:** Default 0.5 is placeholder
- **No FDA approval:** Research/educational use only
- **Not a medical device:** Should not replace professional diagnosis

### **Platform Limitations**
- **Web:** TFLite not supported (dart:ffi unavailable)
- **Windows:** Requires Developer Mode + Visual Studio
- **Symlink Support:** Windows requires Developer Mode enabled

---

## 📊 Performance Considerations

### **Inference Latency**
- **Expected:** <100ms on modern devices (model-dependent)
- **Threads:** 4 (configurable via InterpreterOptions)
- **Quantization:** INT8 for optimal mobile performance

### **Memory Footprint**
- **Model Size:** <6 MB
- **Input Tensor:** ~600 KB (1×3×224×224 float32)
- **App Size:** ~50-80 MB (platform-dependent)

### **Battery Impact**
- On-device inference = minimal battery drain
- No network requests = no radio power consumption
- Camera usage = primary battery consumer

---

## 🧪 Testing Strategy

### **Unit Tests**
- Inference service logic
- Image preprocessing pipeline
- Softmax calculations
- Clinical threshold logic

### **Widget Tests**
- Screen navigation flows
- Camera permission handling
- Result display components
- History list rendering

### **Integration Tests**
- End-to-end scan workflow
- Model loading and inference
- State management transitions
- Platform-specific features

---

## 🗺️ Future Roadmap

### **Planned Features**
- [ ] Multi-image analysis (track lesion changes over time)
- [ ] Export reports as PDF
- [ ] Integration with health apps (Apple Health, Google Fit)
- [ ] Multi-language support
- [ ] Accessibility enhancements (screen reader support)
- [ ] Telemedicine integration
- [ ] AI model updates via OTA

### **Technical Improvements**
- [ ] Model quantization optimization (INT4?)
- [ ] GPU delegate support for faster inference
- [ ] Cloud backup (optional, encrypted)
- [ ] Federated learning integration
- [ ] Advanced image augmentation for robustness

---

## 📚 Medical Disclaimer

**DermaLens is a research and educational tool, NOT a medical device.**

- This app does NOT replace professional medical advice
- Always consult a dermatologist for proper diagnosis
- Do not make treatment decisions based solely on this app
- Early detection requires professional examination
- AI models can make mistakes - human expertise is essential

---

## 👥 Target Audience

1. **General Public:** Skin health awareness and self-monitoring
2. **Medical Students:** Dermatology education and training
3. **Researchers:** Skin lesion classification experiments
4. **Healthcare Providers:** Triage and screening assistance (with proper validation)

---

## 📄 License & Attribution

**Project:** DermaLens  
**Description:** A premium skin health application with a Tactile Apothecary aesthetic  
**Model:** FairDermNet-8class-v1  

*Ensure proper licensing for medical AI applications in your jurisdiction.*

---

## 📞 Support & Contribution

For issues, feature requests, or contributions, please refer to the project repository documentation.

**Created:** 2026  
**Last Updated:** 2026-08-05

---

*This document provides a comprehensive technical and functional overview of the DermaLens application. For implementation details, refer to the source code and inline documentation.*
