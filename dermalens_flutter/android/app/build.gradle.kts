plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.dermalens_app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.example.dermalens_app"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

// Kotlin 2.4 DSL (kotlinOptions {} is deprecated)
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

flutter {
    source = "../.."
}

// tflite_flutter 0.11.0 pins org.tensorflow:tensorflow-lite 2.11.0, whose
// sibling artifacts (-api, -gpu, -gpu-api) all share the "org.tensorflow.lite"
// namespace. TFLite ≥2.16.1 gives them unique namespaces, which AGP 9's
// manifest validation requires — force the newer (binary-compatible) versions.
configurations.all {
    resolutionStrategy.force(
        "org.tensorflow:tensorflow-lite:2.16.1",
        "org.tensorflow:tensorflow-lite-api:2.16.1",
        "org.tensorflow:tensorflow-lite-gpu:2.16.1",
        "org.tensorflow:tensorflow-lite-gpu-api:2.16.1",
    )
}
