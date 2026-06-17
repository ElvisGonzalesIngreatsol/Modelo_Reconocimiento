plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.reconocimiento_manos"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.reconocimiento_manos"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    // ==========================================================================
    // TRUCO DE EMPAQUETADO: Obliga a Gradle a ignorar los duplicados del manifiesto
    // ==========================================================================
    packaging {
        resources {
            merges.add("**/*.xml")
            merges.add("META-INF/**")
        }
    }
}

dependencies {
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)

    // ==========================================================================
    // Librerías de CameraX (Para controlar la cámara)
    // ==========================================================================
    implementation("androidx.camera:camera-core:1.3.1")
    implementation("androidx.camera:camera-camera2:1.3.1")
    implementation("androidx.camera:camera-lifecycle:1.3.1")
    implementation("androidx.camera:camera-view:1.3.1")

    // ==========================================================================
    // LIBRERÍA DE TENSORFLOW LITE ACTUALIZADA (Sin colisión de manifiestos)
    // ==========================================================================
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
}