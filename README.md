# FishGo 🐟

Indonesian Fish Species Detection App for Android

## Overview

FishGo is an MVP Android application designed to help Indonesian fishermen identify and log fish catches using smartphone cameras. The app uses on-device AI (YOLOv8n + TensorFlow Lite) for real-time fish species detection.

## Features

- **Real-time Fish Detection**: Using CameraX and TFLite for on-device inference
- **19 Indonesian Fish Species**: Commercial marine fish species trained model
- **Indonesian Names**: Displays fish names in Bahasa Indonesia
- **Bounding Box Overlay**: Visual detection feedback
- **Modular Design**: Ready for integration into Navigo app

## Supported Fish Species

| Scientific Name | Indonesian Name |
|----------------|-----------------|
| Katsuwonus Pelamis | Cakalang |
| Euthynnus Affinis | Tongkol |
| Lutjanus Malabaricus | Kakap Merah |
| Caranx Ignobilis | Kuwe |
| Chanos Chanos | Bandeng |
| Rastrelliger Kanagurta | Kembung |
| Parastromateus Niger | Bawal Hitam |
| Thunnus Obesus | Tuna Mata Besar |
| Thunnus Alalunga | Albacore Tuna |
| Thunnus Tonggol | Tuna |
| Scomberomorus Guttatus | Tenggiri Papan |
| Alepes Djedaba | Selar Bulat |
| Decapterus Macarellus | Malalugis |
| *...and 6 more species* | |

## Tech Stack

- **Language**: Kotlin
- **Architecture**: MVVM (simplified for MVP)
- **Camera**: CameraX (androidx.camera)
- **AI/ML**: TensorFlow Lite Task Vision
- **Model**: YOLOv8n (fine-tuned, TFLite Int8)

## Requirements

- Android SDK 24+ (Android 7.0 Nougat)
- Camera permission

## Build & Run

```bash
# Clone the repository
git clone https://github.com/amigo-inovasi/fishgo.git

# Build and install
cd fishgo
./gradlew installDebug
```

## Project Structure

```
fishgo/
├── app/
│   ├── src/main/
│   │   ├── assets/
│   │   │   ├── fish_model.tflite    # Trained model
│   │   │   └── labels.txt           # Class labels
│   │   ├── java/com/amigoinovasi/fishgo/
│   │   │   ├── MainActivity.kt
│   │   │   ├── detection/
│   │   │   │   ├── FishDetector.kt
│   │   │   │   └── DetectionResult.kt
│   │   │   └── ui/
│   │   │       └── DetectionOverlayView.kt
│   │   └── res/
│   └── build.gradle.kts
├── gradle/
└── build.gradle.kts
```

## Integration with Navigo

The detection module (`com.amigoinovasi.fishgo.detection`) is designed for easy integration:

1. Copy the `detection/` package to Navigo
2. Add the TFLite model to assets
3. Use `FishDetector` class for inference

## License

Private - Amigo Inovasi Digital

## Contact

- Organization: Amigo Inovasi Digital
- Project: Navigo / FishGo
