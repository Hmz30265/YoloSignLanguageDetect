# Real-Time ASL Translation Using YOLO

[![Conference](https://img.shields.io/badge/ICLR-2025-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

This project implements a **real-time American Sign Language (ASL) alphabet translation system** using the YOLO (You Only Look Once) object detection framework. Designed with accessibility in mind, this tool aims to bridge communication gaps for the deaf and hard-of-hearing community by recognizing static ASL letters from images or video streams.

## 🧠 Motivation

Millions of people rely on sign language as their primary mode of communication. However, the scarcity of real-time translation tools limits inclusion in everyday tasks such as ordering food or attending a doctor's appointment. This project leverages state-of-the-art object detection techniques to accurately recognize ASL letters, laying the groundwork for broader gesture-based translation systems.

## 🎯 Project Goals

- ✅ Accurate detection of all 26 ASL alphabet hand signs.
- 🌐 Robust performance across lighting, background, and skin tone variations.
- 🎥 Real-time letter sequence recognition from webcam streams.
- 🚀 Establish a foundation for future dynamic gesture and sentence-level translation.

## 📁 Dataset

- **Primary**: Kaggle ASL Alphabet YOLO-formatted Dataset  
  [Dataset Link](https://www.kaggle.com/datasets/daskoushik/sign-language-dataset-for-yolov7)
- **Additional**: Ayuraj's ASL Alphabet & Digits Dataset  
  [Dataset Link](https://www.kaggle.com/datasets/ayuraj/asl-dataset)

## 🛠️ Model Overview

### YOLOv8

- Framework: Ultralytics YOLOv8
- Image Size: `640x640`
- Augmentations:
  - Color jitter
  - Rotation, scaling, translation
  - Mosaic and Mixup
- Best Model Accuracy: **mAP@0.5 = 0.971**, **Precision = 0.917**, **Recall = 0.908**

### YOLOv10

- Variants: Nano (n), Small (s), Medium (m)
- Augmented and non-augmented variants tested
- Best Result (YOLOv10s Augmented):
  - **Precision = 0.920**, **mAP@0.5 = 0.924**

## 📊 Performance Summary

| Model         | Augmented | Precision | Recall | F1-score | mAP@0.5 |
|---------------|-----------|-----------|--------|----------|----------|
| YOLOv8        | No        | 0.879     | 0.879  | 0.879    | 0.943    |
| YOLOv8        | Yes       | 0.917     | 0.908  | 0.912    | 0.971    |
| YOLOv10s      | Yes       | 0.920     | 0.865  | -        | 0.924    |

## 🎥 Live Detection Demo

Real-time predictions from webcam input:

- **Single-letter Recognition**: 3-second window per letter, majority vote used.
- **Word Spelling**: Successfully spelled words like `CAT` and `HELLO`.

| Word   | Prediction | Outcome |
|--------|------------|---------|
| CAT    | C–A–T      | ✅      |
| HELLO  | H–E–L–L–O  | ✅      |
| WORLD  | W–O–P–L–K  | ❌      |

## ⚠️ Known Challenges

- **Similar Gestures**: Difficulties distinguishing M/N, P/Q.
- **Lighting**: Indoor conditions may cause classification errors.
- **Fixed Window**: May miss transitions between letters.

## 🔮 Future Work

- Integrate hand-pose keypoint detection to resolve subtle gesture differences.
- Use dynamic frame buffering instead of a fixed voting window.
- Improve segmentation and class isolation for hybrid datasets.
- Expand to dynamic ASL gestures and sentence-level translation.


