# 🖼️ Intel Image Classification - CNN Model

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-81.73%25-brightgreen.svg)]()

## 📋 Project Overview

This project implements a **Convolutional Neural Network (CNN)** from scratch to classify natural scene images into 6 categories. The model achieves **81.73% test accuracy** on the Intel Image Classification dataset.

### 🎯 Objective
Build and evaluate a CNN model for multi-class image classification with data preprocessing, augmentation, and comprehensive performance analysis.

### 📊 Dataset Classes
| Class | Label | Images (Train) |
|-------|-------|----------------|
| 🏢 Buildings | 0 | 2,191 |
| 🌲 Forest | 1 | 2,271 |
| 🏔️ Glacier | 2 | 2,403 |
| ⛰️ Mountain | 3 | 2,502 |
| 🌊 Sea | 4 | 2,273 |
| 🛣️ Street | 5 | 2,386 |

## 🏆 Results Summary

### Test Performance
- **Accuracy:** 81.73%
- **Loss:** 0.5505
- **Macro F1-Score:** 0.82

### Per-Class Performance Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Buildings | 0.91 | 0.75 | 0.82 | 437 |
| Forest | 0.74 | 0.99 | 0.85 | 474 |
| Glacier | 0.87 | 0.74 | 0.80 | 553 |
| Mountain | 0.75 | 0.83 | 0.79 | 525 |
| Sea | 0.84 | 0.87 | 0.85 | 510 |
| Street | 0.86 | 0.72 | 0.78 | 501 |

### Key Insights
- ✅ **Best Performance:** Forest (99% recall) and Sea (87% recall)
- ⚠️ **Challenging Pairs:** Glacier ↔ Mountain (visually similar)
- 📈 **Balanced Performance:** All classes achieve F1 > 0.78

## 🏗️ Model Architecture

```
Input Layer
(150, 150, 3) - RGB Image
        │
        ▼
────────────────────────────────────────────
Conv Block 1
- Conv2D (32, 3×3) + ReLU + BatchNorm
- Conv2D (32, 3×3) + ReLU + BatchNorm
- MaxPooling2D (2×2)
- Dropout (0.25)
────────────────────────────────────────────
        │
        ▼
────────────────────────────────────────────
Conv Block 2
- Conv2D (64, 3×3) + ReLU + BatchNorm
- Conv2D (64, 3×3) + ReLU + BatchNorm
- MaxPooling2D (2×2)
- Dropout (0.25)
────────────────────────────────────────────
        │
        ▼
────────────────────────────────────────────
Conv Block 3
- Conv2D (128, 3×3) + ReLU + BatchNorm
- Conv2D (128, 3×3) + ReLU + BatchNorm
- MaxPooling2D (2×2)
- Dropout (0.25)
────────────────────────────────────────────
        │
        ▼
────────────────────────────────────────────
Fully Connected Layers
- Flatten
- Dense (512) + ReLU + BatchNorm
- Dropout (0.5)
- Dense (6) + Softmax
────────────────────────────────────────────
```

### 📌 Output Classes

* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

### ⚡ Highlights

* Increasing filters: **32 → 64 → 128** for deeper feature extraction
* **Batch Normalization** improves training stability
* **Dropout (0.25 / 0.5)** prevents overfitting
* **Softmax layer** outputs class probabilities

### Model Statistics
- **Total Parameters:** ~1.2 million
- **Trainable Parameters:** ~1.2 million
- **Activation Functions:** ReLU (hidden), Softmax (output)

## 🔧 Data Preprocessing & Augmentation

### Preprocessing Steps
- ✅ Resize all images to **150×150 pixels**
- ✅ Normalize pixel values to **[0, 1] range**
- ✅ Train/Validation split: **80/20**
- ✅ Shuffle training data

### Data Augmentation (Training Only)
| Technique | Parameters |
|-----------|------------|
| Rotation Range | ±20 degrees |
| Width Shift | 20% |
| Height Shift | 20% |
| Zoom Range | 20% |
| Horizontal Flip | Yes |
| Fill Mode | Nearest |

## 📊 Final Results Summary
TEST ACCURACY: 81.73%

Per-Class Performance: 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
Class        Precision    Recall      F1-Score 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
Buildings    0.91         0.75        0.82 
Forest       0.74         0.99        0.85 
Glacier      0.87         0.74        0.80 
Mountain     0.75         0.83        0.79 
Sea          0.84         0.87        0.85 
Street       0.86         0.72        0.78 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
Macro Avg    0.83         0.82        0.82 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
