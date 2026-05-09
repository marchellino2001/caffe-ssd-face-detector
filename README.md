# 😷 Face Mask Detection — CL-SSDXcept
### CSE 429: Computer Vision and Pattern Recognition | E-JUST | Spring 2026

A real-time face mask detection system built with **Xception transfer learning** and the **Caffe SSD face detector**, based on the paper:
> *"CL-SSDXcept: A Novel Face Mask Detection System"* — PMC9878194

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Real-Time Detection](#real-time-detection)
- [Installation](#installation)
- [Team](#team)

---

## Project Overview

The system detects in real-time whether a person is wearing a face mask using a two-stage pipeline:

1. **Face Detection** — A pre-trained Caffe SSD (Single Shot Detector) locates all faces in the webcam frame
2. **Mask Classification** — Each detected face is cropped and passed to a fine-tuned Xception model that classifies it as **With Mask** or **Without Mask**

---

## Results

We trained two versions of the model and compared their performance:

| Model | Strategy | Hardware | Speed | Val Acc | Val Loss | Test Acc |
|---|---|---|---|---|---|---|
| `mask_detector.h5` | Frozen Xception base (CPU) | Windows CPU | ~3 sec/step | 66% | 0.8388 | ~66% |
| `mask_detector_gpu.h5` ✅ | Fine-tuned top 20 layers | Ubuntu RTX 3060 | **~697ms/step** | **~99%** | **0.0554** | **99.0%** |

**The fine-tuned GPU model (`mask_detector_gpu.h5`) is the final submitted model.**

> 📥 **Download pre-trained models:** [Google Drive](https://drive.google.com/drive/folders/1WMSSbDawwZN4mYCB4DPD_kKhb02QSdZa?usp=sharing)
> *(includes `mask_detector_gpu.h5` and `res10_300x300_ssd_iter_140000.caffemodel`)*

### Per-Class Metrics (Fine-Tuned Model — Test Set, 300 images)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| With Mask | 99.3% | 98.7% | 99.0% | 150 |
| Without Mask | 98.7% | 99.3% | 99.0% | 150 |
| **Overall** | **99.0%** | **99.0%** | **99.0%** | **300** |

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

| | Predicted: With Mask | Predicted: Without Mask |
|---|---|---|
| **Actual: With Mask** | ✅ 148 | ❌ 2 |
| **Actual: Without Mask** | ❌ 1 | ✅ 149 |

Only **3 misclassifications out of 300 test images.**

### Training Curves

![Training Curves](results/training_curves.png)

- Training accuracy reached **~100%** by epoch 3 and remained stable throughout
- Validation accuracy started at **98%** and steadily climbed to **~99%** by epoch 20
- Both losses dropped sharply in the first 3 epochs — train loss converged near **0.00**, validation loss near **0.02**
- No signs of overfitting: validation loss kept decreasing alongside training loss

---

## Project Structure

```
Face-Mask-Detection-Project/
│
├── data/
│   └── README.md                   # Instructions: place Cleaned_Data/ here
│
├── models/
│   ├── mask_detector_gpu.h5        # ⬇️ Download from Google Drive
│   ├── deploy.prototxt             # Caffe SSD architecture config
│   └── res10_300x300_ssd_iter_140000.caffemodel  # ⬇️ Download from Google Drive
│
├── results/
│   ├── confusion_matrix.png        # Test set confusion matrix
│   ├── training_curves.png         # Accuracy & loss over 20 epochs
│   └── classification_report.txt  # Precision, Recall, F1 per class
│
├── scripts/
│   ├── preprocess.py               # Person B — CLAHE + augmentation pipeline
│   ├── train_gpu.py                # Person A — Fine-tuned GPU training script
│   ├── evaluate.py                 # Person A — Confusion matrix & metrics
│   └── detect_faces.py             # Person C — Real-time webcam inference
│
└── README.md
```

---

## Dataset & Preprocessing

**Dataset:** [Real-World Masked Face Dataset (RFMD)](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)

**Split:**

| Split | With Mask | Without Mask | Total |
|---|---|---|---|
| Train | 702 | 700 | 1,402 |
| Validation | 150 | 150 | 300 |
| Test | 150 | 150 | 300 |
| **Total** | **1,002** | **1,000** | **2,002** |

### CLAHE Preprocessing (Paper Section 3.1)

Each image was enhanced using **Contrast-Limited Adaptive Histogram Equalization (CLAHE)** before training. This improves local contrast, which is especially important for faces partially obscured by masks.

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```

The process: convert BGR → LAB color space → apply CLAHE to the **L (luminance) channel only** → convert back to BGR. This boosts contrast without distorting colors.

All images were resized to **299×299** to match Xception's required input size.

### Data Augmentation (Training Set Only)

| Augmentation | Value |
|---|---|
| Rotation | ±30° |
| Width / Height Shift | 20% |
| Shear | 20% |
| Zoom | 20% |
| Horizontal Flip | Yes |
| Brightness | 0.8 – 1.2× |

To reproduce preprocessing:
```bash
python scripts/preprocess.py
```

---

## Model Architecture

We use **Xception** (Extreme Inception) with ImageNet pre-trained weights as the base model, following the paper's CL-SSDXcept approach.

```
Input (299×299×3)
      ↓
Xception Base (ImageNet weights, top 20 layers unfrozen)
      ↓
AveragePooling2D (7×7)
      ↓
Flatten
      ↓
Dense(256, ReLU)
      ↓
Dropout(0.5)
      ↓
Dense(2, Softmax)  →  [With Mask, Without Mask]
```

**Why Xception?** Xception uses depthwise separable convolutions which are highly efficient and achieve strong performance on image classification — making it well-suited for fine-grained mask detection.

---

## Training

### Stage 1 — Base Model (Frozen Transfer Learning)

All Xception layers frozen. Only the new classification head is trained.

- Optimizer: Adam (lr = 1e-4)
- Loss: Categorical Crossentropy
- Hardware: Windows CPU (~3 sec/step)
- Epochs: up to 50 with EarlyStopping (patience = 5)
- Val Accuracy: ~66% | Val Loss: 0.8388
- Result: model struggled on CPU — frozen base insufficient for this task
- Saved as: `mask_detector.h5`

### Stage 2 — Fine-Tuned Model (GPU — Final Model ✅)

The top 20 Xception layers were unfrozen and retrained, allowing the model to adapt deeper feature representations to our specific mask detection task.

```bash
python scripts/train_gpu.py \
  --train_dir data/Cleaned_Data/Train \
  --val_dir data/Cleaned_Data/Validation
```

- Hardware: Ubuntu + NVIDIA RTX 3060 (~697ms/step)
- Optimizer: Adam (lr = 1e-4)
- Unfrozen layers: last 20 of Xception base (26 trainable layers total)
- Epochs: 20
- Val Accuracy: ~99% | Val Loss: 0.0554 | Test Accuracy: **99.0%**
- Saved as: `mask_detector_gpu.h5` ✅

---

## Evaluation

Run the evaluation script on the test set to regenerate all metrics:

```bash
python scripts/evaluate.py \
  --model models/mask_detector_gpu.h5 \
  --test_dir data/Cleaned_Data/Test
```

**Outputs** (saved to `results/`):
- `confusion_matrix.png` — True vs Predicted labels heatmap
- `training_curves.png` — Accuracy & loss over all 20 epochs
- `classification_report.txt` — Precision, Recall, F1-score per class

---

## Real-Time Detection

Ensure you have downloaded the model files from the [Google Drive link](https://drive.google.com/drive/folders/1WMSSbDawwZN4mYCB4DPD_kKhb02QSdZa?usp=sharing) and placed them in `models/`.

```bash
python scripts/detect_faces.py
```

**How it works:**
1. The Caffe SSD detects all faces in each webcam frame (confidence threshold: 0.5)
2. Each detected face ROI is cropped, resized to 299×299, and normalized
3. The Xception model predicts mask / no-mask
4. **Green box** = Mask ✅ | **Red box** = No Mask ❌
5. Press `Q` to quit

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Face-Mask-Detection-Project.git
cd Face-Mask-Detection-Project

# Install dependencies
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn

# Download large model files from Google Drive (link above)
# Place them inside models/
```

**Requirements:**
- Python 3.8+
- TensorFlow 2.x
- OpenCV (`opencv-python`)
- NumPy, Matplotlib, Seaborn, Scikit-learn

---

## Team

| Role | Person | Responsibilities |
|---|---|---|
| **ML Engineer** | Person A | Xception architecture, transfer learning, GPU fine-tuning, training loop, evaluation metrics |
| **Data Architect** | Person B | Dataset download, CLAHE preprocessing, data augmentation, train/val/test split |
| **Systems & Lead** | Person C | Caffe SSD face detector, real-time webcam integration, GitHub setup, report webpage |

**Course:** CSE 429 — Computer Vision and Pattern Recognition
**Instructor:** Dr. Ahmed Gomaa
**Institution:** Egypt-Japan University of Science and Technology (E-JUST)
**Date:** Spring 2026
