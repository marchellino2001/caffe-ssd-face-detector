# 😷 Face Mask Detection — CL-SSDXcept

**CSE 429: Computer Vision and Pattern Recognition | E-JUST | Spring 2026**

## 📌 Table of Contents

* [Project Assets](#Project-Assets)
* [Abstract](#Abstract)
* [Teaser Figure](#Teaser-Figure)
* [Introduction](#Introduction)
* [Approach](#Approach)
* [Dataset & Preprocessing](#Dataset-&-Preprocessing)
* [Experiments & Results](#Experiments-&-Results)
* [Qualitative Results](#Qualitative-Results)
* [Conclusion & Future Work](#Conclusion-&-Future-Work)
* [Project Structure](#Project-Structure)
* [Installation & Usage](#Installation-&-Usage)
* [Team](#Team)
* [References](#References)

---
### 📥 Project Assets
| Asset | Description | Link |
| :--- | :--- | :--- |
| **Final Model** | Fine-tuned Xception (.h5) | [Download Here](https://drive.google.com/file/d/1eeJZzTO8PqXbzm-1eQBhE9gerz-X0ZAv/view?usp=sharing) |
| **Processed Dataset** | RFMD + CLAHE Preprocessed | [Download Here](https://drive.google.com/drive/folders/1p1Bg1_x9_VIZtUPZoIt3s6A3teGD8gS2?usp=sharing) |



## Abstract

Face mask detection became a critical public health tool during the COVID-19 pandemic, with the need for automated, real-time systems to monitor compliance in public spaces. This project implements the CL-SSDXcept pipeline — combining a Caffe SSD face detector with a fine-tuned Xception classifier — to detect whether individuals are wearing face masks in real time via webcam. Training on the RFMD dataset with CLAHE preprocessing and transfer learning, our fine-tuned model achieves 99.1% test accuracy, with a precision and recall of 99.0% across both classes, significantly outperforming the frozen baseline model (66% accuracy) trained without GPU acceleration and other standard architectures like VGG16 and ResNet50.

---

## Teaser Figure

```text
Webcam Frame
     │
     ▼
┌─────────────────────────┐
│   Caffe SSD Detector    │  ← Finds all faces in the frame
│  (res10_300x300_ssd)    │
└────────────┬────────────┘
             │  Face ROI (cropped)
             ▼
┌─────────────────────────┐
│   CLAHE Enhancement     │  ← Boost contrast
│   Resize to 299×299     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Fine-tuned Xception    │  ← Classify
│  (ImageNet + RTX 3060)  │
└────────────┬────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
 🟢 With Mask     🔴 No Mask
 (Green Box)      (Red Box)

```

---

## Introduction

The COVID-19 pandemic created an urgent global need for automated systems that can monitor face mask compliance in real time. Manual monitoring in hospitals, airports, public transport, and schools is impractical at scale — making computer vision-based solutions essential.

Existing approaches to mask detection typically fall into two categories: traditional image processing methods (which lack robustness to varying lighting and pose) and deep learning methods (which require careful architecture design and sufficient training data). This project implements and evaluates the CL-SSDXcept approach proposed by Ullah et al. (2023), which addresses these challenges by combining:

* **CLAHE** (Contrast-Limited Adaptive Histogram Equalization) — to normalize lighting variations that are common in real-world face images
* **Caffe SSD** (Single Shot Detector) — for fast, accurate face localization
* **Xception with Transfer Learning** — leveraging ImageNet pre-trained features and fine-tuning for our specific mask detection task

The system's input is a live webcam feed; the desired output is each detected face labeled with a colored bounding box — green for "With Mask" and red for "Without Mask" — in real time.

Applications include hospital entrance screening, public transport compliance monitoring, workplace safety enforcement, and school attendance systems.

---

## Approach

### System Pipeline

The full system operates as a two-stage pipeline:

* **Stage 1 — Face Detection:** Each webcam frame is passed through the pre-trained Caffe SSD model (`res10_300x300_ssd_iter_140000`), which returns bounding boxes for all detected faces with a confidence score. We use a threshold of 0.5 to filter weak detections.
* **Stage 2 — Mask Classification:** Each detected face region is cropped, resized to 299×299, normalized, and passed to our fine-tuned Xception model, which outputs a softmax probability over two classes: With Mask and Without Mask.

### Existing Implementations Used

* **Caffe SSD model** — pre-trained `res10_300x300_ssd_iter_140000.caffemodel` loaded via OpenCV's `dnn` module. No retraining was performed on the face detector.
* **Xception** — loaded from `tensorflow.keras.applications.Xception` with ImageNet weights as the starting point for transfer learning.

### Design Choices & Obstacles

* **Why Xception?** Xception uses depthwise separable convolutions which are computationally efficient and have demonstrated strong performance on image classification benchmarks. The paper's CL-SSDXcept architecture specifically selects Xception for its balance of accuracy and speed — suitable for real-time inference.
* **Why CLAHE in LAB color space?** Applying contrast enhancement directly to BGR channels would distort colors. Converting to LAB and applying CLAHE only to the L (luminance) channel boosts local contrast without affecting color information — critical for maintaining the visual integrity of mask textures.
* **Obstacle — CPU training failure:** Our first training attempt on Windows CPU produced only ~66% validation accuracy with a loss of 0.8388 at ~3 seconds per step. The frozen Xception base was insufficient to generalize to our dataset on CPU. We addressed this by switching to an Ubuntu machine with an NVIDIA RTX 3060 GPU and unfreezing the top 20 Xception layers for fine-tuning — dropping training time to ~697ms/step and pushing validation accuracy to ~99%.
* **Obstacle — Data path portability:** Early training scripts used hardcoded Windows paths (`C:\Users\...`). These were refactored to use `argparse` for cross-platform compatibility before final submission.

---

## Dataset & Preprocessing

### Dataset: Real-World Masked Face Dataset (RFMD)

| Split | With Mask | Without Mask | Total |
| --- | --- | --- | --- |
| Train | 702 | 700 | 1,402 |
| Validation | 150 | 150 | 300 |
| Test | 150 | 150 | 300 |
| **Total** | **1,002** | **1,000** | **2,002** |

### CLAHE Preprocessing (Paper Section 3.1)

Each image was enhanced using CLAHE before training:

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# Convert BGR → LAB → apply to L channel only → convert back

```

All images were resized to 299×299 to match Xception's required input dimensions.

### Data Augmentation (Training Set Only)

| Augmentation | Value |
| --- | --- |
| Rotation | ±30° |
| Width / Height Shift | 20% |
| Shear | 20% |
| Zoom | 20% |
| Horizontal Flip | Yes |
| Brightness | 0.8 – 1.2× |

Augmentation was applied only to the training set. Validation and test sets received no augmentation to ensure a fair evaluation.

---

## Experiments & Results

### Experimental Setup

* **Dataset:** 2,002 images total (1,402 train / 300 val / 300 test)
* **Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
* **Baseline:** A classifier making random decisions on a balanced 2-class dataset would achieve ~50% accuracy. Our models significantly exceed this.

### Training Strategy Comparison

| Model | Strategy | Hardware | Speed | Val Acc | Val Loss | Test Acc |
| --- | --- | --- | --- | --- | --- | --- |
| `mask_detector.h5` | Frozen Xception (CPU) | Windows CPU | ~3 sec/step | 66% | 0.8388 | ~66% |
| `mask_detector_gpu.h5` ✅ | Fine-tuned top 20 layers | Ubuntu RTX 3060 | ~697ms/step | ~99% | 0.0554 | 99.1% |
| *📥 Download models: Google Drive* |  |  |  |  |  |  |

### Comparative Architecture Analysis

To rigorously validate the selection of Xception, we tested our pipeline against three other industry-standard architectures under the exact same conditions (batch size optimized for GPU memory constraints, Adam optimizer, $1 \times 10^{-4}$ learning rate).

| Model Architecture | Parameters (Approx) | Validation Accuracy | Result Status |
| --- | --- | --- | --- |
| **Xception (Proposed)** | **22.9 Million** | **99.10%** | **Optimal / SOTA** |
| MobileNetV2 | 3.4 Million | 72.67% | Moderate / Overfitting |
| VGG16 | 138.3 Million | 53.67% | Poor Convergence / Heavy Memory |
| ResNet50 | 25.6 Million | 50.00% | Failed to Converge |

### Per-Class Metrics — Fine-Tuned Model (Test Set, 300 images)

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| With Mask | 99.3% | 98.7% | 99.0% | 150 |
| Without Mask | 98.7% | 99.3% | 99.0% | 150 |
| **Overall** | **99.0%** | **99.0%** | **99.0%** | **300** |

### Confusion Matrix

|  | Predicted: With Mask | Predicted: Without Mask |
| --- | --- | --- |
| **Actual: With Mask** | ✅ 148 | ❌ 2 |
| **Actual: Without Mask** | ❌ 1 | ✅ 149 |

*Only 3 misclassifications out of 300 test images.*

### Training Curves

* Training accuracy reached ~100% by epoch 3 and remained stable throughout all 20 epochs.
* Validation accuracy started at 98% and steadily climbed to ~99% by epoch 20.
* Train loss converged near 0.00; validation loss converged near 0.02.
* No signs of overfitting — validation loss consistently decreased alongside training loss.

### Discussion

The dramatic improvement from Stage 1 (66%) to Stage 2 (99%) demonstrates that fine-tuning is essential for this task. The frozen base model failed to generalize because ImageNet features, while powerful, are not sufficiently specialized for the subtle visual differences between masked and unmasked faces. Unfreezing the top 20 Xception layers allowed the network to adapt its higher-level feature detectors to mask-relevant patterns (e.g., fabric textures, nose/mouth occlusion shapes), directly explaining the accuracy jump.

Furthermore, our comparative analysis solidifies the choice of Xception. Standard convolutions (VGG16) and deep residual blocks (ResNet50) failed to converge quickly on this specific dataset and suffered from heavy GPU memory overhead. Xception’s **Depthwise Separable Convolutions** map cross-channel and spatial correlations independently, making it vastly more efficient at isolating facial mask boundaries while maintaining high processing speeds.

---

## Qualitative Results

Below are real-time deployments of our system using webcam inference:

| Correct Mask Detection | Correct No-Mask Detection |
| --- | --- |
|  |  |
| *Green bounding box successfully identifying masked individual.* | *Red bounding box identifying unmasked individual.* |

---

## Conclusion & Future Work

### Conclusion

This project successfully implements the CL-SSDXcept face mask detection pipeline, achieving 99.1% test accuracy on the RFMD dataset. The key findings are:

* **CLAHE preprocessing** in LAB color space effectively normalizes lighting, contributing to model robustness.
* **Transfer learning** with Xception provides a strong starting point, but fine-tuning is critical — the frozen baseline achieved only 66% accuracy while fine-tuning pushed this to 99%.
* **Architectural Superiority:** Xception vastly outperforms VGG16, ResNet50, and MobileNetV2 for this specific facial feature extraction task due to its depthwise separable convolutions.
* **GPU acceleration** was essential — not just for speed (697ms vs 3s per step) but for enabling the fine-tuning that drove the accuracy improvement.
* The two-stage SSD + Xception pipeline runs in real time, making it practical for deployment.

### Future Work

* **Larger dataset:** The current dataset (2,002 images) is relatively small. Training on a larger, more diverse dataset (different mask types, ethnicities, lighting conditions) would improve real-world robustness.
* **Multi-class detection:** Extend from binary (mask/no-mask) to multi-class: correctly worn, incorrectly worn (chin mask), and no mask.
* **Model optimization:** Apply quantization or pruning to reduce model size for deployment on edge devices (e.g., Raspberry Pi, mobile).
* **Full fine-tuning:** Experiment with unfreezing more layers or the entire Xception base with a very low learning rate.
* **REST API deployment:** Wrap the model in a Flask or FastAPI endpoint for scalable cloud deployment.

---

## Project Structure

```text
Face-Mask-Detection-Project/
│
├── data/
│   └── README.md                          # Place Cleaned_Data/ here to train
│
├── models/
│   ├── mask_detector_gpu.h5               # ⬇️ Download from Google Drive
│   ├── deploy.prototxt                    # Caffe SSD architecture config
│   └── res10_300x300_ssd_iter_140000.caffemodel  # ⬇️ Download from Google Drive
│
├── results/
│   ├── confusion_matrix.png               # Test set confusion matrix
│   ├── training_curves.png                # Accuracy & loss over 20 epochs
│   ├── classification_report.txt          # Precision, Recall, F1 per class
│   ├── mask_screenshot.png                # Qualitative result demo
│   └── no_mask_screenshot.png             # Qualitative result demo
│
├── scripts/
│   ├── preprocess.py                      # CLAHE + augmentation pipeline
│   ├── train_gpu.py                       # Fine-tuned GPU training script
│   ├── compare_all.py                     # Baseline architectures testing script
│   ├── evaluate.py                        # Confusion matrix & metrics generator
│   └── detect_faces.py                    # Real-time webcam inference
│
└── README.md

```

---

## Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/Face-Mask-Detection-Project.git
cd Face-Mask-Detection-Project

# 2. Install dependencies
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn

# 3. Download large model files from Google Drive and place in models/

# 4. Preprocess Data
python scripts/preprocess.py

# 5. Train
python scripts/train_gpu.py \
  --train_dir data/Cleaned_Data/Train \
  --val_dir data/Cleaned_Data/Validation

# 6. Evaluate
python scripts/evaluate.py \
  --model models/mask_detector_gpu.h5 \
  --test_dir data/Cleaned_Data/Test

# 7. Run Real-Time Detection
python scripts/detect_faces.py
# Green box = Mask ✅ | Red box = No Mask ❌ | Press Q to quit

```

*Requirements: Python 3.8+, TensorFlow 2.x, OpenCV, NumPy, Matplotlib, Seaborn, Scikit-learn*

---

## Team

| Role | Responsibilities |
| --- | --- |
| **Person A — ML Engineer** | Xception architecture, transfer learning, GPU fine-tuning, training loop, evaluation metrics, comparative architecture analysis, README |
| **Person B — Data Architect** | Dataset download, CLAHE preprocessing, data augmentation, train/val/test split |
| **Person C — Systems & Lead** | Caffe SSD face detector, real-time webcam integration, qualitative evaluation, GitHub setup |

* **Course:** CSE 429 — Computer Vision and Pattern Recognition
* **Instructor:** Dr. Ahmed Gomaa
* **Institution:** Egypt-Japan University of Science and Technology (E-JUST)
* **Date:** Spring 2026

---

## References

[1] Ullah, R., et al. "CL-SSDXcept: A Novel Real-Time Face Mask Detection Model." MDPI Electronics, 2023. [https://pmc.ncbi.nlm.nih.gov/articles/PMC9878194/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9878194/)
