# Facial Recognition & Classification Pipeline - Refactored

An end-to-end face recognition pipeline that detects faces from raw images, extracts deep features using a pretrained **InceptionResNetV2** CNN, and classifies identities using classical machine learning models. Originally built as a university project (CSE422 - Artificial Intelligence), this is the **refactored version** with significantly improved performance.

> **Previous version:** [face-recognition-pipeline](https://github.com/MufradMahmud19/face-recognition-pipeline) where raw pixel + wavelet features were used.
>
> **This version:** Uses deep CNN embeddings + data augmentation.

---

## What Changed From the Original

The original pipeline fed **16,384 raw pixel values** (64×64 RGB + Haar Wavelet) directly into classifiers. Raw pixels shift with every change in lighting, pose, or expression, making it impossible for classical ML models to generalize well.

This refactored version fixes that with:

| Improvement | Before | After |
|---|---|---|
| **Feature Extraction** | Raw pixels + wavelet (16,384-D) | InceptionResNetV2 deep embeddings (1,536-D) |
| **Data Augmentation** | None | Horizontal flips, brightness jitter (±30), slight rotation |
| **Train/Test Split** | Random | Stratified (balanced classes) |
| **Preprocessing** | Missing in some models | StandardScaler + PCA in every model |
| **Hyperparameter Tuning** | Defaults only | GridSearchCV on SVM, Random Forest |
| **MLP Architecture** | (512, 128) | (512, 256, 128) with L2 regularization & adaptive LR |

---

## Model Performance

### Training Accuracy

| # | Model | Accuracy |
|---|---|---|
| 1 | SVM (Linear) | 100% |
| 2 | SVM (RBF) | 100% |
| 3 | SVC (Sigmoid) | 80% |
| 4 | Random Forest | 100% |
| 5 | Naive Bayes | 39% |
| 6 | MLP Neural Network (ReLU) | 99% |
| 7 | Ensemble (Soft Voting) | 100% |

### Testing Accuracy

| # | Model | Accuracy |
|---|---|---|
| 1 | SVM (Linear) | 90% |
| 2 | **SVM (RBF)** | **94%** ⭐ |
| 3 | SVC (Sigmoid) | 72% |
| 4 | Random Forest | 89% |
| 5 | Naive Bayes | 39% |
| 6 | MLP Neural Network (ReLU) | 93% |
| 7 | **Ensemble (Soft Voting)** | **94%** ⭐ |

> After testing with 10 random group photos of celebrities from the dataset, the most accurate real-world predictions were from **Random Forest** and **Ensemble**.

---

## Features

- **Automated Face & Eye Detection:** Uses OpenCV's Haar Cascade classifiers to identify and crop faces only when eyes are clearly visible, ensuring high-quality training data.
- **Deep Feature Extraction:** Extracts 1,536-D feature vectors from a pretrained InceptionResNetV2 CNN (transfer learning from ImageNet) — far more robust than raw pixel features.
- **Data Augmentation:** Multiplies training data ~4× with horizontal flips, brightness adjustments, and slight rotations to improve generalization.
- **Model Benchmarking:** Compares 7 scikit-learn classifiers with proper preprocessing pipelines (StandardScaler + PCA) and GridSearchCV hyperparameter tuning.
- **Group Photo Prediction:** Detects and identifies multiple faces in a single group photograph.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras (InceptionResNetV2) |
| Computer Vision | OpenCV (`cv2`) |
| Machine Learning | Scikit-Learn |
| Signal Processing | PyWavelets (`pywt`) |
| Data Manipulation | NumPy |
| Visualization | Matplotlib |

---

## Project Structure

```text
📁 face-recognition-pipeline-refactored/
│
├── 1_prepare_data.py          # Face cropping + CNN feature extraction + augmentation
├── 2a_train_svm_linear.py     # Linear SVM with GridSearchCV
├── 2b_train_svm_rbf.py        # RBF SVM with GridSearchCV
├── 2c_train_svc_sigmoid.py    # Sigmoid SVM with GridSearchCV
├── 2d_train_random_forest.py  # Random Forest with GridSearchCV
├── 2e_train_naive_bayes.py    # Gaussian Naive Bayes
├── 2f_train_mlp.py            # MLP Neural Network
├── 2g_train_ensemble.py       # Soft Voting Ensemble
├── 3_predict_group.py         # Group photo face recognition
├── requirements.txt           # Python dependencies
├── main.py                    # Legacy monolithic script
└── proj.ipynb                 # Legacy notebook
```

---

## Dataset Structure

The pipeline expects a `Dataset/` folder containing subfolders for each individual. The preprocessing script automatically creates a `cropped/` subdirectory with isolated face data.

```text
📁 Dataset/
│
├── 📁 Person_A/          (contains .jpg, .png, .jpeg)
├── 📁 Person_B/
├── ...
│
└── 📁 cropped/           (Generated automatically by 1_prepare_data.py)
    ├── 📁 Person_A/
    └── 📁 Person_B/
```

> **Note:** The dataset is not included in this repository due to size. You can use any face dataset organized in the folder structure above.

---

## Setup & Installation

1. **Clone the repository:**

```bash
git clone https://github.com/MufradMahmud19/face-recognition-pipeline-refactored.git
cd face-recognition-pipeline-refactored
```

2. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

3. **Prepare your dataset:** Place your image folders inside a `Dataset/` directory and update the paths in `1_prepare_data.py` if needed.

---

## Usage

Run the scripts **in order**. Step 1 only needs to be run once — the extracted features are saved to disk.

### Step 1: Data Preparation (run once)

```bash
python 1_prepare_data.py
```

This will:
- Detect and crop faces from all images
- Extract 1,536-D deep features using InceptionResNetV2
- Generate augmented samples (flips, brightness, rotation)
- Save `artifacts/data_splits.pkl` and `artifacts/student_encoded.pkl`

> ⏱️ Takes ~15–30 minutes depending on dataset size and hardware.

### Step 2: Train Models

Run any or all of the training scripts:

```bash
python 2a_train_svm_linear.py
python 2b_train_svm_rbf.py
python 2c_train_svc_sigmoid.py
python 2d_train_random_forest.py
python 2e_train_naive_bayes.py
python 2f_train_mlp.py
python 2g_train_ensemble.py
```

Trained models are saved to the `models/` directory.

### Step 3: Predict on Group Photos

```bash
python 3_predict_group.py
```

Detects faces in a group photo and identifies each person using all 7 models.

---

## How It Works

```
Raw Image → Haar Cascade (Face + Eye Detection) → Cropped Face
    ↓
InceptionResNetV2 (pretrained, frozen) → 1,536-D Feature Vector
    ↓
Data Augmentation (flip, brightness, rotation) → 4× more samples
    ↓
StandardScaler → PCA (95% variance) → Classifier (SVM / RF / MLP / Ensemble)
    ↓
Predicted Identity
```

---
