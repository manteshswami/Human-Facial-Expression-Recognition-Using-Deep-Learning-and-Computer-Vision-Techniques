# **😄 Human Facial Expression Recognition Using Deep Learning and Computer Vision Techniques**
---

## Overview

This project builds a deep learning pipeline for facial expression recognition using computer vision and transfer learning. The model is trained on the Face Emotion Dataset (EMOTIONX-09) and uses ResNet50 as a feature extractor with a custom classification head.

## Dataset

- Dataset: Face Emotion Dataset (EMOTIONX-09) v2
- Structure:
    - `train/`
    - `test/`
- Classes: multiple facial emotion labels
- Image size: 96x96 RGB

**Sample Images From Each Class -**

![Sample Images Per Class](https://raw.githubusercontent.com/manteshswami/Human-Facial-Expression-Recognition-Using-Deep-Learning-and-Computer-Vision-Techniques/main/sample%20images%20per%20class.png)

## Key Steps

1. Load and preprocess images from train/test folders
2. Apply image enhancement:
     - Gaussian blur
     - CLAHE
     - sharpening
3. Normalize image pixel values
4. One-hot encode labels
5. Augment training data using `ImageDataGenerator`
6. Build and train a ResNet50-based model
7. Fine-tune the last layers of ResNet50
8. Evaluate performance using multiple metrics

## Model

- Base model: ResNet50 (ImageNet weights, include_top=False)
- Custom head:
    - GlobalAveragePooling2D
    - Dense(256, activation='relu')
    - Dropout(0.4)
    - Dense(num_classes, activation='softmax')

## Training

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy
- Callbacks:
    - EarlyStopping
    - ReduceLROnPlateau
- Uses class weights to address class imbalance

## Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Log loss
- Confusion matrix
- ROC curves for each class

## Results

- Saved model: `ResNet50_emotion_model.keras`
- Classification report and confusion matrix generated for test set

## Usage

1. Install dependencies:
     - `tensorflow`
     - `opencv-python`
     - `numpy`
     - `pandas`
     - `matplotlib`
     - `seaborn`
     - `scikit-learn`
2. Place dataset and images under the specified `BASE` path, including the `train/` and `test/` folders
3. Run notebook cells in order
4. Use the saved model for inference or further fine-tuning

## Notes

- This notebook is designed for Kaggle or local Jupyter execution.
- Preprocessing is applied to both train and test images before model training.
- The pipeline includes data visualization, preprocessing comparisons, training, fine-tuning, and evaluation.
