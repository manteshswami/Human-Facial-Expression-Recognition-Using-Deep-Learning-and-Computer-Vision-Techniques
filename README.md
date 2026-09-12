<div align="center">

# 🧠 NeuroExpress — Advanced Facial Emotion Recognition


![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.12.1-red?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-green?style=for-the-badge&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-purple?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)

**A state-of-the-art deep learning system for real-time facial expression analysis powered by an intelligent ResNet50 + DenseNet121 soft-voting ensemble**

[🚀 Live Application](https://neuroexpression.streamlit.app/) • [📖 Documentation](#-documentation) • [🤝 Contributing](#-contributing)

</div>

---

## 🎯 Overview

NeuroExpress is a production-grade Streamlit web application that leverages cutting-edge deep learning to classify facial expressions with exceptional accuracy. The system combines two fine-tuned ImageNet models (ResNet50 and DenseNet121) through intelligent soft-voting ensemble methodology, delivering **98.46% accuracy** on a diverse test dataset of 976 samples across 9 distinct emotion classes.

Built for researchers, practitioners, and developers, NeuroExpress provides:
- ✅ **Real-time emotion detection** from uploaded images or live camera feeds
- ✅ **Robust face detection** with automatic crop and preprocessing
- ✅ **Probabilistic confidence scores** for informed decision-making
- ✅ **Mobile-responsive interface** with dark-theme optimization
- ✅ **Production-ready architecture** with fallback mechanisms

---

## ✨ Key Features

### 🎬 Multi-Input Support
- **Photo Upload**: JPG, PNG, and WebP format support
- **Live Camera**: Direct browser-based real-time capture
- **Automatic Face Crop**: OpenCV Haar Cascade detection for optimal ROI isolation
- **Flexible Processing**: Optional auto-crop with fallback to full-image analysis

#### 📷 Live Camera Instructions
For best emotion detection results using the live camera:
- **Position**: Hold camera 12-18 inches (30-45 cm) from your face
- **Framing**: Ensure your face fills most of the frame—forehead to chin should be visible
- **Lighting**: Use natural or bright indoor lighting; avoid shadows across your face
- **Clarity**: Keep the camera lens clean and the image sharp
- **Expression**: Hold your expression steady for 1-2 seconds before capture

### 🧠 Advanced Ensemble Inference
- **Dual Model Architecture**: Combines ResNet50 (94.16% accuracy) and DenseNet121 (98.57% accuracy)
- **Soft-Voting Mechanism**: Equal-weight probability averaging for robust predictions
- **Intelligent Fallback**: Automatic DenseNet121 solo mode if ResNet50 unavailable
- **Model Validation**: Input/output shape verification at startup

### 📊 Comprehensive Analytics
- **Primary Prediction**: Top emotion with confidence percentage
- **Ranked Breakdown**: Top-3 emotions with detailed scores
- **Full Distribution**: All 9 emotion probabilities available on demand
- **Inference Latency**: Real-time performance metrics
- **Visual Feedback**: Clear UI indicators for model mode (ensemble vs. fallback)

### 🎨 Professional Interface
- **Dark Theme**: Optimized for eye comfort and data visualization
- **Responsive Design**: Seamless experience across desktop and mobile devices
- **Clear Controls**: One-click analysis with repeat testing capability
- **Accessibility**: Intuitive navigation and comprehensive error handling

---

## 📋 Emotion Classes

The model recognizes nine distinct emotional states, each with a dedicated confidence metric:

| Index | Emotion | Emoji | Characteristics |
| :---: | --- | :---: | --- |
| 0 | 😠 **Angry** | 😠 | High activation, negative valence |
| 1 | 😰 **Anxiety** | 😰 | Apprehension, worry, concern |
| 2 | 😕 **Confusion** | 😕 | Uncertainty, puzzlement, bewilderment |
| 3 | 🤢 **Disgust** | 🤢 | Revulsion, contempt, disapproval |
| 4 | 😨 **Fear** | 😨 | Alarm, dread, apprehension |
| 5 | 😄 **Happy** | 😄 | Joy, contentment, satisfaction |
| 6 | 😐 **Neutral** | 😐 | No strong emotional expression |
| 7 | 😢 **Sad** | 😢 | Sorrow, melancholy, unhappiness |
| 8 | 😲 **Surprise** | 😲 | Astonishment, wonder, unexpectedness |

> **📌 Note**: The internal model class is labeled `Suprise` (legacy from training dataset) but displays as standard `Surprise` in the UI for professional presentation.

---

## 🏗️ Architecture & Technical Design

### Model Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
│                    Image (H×W×3) or Stream                           │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
         ┌───────────────┴──────────────────┐
         │                                  │
         ▼                                  ▼
┌─────────────────────────┐    ┌──────────────────────────┐
│  FACE DETECTION STAGE   │    │ SKIP (if disabled)       │
│  OpenCV Haar Cascade    │    │ Use original image       │
│  ✓ Multi-scale scan     │    │                          │
│  ✓ Confidence filter    │    └──────────┬───────────────┘
│  ✓ ROI extraction       │               │
└────────────┬────────────┘    ┌──────────┘
             │                 │
             └─────────┬───────┘
                       │
                       ▼
      ┌────────────────────────────────────┐
      │   ADAPTIVE PREPROCESSING STACK      │
      │  ┌──────────────────────────────┐  │
      │  │ 1. Resize: 96×96 RGB         │  │
      │  │    (Bilinear interpolation)  │  │
      │  └──────────────────────────────┘  │
      │  ┌──────────────────────────────┐  │
      │  │ 2. Gaussian Blur             │  │
      │  │    (σ=1.0, kernel=3×3)      │  │
      │  │    → Noise suppression       │  │
      │  └──────────────────────────────┘  │
      │  ┌──────────────────────────────┐  │
      │  │ 3. CLAHE Enhancement         │  │
      │  │    • Convert RGB→LAB         │  │
      │  │    • Adaptive histogram eq.  │  │
      │  │    • clipLimit=2.0           │  │
      │  │    • tileGridSize=8×8        │  │
      │  │    → Contrast normalization  │  │
      │  └──────────────────────────────┘  │
      │  ┌──────────────────────────────┐  │
      │  │ 4. Detail Enhancement        │  │
      │  │    (3×3 Laplacian filter)    │  │
      │  │    → Feature sharpening      │  │
      │  └──────────────────────────────┘  │
      │  ┌──────────────────────────────┐  │
      │  │ 5. Normalization             │  │
      │  │    • float32 conversion      │  │
      │  │    • Pixel range: [0, 1]     │  │
      │  └──────────────────────────────┘  │
      └────────────┬───────────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
         ▼                    ▼
    ┌─────────────┐      ┌──────────────┐
    │ ResNet50    │      │ DenseNet121  │
    │ Backbone    │      │ Backbone     │
    │             │      │              │
    │ Features:   │      │ Features:    │
    │ • 50 layers │      │ • 121 layers │
    │ • Skip conn │      │ • Dense conn │
    │ • Residual  │      │ • Efficiency │
    │   blocks    │      │              │
    │             │      │              │
    │ Output:     │      │ Output:      │
    │ 9-class     │      │ 9-class      │
    │ softmax     │      │ softmax      │
    │ P_r ∈ℝ⁹    │      │ P_d ∈ℝ⁹     │
    │ Acc: 94.16% │      │ Acc: 98.57%  │
    └──────┬──────┘      └──────┬───────┘
           │                    │
           └─────────┬──────────┘
                     │
        ┌────────────▼────────────┐
        │  SOFT-VOTING ENSEMBLE   │
        │  ──────────────────     │
        │  P_e = 0.5·P_r +        │
        │        0.5·P_d          │
        │                         │
        │  • Equal weight fusion  │
        │  • Probabilistic avg    │
        │  • Robust to outliers   │
        │  • Accuracy: 98.46%     │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ PREDICTION & RANKING    │
        │ ──────────────────────  │
        │ pred_class = argmax(P_e)│
        │ confidence = max(P_e)   │
        │ ranking = sort(P_e)     │
        │                         │
        │ Output Vector:          │
        │ [class_idx, confidence, │
        │  top_3_emotions,        │
        │  full_distribution]     │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │  INFERENCE LATENCY      │
        │  ─────────────────      │
        │  CPU: 50-100ms          │
        │  GPU: 10-20ms           │
        └─────────────────────────┘
```

**Pipeline Characteristics:**
- **Deterministic**: Identical preprocessing ensures reproducible results
- **Robust**: Multi-stage enhancement handles varied input conditions
- **Efficient**: Optimized tensor operations for real-time inference
- **Validated**: Shape verification at model initialization

### Preprocessing Specification

Every input image undergoes this standardized pipeline (identical to training) to ensure consistency:

```python
1. Resize        → 96 × 96 RGB image
2. Blur          → Gaussian kernel 3×3 (noise reduction)
3. Enhancement   → CLAHE in LAB color space (clipLimit=2.0, tileGridSize=8×8)
4. Sharpening    → Laplacian kernel 3×3 (detail enhancement)
5. Normalization → float32, pixel range [0, 1]
```

**⚠️ Critical**: Do not replace this pipeline with backbone-specific `preprocess_input` unless models are retrained with that function.

### Ensemble Inference Algorithm

```
Ensemble Output = (0.5 × ResNet50_probs) + (0.5 × DenseNet121_probs)
Predicted Class = argmax(Ensemble Output)
Confidence      = max(Ensemble Output) × 100%
```

---

## 📊 Model Performance

Evaluated on a **held-out test set of 976 samples** across all 9 emotion classes:

### Overall Metrics Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Log Loss |
| --- | :---: | :---: | :---: | :---: | :---: |
| ResNet50 (Fine-tuned) | **94.16%** | 94.34% | 94.16% | 94.12% | 0.1442 |
| **DenseNet121** (Fine-tuned) | **98.57%** ⭐ | **98.71%** | **98.57%** | **98.58%** | **0.0328** |
| **Ensemble** (Production) | **98.46%** 🏆 | **98.50%** | **98.46%** | **98.47%** | **0.0727** |

### Per-Class Breakdown — DenseNet121

| Emotion | Precision | Recall | F1-Score | Support |
| --- | :---: | :---: | :---: | :---: |
| Angry | 100% | 100% | 100% | 97 |
| Anxiety | 90% | 100% | 95% | 123 |
| Confusion | 100% | 99% | 100% | 112 |
| Disgust | 100% | 100% | 100% | 65 |
| Fear | 100% | 94% | 97% | 198 |
| Happy | 100% | 99% | 99% | 94 |
| Neutral | 100% | 100% | 100% | 132 |
| Sad | 100% | 100% | 100% | 80 |
| Surprise | 100% | 100% | 100% | 75 |
| **Macro Average** | **99%** | **99%** | **99%** | **976** |

### Per-Class Breakdown — Ensemble (Production)

| Emotion | Precision | Recall | F1-Score | Support |
| --- | :---: | :---: | :---: | :---: |
| Angry | 99% | 100% | 99% | 97 |
| Anxiety | 92% | 97% | 94% | 123 |
| Confusion | 100% | 99% | 100% | 112 |
| Disgust | 100% | 100% | 100% | 65 |
| Fear | 98% | 95% | 97% | 198 |
| Happy | 100% | 99% | 99% | 94 |
| Neutral | 100% | 100% | 100% | 132 |
| Sad | 100% | 100% | 100% | 80 |
| Surprise | 100% | 100% | 100% | 75 |
| **Macro Average** | **99%** | **99%** | **99%** | **976** |

---

## 🎮 Getting Started

### Prerequisites

- Python 3.10 or higher
- 4GB+ RAM for inference
- Webcam (optional, for live capture)

### Installation

#### 1️⃣ Clone & Setup Environment

```bash
git clone https://github.com/yourusername/NeuroExpress.git
cd NeuroExpress
python -m venv .venv
```

**Activate Virtual Environment:**

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

#### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3️⃣ Run Application

```bash
streamlit run app.py
```

The application launches at `http://localhost:8501`. Grant camera permissions when prompted for live capture mode.

---

## 📁 Project Structure

```
NeuroExpress/
│
├── 📂 .streamlit/
│   └── config.toml                          # Dark theme & production config
│
├── 📂 models/
│   ├── resnet50_finetuned.keras             # ResNet50 model (~217 MB)
│   ├── densenet121_finetuned.keras          # DenseNet121 model (~92 MB)
│   ├── ensemble_config.json                 # Metadata & architecture
│   ├── class_names.json                     # 9 emotion labels
│   └── model_info.txt                       # Human-readable summary
│
├── 📂 assets/
│   ├── ui.png                               # Application preview
│   ├── Happy Test Case.png                  # Happy prediction example
│   └── Angry Test Case.png                  # Angry prediction example
│
├── 📄 app.py                                # Main Streamlit application
├── 📄 training_notebook.ipynb               # Model training & evaluation
├── 📄 requirements.txt                      # Python dependencies
└── 📄 README.md                             # This file
```

---

## 🚀 Deployment Guide

### Model Storage with Git LFS

The ensemble models total **~309 MB**. NeuroExpress uses **Git Large File Storage (LFS)** for reliable model versioning and distribution.

#### Setup Git LFS

```bash
# Install Git LFS (one-time setup)
git lfs install

# Track model files
git add .gitattributes
git add models/resnet50_finetuned.keras
git add models/densenet121_finetuned.keras

# Commit and push
git commit -m "Add ensemble models via Git LFS"
git push
```

#### Verify LFS Configuration

```bash
# Check tracked files
git lfs ls-files

# Expected output:
# models/resnet50_finetuned.keras (217 MB)
# models/densenet121_finetuned.keras (92 MB)
```

### Streamlit Community Cloud Deployment

1. **Repository**: Ensure models tracked via Git LFS on GitHub
2. **Deploy**: Connect repository to Streamlit Cloud
3. **Python Version**: Select **3.10** in Advanced Settings
4. **Secrets** (optional): Add environment variables if needed:
   ```toml
   STREAMLIT_SERVER_MAXUPLOADSIZE = 200
   ```
5. **Validation**: App validates models at startup and reports status

### Production Checklist

- ✅ Git LFS properly configured in local repo and GitHub
- ✅ `.gitattributes` file committed with LFS rules
- ✅ Both model files tracked and pushed to remote
- ✅ DenseNet121 model is **required** (validation at startup)
- ✅ ResNet50 optional; missing triggers automatic fallback mode
- ✅ Model paths resolved relative to `app.py` directory
- ✅ Input/output shape validation on app launch
- ✅ UI displays active inference mode (ensemble vs. fallback)
- ✅ Dependencies pinned to exact versions in `requirements.txt`
- ✅ `config.toml` configured for production dark theme

---

## 📚 Documentation

### API & Model Details

- **Input Shape**: `96 × 96 × 3` RGB image
- **Output Shape**: `(9,)` probability vector
- **Inference Time**: ~50-100ms per image (GPU: ~10-20ms)
- **Model Size**: 309 MB total (ResNet: 217 MB, DenseNet: 92 MB)

### Configuration Files

| File | Purpose |
| --- | --- |
| `ensemble_config.json` | Model weights, architecture metadata |
| `class_names.json` | Emotion label mappings |
| `model_info.txt` | Human-readable model summary |

### Training & Fine-tuning

See `training_notebook.ipynb` for:
- Dataset preparation and augmentation
- Transfer learning setup
- Training loop and hyperparameters
- Evaluation metrics and confusion matrices
- Model export and validation

---

## 🧪 Test Results

### Happy Prediction
![Happy test case](assets/Happy%20Test%20Case.png)

### Angry Prediction
![Angry test case](assets/Angry%20Test%20Case.png)

---

## ⚠️ Disclaimer

NeuroExpress is a **research and demonstration project** for facial expression classification. Results are probabilistic predictions based on visual input. Users are responsible for validating model performance in their specific application context before production deployment.

---

## 🔧 Troubleshooting

| Issue | Solution |
| --- | --- |
| Models not found | Verify Git LFS installed: `git lfs install` |
| Input shape mismatch | Check preprocessing pipeline in code |
| Low accuracy on test images | Ensure proper lighting and face visibility |
| Camera not working | Grant browser permission & check HTTPS |
| Slow inference | GPU recommended; check CUDA installation |

---

## 📦 Dependencies

See `requirements.txt` for pinned versions:

```
tensorflow==2.16.1
keras==3.12.1
streamlit>=1.28.0
opencv-python==4.11.0.0
numpy==1.26.4
scikit-image>=0.20.0
Pillow>=10.0.0
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

## 👨‍💻 Authors & Acknowledgments

Built with ❤️ using:
- **TensorFlow & Keras** for deep learning
- **Streamlit** for rapid web development
- **OpenCV** for computer vision
- **NumPy** for numerical computing

---

## 📞 Support & Contact

- 📧 **Email**: swamimantesh215@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/NeuroExpress/issues)
- 🌐 **Live Demo**: [NeuroExpress Application](https://neuroexpression.streamlit.app/)

---

<div align="center">

---

**Facial Emotion Recognition with Advanced Deep Learning**

Made with ❤️ by Swami Mantesh

[⭐ Star the repository](https://github.com/yourusername/NeuroExpress) if you found this project useful!

</div>
