# NeuroExpress — Facial Emotion Recognition

NeuroExpress is a Streamlit web application for classifying facial expressions with a fine-tuned **ResNet50** model. It predicts one of nine emotion classes from a portrait image or a live camera capture, then presents the top result and a ranked confidence breakdown.

The application is built to reproduce the preprocessing and label order used in the accompanying training notebook, so browser predictions are consistent with the trained model pipeline.

## Web App Preview

<!-- Add your deployed-app screenshot here, for example:
![NeuroExpress web app](assets/neuroexpress-preview.png)
-->

## Features

- Photo upload for JPG and PNG portraits
- Live camera capture directly in the browser
- Nine-class facial expression prediction
- Primary prediction with confidence score and inference time
- Ranked top-three emotion signals, with all remaining scores available on demand
- Clear analysis control for fast repeat testing
- Responsive dark interface for desktop and mobile use

## Emotion Classes

| Index | Model class | Display label |
| ---: | --- | --- |
| 0 | Angry | Angry |
| 1 | Anxiety | Anxiety |
| 2 | Confusion | Confusion |
| 3 | Disgust | Disgust |
| 4 | Fear | Fear |
| 5 | Happy | Happy |
| 6 | Neutral | Neutral |
| 7 | Sad | Sad |
| 8 | Suprise | Surprise |

> The model was trained on a dataset directory named `Suprise`. The application preserves this internal class order to keep predictions correct, while displaying the standard spelling, **Surprise**, in the interface.

## Model Pipeline

The model was trained in `code-ipynb.ipynb` using an ImageNet-initialized ResNet50 backbone with a nine-class softmax output.

Each image follows this preprocessing sequence before prediction:

1. Resize to `96 × 96` RGB
2. Apply Gaussian blur with a `3 × 3` kernel
3. Apply CLAHE contrast enhancement in LAB color space
4. Apply image sharpening
5. Convert to `float32` and normalize pixels to `[0, 1]`

This same sequence is implemented in `app.py`. Do not replace it with a backbone-specific `preprocess_input` function unless the model is retrained using that function.

## Project Structure

```text
Facial Web APP/
├── .streamlit/
│   └── config.toml              # Production dark-theme configuration
├── app.py                       # Streamlit application
├── code-ipynb.ipynb             # Training and evaluation notebook
├── resnet50_model.keras         # Exported trained model
├── requirements.txt             # Python dependencies
└── README.md
```

## Run Locally

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal. For live camera mode, grant the browser permission to use your camera.

## Deployment Notes

The model file is approximately 217 MB, which exceeds GitHub’s normal 100 MB file limit. Before deploying, choose one of these options:

- Store the model using **Git LFS**.
- Store the model in cloud object storage and download it securely when the app starts.
- Deploy with Docker or another platform that allows the model artifact to be included outside a standard GitHub push.

For Streamlit Community Cloud, keep `app.py`, `requirements.txt`, and `.streamlit/config.toml` in the repository root. Ensure the deployed environment can access `resnet50_model.keras` through one of the approaches above.

## Responsible Use

NeuroExpress is a demonstration and research project. Facial-expression classification is probabilistic and can be affected by lighting, pose, image quality, demographic variation, and dataset limitations. It must not be used for medical, psychological, employment, security, or other high-impact decisions.

