"""NeuroExpress — compact, single-page facial emotion recognition workspace."""

import io
import time
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

MODEL_PATHS = (
    "models/resnet50_finetuned.keras",
    "models/densenet121_finetuned.keras",
)
MODEL_WEIGHTS = np.array([0.5, 0.5], dtype="float32")
IMG_SIZE = (96, 96)


_original_dense_from_config = tf.keras.layers.Dense.from_config


@classmethod
def _dense_from_config_without_quantization(cls, config):
    config = dict(config)
    config.pop("quantization_config", None)
    return _original_dense_from_config(config)


tf.keras.layers.Dense.from_config = _dense_from_config_without_quantization

# Alphabetical class order from training dataset
CLASS_NAMES = ["Angry", "Anxiety", "Confusion", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Suprise"]

EMOJI_MAP = {
    "Angry": "😠", "Anxiety": "😰", "Confusion": "😕", "Disgust": "🤢",
    "Fear": "😨", "Happy": "😄", "Neutral": "😐", "Sad": "😢", "Suprise": "😲",
}

COLOR_MAP = {
    "Angry": "#fb7185", "Anxiety": "#f59e0b", "Confusion": "#a78bfa", "Disgust": "#a3e635",
    "Fear": "#8b5cf6", "Happy": "#34d399", "Neutral": "#8ea0bd", "Sad": "#60a5fa", "Suprise": "#f472b6",
}

st.set_page_config(
    page_title="NeuroExpress | Emotion AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom single-page CSS without scroll requirements
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

#MainMenu, header, footer { visibility: hidden; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; overflow-x: hidden; }
.stApp, [data-testid="stAppViewContainer"] { background: #080b14 !important; color: #f8fafc; }
.stApp:before {
    content: ""; position: fixed; inset: 0; pointer-events: none;
    background: radial-gradient(circle at 15% 5%, rgba(108, 92, 231, 0.15), transparent 28rem),
                radial-gradient(circle at 85% 20%, rgba(244, 114, 182, 0.08), transparent 25rem);
}

/* Single page container fitting viewport */
.block-container {
    max-width: 1200px;
    padding: 1rem 1.75rem 1rem !important;
    position: relative;
}

/* Header Navbar */
.nav-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.8rem;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
}
.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.05rem;
    margin: 0;
    line-height: 1;
}
.brand-dot { color: #a78bfa; }
.brand-tag {
    font-size: 0.76rem;
    color: #94a3b8;
    margin-left: 0.5rem;
    font-weight: 500;
}
.privacy-badge {
    color: #c4b5fd;
    border: 1px solid rgba(167, 139, 250, 0.3);
    background: rgba(139, 92, 246, 0.1);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Sections */
.section-kicker { color: #a78bfa; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 0.2rem; }
.section-title { font-family: 'Manrope', sans-serif; font-size: 1.1rem; font-weight: 700; margin: 0; }
.surface {
    background: linear-gradient(145deg, rgba(22, 28, 44, 0.85), rgba(14, 18, 29, 0.9));
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 16px;
    padding: 1rem 1.15rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

/* Uploader & Tabs */
[data-testid="stFileUploader"] { margin: 0.4rem 0 0.6rem; padding: 0.4rem; background: rgba(139, 92, 246, 0.04); border-radius: 12px; }
[data-testid="stFileUploader"] section, div[data-testid="stFileUploaderDropzone"] {
    min-height: 80px !important;
    background: rgba(139, 92, 246, 0.05) !important;
    border: 1.5px dashed rgba(167, 139, 250, 0.4) !important;
    border-radius: 10px !important;
    padding: 0.6rem !important;
}
[data-testid="stFileUploader"] button {
    background: #765ce7 !important; color: #fff !important; border: 0 !important;
    border-radius: 8px !important; font-weight: 700 !important; font-size: 0.8rem !important;
}

/* Preview Image Frame */
.preview-frame {
    height: 230px;
    display: flex; align-items: center; justify-content: center;
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 14px; overflow: hidden;
    background: linear-gradient(135deg, #0f172a, #090d16);
}
.preview-empty { text-align: center; color: #64748b; font-size: 0.85rem; }
.preview-empty span { display: block; font-size: 1.8rem; margin-bottom: 0.3rem; }
div[data-testid="stImage"] img {
    width: 100%; height: 230px; object-fit: contain;
    border-radius: 12px; background: #090d16;
}

/* Result Cards */
.result-hero {
    border-radius: 16px;
    padding: 1rem 1.25rem;
    text-align: center;
    background: linear-gradient(140deg, rgba(108, 92, 231, 0.32), rgba(30, 20, 50, 0.85));
    border: 1px solid rgba(196, 181, 253, 0.25);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.result-emoji { font-size: 2.8rem; line-height: 1; filter: drop-shadow(0 6px 8px rgba(0,0,0,0.3)); }
.result-name { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.65rem; letter-spacing: -0.04rem; margin: 0.25rem 0 0.1rem; }
.result-meta { color: #cbd5e1; font-size: 0.8rem; }
.confidence { color: #fff; font-weight: 700; }
.insight { margin-top: 0.4rem; color: #94a3b8; font-size: 0.8rem; }

.waiting { color: #64748b; text-align: center; padding: 3rem 1rem; }
.waiting-icon { font-size: 2rem; margin-bottom: 0.5rem; color: #a78bfa; }

/* Emotion score bars */
.rank-row {
    display: grid; grid-template-columns: 24px 1fr auto;
    align-items: center; gap: 0.6rem; padding: 0.42rem 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.08);
}
.rank-row:last-child { border-bottom: 0; }
.rank-number { color: #64748b; font-size: 0.72rem; font-weight: 700; }
.rank-label { color: #e2e8f0; font-size: 0.82rem; font-weight: 600; display: flex; align-items: center; gap: 0.35rem; }
.rank-value { color: #cbd5e1; font-size: 0.78rem; font-variant-numeric: tabular-nums; font-weight: 600; }
.bar-track { height: 6px; background: rgba(148, 163, 184, 0.12); border-radius: 10px; overflow: hidden; margin-top: 0.25rem; }
.bar-fill { height: 100%; border-radius: 10px; transition: width 0.4s ease; }

/* Compact buttons and UI */
div[data-testid="stButton"] > button {
    width: 100%; background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    color: #cbd5e1 !important; border-radius: 10px !important;
    font-size: 0.8rem !important; font-weight: 600 !important; padding: 0.35rem !important;
    margin-top: 0.4rem !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: rgba(167, 139, 250, 0.6) !important;
    color: #fff !important; background: rgba(139, 92, 246, 0.12) !important;
}
[data-testid="stTabs"] button { color: #94a3b8 !important; font-weight: 600 !important; font-size: 0.82rem !important; padding: 0.3rem 0.8rem !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #f8fafc !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: #8b5cf6 !important; }
[data-testid="stCheckbox"] label { font-size: 0.8rem !important; color: #a78bfa !important; }

/* Status indicator badge */
.status-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-size: 0.72rem; color: #34d399; background: rgba(52, 211, 153, 0.1);
    border: 1px solid rgba(52, 211, 153, 0.25); border-radius: 999px;
    padding: 0.15rem 0.5rem; margin-top: 0.3rem;
}

@media(max-width: 800px) {
    .block-container { padding: 0.8rem !important; }
    .preview-frame, div[data-testid="stImage"] img { height: 180px; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner=False)
def load_ensemble(paths: tuple[str, ...]):
    return tuple(load_model(path) for path in paths)


@st.cache_resource
def get_face_cascade():
    """Load OpenCV's optional Haar cascade without blocking inference.

    Some cloud OpenCV builds do not ship ``cv2.data.haarcascades``. Face
    cropping is an enhancement, not a model requirement, so return ``None``
    and safely analyze the full image when the detector is unavailable.
    """
    cv2_data = getattr(cv2, "data", None)
    cascade_directory = getattr(cv2_data, "haarcascades", None)
    if not cascade_directory:
        return None

    try:
        cascade = cv2.CascadeClassifier(
            f"{cascade_directory}haarcascade_frontalface_default.xml"
        )
        return None if cascade.empty() else cascade
    except (AttributeError, cv2.error):
        return None


def detect_and_crop_face(image: Image.Image):
    """Detect primary face and crop with comfortable margin for accurate ResNet input."""
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    cascade = get_face_cascade()
    if cascade is None:
        return image, False
    
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )
    
    if len(faces) > 0:
        # Select largest detected face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        
        # Add 18% margin around face to match portrait dataset style
        margin_w = int(w * 0.18)
        margin_h = int(h * 0.18)
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img_np.shape[1], x + w + margin_w)
        y2 = min(img_np.shape[0], y + h + margin_h)
        
        cropped_np = img_np[y1:y2, x1:x2]
        return Image.fromarray(cropped_np), True
    
    return image, False


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Reproduce training preprocessing pipeline: Resize -> Blur -> CLAHE -> Sharpen -> Normalize."""
    rgb_image = np.asarray(image.convert("RGB").resize(IMG_SIZE))
    blurred = cv2.GaussianBlur(rgb_image, (3, 3), 0)

    lab_image = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(
        cv2.merge([clahe.apply(lightness), channel_a, channel_b]),
        cv2.COLOR_LAB2RGB,
    )

    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpening_kernel)
    return np.expand_dims(sharpened.astype("float32") / 255.0, axis=0)


def predict(models, image: Image.Image, auto_crop: bool = True):
    started = time.perf_counter()
    face_detected = False
    
    if auto_crop:
        processed_img, face_detected = detect_and_crop_face(image)
    else:
        processed_img = image
        
    processed_batch = preprocess_image(processed_img)
    model_probabilities = np.stack(
        [model.predict(processed_batch, verbose=0)[0] for model in models]
    )
    probabilities = np.average(model_probabilities, axis=0, weights=MODEL_WEIGHTS)
    latency = (time.perf_counter() - started) * 1000
    return probabilities, np.argsort(probabilities)[::-1], latency, face_detected, processed_img


def score_row(rank: int, label: str, value: float):
    color = COLOR_MAP[label]
    display_label = "Surprise" if label == "Suprise" else label
    st.markdown(f'''
    <div class="rank-row">
        <div class="rank-number">0{rank}</div>
        <div>
            <div class="rank-label"><span>{EMOJI_MAP[label]}</span> {display_label}</div>
            <div class="bar-track"><div class="bar-fill" style="width:{value:.1f}%; background:{color}"></div></div>
        </div>
        <div class="rank-value">{value:.1f}%</div>
    </div>
    ''', unsafe_allow_html=True)


def clear_analysis():
    st.session_state.pop("emotion_upload", None)
    st.session_state.pop("camera_capture", None)
    st.session_state["input_mode"] = "upload"


def activate_source(source: str):
    st.session_state["input_mode"] = source


# --- TOP HEADER ---
st.markdown('''
<div class="nav-header">
    <div>
        <h1 class="brand">Neuro<span class="brand-dot">Express</span> <span class="brand-tag">v2.0 • Emotion AI</span></h1>
    </div>
    <div class="privacy-badge">✦ Real-time Neural Engine</div>
</div>
''', unsafe_allow_html=True)

# Model Loading
try:
    with st.spinner("Initializing ResNet50 + DenseNet121 Ensemble…"):
        models = load_ensemble(MODEL_PATHS)
except Exception as error:
    st.error("Failed to load the ensemble models. Ensure both model files exist.")
    with st.expander("Error details"):
        st.code(str(error))
    st.stop()

# --- MAIN COMPACT 2-COLUMN VIEWPORT ---
upload_col, results_col = st.columns([0.48, 0.52], gap="large")

with upload_col:
    st.markdown('<p class="section-kicker">Input Selection</p><h2 class="section-title">Analyze Facial Expression</h2>', unsafe_allow_html=True)
    
    upload_tab, camera_tab = st.tabs(["⌁  Upload Image", "◉  Live Camera"])
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="emotion_upload",
            on_change=activate_source,
            args=("upload",)
        )
    with camera_tab:
        camera_capture = st.camera_input(
            "Capture camera photo",
            label_visibility="collapsed",
            key="camera_capture",
            on_change=activate_source,
            args=("camera",)
        )

    source = st.session_state.get("input_mode", "upload")
    selected_file = camera_capture if source == "camera" else uploaded_file
    image = Image.open(io.BytesIO(selected_file.getvalue())) if selected_file is not None else None

    # Face Auto-Crop Option
    auto_crop = st.checkbox("Auto Face Detection & Crop", value=True, help="Automatically isolates face ROI to boost emotion detection accuracy.")

    if image is not None:
        st.image(image, use_container_width=True)
        st.button("↺  Clear & Upload New", on_click=clear_analysis, use_container_width=True)
    else:
        st.markdown('''
        <div class="preview-frame">
            <div class="preview-empty">
                <span>📷</span>Select or drop a photo to begin instant evaluation
            </div>
        </div>
        ''', unsafe_allow_html=True)

with results_col:
    st.markdown('<p class="section-kicker">Real-time Inference</p><h2 class="section-title">Emotion Prediction</h2>', unsafe_allow_html=True)
    
    if image is None:
        st.markdown('''
        <div class="result-hero waiting">
            <div class="waiting-icon">✦</div>
            <strong style="color: #f1f5f9; font-size: 1.1rem;">Awaiting Image Input</strong><br>
            Upload a portrait photo or use live camera capture.
        </div>
        <div class="surface" style="margin-top: 0.8rem;">
            <div class="scores-heading" style="display:flex; justify-space-between; align-items:center;">
                <h3 class="section-title">Emotion Signals</h3>
                <span style="color:#64748b; font-size:0.75rem;">9 Classes</span>
            </div>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0.4rem 0 0;">Ranked confidence probabilities will display here automatically.</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        probabilities, ordered_indices, latency_ms, face_detected, cropped_img = predict(models, image, auto_crop=auto_crop)
        
        top_index = ordered_indices[0]
        top_label = CLASS_NAMES[top_index]
        top_value = probabilities[top_index] * 100
        display_label = "Surprise" if top_label == "Suprise" else top_label
        
        status_html = ""
        if auto_crop and face_detected:
            status_html = '<div class="status-badge">✓ Primary Face Auto-Cropped</div>'
        elif auto_crop and not face_detected:
            status_html = '<div class="status-badge" style="color:#f59e0b; background:rgba(245,158,11,0.1); border-color:rgba(245,158,11,0.3);">⚡ Full Image Analyzed (No distinct face box)</div>'
        
        st.markdown(f'''
        <div class="result-hero">
            <div class="result-emoji">{EMOJI_MAP[top_label]}</div>
            <div class="result-name">{display_label}</div>
            <div class="result-meta">
                <span class="confidence">{top_value:.1f}% Confidence</span> &nbsp;•&nbsp; {latency_ms:.0f} ms inference
            </div>
            {status_html}
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="surface" style="margin-top: 0.6rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <h3 class="section-title">Leading Signals</h3>
                <span style="color: #94a3b8; font-size: 0.75rem;">Top 3 of 9</span>
            </div>
        ''', unsafe_allow_html=True)
        
        for rank, index in enumerate(ordered_indices[:3], start=1):
            score_row(rank, CLASS_NAMES[index], probabilities[index] * 100)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.expander("View All 9 Emotion Scores"):
            for rank, index in enumerate(ordered_indices[3:], start=4):
                score_row(rank, CLASS_NAMES[index], probabilities[index] * 100)
