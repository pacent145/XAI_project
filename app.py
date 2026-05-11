import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os
import time
from PIL import Image

st.set_page_config(
    page_title="RxAuth — Prescription Authenticator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #080c14;
        color: #e2e8f0;
    }

    .main { background-color: #080c14; }
    .block-container { padding: 2rem 3rem; }

    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -1px;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }

    .hero-sub {
        font-size: 1rem;
        color: #64748b;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #0f1729 0%, #111827 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1;
        margin-bottom: 0.3rem;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }

    .authentic-badge {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #16a34a;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #4ade80;
        letter-spacing: 2px;
    }

    .suspicious-badge {
        background: linear-gradient(135deg, #2d0a0a, #450a0a);
        border: 1px solid #dc2626;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f87171;
        letter-spacing: 2px;
    }

    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e293b;
    }

    .prob-bar-container {
        margin-bottom: 0.5rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        width: 100%;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #38bdf8, #0ea5e9);
        transform: translateY(-1px);
    }

    .sidebar-info {
        background: #0f1729;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    [data-testid="stSidebar"] {
        background-color: #0a0f1e;
        border-right: 1px solid #1e293b;
    }

    .divider { border-top: 1px solid #1e293b; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if os.path.exists("outputs/prescription_model.keras"):
        return keras.models.load_model("outputs/prescription_model.keras")
    st.error("Model not found. Run prescription_auth.py first to train and save the model.")
    st.stop()


def load_metrics():
    if os.path.exists("outputs/metrics.json"):
        with open("outputs/metrics.json") as f:
            return json.load(f)
    return None


def make_gradcam(img_array, model, layer_name='conv3'):
    img_tensor = tf.cast(img_array, tf.float32)
    conv_layer = model.get_layer(layer_name)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        x = img_tensor
        for layer in model.layers:
            x = layer(x)
            if layer.name == layer_name:
                conv_out = x
                tape.watch(conv_out)
        preds = x
        class_idx = tf.argmax(preds[0])
        score = preds[:, class_idx]

    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def preprocess(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((28, 28))
    arr = np.array(img) / 255.0
    arr = arr[..., np.newaxis]
    arr = np.expand_dims(arr, axis=0)
    return arr, img


model = load_model()
metrics = load_metrics()

with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.4rem;">RxAuth</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="font-size:0.7rem; margin-bottom:1.5rem;">Prescription AI System</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Info</div>', unsafe_allow_html=True)

    if metrics:
        st.markdown(f"""
        <div class="sidebar-info">
            <b>Architecture:</b> CNN (3 Conv layers)<br>
            <b>Dataset:</b> MNIST (60,000 samples)<br>
            <b>Test Accuracy:</b> {metrics['test_accuracy']}%<br>
            <b>Training Time:</b> {metrics['training_time_seconds']}s<br>
            <b>Memory Used:</b> +{metrics['memory_increase_mb']} MB<br>
            <b>CPU Before:</b> {metrics['cpu_before']}%<br>
            <b>CPU After:</b> {metrics['cpu_after']}%
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1rem;">Auth Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("Confidence Threshold (%)", min_value=50, max_value=99, value=85, step=1)
    st.caption("Predictions above this value = AUTHENTIC")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info">
        This system uses a CNN trained on handwritten digits to authenticate prescription entries.
        Grad-CAM highlights exactly which pixels the model analysed when making its decision —
        making the AI decision transparent and explainable.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="hero-title">Prescription<br>Authenticator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Explainable AI — Digit Recognition System</div>', unsafe_allow_html=True)

if metrics:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["test_accuracy"]}%</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["training_time_seconds"]}s</div><div class="metric-label">Training Time</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["memory_increase_mb"]}MB</div><div class="metric-label">Memory Used</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["test_loss"]}</div><div class="metric-label">Test Loss</div></div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1.6], gap="large")

with left_col:
    st.markdown('<div class="section-title">Upload Prescription Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    st.markdown('<div class="section-title" style="margin-top:1rem;">Or test with MNIST sample</div>', unsafe_allow_html=True)
    use_sample = st.button("▶  Run Random Sample")

    if metrics:
        st.markdown('<div class="section-title" style="margin-top:1.5rem;">Training Curves</div>', unsafe_allow_html=True)
        if os.path.exists("outputs/training_curves.png"):
            st.image("outputs/training_curves.png", use_container_width=True)

with right_col:
    st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)

    run_image = None
    source_label = ""

    if uploaded is not None:
        run_image = uploaded
        source_label = "uploaded image"
    elif use_sample and os.path.exists("outputs/temp_test.png"):
        run_image = "outputs/temp_test.png"
        source_label = "MNIST sample"

    if run_image is not None:
        with st.spinner("Analysing..."):
            img_array, pil_img = preprocess(run_image)

            infer_start = time.time()
            preds = model.predict(img_array, verbose=0)
            infer_time = (time.time() - infer_start) * 1000

            predicted_digit = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0])) * 100
            status = "AUTHENTIC" if confidence >= threshold else "SUSPICIOUS"

            heatmap = make_gradcam(img_array, model)
            heatmap_resized = cv2.resize(heatmap, (28, 28))
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
            original_rgb = np.repeat(img_array[0], 3, axis=2)
            overlay = np.clip(0.4 * heatmap_colored + 0.6 * original_rgb, 0, 1)

        badge_class = "authentic-badge" if status == "AUTHENTIC" else "suspicious-badge"
        icon = "✅" if status == "AUTHENTIC" else "❌"
        st.markdown(f'<div class="{badge_class}">{icon} &nbsp; {status}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:2.5rem;">{predicted_digit}</div><div class="metric-label">Predicted Digit</div></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{confidence:.1f}%</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{infer_time:.1f}ms</div><div class="metric-label">Inference Time</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        img_col, heat_col, over_col = st.columns(3)
        with img_col:
            st.markdown('<div class="section-title">Input</div>', unsafe_allow_html=True)
            st.image(np.array(pil_img), use_container_width=True)
        with heat_col:
            st.markdown('<div class="section-title">Grad-CAM</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(3, 3))
            fig.patch.set_facecolor('#080c14')
            ax.imshow(heatmap_resized, cmap='jet')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        with over_col:
            st.markdown('<div class="section-title">Overlay</div>', unsafe_allow_html=True)
            st.image(overlay, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Probability Breakdown — All Digits</div>', unsafe_allow_html=True)

        probs = {i: float(preds[0][i]) * 100 for i in range(10)}
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        for digit, prob in sorted_probs:
            bar_color = "#38bdf8" if digit == predicted_digit else "#1e293b"
            filled = int(prob / 2)
            bar = "█" * filled + "░" * (50 - filled)
            st.markdown(f"""
            <div class="prob-bar-container">
                <span style="font-family:'Space Mono',monospace; font-size:0.8rem; color:{'#38bdf8' if digit == predicted_digit else '#64748b'};">
                    Digit {digit} &nbsp; {bar} &nbsp; {prob:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#0f1729; border:1px dashed #1e293b; border-radius:12px; padding:3rem; text-align:center; color:#334155;">
            <div style="font-size:3rem; margin-bottom:1rem;">🏥</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.9rem; letter-spacing:0.1em;">
                UPLOAD AN IMAGE OR RUN A SAMPLE<br>TO BEGIN ANALYSIS
            </div>
        </div>
        """, unsafe_allow_html=True)

if metrics:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">System Performance Analysis</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1:
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#0f1729')
        ax.set_facecolor('#0f1729')
        epochs_r = range(1, 6)
        ax.plot(epochs_r, metrics['train_accuracy'], 'o-', color='#38bdf8', label='Train', linewidth=2)
        ax.plot(epochs_r, metrics['val_accuracy'],   'o-', color='#f472b6', label='Validation', linewidth=2)
        ax.set_title('Accuracy', color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', color='#64748b', fontsize=9)
        ax.set_ylabel('Accuracy (%)', color='#64748b', fontsize=9)
        ax.tick_params(colors='#64748b')
        ax.legend(facecolor='#1e293b', labelcolor='white', fontsize=8)
        ax.grid(True, alpha=0.1, color='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')
        st.pyplot(fig)
        plt.close()

    with p2:
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#0f1729')
        ax.set_facecolor('#0f1729')
        ax.plot(epochs_r, metrics['train_loss'], 'o-', color='#38bdf8', label='Train', linewidth=2)
        ax.plot(epochs_r, metrics['val_loss'],   'o-', color='#f472b6', label='Validation', linewidth=2)
        ax.set_title('Loss', color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', color='#64748b', fontsize=9)
        ax.set_ylabel('Loss', color='#64748b', fontsize=9)
        ax.tick_params(colors='#64748b')
        ax.legend(facecolor='#1e293b', labelcolor='white', fontsize=8)
        ax.grid(True, alpha=0.1, color='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')
        st.pyplot(fig)
        plt.close()