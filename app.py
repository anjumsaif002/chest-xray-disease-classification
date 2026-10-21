import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras

# Set page configuration with a wide layout and medical title
st.set_page_config(
    page_title="PulmoScan AI - Chest CT & X-Ray Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern medical UI styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header card */
    .header-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
        color: #38bdf8;
    }
    .header-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    
    /* Result cards */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Diagnosis banners */
    .verdict-normal {
        background: #ecfdf5;
        border-left: 6px solid #10b981;
        padding: 1.2rem;
        border-radius: 8px;
        color: #065f46;
        margin-bottom: 1.2rem;
    }
    .verdict-cancer {
        background: #fff1f2;
        border-left: 6px solid #ef4444;
        padding: 1.2rem;
        border-radius: 8px;
        color: #991b1b;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "normal",
    "squamous_cell_carcinoma"
]

DISPLAY_NAMES = {
    "adenocarcinoma": "Adenocarcinoma",
    "large_cell_carcinoma": "Large Cell Carcinoma",
    "normal": "Normal (Healthy Lung)",
    "squamous_cell_carcinoma": "Squamous Cell Carcinoma"
}

CANCER_CLASSES = ["adenocarcinoma", "large_cell_carcinoma", "squamous_cell_carcinoma"]

# Cached model loader
@st.cache_resource
def load_model():
    candidates = [
        "best_lung_cancer_mobilenetv2.keras",
        "final_lung_cancer_mobilenetv2_model.keras",
        "best_model.keras"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return keras.models.load_model(path), path
            except Exception as e:
                continue
    return None, None

model, model_path = load_model()

# Header Banner
st.markdown("""
<div class="header-box">
    <div class="header-title">🫁 PulmoScan AI • Deep Diagnostic Platform</div>
    <div class="header-subtitle">
        Automated Pulmonary Lesion Classification & Malignancy Screening from Chest CT / X-Ray Images.<br>
        Powered by Transfer Learning (<strong>MobileNetV2</strong>) & Multi-Class Feature Analysis.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with model info & sample selector
with st.sidebar:
    st.image("https://img.shields.io/badge/Model-MobileNetV2%20CNN-38bdf8?style=for-the-badge&logo=tensorflow", use_container_width=True)
    st.markdown("### ⚙️ System Status")
    if model is not None:
        st.success(f"✅ Diagnostic Model Loaded\n`{os.path.basename(model_path)}`")
    else:
        st.error("❌ No trained model weights found.")
        
    st.divider()
    st.markdown("### 🔬 About the 4 Classes")
    st.markdown("""
    - **Adenocarcinoma**: Peripheral lung malignant tumor, most prevalent subtype.
    - **Large Cell Carcinoma**: Fast-proliferating aggressive malignant cells.
    - **Squamous Cell Carcinoma**: Malignancy emerging in central bronchial airways.
    - **Normal**: Unremarkable, healthy pulmonary tissue.
    """)
    st.divider()
    st.caption("Developed by Sara & Anjum • Medical AI Portfolio Prototype")

# Main Interface Layout
col_input, col_results = st.columns([1, 1.2], gap="large")

# 1. Input Column
with col_input:
    st.markdown("### 📥 1. Select or Upload Scan")
    
    input_mode = st.radio(
        "Choose Scan Source:",
        ["📁 Select from Clinical Test Scans", "📤 Upload Custom Image"],
        horizontal=True
    )
    
    selected_image = None
    image_name = ""
    
    samples_dir = "sample_images"
    available_samples = []
    if os.path.exists(samples_dir):
        available_samples = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if input_mode == "📁 Select from Clinical Test Scans":
        if available_samples:
            sample_choice = st.selectbox(
                "Choose a validated patient CT scan slice:",
                available_samples,
                format_func=lambda x: x.replace("_", " ").replace(".png", "")
            )
            selected_image = Image.open(os.path.join(samples_dir, sample_choice)).convert("RGB")
            image_name = sample_choice
        else:
            st.info("No sample images found in `sample_images/` directory.")
    else:
        uploaded_file = st.file_uploader(
            "Upload chest scan (CT or X-Ray):",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name

    if selected_image is not None:
        st.markdown("#### Scan Inspection")
        st.image(selected_image, caption=f"Active Scan: {image_name}", use_container_width=True)
        st.caption(f"Dimensions: {selected_image.size[0]} x {selected_image.size[1]} px | Format: {selected_image.format or 'RGB'}")
    else:
        st.info("Please select a sample scan above or upload an image to begin evaluation.")

# 2. Results & Diagnosis Column
with col_results:
    st.markdown("### 📊 2. AI Diagnostic Assessment")
    
    if selected_image is not None:
        if model is None:
            st.error("Cannot run inference: Model weights could not be loaded. Please ensure `best_lung_cancer_mobilenetv2.keras` is in the project folder.")
        else:
            with st.spinner("Analyzing anatomical features through MobileNetV2..."):
                # Preprocess image to model input shape
                resized = selected_image.resize((224, 224))
                img_arr = np.array(resized, dtype=np.float32) / 255.0
                img_batch = np.expand_dims(img_arr, axis=0)
                
                # Run prediction
                preds = model.predict(img_batch, verbose=0)[0]
                top_idx = int(np.argmax(preds))
                top_class = CLASS_NAMES[top_idx]
                confidence = float(preds[top_idx] * 100)
                
                # Compute Aggregate Cancer Risk
                cancer_risk = float(sum(preds[CLASS_NAMES.index(c)] for c in CANCER_CLASSES) * 100)
                normal_prob = float(preds[CLASS_NAMES.index("normal")] * 100)
                
                # Confidence rating
                if confidence >= 75:
                    conf_rating = "High Confidence"
                elif confidence >= 50:
                    conf_rating = "Moderate Confidence"
                else:
                    conf_rating = "Borderline / Low Confidence"

            # Primary Verdict Banner
            if top_class == "normal":
                st.markdown(f"""
                <div class="verdict-normal">
                    <h3 style="margin:0; color:#065f46;">✅ Negative for Pulmonary Malignancy</h3>
                    <p style="margin:0.4rem 0 0 0; font-size:1.05rem;">
                        Scan classified as <strong>Normal Healthy Tissue</strong> with <strong>{confidence:.1f}%</strong> confidence.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-cancer">
                    <h3 style="margin:0; color:#991b1b;">⚠️ Pulmonary Neoplasm / Malignancy Detected</h3>
                    <p style="margin:0.4rem 0 0 0; font-size:1.05rem;">
                        Primary Subtype: <strong>{DISPLAY_NAMES[top_class]}</strong> ({confidence:.1f}% confidence).
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Key Metrics Cards
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color: {'#10b981' if top_class == 'normal' else '#ef4444'};">
                        {confidence:.1f}%
                    </div>
                    <div class="metric-lbl">Confidence ({conf_rating})</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color: #dc2626;">{cancer_risk:.1f}%</div>
                    <div class="metric-lbl">Combined Cancer Risk</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color: #059669;">{normal_prob:.1f}%</div>
                    <div class="metric-lbl">Healthy Tissue Score</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown("#### 🔬 Detailed Probability Distribution")
            
            # Sorted Probability Progress Bars
            sorted_indices = np.argsort(preds)[::-1]
            for idx in sorted_indices:
                c_name = CLASS_NAMES[idx]
                p_val = float(preds[idx])
                display_label = DISPLAY_NAMES[c_name]
                
                col_lbl, col_bar = st.columns([1.2, 2.5])
                with col_lbl:
                    is_top = (idx == top_idx)
                    st.write(f"**{'👉 ' if is_top else ''}{display_label}**")
                with col_bar:
                    st.progress(p_val, text=f"{p_val * 100:.1f}%")

            # Clinical Guidance Box
            st.divider()
            st.markdown("#### 📋 Clinical Review Recommendations")
            if top_class == "normal":
                st.success("• Routine surveillance: No acute pulmonary mass or infiltration observed. Correlate with clinical symptoms.")
            else:
                st.warning(f"• High clinical suspicion of **{DISPLAY_NAMES[top_class]}**.\n• Follow-up contrast CT/PET-CT and histopathological tissue biopsy strongly advised.")

            with st.expander("⚠️ Medical AI Prototype Disclaimer"):
                st.caption(
                    "This software is an academic engineering demonstration built using TensorFlow and Keras. "
                    "It is NOT a medical device and must never substitute professional radiological diagnosis."
                )
    else:
        st.markdown("""
        <div style="border: 2px dashed #cbd5e1; border-radius: 12px; padding: 3rem; text-align: center; color: #64748b; background: white;">
            <p style="font-size: 1.2rem; font-weight: 600;">No Scan Loaded</p>
            <p style="font-size: 0.95rem;">Select one of the validated clinical samples on the left or upload your own scan to view real-time diagnostic classification.</p>
        </div>
        """, unsafe_allow_html=True)
