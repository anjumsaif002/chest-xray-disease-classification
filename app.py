import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras

CLASS_NAMES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "normal",
    "squamous_cell_carcinoma"
]

CANCER_CLASSES = ["adenocarcinoma", "large_cell_carcinoma", "squamous_cell_carcinoma"]

st.set_page_config(
    page_title="Chest X-Ray / CT Disease Classifier",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Pulmonary Disease & Lung Cancer Detection")
st.markdown("""
This application uses a Deep Learning Convolutional Neural Network (**MobileNetV2** transfer learning)
to classify chest CT scan and X-Ray images into 4 diagnostic categories:
- **Adenocarcinoma**
- **Large Cell Carcinoma**
- **Squamous Cell Carcinoma**
- **Normal (Healthy Lung Tissue)**
""")

@st.cache_resource
def load_trained_model(model_path="best_lung_cancer_mobilenetv2.keras"):
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    # Check alternate filenames
    for alt in ["final_lung_cancer_mobilenetv2_model.keras", "best_model.keras"]:
        if os.path.exists(alt):
            return keras.models.load_model(alt)
    return None

model = load_trained_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Chest Image")
    uploaded_file = st.file_uploader(
        "Choose a chest CT Scan or X-Ray image (.png, .jpg, .jpeg)...",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Scan", use_container_width=True)

with col2:
    st.subheader("🔍 Diagnostic Evaluation")
    if uploaded_file is not None:
        if model is None:
            st.warning("⚠️ No trained `.keras` model file found in directory. Please train a model or place `best_lung_cancer_mobilenetv2.keras` in the root folder.")
        else:
            with st.spinner("Analyzing image features..."):
                resized_image = image.resize((224, 224))
                img_array = np.array(resized_image, dtype=np.float32) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)

                predictions = model.predict(img_batch)[0]
                pred_idx = np.argmax(predictions)
                pred_class = CLASS_NAMES[pred_idx]
                confidence = float(predictions[pred_idx] * 100)

                cancer_risk = float(sum(predictions[CLASS_NAMES.index(c)] for c in CANCER_CLASSES) * 100)
                normal_prob = float(predictions[CLASS_NAMES.index("normal")] * 100)

                if pred_class == "normal":
                    st.success(f"### Result: Normal / Healthy Tissue ({confidence:.1f}%)")
                else:
                    st.error(f"### Result: {pred_class.replace('_', ' ').title()} Detected ({confidence:.1f}%)")

                st.markdown(f"**Cancer Risk Score:** `{cancer_risk:.1f}%` | **Normal Score:** `{normal_prob:.1f}%`")

                st.write("#### Class Probabilities Breakdown:")
                for name, prob in zip(CLASS_NAMES, predictions):
                    label = name.replace("_", " ").title()
                    st.write(f"**{label}**: {prob*100:.1f}%")
                    st.progress(float(prob))

                st.caption("ℹ️ *Disclaimer: This AI tool is for research & educational demonstration only. Clinical diagnosis must be confirmed by a licensed medical professional.*")
    else:
        st.info("Please upload a scan to run disease classification.")
