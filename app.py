import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path

# ===================== Page Config (FIRST Streamlit command) =====================
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱", layout="wide")

# ===================== Paths =====================
WORK_DIR = Path("E:/Nibm/Degree/4th_Year/CV/Individual_CW/CV_CW/CV_CW")
MODEL_ONE_PATH = WORK_DIR / "models/model_one.keras" # Renamed for clarity
MODEL_TWO_PATH = WORK_DIR / "models/model_two.keras" # New path for second model
LABEL_MAP_PATH = WORK_DIR / "label_map.json"

# ===================== Load Models =====================
@st.cache_resource
def load_model(model_path):
    """Loads a Keras model from a given path."""
    model = tf.keras.models.load_model(str(model_path))
    return model

model_one = load_model(MODEL_ONE_PATH) # Load first model
model_two = load_model(MODEL_TWO_PATH) # Load second model

# ===================== Load Label Map =====================
with open(LABEL_MAP_PATH, 'r') as f:
    inv_class_indices = json.load(f)

IMG_SIZE = (224, 224)

# ===================== Preprocess =====================
def preprocess_image(img: Image.Image):
    img = img.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr

# ===================== Predict =====================
def predict(img: Image.Image, model: tf.keras.Model):
    """
    Performs prediction using a specified model.
    Returns: predicted_label, confidence, top3_predictions
    """
    arr = preprocess_image(img)
    preds = model.predict(arr)[0]
    top_idx = int(np.argmax(preds))
    predicted_label = inv_class_indices[str(top_idx)]
    confidence = float(preds[top_idx])

    # top-3 predictions
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [(inv_class_indices[str(int(i))], float(preds[int(i)])) for i in top3_idx]

    return predicted_label, confidence, top3

# ===================== Streamlit UI =====================

# Sidebar
st.sidebar.title("📌 Instructions")
st.sidebar.info(
    "1. Upload a **leaf image** (JPG/PNG).\n"
    "2. Click **Predict** to analyze.\n"
    "3. View the **disease name and top-3 predictions** from **both models**."
)
st.sidebar.markdown("---")

# Main Title
st.title("🌱 Plant Disease Detection (Dual Model Analysis)")
st.markdown("Upload a clear **leaf image** and let **two AI models** predict the disease for comparison.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)

    col_img, col_pred = st.columns([1, 2])
    
    with col_img:
        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    with col_pred:
        st.subheader("🔍 Prediction Results")

        if st.button("🚀 Run Dual Prediction", key="predict_button"):
            
            # Run prediction on both models
            label_one, conf_one, top3_one = predict(image, model_one)
            label_two, conf_two, top3_two = predict(image, model_two)

            # --- Results for Model One ---
            st.markdown("---")
            st.markdown("### 🤖 Model One Results(CNN)")
            st.info(f"**Main Prediction:** **{label_one}**")
            st.progress(int(conf_one * 100))
            st.write(f"Confidence: **{conf_one:.2f}**")

            st.markdown("#### Top 3 Predictions")
            col_l1, col_c1 = st.columns(2)
            for i, (l, c) in enumerate(top3_one):
                col_l1.write(f"**{i+1}.** {l}")
                col_c1.progress(int(c * 100))

            # --- Results for Model Two ---
            st.markdown("---")
            st.markdown("### ⚙️ Model Two Results(MobileNetV2)")
            st.info(f"**Main Prediction:** **{label_two}**")
            st.progress(int(conf_two * 100))
            st.write(f"Confidence: **{conf_two:.2f}**")

            st.markdown("#### Top 3 Predictions")
            col_l2, col_c2 = st.columns(2)
            for i, (l, c) in enumerate(top3_two):
                col_l2.write(f"**{i+1}.** {l}")
                col_c2.progress(int(c * 100))

else:
    st.warning("⬆️ Please upload an image to start dual prediction.")