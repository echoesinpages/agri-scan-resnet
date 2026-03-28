import streamlit as st
import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================

st.set_page_config(
    page_title="Agri-Scan Galaxy",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

page_bg_css = """
<style>
/* 1. Main Background: Deep Space Gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #000000, #0f0c29, #302b63, #24243e);
    color: white;
}

/* 2. Sidebar Background: Semi-transparent dark */
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* 3. Text Colors */
h1, h2, h3, h4, h5, h6, p, li, span, div, label {
    color: #e0e0e0 !important;
}

/* 4. Neon Glow Buttons */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 20px;
    border: none;
    box-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    box-shadow: 0 0 20px rgba(76, 175, 80, 0.8);
    transform: scale(1.05);
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# ==========================================
# 2. LOAD CLASSES & MODEL
# ==========================================

# FIX: Corrected to 38 classes — the actual count in the New Plant Diseases Dataset.
# These are sorted alphabetically, matching how ImageDataGenerator assigns class indices.
FALLBACK_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]


def get_class_names():
    """Try to load class names from dataset folder; fall back to hardcoded list."""
    search_paths = [
        'data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
        'data/train',
        'data',
    ]
    for path in search_paths:
        if os.path.exists(path):
            classes = sorted([
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
            ])
            if len(classes) > 10:
                return classes
    return FALLBACK_CLASSES


CLASS_NAMES = get_class_names()


@st.cache_resource
def load_model():
    """Load the trained model. Returns None if model.h5 is not found."""
    if not os.path.exists('model.h5'):
        return None
    return tf.keras.models.load_model('model.h5')


model = load_model()

# ==========================================
# 3. SIDEBAR
# ==========================================

with st.sidebar:
    st.title("✨ Agri-Scan")
    st.markdown("---")
    st.info("**How to use:**")
    st.markdown("1. 📸 Take a photo of a leaf.")
    st.markdown("2. 📤 Upload it.")
    st.markdown("3. 🤖 AI analyzes it.")

    st.markdown("---")
    if model is not None:
        st.success(f"✅ System Online\n({len(CLASS_NAMES)} Classes)")
    else:
        st.error("❌ Model not found.\nDownload model.h5 from the Releases page and place it in the project root.")

    st.markdown("---")
    st.caption("v2.1 Galaxy Edition")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================

st.title("🌿 Plant Disease AI")
st.markdown("#### Advanced Diagnosis System")

# FIX: Show a clear warning at the top if model is missing, before the uploader.
if model is None:
    st.warning(
        "⚠️ **Model not loaded.** Please download `model.h5` from the "
        "[Releases page](https://github.com/echoesinpages/agri-scan-resnet/releases) "
        "and place it in the project root folder, then restart the app."
    )

uploaded_file = st.file_uploader("Upload Leaf Photo", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # FIX: Guard against missing model before attempting prediction.
    if model is None:
        st.error("Cannot run prediction — model.h5 is missing. See the warning above.")
    else:
        c1, c2 = st.columns([1, 1])

        image = Image.open(uploaded_file).convert('RGB')

        with c1:
            st.image(image, caption='Uploaded Leaf', use_container_width=True)

        with c2:
            st.write("#### 🔍 Analysis")

            # Scanning animation
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            my_bar.empty()

            # FIX: Use resnet50.preprocess_input — matches training preprocessing in train.py.
            # Do NOT use img / 255.0 or rescale — that would give wrong predictions.
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            confidence = float(np.max(predictions)) * 100
            class_index = int(np.argmax(predictions))

            if class_index < len(CLASS_NAMES):
                predicted_label = CLASS_NAMES[class_index]
            else:
                predicted_label = f"Unknown class #{class_index}"

            # Results
            if confidence > 50:
                if "healthy" in predicted_label.lower():
                    st.snow()
                    st.success("### 🌱 Healthy Plant!")
                    st.metric("Confidence", f"{confidence:.1f}%")
                else:
                    st.error("### ⚠️ Disease Detected")
                    st.markdown(f"**{predicted_label.replace('___', ' → ')}**")
                    st.metric("Confidence", f"{confidence:.1f}%")
            else:
                st.warning("### ⚠️ Low Confidence")
                st.write(f"Best guess: **{predicted_label}** ({confidence:.1f}%)")
                st.write("Try uploading a clearer, closer photo of the leaf.")

        # Top-5 probability chart
        st.markdown("---")
        st.markdown("### 📊 Top 5 Predictions")

        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_labels = [CLASS_NAMES[i].replace('___', ' → ') for i in top_5_indices if i < len(CLASS_NAMES)]
        top_5_scores = [float(predictions[0][i]) * 100 for i in top_5_indices if i < len(CLASS_NAMES)]

        st.bar_chart(
            {"Disease": top_5_labels, "Probability (%)": top_5_scores},
            x="Disease",
            y="Probability (%)",
            color="#4CAF50"
        )
