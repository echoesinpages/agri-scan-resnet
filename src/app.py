import streamlit as st
import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image

# 1. PAGE CONFIGURATION & THEME

st.set_page_config(
    page_title="Agri-Scan Galaxy",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# BackGround (Dark Space Gradient) 
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
    backdrop-filter: blur(10px); /* Frosted glass effect */
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* 3. Text Colors: Force white text to pop against dark mode */
h1, h2, h3, h4, h5, h6, p, li, span, div, label {
    color: #e0e0e0 !important;
}

/* 4. Make buttons look cool (Neon Glow) */
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

# 2. LOAD CLASSES & MODEL

FALLBACK_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def get_class_names():
    search_paths = ['data', 'data/train', 'data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train']
    for path in search_paths:
        if os.path.exists(path):
            classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            if len(classes) > 10:
                return classes
    return FALLBACK_CLASSES

CLASS_NAMES = get_class_names()

@st.cache_resource
def load_model():
    if not os.path.exists('model.h5'):
        return None
    return tf.keras.models.load_model('model.h5')

model = load_model()

# 3. SIDEBAR DASHBOARD

with st.sidebar:
    st.title("✨ Agri-Scan")
    st.markdown("---")
    st.info(" **How to use:**")
    st.markdown("1. 📸 Take a photo of a leaf.")
    st.markdown("2. 📤 Upload it.")
    st.markdown("3. 🤖 AI analyzes it.")
    
    st.markdown("---")
    if model:
        st.success(f"✅ System Online\n({len(CLASS_NAMES)} Classes)")
    else:
        st.error("❌ Brain Missing")
    
    st.markdown("---")
    st.caption("v2.0 Galaxy Edition")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================

st.title("🌿 Plant Disease AI")
st.markdown("#### Advanced Diagnosis System")

uploaded_file = st.file_uploader("Upload Leaf Photo", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    c1, c2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file).convert('RGB')
    
    with c1:
        st.image(image, caption='Sample', use_container_width=True)

    with c2:
        st.write("#### 🔍 Analysis")
        
        # Fake "Scanning" Animation
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        my_bar.empty()

        # PREDICTION
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        confidence = np.max(predictions) * 100
        class_index = np.argmax(predictions)
        predicted_label = CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else f"Unknown #{class_index}"

        # --- RESULTS ---
        if confidence > 50:
            if "healthy" in predicted_label.lower():
                st.snow() # <--- Falling "Star Dust" animation
                st.success(f"### 🌱 Healthy Plant!")
                st.metric("Confidence", f"{confidence:.1f}%")
            else:
                st.error(f"### ⚠️ Disease Found")
                st.markdown(f"**{predicted_label}**")
                st.metric("Confidence", f"{confidence:.1f}%")
        else:
            st.warning("### ⚠️ Uncertain")
            st.write(f"Best guess: **{predicted_label}** ({confidence:.1f}%)")

    # --- CHART SECTION ---
    st.markdown("---")
    st.markdown("### 📊 Probability Data")
    
    # Sort top 5 predictions
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_labels = [CLASS_NAMES[i] for i in top_5_indices]
    top_5_scores = [predictions[0][i] for i in top_5_indices]

    st.bar_chart({"Disease": top_5_labels, "Probability": top_5_scores}, x="Disease", y="Probability", color="#ffffff")
