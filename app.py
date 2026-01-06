import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os  

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# -------------------- LOAD MODEL --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "tomato_leaf_disease_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)


class_names = [
    'Bacterial Spot',
    'Early Blight',
    'Healthy',
    'Late Blight',
    'Leaf Mold',
    'Mosaic Virus',
    'Septoria Leaf Spot',
    'Spider Mites',
    'Target Spot',
    'Yellow Leaf Curl Virus'
]

disease_info = {
    "Bacterial Spot": {
        "desc": "A bacterial disease causing small dark spots on leaves.",
        "treatment": "Use disease-free seeds and copper-based bactericides."
    },
    "Early Blight": {
        "desc": "A fungal disease with concentric rings on older leaves.",
        "treatment": "Remove infected leaves and apply fungicides."
    },
    "Healthy": {
        "desc": "The leaf is healthy with no visible disease symptoms.",
        "treatment": "No treatment required. Maintain proper care."
    },
    "Late Blight": {
        "desc": "A severe fungal disease causing dark lesions.",
        "treatment": "Use resistant varieties and fungicides."
    },
    "Leaf Mold": {
        "desc": "Fungal disease causing yellow spots and mold growth.",
        "treatment": "Reduce humidity and improve ventilation."
    },
    "Mosaic Virus": {
        "desc": "Viral infection causing mottled leaf patterns.",
        "treatment": "Remove infected plants and control insects."
    },
    "Septoria Leaf Spot": {
        "desc": "Fungal disease with circular dark-bordered spots.",
        "treatment": "Avoid wet foliage and apply fungicides."
    },
    "Spider Mites": {
        "desc": "Pests causing yellowing and stippling of leaves.",
        "treatment": "Use insecticidal soap or natural predators."
    },
    "Target Spot": {
        "desc": "Fungal disease producing target-like spots.",
        "treatment": "Remove infected leaves and rotate crops."
    },
    "Yellow Leaf Curl Virus": {
        "desc": "Viral disease causing leaf curling and yellowing.",
        "treatment": "Control whiteflies and use resistant varieties."
    }
}

# -------------------- UI TEXT (LANGUAGE) --------------------
ui_text = {
    "English": {
        "title": "Tomato Leaf Disease Detection System",
        "upload": "Upload Tomato Leaf Image",
        "prediction": "Prediction Result",
        "confidence": "Prediction Confidence",
        "severity": "Severity Level",
        "disease_info": "Disease Information",
        "description": "Description",
        "treatment": "Recommended Action"
    },
    "Tamil": {
        "title": "தக்காளி இலை நோய் கண்டறிதல் அமைப்பு",
        "upload": "தக்காளி இலை படத்தை பதிவேற்றவும்",
        "prediction": "முன்னறிவு முடிவு",
        "confidence": "நம்பிக்கை அளவு",
        "severity": "தீவிர நிலை",
        "disease_info": "நோய் தகவல்",
        "description": "விளக்கம்",
        "treatment": "பரிந்துரைக்கப்பட்ட நடவடிக்கை"
    }
}

# -------------------- SIDEBAR --------------------
st.sidebar.title("🌱 Project Information")

language = st.sidebar.selectbox(
    "🌐 Select Language",
    ["English", "Tamil"]
)

st.sidebar.markdown("""
**Project:** Tomato Leaf Disease Detection  
**Technology:** Deep Learning (CNN)  
**Framework:** TensorFlow & Streamlit  
""")

# -------------------- MAIN TITLE --------------------
st.markdown(
    f"<h1 style='text-align:center;'>🍅 {ui_text[language]['title']}</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------- IMAGE UPLOAD --------------------
uploaded_file = st.file_uploader(
    f"📤 {ui_text[language]['upload']}",
    type=["jpg", "jpeg", "png"]
)

# -------------------- PREDICTION --------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Severity logic
    if confidence < 60:
        severity = "Low"
        sev_color = "#facc15"
    elif confidence < 85:
        severity = "Medium"
        sev_color = "#fb923c"
    else:
        severity = "High"
        sev_color = "#dc2626"

    st.markdown("---")

    # Result card
    st.markdown(
        f"""
        <div style="padding:20px;border-radius:10px;
        background-color:#e6f4ea;border-left:6px solid #2e7d32;color:#1b1b1b;">
        <h3>{ui_text[language]['prediction']}</h3>
        <p><b>Disease:</b> {predicted_class}</p>
        <p><b>Confidence:</b> {confidence:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Confidence bar
    st.markdown(f"### 📊 {ui_text[language]['confidence']}")
    st.progress(int(confidence))

    st.markdown(
        f"""
        <div style="margin-top:10px;padding:15px;border-radius:8px;
        background-color:#f9fafb;border-left:6px solid {sev_color};color:#111827;">
        <p><b>{ui_text[language]['severity']}:</b>
        <span style="color:{sev_color};font-weight:bold;"> {severity}</span></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Disease information
    info = disease_info[predicted_class]
    st.markdown(
        f"""
        <div style="margin-top:15px;padding:20px;border-radius:10px;
        background-color:#eef2ff;border-left:6px solid #1e40af;color:#111827;">
        <h4>{ui_text[language]['disease_info']}</h4>
        <p><b>{ui_text[language]['description']}:</b> {info['desc']}</p>
        <p><b>{ui_text[language]['treatment']}:</b> {info['treatment']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Academic Machine Learning Project</p>", unsafe_allow_html=True)
