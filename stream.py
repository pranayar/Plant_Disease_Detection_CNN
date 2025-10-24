import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import base64
import time

# -------------------------------
# 1️⃣ Model + Labels
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_cnn7_best.keras", compile=False)

model = load_model()

LABELS = [
    "Apple - Apple Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust", "Apple - Healthy",
    "Blueberry - Healthy", "Cherry - Powdery Mildew", "Cherry - Healthy",
    "Corn (Maize) - Cercospora Leaf Spot / Gray Leaf Spot", "Corn (Maize) - Common Rust",
    "Corn (Maize) - Northern Leaf Blight", "Corn (Maize) - Healthy",
    "Grape - Black Rot", "Grape - Black Measles (Esca)", "Grape - Leaf Blight (Isariopsis Leaf Spot)", "Grape - Healthy",
    "Orange - Citrus Greening (Huanglongbing)", "Peach - Bacterial Spot", "Peach - Healthy",
    "Bell Pepper - Bacterial Spot", "Bell Pepper - Healthy",
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy",
    "Raspberry - Healthy", "Soybean - Healthy", "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch", "Strawberry - Healthy",
    "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight",
    "Tomato - Leaf Mold", "Tomato - Septoria Leaf Spot", "Tomato - Spider Mites (Two-Spotted Spider Mite)",
    "Tomato - Target Spot", "Tomato - Yellow Leaf Curl Virus", "Tomato - Mosaic Virus", "Tomato - Healthy"
]

IMG_SIZE = 224

# -------------------------------
# 2️⃣ Page Configuration
# -------------------------------
st.set_page_config(page_title="🌿 GreenLeaf AI", page_icon="🌱", layout="wide")

st.markdown("""
<style>
/* Background and typography */
body {
    background: linear-gradient(135deg, #E8F5E9 0%, #E3F2FD 100%);
    font-family: 'Poppins', sans-serif;
    color: #2E7D32;
}
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    margin-top: 1rem;
    background: -webkit-linear-gradient(#2E7D32, #388E3C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.sidebar .sidebar-content {
    background: #F1F8E9;
}
.prediction-card {
    background-color: white;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    padding: 25px;
    text-align: center;
    margin-top: 20px;
}
.confidence-bar {
    height: 10px;
    border-radius: 5px;
    background: linear-gradient(90deg, #81C784, #2E7D32);
}
.camera-frame {
    display: flex;
    justify-content: center;
    margin-top: 15px;
}
.camera-feed-img {
    width: 75%;
    max-width: 640px;
    border-radius: 20px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 3️⃣ Header Section
# -------------------------------
st.markdown("<h1 class='main-title'>🌿 GreenLeaf AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Real-Time Plant Disease Detection</p>", unsafe_allow_html=True)

# -------------------------------
# 4️⃣ Sidebar Navigation
# -------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/619/619153.png", width=100)
    st.header("🌱 Navigation")
    mode = st.radio("Select Mode", ["🖼️ Upload Image", "🎥 Real-Time Camera"])
    st.markdown("---")
    st.markdown("**About GreenLeaf**")
    st.info("""
        GreenLeaf uses a deep learning model to detect **38+ plant diseases** 
        from crops like Apple, Tomato, Corn, Potato, and more.
        Upload an image or use your webcam for instant detection.
    """)

# -------------------------------
# 5️⃣ Prediction Function
# -------------------------------
def predict(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(x, verbose=0)[0]
    idx = np.argmax(preds)
    conf = np.max(preds) * 100
    return LABELS[idx], conf, preds

# -------------------------------
# 6️⃣ Upload Mode
# -------------------------------
if mode == "🖼️ Upload Image":
    st.subheader("📸 Upload a Leaf Image for Analysis")
    uploaded = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(img, caption="Uploaded Image", width=300)

        with col2:
            st.write("🧠 **Analyzing with Deep Learning...**")
            time.sleep(0.5)
            label, conf, _ = predict(img)

            st.markdown(f"""
                <div class='prediction-card'>
                    <h3>🌿 Prediction: {label}</h3>
                    <p style='font-size:1.2rem;'>Confidence: {conf:.2f}%</p>
                    <div class='confidence-bar' style='width:{conf}%;'></div>
                </div>
            """, unsafe_allow_html=True)

# -------------------------------
# 7️⃣ Real-Time Detection Mode
# -------------------------------
else:
    st.subheader("🎥 Real-Time Detection")
    st.info("Click below to start your camera and begin AI-based live detection.")

    run = st.toggle("📷 Start Real-Time Camera")
    camera_feed = st.empty()
    prediction_area = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("⚠️ Could not access webcam.")
            run = False

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            label, conf, _ = predict(img)

            cv2.putText(frame_rgb, f"{label} ({conf:.1f}%)",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (46, 204, 113), 3)

            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            b64 = base64.b64encode(buffer).decode()
            html = f"<div class='camera-frame'><img src='data:image/jpg;base64,{b64}' class='camera-feed-img'/></div>"

            camera_feed.markdown(html, unsafe_allow_html=True)
            prediction_area.markdown(f"""
                <div class='prediction-card'>
                    <h3>{label}</h3>
                    <p>Confidence: {conf:.2f}%</p>
                    <div class='confidence-bar' style='width:{conf}%;'></div>
                </div>
            """, unsafe_allow_html=True)

        cap.release()
