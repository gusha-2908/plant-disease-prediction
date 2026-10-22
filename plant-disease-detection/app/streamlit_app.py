import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from PIL import Image

from src.gradcam import load_checkpoint, predict_with_cam

CKPT = "models/mobilenet_v2_best.pt"

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detector")
st.write("Upload a clear photo of a single leaf to get a diagnosis.")


@st.cache_resource
def load():
    return load_checkpoint(CKPT)


model, classes, name = load()
file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    label, conf, overlay = predict_with_cam(model, name, classes, img)
    c1, c2 = st.columns(2)
    c1.image(img, caption="Uploaded", use_container_width=True)
    c2.image(overlay, caption="Model focus (Grad-CAM)", use_container_width=True)
    st.subheader(label.replace("___", " - ").replace("_", " "))
    st.progress(conf, text=f"Confidence: {conf:.1%}")
    if conf < 0.6:
        st.warning("Low confidence. Try a clearer, well-lit photo of one leaf.")
