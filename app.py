"""
Streamlit demo app — droplet detection & hydrophobicity-proxy metrics.

Loads the trained YOLOv8 model and lets the user run inference on a test
image (or upload their own), visualizing detections and computing the
geometric contact-angle-proxy metric per droplet.

Note: the original project's interactive demo was built in Gradio (see
notebooks/droplet_detection_pipeline.ipynb for that version's code) —
this Streamlit version exists purely for free, reliable public hosting
via Streamlit Community Cloud, since Hugging Face Spaces now requires a
paid plan for Gradio apps on CPU Basic hardware.

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent / "src"))
from geometry import compute_droplet_geometry

st.set_page_config(page_title="Droplet Detection — YOLOv8 Demo", layout="wide")

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"
TEST_IMAGES_DIR = Path(__file__).parent / "data" / "test" / "images"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Trained model not found at {MODEL_PATH}.")
        st.stop()
    return YOLO(str(MODEL_PATH))


def detect_droplets(image: Image.Image, conf_threshold: float):
    model = load_model()
    results = model.predict(image, conf=conf_threshold, verbose=False)[0]

    annotated = Image.fromarray(results.plot()[:, :, ::-1])  # BGR -> RGB

    rows = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x1, y2 - y1
        geom = compute_droplet_geometry(w, h)
        rows.append({
            "confidence": round(float(box.conf[0]), 3),
            "width_px": round(w, 1),
            "height_px": round(h, 1),
            "aspect_ratio (b/a)": round(geom.aspect_ratio, 3),
            "contact_angle_proxy (deg)": round(geom.contact_angle_proxy_deg, 1),
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["confidence", "width_px", "height_px", "aspect_ratio (b/a)", "contact_angle_proxy (deg)"]
    )
    return annotated, df


st.title("💧 Droplet Detection & Hydrophobicity-Proxy Metrics")
st.caption(
    "Demo of a YOLOv8 model trained to detect droplets and derive a geometric "
    "contact-angle-proxy metric from each detection's bounding box. "
    "**All images here are synthetic** — see the README for details."
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    source = st.radio("Image source", ["Use a test example", "Upload my own"], horizontal=True)

    image = None
    if source == "Use a test example":
        if TEST_IMAGES_DIR.exists():
            example_files = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
            if example_files:
                choice = st.selectbox("Choose an example", [p.name for p in example_files])
                image = Image.open(TEST_IMAGES_DIR / choice)
        else:
            st.warning("No test images found in data/test/images/.")
    else:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(uploaded)

    conf_threshold = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)

    if image is not None:
        st.image(image, caption="Input image", use_container_width=True)

with col2:
    st.subheader("Detections")
    if image is not None:
        annotated, df = detect_droplets(image, conf_threshold)
        st.image(annotated, caption="Detected droplets", use_container_width=True)

        n = len(df)
        if n > 0:
            st.markdown(
                f"**{n} droplet(s) detected** · mean contact-angle proxy: "
                f"{df['contact_angle_proxy (deg)'].mean():.1f}°"
            )
        else:
            st.markdown("**No droplets detected** at this confidence threshold.")

        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Select or upload an image to run detection.")
