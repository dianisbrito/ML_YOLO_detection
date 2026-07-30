"""
Gradio demo app — droplet detection & hydrophobicity-proxy metrics.

Loads the trained YOLOv8 model and lets the user run inference on a test
image (or upload their own), visualizing detections and computing the
geometric contact-angle-proxy metric per droplet — reproducing the
interactive-demo component of the original project.

Run with: python app.py
"""

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from src.geometry import compute_droplet_geometry

MODEL_PATH = Path(__file__).parent / "runs" / "detect" / "train" / "weights" / "best.pt"
TEST_IMAGES_DIR = Path(__file__).parent / "data" / "test" / "images"

_model_cache = {}


def get_model():
    if "model" not in _model_cache:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run `python src/train.py` first."
            )
        _model_cache["model"] = YOLO(str(MODEL_PATH))
    return _model_cache["model"]


def detect_droplets(image: Image.Image, conf_threshold: float = 0.25):
    model = get_model()
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

    summary = (
        f"**{len(rows)} droplet(s) detected**"
        + (f" · mean contact-angle proxy: {df['contact_angle_proxy (deg)'].mean():.1f}°" if rows else "")
    )

    return annotated, df, summary


def build_interface():
    example_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))[:6] if TEST_IMAGES_DIR.exists() else []

    with gr.Blocks(title="Droplet Detection — YOLOv8 Demo") as demo:
        gr.Markdown(
            "# 💧 Droplet Detection & Hydrophobicity-Proxy Metrics\n"
            "Demo of a YOLOv8 model trained to detect droplets and derive a geometric "
            "contact-angle-proxy metric from each detection's bounding box. "
            "**All images here are synthetic** — see the README for details.\n"
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input image")
                conf_slider = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
                run_btn = gr.Button("Detect droplets", variant="primary")
                if example_images:
                    gr.Examples(examples=[[str(p)] for p in example_images], inputs=image_input)
            with gr.Column():
                image_output = gr.Image(type="pil", label="Detections")
                summary_output = gr.Markdown()
                table_output = gr.Dataframe(label="Per-droplet geometry")

        run_btn.click(
            fn=detect_droplets,
            inputs=[image_input, conf_slider],
            outputs=[image_output, table_output, summary_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
