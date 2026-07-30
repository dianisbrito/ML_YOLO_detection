# Droplet Detection & Hydrophobicity-Proxy Metrics — YOLOv8

**Author:** Diana Brito Hoyos — Biologist & Biostatistician | Data Analyst

📓 **[Full pipeline notebook](./notebooks/droplet_detection_pipeline.ipynb)** — data generation, training, evaluation, and geometry extraction, with real (executed) outputs

An end-to-end computer-vision pipeline: training a YOLOv8 object detector to identify liquid droplets, evaluating it with standard object-detection metrics, and extracting a geometric hydrophobicity-proxy metric from each detection — plus an interactive Streamlit demo.

> ⚠️ **Note on data and origin:** This project reproduces the methodology of a real droplet-hydrophobicity video-analysis pipeline I built (originally for lichen surface wetting characterization), where droplets were tracked across real video recordings, labeled via Roboflow, and analyzed with a trained YOLO model. **That original video/image data is not public.** Everything in this repository — every training image, its bounding-box labels, and the trained model — is generated and trained **from scratch on fully synthetic data** (see [`data/generate_synthetic_dataset.py`](./data/generate_synthetic_dataset.py)). No real specimen photos, videos, or Roboflow-annotated data are used anywhere. All reported metrics (precision, recall, mAP) are genuine — computed on synthetic images the model never saw during training, not simulated or invented numbers.

---

## Why this demonstrates ML engineering expertise

- **A real, trained model** — not just code that would theoretically work. YOLOv8 was fine-tuned in this repository's own CI-reproducible pipeline, achieving mAP50 = 0.995 and mAP50-95 = 0.879 on a held-out synthetic test set.
- **Physically-reasoned data augmentation**, not library defaults: every augmentation parameter (`hsv_v`, `flipud`, `shear`, `mosaic`, etc.) is chosen and documented for a specific physical reason related to how a droplet actually behaves under gravity — see [`src/train.py`](./src/train.py) for the fully annotated configuration.
- **Standard, correctly-interpreted detection metrics**: precision, recall, mAP@50, and the stricter mAP@50-95 (averaged across IoU thresholds), evaluated on a genuinely held-out test split — plus the automatically-generated PR curve and confusion matrix.
- **Downstream geometric feature engineering**: detections aren't just boxes — [`src/geometry.py`](./src/geometry.py) converts each bounding box into an interpretable, domain-relevant hydrophobicity-proxy metric (a contact-angle approximation from box aspect ratio), the actual scientific goal of the original project.
- **A usable interactive demo** ([`app.py`](./app.py), Streamlit) — not just a model checkpoint, but a tool a non-technical user could point at an image and get results from.

## Project structure

```
lichen-droplet-detection-yolo/
├── README.md
├── app.py                              ← Streamlit interactive demo (see notebooks/ for the original Gradio version)
├── data/
│   └── generate_synthetic_dataset.py   ← synthetic droplet image + YOLO-label generator
├── src/
│   ├── train.py                        ← YOLOv8 training (documented augmentation reasoning)
│   └── geometry.py                     ← contact-angle-proxy geometric feature extraction
├── models/
│   └── best.pt                         ← trained YOLOv8 weights
└── notebooks/
    └── droplet_detection_pipeline.ipynb ← full pipeline, executed end-to-end
```

## Running locally

```bash
pip install -r requirements.txt

# 1. Generate the synthetic dataset (skipped automatically if data/ already exists)
python data/generate_synthetic_dataset.py

# 2. Train (skipped if models/best.pt already exists — see notebooks/ for the full run)
python src/train.py

# 3. Launch the interactive demo
streamlit run app.py
```

## Tech stack

`Python` · `ultralytics` (YOLOv8) · `PyTorch` · `Streamlit` · `PIL` / `NumPy` · `pandas` · Jupyter

## About the author

Biologist with 12+ years of experience integrating remote sensing, advanced statistics, and scientific programming across public health, ecology, agriculture, and environmental research — including computer-vision pipelines (YOLO) for animal behavior and morphological/physical characterization. See full profile on [GitHub](https://github.com/dianisbrito) · [LinkedIn](TU-LINK-AQUI).
