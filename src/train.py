"""
YOLOv8 training script for the synthetic droplet-detection dataset.

Reproduces the augmentation philosophy of the original real-data project:
every augmentation parameter is chosen deliberately based on the physical
plausibility of the object being detected (a gravity-affected liquid
droplet), not left at library defaults. See inline comments for the
reasoning behind each choice.

Run with: python src/train.py
"""

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = str(Path(__file__).parent.parent / "data" / "data.yaml")


def train_model(epochs=30, imgsz=416, batch=16):
    model = YOLO("yolov8n.pt")  # nano: appropriate for a small, single-class dataset

    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=15,   # early stopping: with a small dataset, further epochs
                       # past a plateau just risk overfitting rather than helping

        # ---------------------------------------------------------------
        # Photometric augmentations (color / lighting)
        # ---------------------------------------------------------------
        hsv_h=0.010,   # hue variation: kept LOW. Droplet/surface color is a
                       # genuinely informative signal here, not noise — heavy
                       # hue distortion would remove information the model
                       # could legitimately use.
        hsv_s=0.5,     # saturation variation: helps generalize across droplets
                       # that appear more or less reflective/saturated depending
                       # on lighting angle.
        hsv_v=0.4,     # brightness variation: the MOST important photometric
                       # augmentation here — real capture sessions have very
                       # different ambient light, and the model should not
                       # depend on one exact brightness level.

        # ---------------------------------------------------------------
        # Geometric augmentations
        # ---------------------------------------------------------------
        degrees=5.0,     # small rotation only (+/-5°): the camera may not be
                         # perfectly level between sessions, but large rotations
                         # have no physical justification (the droplet is always
                         # viewed "from the side", under gravity).
        translate=0.1,   # random shift — the droplet can land anywhere in frame.
        scale=0.3,       # random zoom — simulates different camera distances
                         # or droplet sizes.
        shear=0.0,       # NO shear: a droplet does not physically shear.
        perspective=0.0, # NO perspective warp: fixed side-on camera assumed,
                         # no varying viewing angle to simulate.

        flipud=0.0,      # NEVER vertical flip. Gravity defines the droplet's
                         # shape (flatter at the bottom, curved at the top);
                         # flipping it vertically creates a physically
                         # impossible geometry that would corrupt training.
        fliplr=0.5,      # horizontal flip IS valid: a droplet is approximately
                         # left-right symmetric, so this is "free", physically
                         # valid augmentation.

        # ---------------------------------------------------------------
        # Compositional augmentations (image mixing)
        # ---------------------------------------------------------------
        mosaic=0.5,      # 4-image mosaic, at reduced probability (default is
                         # 1.0): with a small, single-object-type dataset (no
                         # complex multi-object scenes), full-strength mosaic
                         # risks generating unrealistic compositions.
        mixup=0.0,       # disabled: alpha-blending two images tends to confuse
                         # a model detecting a specific-texture object like a
                         # droplet rather than help.
        copy_paste=0.0,  # disabled: designed for scenes with multiple distinct
                         # object types; not useful for a single-class dataset.

        close_mosaic=max(1, epochs // 3),  # disable mosaic for the final third
                         # of training, so the model finishes fine-tuning on
                         # "real" (non-composited) image statistics.

        verbose=True,
    )
    return model, results


if __name__ == "__main__":
    model, results = train_model()
    print("Training complete. Best weights saved under runs/detect/train/weights/best.pt")
