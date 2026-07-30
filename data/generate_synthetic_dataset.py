"""
Synthetic droplet-detection dataset generator (YOLO format).

Generates fully synthetic images of ellipsoidal "droplets" on a textured
lichen-like surface, with corresponding YOLO-format bounding-box labels
(class 'gota', single class) — reproducing the general structure and
physical realism of a real droplet-hydrophobicity video-annotation
project, WITHOUT using any real photos, videos, or specimen data.

Physical realism choices (matching the reasoning in the original project):
  - Droplets are rendered as ellipses flattened at the bottom (gravity
    effect: a real droplet on a surface is not a perfect circle).
  - Background texture varies in brightness/contrast to simulate
    different lighting/surface conditions across "sessions".
  - 1-4 droplets per image, non-overlapping, at random positions/sizes.

Run with: python data/generate_synthetic_dataset.py
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

random.seed(42)
np.random.seed(42)

IMG_SIZE = 416
N_IMAGES = {"train": 140, "val": 40, "test": 20}


def make_background(size=IMG_SIZE):
    """Textured, mottled background simulating a lichen surface under
    variable lighting — Perlin-ish noise via layered random blobs."""
    base_gray = random.randint(90, 160)
    img = Image.new("RGB", (size, size), (base_gray, base_gray - 10, base_gray - 25))
    draw = ImageDraw.Draw(img)

    # Mottled texture: random soft blobs of slightly different tone
    for _ in range(random.randint(25, 45)):
        x, y = random.randint(0, size), random.randint(0, size)
        r = random.randint(10, 40)
        tone_shift = random.randint(-25, 25)
        color = tuple(np.clip(np.array(img.getpixel((min(x, size-1), min(y, size-1)))) + tone_shift, 0, 255).tolist())
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    img = img.filter(ImageFilter.GaussianBlur(radius=3))

    # Global lighting variation across "sessions"
    brightness_shift = random.uniform(0.75, 1.25)
    arr = np.array(img).astype(np.float32) * brightness_shift
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def draw_droplet(draw, cx, cy, a, b, base_tone):
    """Draw a gravity-flattened droplet (ellipse, slightly flatter at
    the bottom) with a soft highlight to look mildly 3D/reflective."""
    # Slightly flatten the bottom by drawing two half-ellipses of
    # different vertical radius (top vs bottom)
    b_top = b * 1.05
    b_bottom = b * 0.85

    fill = tuple(np.clip(np.array(base_tone) - 40, 0, 255).astype(int).tolist())
    outline = tuple(np.clip(np.array(base_tone) - 70, 0, 255).astype(int).tolist())

    draw.ellipse([cx - a, cy - b_top, cx + a, cy + b_bottom], fill=fill, outline=outline, width=2)

    # Small highlight (specular-ish) for visual realism
    hl_r = max(2, int(a * 0.25))
    draw.ellipse([cx - a * 0.35 - hl_r, cy - b_top * 0.4 - hl_r,
                  cx - a * 0.35 + hl_r, cy - b_top * 0.4 + hl_r],
                 fill=tuple(np.clip(np.array(base_tone) + 60, 0, 255).astype(int).tolist()))


def generate_image(size=IMG_SIZE):
    img = make_background(size)
    draw = ImageDraw.Draw(img)
    sample_px = np.array(img)[size // 4, size // 4]

    n_droplets = random.randint(1, 4)
    boxes = []
    attempts = 0
    while len(boxes) < n_droplets and attempts < 30:
        attempts += 1
        a = random.randint(18, 45)  # semi-axis horizontal
        b = random.randint(12, 35)  # semi-axis vertical (droplets wider than tall -> gravity flattening)
        cx = random.randint(a + 5, size - a - 5)
        cy = random.randint(b + 5, size - b - 5)

        # avoid heavy overlap with existing droplets
        overlap = any(
            abs(cx - bx) < (a + ba) * 0.8 and abs(cy - by) < (b + bb) * 0.8
            for (bx, by, ba, bb) in boxes
        )
        if overlap:
            continue

        boxes.append((cx, cy, a, b))

    for (cx, cy, a, b) in boxes:
        draw_droplet(draw, cx, cy, a, b, sample_px)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))  # slight softening, camera-like

    # YOLO format: class cx_norm cy_norm w_norm h_norm
    labels = []
    for (cx, cy, a, b) in boxes:
        w, h = 2 * a, 2 * b * 1.0  # bounding box height approximated from vertical extent
        labels.append(f"0 {cx/size:.6f} {cy/size:.6f} {w/size:.6f} {h/size:.6f}")

    return img, labels


def build_dataset(output_dir="data"):
    out = Path(output_dir)
    for split, n in N_IMAGES.items():
        img_dir = out / split / "images"
        lbl_dir = out / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            img, labels = generate_image()
            fname = f"{split}_{i:04d}"
            img.save(img_dir / f"{fname}.jpg", quality=90)
            with open(lbl_dir / f"{fname}.txt", "w") as f:
                f.write("\n".join(labels))

    # data.yaml for YOLO training
    yaml_content = f"""path: {out.resolve()}
train: train/images
val: val/images
test: test/images

names:
  0: gota
"""
    with open(out / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Synthetic dataset created in '{out}': " +
          ", ".join(f"{split}={n}" for split, n in N_IMAGES.items()))


if __name__ == "__main__":
    build_dataset()
