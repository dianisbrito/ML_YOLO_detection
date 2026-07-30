"""
Geometric feature extraction from YOLO detections — droplet
hydrophobicity proxy metrics.

Reproduces the geometric reasoning from the original project: each
detected droplet's bounding box is treated as approximating an ellipse
(semi-axis a = width/2, semi-axis b = height/2). A contact-angle PROXY is
derived from the aspect ratio of that ellipse — not a true physical
contact angle measurement, but a fast, camera-agnostic indicator of how
"flattened" (hydrophilic-leaning) vs. "rounded" (hydrophobic-leaning) a
droplet appears, trackable frame-to-frame.

    geometric_angle(phi) = arctan( (height/width) * tan(phi) )

At the reference phi=45°, this reduces to arctan(height/width):
  - A circular droplet (height ≈ width) gives ~45°.
  - A flattened, spread droplet (height << width) approaches 0°.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DropletGeometry:
    a: float  # semi-axis horizontal (width / 2)
    b: float  # semi-axis vertical (height / 2)
    contact_angle_proxy_deg: float  # arctan(b/a) in degrees, the phi=45 reference case
    aspect_ratio: float  # b / a


def compute_droplet_geometry(box_width: float, box_height: float) -> DropletGeometry:
    """Compute geometric hydrophobicity-proxy metrics from a single
    YOLO bounding box (width, height in any consistent unit — pixels
    or normalized)."""
    a = box_width / 2
    b = box_height / 2
    aspect_ratio = b / a if a > 0 else np.nan
    contact_angle_proxy = math.degrees(math.atan(aspect_ratio)) if a > 0 else np.nan

    return DropletGeometry(a=a, b=b, contact_angle_proxy_deg=contact_angle_proxy, aspect_ratio=aspect_ratio)


def geometric_angle_at_phi(box_width: float, box_height: float, phi_deg: float) -> float:
    """General form: geometric angle at an arbitrary reference angle phi
    (not just the phi=45 special case), matching the original project's
    multi-angle characterization (30°, 45°, 60° reference rays)."""
    aspect_ratio = (box_height / box_width) if box_width > 0 else np.nan
    phi_rad = math.radians(phi_deg)
    return math.degrees(math.atan(aspect_ratio * math.tan(phi_rad)))


def summarize_detections(boxes_wh: list[tuple[float, float]]) -> dict:
    """Summarize hydrophobicity-proxy geometry across multiple detections
    (e.g. all droplets in a single frame, or a single droplet tracked
    across frames)."""
    geoms = [compute_droplet_geometry(w, h) for (w, h) in boxes_wh]
    angles = [g.contact_angle_proxy_deg for g in geoms]
    aspect_ratios = [g.aspect_ratio for g in geoms]

    return {
        "n_droplets": len(geoms),
        "mean_contact_angle_proxy_deg": float(np.mean(angles)) if angles else np.nan,
        "sd_contact_angle_proxy_deg": float(np.std(angles)) if angles else np.nan,
        "mean_aspect_ratio": float(np.mean(aspect_ratios)) if aspect_ratios else np.nan,
        "geometries": geoms,
    }
