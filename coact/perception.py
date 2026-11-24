"""Per-camera perception: detections come out of pixels, not out of the simulator.

Each agent runs this on ITS OWN rendered camera image. The output is a list of
detections in that camera's image frame — nothing is read back from MuJoCo's
state. That matters for the research claim: if both agents were handed the same
world-frame object list, "viewpoint disagreement" would be a scripted effect
rather than a measured one. Here each agent genuinely sees a different picture
and genuinely assigns "left" differently.

This is classical CV (colour segmentation + contour geometry), not a VLM. It is
a deliberate trade: it is reproducible, it needs no network at demo time, and
it keeps the contribution squarely on the consensus gate. Swapping in a VLM
detector means replacing `detect()` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import math

import cv2
import numpy as np

# Table surface is a mid-brown; plates are off-white; cups are saturated.
# Area thresholds, expressed at the reference resolution below and scaled to
# whatever the camera actually renders. Absolute pixel counts would silently
# break the detector on a smaller frame — halve the resolution and every blob
# falls under the floor.
REF_W, REF_H = 640, 480
MIN_AREA = 420          # px at REF_W x REF_H; rejects speckle and distant clutter
MAX_AREA = 90_000       # px at REF_W x REF_H; rejects the table itself


def _area_bounds(shape) -> tuple[float, float]:
    scale = (shape[0] * shape[1]) / float(REF_W * REF_H)
    return MIN_AREA * scale, MAX_AREA * scale


@dataclass
class Detection:
    """One object as THIS camera sees it. All geometry is image-frame."""

    id: str                 # provisional, assigned by the detector
    label: str              # plate | cup
    color: str
    cx: float               # centroid x in pixels
    cy: float
    area_px: float
    u: float                # cx normalised to [-1, +1]; -1 is the image's left edge
    v: float                # cy normalised; -1 is the top
    apparent_size: float    # area normalised by the largest detection in this view
    left_px: tuple[float, float] = (0.0, 0.0)    # silhouette extremes, for metric sizing
    right_px: tuple[float, float] = (0.0, 0.0)
    metric_radius_m: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _masks(bgr: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Crockery: bright and desaturated. The robot's white plastic is the main
    # confuser, so it is rejected later on contour shape, not on colour.
    white = ((s < 46) & (v > 150)).astype(np.uint8) * 255
    red = (((h < 8) | (h > 172)) & (s > 110) & (v > 60)).astype(np.uint8) * 255
    blue = ((h > 100) & (h < 130) & (s > 110) & (v > 50)).astype(np.uint8) * 255
    green = ((h > 40) & (h < 85) & (s > 90) & (v > 50)).astype(np.uint8) * 255

    k = np.ones((5, 5), np.uint8)
    out = {}
    for name, m in (("white", white), ("red", red), ("blue", blue), ("green", green)):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        out[name] = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return out


def _contours(mask: np.ndarray) -> list[np.ndarray]:
    lo, hi = _area_bounds(mask.shape)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if lo < cv2.contourArea(c) < hi]


def _is_plate(c: np.ndarray) -> bool:
    """A plate reads as a wide, solid, elliptical blob.

    The arm's white plastic is bright and desaturated too, so colour alone is
    not enough. Links are thin and ragged: they fail on solidity and on the
    width:height ratio of a disc seen at a shallow angle.
    """
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    solidity = area / max(cv2.contourArea(hull), 1.0)
    x, y, w, h = cv2.boundingRect(c)
    aspect = w / max(h, 1)
    fill = area / max(w * h, 1)
    return solidity > 0.68 and 0.60 < aspect < 5.5 and fill > 0.48


def _is_cup(c: np.ndarray, shape) -> bool:
    x, y, w, h = cv2.boundingRect(c)
    aspect = w / max(h, 1)
    area = cv2.contourArea(c)
    return 0.35 < aspect < 2.2 and area > _area_bounds(shape)[0]


def detect(rgb: np.ndarray) -> list[Detection]:
    """Detect crockery in one camera image. Returns image-frame detections."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    masks = _masks(bgr)
    raw: list[tuple[str, str, np.ndarray]] = []

    for c in _contours(masks["white"]):
        if _is_plate(c):
            raw.append(("plate", "white", c))
    for color in ("red", "blue", "green"):
        for c in _contours(masks[color]):
            if _is_cup(c, bgr.shape):
                raw.append(("cup", color, c))

    dets: list[Detection] = []
    for label, color, c in raw:
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        area = float(cv2.contourArea(c))
        pts = c.reshape(-1, 2)
        left = pts[int(np.argmin(pts[:, 0]))]
        right = pts[int(np.argmax(pts[:, 0]))]
        dets.append(
            Detection(
                id="", label=label, color=color, cx=cx, cy=cy, area_px=area,
                u=(cx / W) * 2 - 1, v=(cy / H) * 2 - 1, apparent_size=0.0,
                left_px=(float(left[0]), float(left[1])),
                right_px=(float(right[0]), float(right[1])),
            )
        )

    # Normalise apparent size within each category, so "large"/"small" is a
    # judgement this camera makes about what it can see — which is exactly how
    # a size term should behave for an embodied agent.
    for label in {d.label for d in dets}:
        group = [d for d in dets if d.label == label]
        biggest = max(d.area_px for d in group)
        for d in group:
            d.apparent_size = d.area_px / biggest

    dets.sort(key=lambda d: (d.label, -d.u))
    for i, d in enumerate(dets):
        d.id = f"{d.label}_{i}"
    return dets


def size_word(d: Detection, group: list[Detection]) -> str:
    """Call a detection large or small relative to its category in this view."""
    same = [g for g in group if g.label == d.label]
    if len(same) < 2:
        return "medium"
    biggest, smallest = max(same, key=lambda g: g.area_px), min(same, key=lambda g: g.area_px)
    if biggest.area_px / max(smallest.area_px, 1) < 1.25:
        return "medium"          # too close to call from this viewpoint
    if d is biggest:
        return "large"
    if d is smallest:
        return "small"
    return "medium"


def annotate(rgb: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Draw detections for the dashboard and for debugging."""
    img = rgb.copy()
    for d in dets:
        col = (80, 220, 240) if d.label == "plate" else (240, 150, 90)
        cv2.circle(img, (int(d.cx), int(d.cy)), 9, col, 2)
        cv2.putText(img, f"{d.label}:{size_word(d, dets)}", (int(d.cx) - 40, int(d.cy) - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------
# Cross-view correspondence
# --------------------------------------------------------------------------
# Each agent grounds language in its OWN image frame — that is where the
# disagreement comes from and it must stay that way. But the consensus gate has
# to decide whether two agents picked the same physical object, which needs a
# shared frame. We get one the way any calibrated multi-camera rig does: known
# extrinsics plus a known support plane. Each detection's centroid is a ray;
# intersect it with the table surface and you have a metric estimate of where
# that object is, derived from the image, not read out of the simulator.

TABLE_Z = 0.736


def backproject(cx: float, cy: float, cam_pos: np.ndarray, cam_mat: np.ndarray,
                fovy_deg: float, width: int, height: int,
                plane_z: float = TABLE_Z) -> np.ndarray:
    """Pixel -> world (x, y) on the table plane, via the camera's pose."""
    f = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    ray_cam = np.array([cx - width / 2.0, -(cy - height / 2.0), -f], dtype=float)
    ray_cam /= np.linalg.norm(ray_cam)
    ray_world = cam_mat.reshape(3, 3) @ ray_cam
    if abs(ray_world[2]) < 1e-9:
        return np.array([np.nan, np.nan])
    t = (plane_z - cam_pos[2]) / ray_world[2]
    hit = cam_pos + t * ray_world
    return hit[:2]


def metric_radius(d: Detection, cam_pos: np.ndarray, cam_mat: np.ndarray,
                  fovy_deg: float, width: int, height: int,
                  plane_z: float = TABLE_Z) -> float:
    """Half the object's silhouette chord, back-projected onto the table.

    This is the ablation that matters. Apparent size is viewpoint-dependent, so
    a nearby small plate out-measures a distant large one and the two agents
    disagree about which is "the large plate". Back-projecting the silhouette
    through the same calibration the gate already uses gives a size in metres
    that both agents can compute for themselves and that does not depend on
    where they are standing.
    """
    a = backproject(*d.left_px, cam_pos, cam_mat, fovy_deg, width, height, plane_z)
    b = backproject(*d.right_px, cam_pos, cam_mat, fovy_deg, width, height, plane_z)
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    return float(np.linalg.norm(a - b) / 2.0)


def size_word_metric(d: Detection, group: list[Detection]) -> str:
    """Large/small by measured radius rather than by pixel area."""
    same = [g for g in group
            if g.label == d.label and g.metric_radius_m and not math.isnan(g.metric_radius_m)]
    if len(same) < 2 or d.metric_radius_m is None or math.isnan(d.metric_radius_m):
        return "medium"
    biggest = max(same, key=lambda g: g.metric_radius_m)
    smallest = min(same, key=lambda g: g.metric_radius_m)
    if biggest.metric_radius_m / max(smallest.metric_radius_m, 1e-6) < 1.15:
        return "medium"
    if d is biggest:
        return "large"
    if d is smallest:
        return "small"
    return "medium"
