"""An embodied agent: one camera, one arm, one opinion.

An agent never sees the simulator's state. It renders its own camera, detects
objects in that image, and grounds the instruction in its own image frame —
so "left" means left *to it*. Only the chosen target's position is converted
into the shared table frame, and only so the consensus gate can ask whether
the two agents picked the same physical object.

Scoring runs through the OpenVINO kernel in `ov_backend`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

from .grounding import Constraints
from .ov_backend import get_backend
from .perception import (Detection, backproject, detect, metric_radius,
                         size_word, size_word_metric)

FEATURE_DIM = 8


@dataclass
class Percept:
    """One detection, with the agent's own reading of it."""

    det: Detection
    size_word: str
    world_xy: tuple[float, float]

    def as_dict(self) -> dict:
        return {
            "label": self.det.label,
            "color": self.det.color,
            "u": round(self.det.u, 3),
            "v": round(self.det.v, 3),
            "area_px": round(self.det.area_px, 1),
            "size_word": self.size_word,
            "metric_radius_m": (round(self.det.metric_radius_m, 4)
                                if self.det.metric_radius_m is not None else None),
            "world_xy": [round(self.world_xy[0], 4), round(self.world_xy[1], 4)],
        }


@dataclass
class Interpretation:
    agent_id: str
    viewpoint: str
    target_world_xy: tuple[float, float] | None
    target_label: str | None
    target_attribute: str | None
    target_u: float | None
    confidence: float
    margin: float
    reasoning: str
    constraints: dict
    percepts: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    latency_ms: float | None = None
    backend: str = ""
    n_detections: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        if d["target_world_xy"] is not None:
            d["target_world_xy"] = [round(v, 4) for v in d["target_world_xy"]]
        return d


def _features(p: Percept, c: Constraints) -> np.ndarray:
    """Image-frame features. Dimensions 4-7 are what make agents disagree."""
    u, v = p.det.u, p.det.v
    return np.array(
        [
            1.0 if (c.noun and p.det.label == c.noun) else 0.0,
            1.0 if (c.color and p.det.color == c.color) else 0.0,
            1.0 if (c.size and p.size_word == c.size) else 0.0,
            p.det.apparent_size,
            -u,          # leftness: u is -1 at this camera's left edge
            u,           # rightness
            -v,          # nearness is lower in the image for a table-top view
            v,
        ],
        dtype=np.float32,
    )


def _query(c: Constraints) -> np.ndarray:
    q = np.zeros(FEATURE_DIM, dtype=np.float32)
    if c.noun:
        q[0] = 3.0
    if c.color:
        q[1] = 2.5
    if c.size:
        q[2] = 2.5
    if c.spatial == "left":
        q[4] = 2.0
    elif c.spatial == "right":
        q[5] = 2.0
    elif c.spatial == "near":
        q[6] = 2.0
    elif c.spatial == "far":
        q[7] = 2.0
    return q


class Agent:
    def __init__(self, agent_id: str, runtime, metric_sizing: bool = True):
        import mujoco

        # metric_sizing=False reproduces the naive baseline: size judged from
        # pixel area, which is viewpoint-dependent. The benchmark runs both.
        self.metric_sizing = metric_sizing
        self.id = agent_id
        self.rt = runtime
        self.cam = f"cam_{agent_id}"
        self.cam_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam)
        self.name = f"Arm {agent_id} — {'west' if agent_id == 'A' else 'east'} head camera"
        self.percepts: list[Percept] = []
        self.last_image: np.ndarray | None = None

    # ------------------------------------------------------------------ see
    def observe(self) -> list[Percept]:
        img = self.rt.render(self.cam)
        self.last_image = img
        H, W = img.shape[:2]
        dets = detect(img)
        fovy = float(self.rt.model.cam_fovy[self.cam_id])
        pos = self.rt.data.cam_xpos[self.cam_id]
        mat = self.rt.data.cam_xmat[self.cam_id]
        for d in dets:
            d.metric_radius_m = metric_radius(d, pos, mat, fovy, W, H)
        sizer = size_word_metric if self.metric_sizing else size_word
        self.percepts = [
            Percept(
                det=d,
                size_word=sizer(d, dets),
                world_xy=tuple(backproject(d.cx, d.cy, pos, mat, fovy, W, H)),
            )
            for d in dets
        ]
        return self.percepts

    # --------------------------------------------------------------- ground
    def ground(self, c: Constraints) -> Interpretation:
        if not self.percepts:
            self.observe()
        pool = [p for p in self.percepts if not c.noun or p.det.label == c.noun]
        if not pool:
            pool = list(self.percepts)

        # An explicit attribute is a hard requirement, not a tie-breaker. If the
        # instruction says "the small plate" and nothing in this view reads as
        # small, the honest answer is "I cannot see what you mean" — not the
        # only plate that happens to be visible. Without this, an agent whose
        # view is occluded reports a confident target for an object it cannot
        # see, and if the other agent happens to pick the same one, the gate
        # opens on a referent neither of them actually resolved.
        if c.size and not any(p.size_word == c.size for p in pool):
            pool = []
        elif c.color and not any(p.det.color == c.color for p in pool):
            pool = []

        if not pool:
            return Interpretation(
                agent_id=self.id, viewpoint=self.name, target_world_xy=None,
                target_label=None, target_attribute=None, target_u=None,
                confidence=0.0, margin=0.0,
                reasoning=("nothing matching this instruction is visible from here"
                           + (f" (no {c.size} {c.noun or 'object'})" if c.size else "")),
                constraints=c.as_dict(), percepts=[], candidates=[],
                backend="n/a", n_detections=len(self.percepts),
            )

        backend = get_backend()
        feats = np.stack([_features(p, c) for p in pool])
        scores = backend.score(feats, _query(c))
        order = np.argsort(-scores)
        best = pool[int(order[0])]
        top = float(scores[int(order[0])])
        second = float(scores[int(order[1])]) if len(order) > 1 else top - 2.0
        margin = top - second
        conf = float(1.0 / (1.0 + np.exp(-1.6 * margin)))

        bits = []
        if c.spatial:
            side = "left" if best.det.u < 0 else "right"
            bits.append(f"'{c.spatial}' resolved in this camera's image frame "
                        f"(target at u={best.det.u:+.2f}, i.e. my {side})")
        if c.size:
            bits.append(f"apparent size reads as {best.size_word}")
        if c.color:
            bits.append(f"colour {best.det.color} matched")
        if not bits:
            bits.append("highest-scoring detection for this noun")

        return Interpretation(
            agent_id=self.id, viewpoint=self.name,
            target_world_xy=best.world_xy, target_label=best.det.label,
            target_attribute=best.size_word, target_u=round(best.det.u, 3),
            confidence=round(min(max(conf, 0.05), 0.99), 3),
            margin=round(margin, 3), reasoning="; ".join(bits),
            constraints=c.as_dict(),
            percepts=[p.as_dict() for p in self.percepts],
            candidates=[
                {"label": pool[i].det.label, "u": round(pool[i].det.u, 3),
                 "size": pool[i].size_word, "score": round(float(scores[i]), 3)}
                for i in order[:3]
            ],
            latency_ms=(round(backend.last_latency_ms, 3)
                        if backend.last_latency_ms is not None else None),
            backend="OpenVINO" if backend.available else "NumPy (CPU fallback)",
            n_detections=len(self.percepts),
        )
