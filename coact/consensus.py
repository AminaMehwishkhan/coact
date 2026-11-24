"""The consensus gate — CoAct's core contribution.

Two independently-grounded interpretations go in; a verdict comes out. If the
verdict is not CONSENSUS, nothing downstream is permitted to touch an actuator.

The agents do not share an object index — each one only has detections in its
own image. So agreement is decided geometrically: both back-project their
chosen target onto the table plane, and they agree when those two estimates
fall within `AGREE_TOL_M` of each other. That tolerance has to clear the
back-projection error (measured at roughly 1 cm on this rig) while staying far
below the spacing between candidate objects (about 18 cm), which it does by
more than an order of magnitude on each side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

from .agent import Interpretation

CONFIDENCE_FLOOR = 0.55
AGREE_TOL_M = 0.05


@dataclass
class ConsensusResult:
    agree: bool
    target_world_xy: list[float] | None
    verdict: str                 # CONSENSUS | DISAGREEMENT | LOW_CONFIDENCE | NO_TARGET
    diagnosis: str
    disagreement_type: str | None
    joint_confidence: float
    separation_m: float | None
    per_agent: dict

    def as_dict(self) -> dict:
        return asdict(self)


def _separation(a: Interpretation, b: Interpretation) -> float | None:
    if a.target_world_xy is None or b.target_world_xy is None:
        return None
    return math.dist(a.target_world_xy, b.target_world_xy)


def _diagnose(a: Interpretation, b: Interpretation) -> tuple[str, str]:
    """Explain WHY the viewpoints disagreed. A flag is not a diagnosis."""
    ca, cb = a.constraints, b.constraints
    if ca.get("spatial") and ca.get("spatial") == cb.get("spatial"):
        return (
            "perspective",
            f"Both agents applied '{ca['spatial']}' correctly, but in opposing image "
            f"frames: A placed its target at u={a.target_u:+.2f} and B at "
            f"u={b.target_u:+.2f}. The instruction has no viewpoint-independent "
            f"referent, so neither agent is wrong.",
        )
    if a.target_attribute != b.target_attribute:
        return (
            "attribute",
            f"The two cameras read the same scene differently: A judged its target "
            f"{a.target_attribute}, B judged its target {b.target_attribute}. "
            f"Apparent size is viewpoint-dependent.",
        )
    return (
        "object",
        "Both agents picked an object of the requested category, but not the same "
        "one, and the instruction carries no constraint that separates them.",
    )


def evaluate(a: Interpretation, b: Interpretation) -> ConsensusResult:
    per_agent = {
        a.agent_id: {"xy": a.target_world_xy, "attr": a.target_attribute, "u": a.target_u},
        b.agent_id: {"xy": b.target_world_xy, "attr": b.target_attribute, "u": b.target_u},
    }
    joint = round(min(a.confidence, b.confidence), 3)
    sep = _separation(a, b)

    if sep is None:
        return ConsensusResult(
            agree=False, target_world_xy=None, verdict="NO_TARGET",
            diagnosis="At least one agent found nothing matching the instruction in its "
                      "own view. An agent that cannot see the referent cannot consent "
                      "to acting on it.",
            disagreement_type="no_target", joint_confidence=joint,
            separation_m=None, per_agent=per_agent,
        )

    if sep > AGREE_TOL_M:
        dtype, diagnosis = _diagnose(a, b)
        return ConsensusResult(
            agree=False, target_world_xy=None, verdict="DISAGREEMENT",
            diagnosis=diagnosis, disagreement_type=dtype, joint_confidence=joint,
            separation_m=round(sep, 4), per_agent=per_agent,
        )

    if joint < CONFIDENCE_FLOOR:
        return ConsensusResult(
            agree=False, target_world_xy=None, verdict="LOW_CONFIDENCE",
            diagnosis=f"Both agents converged on the same point ({sep * 100:.1f} cm apart), "
                      f"but the weakest decision margin gives confidence {joint}, below the "
                      f"{CONFIDENCE_FLOOR} floor. Agreeing for a weak reason is still not a "
                      f"licence to move.",
            disagreement_type="low_confidence", joint_confidence=joint,
            separation_m=round(sep, 4), per_agent=per_agent,
        )

    mid = [
        (a.target_world_xy[0] + b.target_world_xy[0]) / 2,
        (a.target_world_xy[1] + b.target_world_xy[1]) / 2,
    ]
    return ConsensusResult(
        agree=True, target_world_xy=[round(v, 4) for v in mid], verdict="CONSENSUS",
        diagnosis=f"Both embodied viewpoints independently localised the referent to within "
                  f"{sep * 100:.1f} cm of the same point on the table.",
        disagreement_type=None, joint_confidence=joint,
        separation_m=round(sep, 4), per_agent=per_agent,
    )
