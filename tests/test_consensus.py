"""Consensus engine unit tests, with synthetic interpretations."""

from coact.agent import Interpretation
from coact.consensus import AGREE_TOL_M, evaluate


def interp(agent_id, xy, *, attr="large", u=0.1, conf=0.9, spatial=None, size=None):
    return Interpretation(
        agent_id=agent_id, viewpoint=f"cam {agent_id}", target_world_xy=xy,
        target_label="plate", target_attribute=attr, target_u=u, confidence=conf,
        margin=1.0, reasoning="test",
        constraints={"noun": "plate", "color": None, "size": size, "spatial": spatial},
    )


def test_same_point_agrees():
    r = evaluate(interp("A", (0.0, 0.09)), interp("B", (0.004, 0.093)))
    assert r.agree and r.verdict == "CONSENSUS"
    assert r.separation_m < AGREE_TOL_M


def test_opposite_plates_disagree_and_are_diagnosed_as_perspective():
    a = interp("A", (0.0, 0.09), u=-0.15, spatial="left")
    b = interp("B", (0.0, -0.09), u=-0.14, spatial="left")
    r = evaluate(a, b)
    assert not r.agree
    assert r.disagreement_type == "perspective"
    assert "no viewpoint-independent referent" in r.diagnosis


def test_apparent_size_conflict_is_diagnosed_as_attribute():
    a = interp("A", (0.0, 0.09), attr="large", size="large")
    b = interp("B", (0.0, -0.09), attr="small", size="large")
    assert evaluate(a, b).disagreement_type == "attribute"


def test_agreement_below_the_confidence_floor_still_blocks():
    r = evaluate(interp("A", (0.0, 0.09), conf=0.4), interp("B", (0.0, 0.09), conf=0.4))
    assert not r.agree and r.verdict == "LOW_CONFIDENCE"


def test_an_agent_that_sees_nothing_cannot_consent():
    r = evaluate(interp("A", (0.0, 0.09)), interp("B", None))
    assert not r.agree and r.verdict == "NO_TARGET"


def test_tolerance_sits_well_clear_of_both_error_and_object_spacing():
    """The tolerance must beat back-projection error and lose to object spacing."""
    assert 0.015 < AGREE_TOL_M < 0.17
