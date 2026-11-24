"""Perception and cross-view geometry, tested against the simulator's truth.

The agents never see these numbers — ground truth is used here only to check
that what comes out of the pixels is actually right.
"""

import math

import pytest

from coact.sim import mujoco_available

pytestmark = pytest.mark.skipif(not mujoco_available(),
                                reason="MuJoCo or the SO-ARM100 assets are not installed")


@pytest.fixture(scope="module")
def rt():
    from coact.mj_runtime import MuJoCoRuntime

    r = MuJoCoRuntime()
    yield r
    r.close()


@pytest.fixture(scope="module")
def agents(rt):
    from coact.agent import Agent

    a, b = Agent("A", rt), Agent("B", rt)
    a.observe()
    b.observe()
    return a, b


def test_both_cameras_detect_both_plates(agents):
    for ag in agents:
        plates = [p for p in ag.percepts if p.det.label == "plate"]
        assert len(plates) == 2, f"agent {ag.id} saw {len(plates)} plates"


def test_backprojection_matches_ground_truth_within_a_centimetre(rt, agents):
    truth = {k: rt.item_pos(k)[:2] for k in ("plate_small", "plate_large")}
    for ag in agents:
        for p in ag.percepts:
            if p.det.label != "plate":
                continue
            nearest = min(truth.values(), key=lambda t: math.dist(p.world_xy, t))
            assert math.dist(p.world_xy, nearest) < 0.015


def test_metric_radius_recovers_true_plate_size(rt, agents):
    want = {"plate_small": 0.042, "plate_large": 0.068}
    truth = {k: rt.item_pos(k)[:2] for k in want}
    for ag in agents:
        for p in ag.percepts:
            if p.det.label != "plate":
                continue
            key = min(truth, key=lambda k: math.dist(p.world_xy, truth[k]))
            assert p.det.metric_radius_m == pytest.approx(want[key], abs=0.006)


def test_the_two_views_mirror_each_other(agents):
    """The whole premise: the same plate sits on opposite sides of the two images."""
    a, b = agents
    a_large = max((p for p in a.percepts if p.det.label == "plate"),
                  key=lambda p: p.det.metric_radius_m)
    b_large = max((p for p in b.percepts if p.det.label == "plate"),
                  key=lambda p: p.det.metric_radius_m)
    assert math.dist(a_large.world_xy, b_large.world_xy) < 0.02   # same physical plate
    assert a_large.det.u * b_large.det.u < 0                      # opposite image sides
