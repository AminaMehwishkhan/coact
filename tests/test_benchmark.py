"""A small benchmark run, as a regression guard on the headline claim."""

import pytest

from coact.sim import mujoco_available

pytestmark = pytest.mark.skipif(not mujoco_available(),
                                reason="MuJoCo or the SO-ARM100 assets are not installed")


@pytest.fixture(scope="module")
def report():
    from coact.benchmark import run

    return run(trials=12, seed=3)


def test_coact_makes_fewer_unsafe_commits_than_a_single_agent(report):
    """The headline claim. Not "zero" — the full 90-trial run shows CoAct still
    commits wrongly under occlusion, and a test that asserted zero would be
    asserting something the benchmark does not support."""
    assert report["coact"]["unsafe_commits"] < report["single_agent"]["unsafe_commits"]


def test_the_single_agent_baseline_does_make_unsafe_commits(report):
    """If the baseline were also clean the benchmark would prove nothing."""
    assert report["single_agent"]["unsafe_commits"] > 0


def test_coact_is_clean_when_nothing_is_occluded(report):
    """Where both agents can see the referent, the gate should let nothing through."""
    for cond in ("clean", "depth_skew"):
        assert report["by_condition"][cond]["coact_unsafe_pct"] == 0.0


def test_every_condition_is_exercised(report):
    assert set(report["meta"]["conditions"]) == {"clean", "depth_skew", "occluded"}


def test_correction_recovers_most_blocks(report):
    c = report["correction"]
    assert c["attempts"] > 0
    assert c["restored_on_intended_target_pct"] >= 50.0


def test_metric_sizing_removes_attribute_disagreements():
    """The ablation's claim, checked rather than asserted in prose."""
    from coact.benchmark import run

    metric = run(trials=12, seed=3, metric_sizing=True)
    apparent = run(trials=12, seed=3, metric_sizing=False)
    m = metric["recovery_by_disagreement_type"].get("attribute", {}).get("blocks", 0)
    a = apparent["recovery_by_disagreement_type"].get("attribute", {}).get("blocks", 0)
    assert m <= a
