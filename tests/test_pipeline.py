"""End-to-end episodes on real physics, and the parsing they depend on."""

import pytest

from coact.grounding import parse_constraints, parse_source
from coact.sim import mujoco_available
from coact.state_machine import SafetyViolation

CLEAR = "Place the red cup on the large plate."
AMBIGUOUS = "Put the cup on the plate on the left."
CORRECTION = "No, the large plate."


# ------------------------------------------------------------------ parsing
def test_destination_clause_keeps_its_head_noun():
    c = parse_constraints(AMBIGUOUS)
    assert (c.noun, c.spatial, c.size) == ("plate", "left", None)


def test_attribute_instruction_parses_size():
    c = parse_constraints(CLEAR)
    assert (c.noun, c.size) == ("plate", "large")


def test_source_clause_is_separate_from_destination():
    assert parse_source(CLEAR).color == "red"
    assert parse_constraints(CLEAR).color is None


def test_a_correction_retires_the_spatial_term_that_caused_the_ambiguity():
    merged = parse_constraints(AMBIGUOUS).merged_with(parse_constraints(CORRECTION))
    assert merged.size == "large"
    assert merged.spatial is None


# ------------------------------------------------------------- episodes
pytestmark_physics = pytest.mark.skipif(
    not mujoco_available(), reason="MuJoCo or the SO-ARM100 assets are not installed")


@pytest.fixture
def session():
    from coact.pipeline import Session

    s = Session()
    assert s.physics, "these tests need the physics runtime"
    return s


@pytestmark_physics
def test_clear_instruction_reaches_consensus_and_executes(session):
    snap = session.instruct(CLEAR)
    assert snap["consensus"]["agree"] is True
    assert snap["execution_authorized"] is True
    assert session.execute()["state"] == "COMPLETE"


@pytestmark_physics
def test_ambiguous_instruction_blocks_execution(session):
    snap = session.instruct(AMBIGUOUS)
    assert snap["consensus"]["agree"] is False
    assert snap["consensus"]["disagreement_type"] == "perspective"
    assert snap["execution_authorized"] is False
    assert snap["state"] == "LISTENING"
    with pytest.raises(SafetyViolation):
        session.execute()


@pytestmark_physics
def test_the_arm_hovers_but_does_not_close_while_the_gate_is_shut(session):
    snap = session.instruct(AMBIGUOUS)
    assert snap["hover_target"] is not None          # it did approach
    assert snap["last_action"]["action"] == "hover"  # and only approached
    assert snap["last_action"]["jaw"] == "open"
    assert snap["gate"]["consensus"] is False


@pytestmark_physics
def test_correction_restores_consensus_then_executes(session):
    session.instruct(AMBIGUOUS)
    snap = session.correct(CORRECTION, source_label="DEMO REPLAY")
    assert snap["consensus"]["agree"] is True
    assert snap["consensus"]["separation_m"] < 0.05
    assert snap["execution_authorized"] is True
    assert session.execute()["state"] == "COMPLETE"


@pytestmark_physics
def test_timeline_records_the_whole_arc(session):
    session.instruct(AMBIGUOUS)
    session.correct(CORRECTION)
    session.execute()
    msgs = " | ".join(e["message"] for e in session.snapshot()["events"])
    for needle in ["Instruction received", "DISAGREEMENT", "Safety gate locked",
                   "Hover confirmation", "constraint updated", "Re-grounding",
                   "CONSENSUS RESTORED", "Execution authorized", "Task complete"]:
        assert needle in msgs, needle


@pytestmark_physics
def test_repeated_episodes_do_not_leak_state(session):
    session.instruct(AMBIGUOUS)
    session.correct(CORRECTION)
    session.execute()
    snap = session.instruct(AMBIGUOUS)
    assert snap["consensus"]["agree"] is False       # no stale size constraint
    assert snap["destination_constraints"]["size"] is None


@pytestmark_physics
def test_correction_outside_the_confirmation_window_is_ignored(session):
    session.instruct(CLEAR)
    before = session.snapshot()["consensus"]["target_world_xy"]
    session.correct("No, the small plate.")
    assert session.snapshot()["consensus"]["target_world_xy"] == before


# ------------------------------------------------------------- fallbacks
def test_openvino_backend_reports_itself_honestly():
    from coact.ov_backend import get_backend

    t = get_backend().telemetry()
    assert t["available"] in (True, False)
    if not t["available"]:
        assert "fallback" in t["backend"].lower()
    assert t["last_latency_ms"] is None or t["last_latency_ms"] >= 0


def test_speechmatics_falls_back_without_a_key(monkeypatch):
    from coact import speech

    monkeypatch.delenv("SPEECHMATICS_API_KEY", raising=False)
    st = speech.status()
    assert st["configured"] is False
    assert st["mode"] == "DEMO REPLAY"


def test_speechmatics_token_request_without_a_key_explains_itself():
    import asyncio

    from coact import speech

    res = asyncio.run(speech.mint_temp_key())
    assert res["ok"] is False and "not configured" in res["reason"]


def test_session_degrades_cleanly_with_no_physics(monkeypatch):
    monkeypatch.setenv("COACT_FORCE_NULL_RUNTIME", "1")
    from coact.pipeline import Session

    s = Session()
    assert s.physics is False
    snap = s.instruct(CLEAR)
    assert snap["state"] == "ERROR"          # says so instead of inventing a result
    assert snap["execution_authorized"] is False
