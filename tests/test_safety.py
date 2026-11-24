"""The safety invariant. These are the tests that justify the architecture."""

import pytest

from coact.state_machine import (
    LEGAL,
    SafetyStateMachine,
    SafetyViolation,
    State,
)


def test_executing_is_not_in_disagreements_transition_set():
    assert State.EXECUTING not in LEGAL[State.DISAGREEMENT]


def test_forcing_the_consensus_flag_from_disagreement_still_raises():
    sm = SafetyStateMachine()
    sm.state = State.DISAGREEMENT
    sm.ctx.consensus = True          # someone sets the flag by hand
    with pytest.raises(SafetyViolation):
        sm.to(State.EXECUTING)


def test_execution_blocked_without_consensus():
    sm = SafetyStateMachine()
    sm.state = State.HOVER_CONFIRMATION
    sm.ctx.consensus = False
    assert not sm.execution_authorized()
    with pytest.raises(SafetyViolation):
        sm.to(State.EXECUTING)


def test_execution_blocked_while_confirmation_pending():
    sm = SafetyStateMachine()
    sm.state = State.HOVER_CONFIRMATION
    sm.ctx.consensus = True
    sm.ctx.confirmation_required = True
    sm.ctx.confirmation_completed = False
    with pytest.raises(SafetyViolation):
        sm.to(State.EXECUTING)


def test_execution_allowed_once_confirmation_completed():
    sm = SafetyStateMachine()
    sm.state = State.HOVER_CONFIRMATION
    sm.ctx.consensus = True
    sm.ctx.confirmation_required = True
    sm.ctx.confirmation_completed = True
    assert sm.to(State.EXECUTING) is State.EXECUTING


def test_only_three_states_can_lead_to_executing():
    allowed = {State.CONSENSUS, State.HOVER_CONFIRMATION, State.LISTENING}
    for src, targets in LEGAL.items():
        if State.EXECUTING in targets:
            assert src in allowed, f"{src} must not lead to EXECUTING"


def test_reset_clears_the_gate():
    sm = SafetyStateMachine()
    sm.ctx.consensus = True
    sm.ctx.confirmation_completed = True
    sm.state = State.COMPLETE
    sm.reset()
    assert sm.state is State.IDLE
    assert not sm.execution_authorized()


def test_illegal_transitions_raise_rather_than_warn():
    sm = SafetyStateMachine()
    with pytest.raises(SafetyViolation):
        sm.to(State.EXECUTING)      # straight out of IDLE
