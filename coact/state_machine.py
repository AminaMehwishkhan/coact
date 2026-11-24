"""Explicit safety state machine.

Critical invariant, enforced structurally rather than by convention:

    EXECUTING is unreachable from DISAGREEMENT, and unreachable at all unless
        consensus is True
        AND (confirmation_required is False OR confirmation_completed is True)

Any attempt to violate this raises SafetyViolation. It is not logged and
swallowed — it stops the transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    GROUNDING = "GROUNDING"
    COMPARING = "COMPARING"
    CONSENSUS = "CONSENSUS"
    DISAGREEMENT = "DISAGREEMENT"
    HOVER_CONFIRMATION = "HOVER_CONFIRMATION"
    LISTENING = "LISTENING"
    INTERRUPTED = "INTERRUPTED"
    REGROUNDING = "REGROUNDING"
    EXECUTING = "EXECUTING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class SafetyViolation(RuntimeError):
    pass


LEGAL: dict[State, set[State]] = {
    State.IDLE: {State.OBSERVING, State.ERROR},
    State.OBSERVING: {State.GROUNDING, State.ERROR, State.IDLE},
    State.GROUNDING: {State.COMPARING, State.ERROR},
    State.COMPARING: {State.CONSENSUS, State.DISAGREEMENT, State.ERROR},
    # NOTE: EXECUTING is deliberately absent here.
    State.DISAGREEMENT: {State.HOVER_CONFIRMATION, State.ERROR, State.IDLE},
    State.CONSENSUS: {State.HOVER_CONFIRMATION, State.EXECUTING, State.ERROR, State.IDLE},
    State.HOVER_CONFIRMATION: {State.LISTENING, State.INTERRUPTED, State.EXECUTING,
                               State.ERROR, State.IDLE},
    State.LISTENING: {State.INTERRUPTED, State.HOVER_CONFIRMATION, State.EXECUTING,
                      State.ERROR, State.IDLE},
    State.INTERRUPTED: {State.REGROUNDING, State.ERROR, State.IDLE},
    State.REGROUNDING: {State.COMPARING, State.ERROR},
    State.EXECUTING: {State.COMPLETE, State.INTERRUPTED, State.ERROR},
    State.COMPLETE: {State.IDLE, State.OBSERVING},
    State.ERROR: {State.IDLE},
}


@dataclass
class SafetyContext:
    consensus: bool = False
    confirmation_required: bool = False
    confirmation_completed: bool = False


@dataclass
class SafetyStateMachine:
    state: State = State.IDLE
    ctx: SafetyContext = field(default_factory=SafetyContext)
    history: list[State] = field(default_factory=lambda: [State.IDLE])

    def execution_authorized(self) -> bool:
        if not self.ctx.consensus:
            return False
        if self.ctx.confirmation_required and not self.ctx.confirmation_completed:
            return False
        return True

    def can(self, target: State) -> bool:
        if target not in LEGAL[self.state]:
            return False
        if target is State.EXECUTING and not self.execution_authorized():
            return False
        return True

    def to(self, target: State) -> State:
        if target not in LEGAL[self.state]:
            raise SafetyViolation(f"Illegal transition {self.state.value} -> {target.value}")
        if target is State.EXECUTING and not self.execution_authorized():
            raise SafetyViolation(
                "Execution blocked by consensus gate: "
                f"consensus={self.ctx.consensus} "
                f"confirmation_required={self.ctx.confirmation_required} "
                f"confirmation_completed={self.ctx.confirmation_completed}"
            )
        self.state = target
        self.history.append(target)
        return target

    def reset(self) -> None:
        self.state = State.IDLE
        self.ctx = SafetyContext()
        self.history = [State.IDLE]
