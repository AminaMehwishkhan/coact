"""CoAct session orchestrator.

OBSERVE -> INDEPENDENT GROUNDING -> COMPARE -> CONSENSUS GATE -> CONFIRM -> EXECUTE
and, when the viewpoints disagree:
OBSERVE -> DISAGREE -> BLOCK -> HOVER -> CORRECTION -> RE-GROUND -> RE-CHECK -> EXECUTE

The session owns the physics, the two agents and the safety state machine. It
is the only place that is allowed to call `commit()` on the runtime, and it can
only do so through `SafetyStateMachine.to(EXECUTING)`, which refuses unless the
gate has opened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .agent import Agent, Interpretation
from .consensus import ConsensusResult, evaluate
from .grounding import Constraints, parse_constraints, parse_source
from .ov_backend import get_backend
from .state_machine import SafetyStateMachine, State


@dataclass
class Event:
    t: float
    state: str
    message: str
    level: str = "info"   # info | warn | block | ok

    def as_dict(self) -> dict:
        return {
            "t": f"{int(self.t // 60):02d}:{int(self.t % 60):02d}",
            "t_seconds": round(self.t, 2),
            "state": self.state,
            "message": self.message,
            "level": self.level,
        }


class Session:
    """One episode: an instruction, two opinions, and a gate between them."""

    def __init__(self, runtime=None, metric_sizing: bool = True):
        from .sim import get_runtime

        self.rt = runtime if runtime is not None else get_runtime()
        self.physics = getattr(self.rt, "connected", False)
        self.agents: dict[str, Agent] = {}
        if self.physics:
            self.agents = {
                k: Agent(k, self.rt, metric_sizing=metric_sizing) for k in ("A", "B")
            }
        self.sm = SafetyStateMachine()
        self.events: list[Event] = []
        self.instruction: str | None = None
        self.source: Constraints | None = None
        self.destination: Constraints | None = None
        self.interpretations: dict[str, Interpretation] = {}
        self.consensus: ConsensusResult | None = None
        self.hover_target: list[float] | None = None
        self.corrections: list[str] = []
        self.last_action: dict | None = None
        self._t0 = time.monotonic()

    # ------------------------------------------------------------------ utils
    def _log(self, message: str, level: str = "info") -> None:
        self.events.append(
            Event(time.monotonic() - self._t0, self.sm.state.value, message, level)
        )

    def reset(self) -> None:
        self.sm.reset()
        self.events.clear()
        self.instruction = None
        self.source = None
        self.destination = None
        self.interpretations.clear()
        self.consensus = None
        self.hover_target = None
        self.corrections.clear()
        self.last_action = None
        self._t0 = time.monotonic()
        if self.physics:
            self.rt.reset()
            for a in self.agents.values():
                a.percepts.clear()
        self._log("Session reset — physics and state cleared", "ok")

    # ----------------------------------------------------------------- stages
    def instruct(self, text: str) -> dict:
        self.reset()
        self.instruction = text
        self._log(f'Instruction received: "{text}"')

        self.sm.to(State.OBSERVING)
        if self.physics:
            counts = {k: len(a.observe()) for k, a in self.agents.items()}
            self._log(
                "Shared workspace observed from two cameras — "
                f"Agent A detected {counts['A']} objects, Agent B detected {counts['B']}"
            )
        else:
            self._log("No physics runtime available; nothing to observe", "warn")
            self.sm.to(State.ERROR)
            return self.snapshot()

        self.source = parse_source(text)
        self.destination = parse_constraints(text, "destination")
        return self._ground_and_compare(first_pass=True)

    def _ground_and_compare(self, first_pass: bool) -> dict:
        self.sm.to(State.GROUNDING if first_pass else State.REGROUNDING)
        assert self.destination is not None

        for k, agent in self.agents.items():
            interp = agent.ground(self.destination)
            self.interpretations[k] = interp
            where = (
                f"u={interp.target_u:+.2f}" if interp.target_u is not None else "no target"
            )
            self._log(
                f"Agent {k} grounding complete → {interp.target_label or '—'} "
                f"({interp.target_attribute}, {where}, conf {interp.confidence}, "
                f"{interp.backend} {interp.latency_ms} ms)"
            )

        self.sm.to(State.COMPARING)
        result = evaluate(self.interpretations["A"], self.interpretations["B"])
        self.consensus = result
        self.sm.ctx.consensus = result.agree

        if result.agree:
            self.sm.to(State.CONSENSUS)
            self._log(
                ("✓ CONSENSUS RESTORED" if not first_pass else "✓ CONSENSUS")
                + f" — both viewpoints localised the referent to within "
                  f"{result.separation_m * 100:.1f} cm of the same point",
                "ok",
            )
            if not first_pass:
                self.sm.ctx.confirmation_completed = True
            self.hover_target = result.target_world_xy
            self._log("▶ Execution authorized by the consensus gate", "ok")
        else:
            self.sm.to(State.DISAGREEMENT)
            self.sm.ctx.confirmation_required = True
            self.sm.ctx.confirmation_completed = False
            self._log(f"⚠ {result.verdict} — {result.diagnosis}", "warn")
            self._log("⏸ Safety gate locked — physical commitment blocked", "block")

            self.sm.to(State.HOVER_CONFIRMATION)
            self.hover_target = self.interpretations["A"].target_world_xy
            if self.hover_target is not None:
                self.last_action = self._hover_over(self.hover_target)
                self._log(
                    f"👁 Hover confirmation at "
                    f"({self.hover_target[0]:+.3f}, {self.hover_target[1]:+.3f}) — "
                    f"approach only, jaw open, nothing grasped or placed",
                    "warn",
                )
            self.sm.to(State.LISTENING)
            self._log("🎙 Listening for a human correction", "warn")

        return self.snapshot()

    def _hover_over(self, xy) -> dict:
        """Approach, never commit. The only motion allowed while the gate is shut."""
        import numpy as np

        from .perception import TABLE_Z

        goal = np.array([xy[0], xy[1], TABLE_Z + 0.105])
        err = self.rt.move_to("A", goal, 420)
        return {"arm": "A", "action": "hover", "position_error_m": round(err, 4),
                "jaw": "open", "simulated": True}

    def correct(self, text: str, source_label: str = "DEMO REPLAY") -> dict:
        if self.sm.state not in (State.LISTENING, State.HOVER_CONFIRMATION):
            self._log(
                f"Correction ignored — not awaiting confirmation (state {self.sm.state.value})",
                "warn",
            )
            return self.snapshot()

        self.corrections.append(text)
        self._log(f'Speechmatics [{source_label}]: "{text}"')
        self.sm.to(State.INTERRUPTED)
        self._log("↳ Correction detected — pending action cancelled, arm retracting", "warn")
        if self.physics:
            self.last_action = self.rt.retract("A")
            for a in self.agents.values():
                a.percepts.clear()

        merged = self.destination.merged_with(parse_constraints(text, "destination"))
        self.destination = merged
        self.hover_target = None
        self._log(f"↳ Semantic constraint updated → {merged.summary()}")
        self._log("↻ Re-grounding both agents against fresh camera frames")
        return self._ground_and_compare(first_pass=False)

    def execute(self) -> dict:
        """Physical commitment. Raises SafetyViolation unless the gate opened."""
        self.sm.to(State.EXECUTING)
        target = self.consensus.target_world_xy if self.consensus else None
        pretty = ("(%.3f, %.3f)" % tuple(target)) if target else "none"
        self._log(f"Executing bimanual place → target {pretty}", "ok")
        if self.physics and target is not None:
            source_id = self._pick_source()
            self.last_action = self._commit_to(source_id, target)
            self._log(
                f"Placed {source_id}: final gripper error "
                f"{self.last_action['position_error_m']} m",
                "ok",
            )
        self.sm.to(State.COMPLETE)
        self._log("✓ Task complete", "ok")
        return self.snapshot()

    def _pick_source(self) -> str:
        """Which object to move. Colour from the instruction when it gave one."""
        want = self.source.color if self.source else None
        for item in self.rt.items:
            if item.graspable and (want is None or item.color == want):
                return item.id
        return next(i.id for i in self.rt.items if i.graspable)

    def _commit_to(self, source_id: str, xy) -> dict:
        import numpy as np

        from .perception import TABLE_Z

        jaw = self.rt.arm_dofs["A"][5]
        src = self.rt.item_pos(source_id)
        self.rt.move_to("A", src + np.array([0, 0, 0.105]), 320)
        self.rt.move_to("A", src + np.array([0, 0, 0.032]), 300)
        self.rt.data.ctrl[jaw] = 0.05
        self.rt.settle(140)
        self.rt.move_to("A", src + np.array([0, 0, 0.15]), 260)
        goal = np.array([xy[0], xy[1], TABLE_Z + 0.11])
        err = self.rt.move_to("A", goal, 460)
        self.rt.data.ctrl[jaw] = 1.5
        self.rt.settle(220)
        return {"arm": "A", "action": "commit", "source": source_id,
                "target_xy": [round(float(v), 4) for v in xy],
                "position_error_m": round(err, 4), "simulated": True,
                "sim_time_s": round(float(self.rt.data.time), 3)}

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        return {
            "state": self.sm.state.value,
            "state_history": [s.value for s in self.sm.history],
            "execution_authorized": self.sm.execution_authorized(),
            "physics": self.physics,
            "gate": {
                "consensus": self.sm.ctx.consensus,
                "confirmation_required": self.sm.ctx.confirmation_required,
                "confirmation_completed": self.sm.ctx.confirmation_completed,
            },
            "instruction": self.instruction,
            "source_constraints": self.source.as_dict() if self.source else None,
            "destination_constraints": self.destination.as_dict() if self.destination else None,
            "agents": {k: v.as_dict() for k, v in self.interpretations.items()},
            "consensus": self.consensus.as_dict() if self.consensus else None,
            "hover_target": self.hover_target,
            "last_action": self.last_action,
            "events": [e.as_dict() for e in self.events],
            "telemetry": get_backend().telemetry(),
        }
