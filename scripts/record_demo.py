"""Record the CoAct episode to video, with the gate state burned into the frame.

Produces, into --out:
  coact_demo.mp4      the full arc: disagree -> block -> hover -> correct -> execute
  agent_views.png     both camera images with detections drawn, side by side

    python scripts/record_demo.py --out results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from coact.agent import Agent
from coact.consensus import evaluate
from coact.grounding import parse_constraints
from coact.mj_runtime import HOME, MuJoCoRuntime
from coact.perception import TABLE_Z, annotate

W, H = 960, 720
FPS = 30

INK = (236, 240, 246)
OK = (120, 224, 140)
WARN = (32, 176, 255)      # BGR
STOP = (109, 77, 255)


def banner(frame: np.ndarray, state: str, line: str, tone: str) -> np.ndarray:
    col = {"ok": OK, "warn": WARN, "block": STOP}.get(tone, INK)
    out = frame.copy()
    strip = out[:96].astype(np.float32) * 0.22
    out[:96] = strip.astype(np.uint8)
    cv2.putText(out, state, (24, 42), cv2.FONT_HERSHEY_DUPLEX, 0.95, col, 2, cv2.LINE_AA)
    cv2.putText(out, line[:78], (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.56, INK, 1, cv2.LINE_AA)
    return out


class Recorder:
    def __init__(self, rt: MuJoCoRuntime):
        self.rt = rt
        self.frames: list[np.ndarray] = []
        self.state = "IDLE"
        self.line = ""
        self.tone = "info"

    def say(self, state: str, line: str, tone: str = "info", hold: float = 1.6) -> None:
        self.state, self.line, self.tone = state, line, tone
        print(f"  {state:<20} {line}")
        self.hold(hold)

    def grab(self) -> None:
        rgb = self.rt.render("cam_obs")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.frames.append(banner(bgr, self.state, self.line, self.tone))

    def hold(self, seconds: float) -> None:
        for _ in range(int(seconds * FPS)):
            self.rt.settle(4)
            self.grab()

    def move(self, arm: str, goal: np.ndarray, steps: int = 420) -> float:
        q_goal, _ = self.rt.solve_ik(arm, goal)
        acts = self.rt.arm_dofs[arm][:5]
        q0 = np.array([self.rt.data.ctrl[a] for a in acts])
        for i in range(steps):
            alpha = min(1.0, (i + 1) / max(1, steps * 0.6))
            for k, a in enumerate(acts):
                self.rt.data.ctrl[a] = float(q0[k] + alpha * (q_goal[k] - q0[k]))
            self.rt.mj.mj_step(self.rt.model, self.rt.data)
            if i % 7 == 0:
                self.grab()
        return float(np.linalg.norm(goal - self.rt.gripper_pos(arm)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rt = MuJoCoRuntime(width=W, height=H)
    A, B = Agent("A", rt), Agent("B", rt)
    rec = Recorder(rt)

    rec.say("CoAct", "Two SO-ARM100 arms, one table, two camera viewpoints", hold=1.8)

    # --- the two views, as the agents actually see them --------------------
    A.observe()
    B.observe()
    pair = np.concatenate(
        [annotate(A.last_image, [p.det for p in A.percepts]),
         annotate(B.last_image, [p.det for p in B.percepts])], axis=1)
    cv2.imwrite(str(out / "agent_views.png"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))

    # --- ambiguous instruction ---------------------------------------------
    instruction = "Put the cup on the plate on the left."
    rec.say("INSTRUCTION", instruction, hold=2.0)

    c = parse_constraints(instruction)
    ia, ib = A.ground(c), B.ground(c)
    res = evaluate(ia, ib)
    rec.say("GROUNDING",
            f"A -> plate at u={ia.target_u:+.2f} ({ia.target_attribute})   "
            f"B -> plate at u={ib.target_u:+.2f} ({ib.target_attribute})", hold=2.0)
    rec.say("DISAGREEMENT",
            f"{res.separation_m * 100:.0f} cm apart - both agents applied 'left' correctly",
            "warn", hold=2.2)
    rec.say("EXECUTION BLOCKED", "Safety gate locked. Nothing may be grasped or placed.",
            "block", hold=2.0)

    # --- hover confirmation: approach, never commit ------------------------
    rec.state, rec.line, rec.tone = ("HOVER CONFIRMATION",
                                     "Approaching Agent A's candidate. Jaw open.", "warn")
    goal = np.array([ia.target_world_xy[0], ia.target_world_xy[1], TABLE_Z + 0.105])
    rec.move("A", goal, 420)
    rec.say("LISTENING FOR CORRECTION", "Awaiting a human correction...", "warn", hold=2.0)

    # --- correction ---------------------------------------------------------
    correction = "No, the large plate."
    rec.say("CORRECTION", f'Speechmatics [DEMO REPLAY]: "{correction}"', "warn", hold=1.8)

    rec.state, rec.line, rec.tone = ("INTERRUPTED", "Pending action cancelled. Retracting.",
                                     "warn")
    for k, act in enumerate(rt.arm_dofs["A"][:5]):
        rt.data.ctrl[act] = float(HOME[k])
    for i in range(280):
        rt.mj.mj_step(rt.model, rt.data)
        if i % 7 == 0:
            rec.grab()

    A.percepts.clear()
    B.percepts.clear()
    A.observe()
    B.observe()
    c2 = c.merged_with(parse_constraints(correction))
    ra, rb = A.ground(c2), B.ground(c2)
    res2 = evaluate(ra, rb)
    rec.say("RE-GROUNDING", "Both agents re-grounded against fresh camera frames", hold=1.6)

    if not res2.agree:
        rec.say("STILL BLOCKED", res2.diagnosis[:70], "block", hold=2.5)
        _write(rec, out)
        return

    rec.say("CONSENSUS RESTORED",
            f"Both viewpoints agree to within {res2.separation_m * 100:.1f} cm", "ok", hold=2.2)
    rec.say("EXECUTION AUTHORIZED", "The gate opens. Only now may the arm commit.", "ok",
            hold=1.8)

    # --- commit --------------------------------------------------------------
    jaw = rt.arm_dofs["A"][5]
    src = rt.item_pos("cup_red")
    rec.state, rec.line, rec.tone = "EXECUTING", "Grasping the red cup", "ok"
    rec.move("A", src + np.array([0, 0, 0.105]), 300)
    rec.move("A", src + np.array([0, 0, 0.032]), 280)
    rt.data.ctrl[jaw] = 0.05
    rec.hold(0.6)
    rec.state, rec.line = "EXECUTING", "Placing on the agreed plate"
    rec.move("A", src + np.array([0, 0, 0.15]), 240)
    err = rec.move("A", np.array([res2.target_world_xy[0], res2.target_world_xy[1],
                                  TABLE_Z + 0.11]), 440)
    rt.data.ctrl[jaw] = 1.5
    rec.hold(0.8)
    rec.say("COMPLETE", f"Placed. Final gripper error {err * 100:.1f} cm", "ok", hold=2.4)

    _write(rec, out)


def _write(rec: Recorder, out: Path) -> None:
    import imageio.v2 as iio

    path = out / "coact_demo.mp4"
    with iio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=1) as w:
        for f in rec.frames:
            w.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    print(f"\n{len(rec.frames)} frames -> {path}")
    print(f"agent views -> {out / 'agent_views.png'}")


if __name__ == "__main__":
    main()
