"""MuJoCo runtime: real physics, real cameras, damped-least-squares IK.

This is what `coact/sim.py` returns when MuJoCo is installed. It owns the
physics, holds the two arms at a ready posture, moves an arm's gripper to a
Cartesian target, and renders the two embodied camera views that the agents
ground against.

The safety-critical distinction lives here too: `hover()` approaches a target
and stops above it. `commit()` is the only method that descends and closes the
jaw. The consensus gate decides which one is allowed to run.
"""

from __future__ import annotations

import os

import numpy as np

# EGL is the headless GL backend that works in containers and on most cloud
# hosts. Set before mujoco is imported anywhere.
os.environ.setdefault("MUJOCO_GL", "egl")

from .world import Item, build_mjcf, default_items  # noqa: E402

# Found by search: the only candidate ready-pose with zero actuator
# saturation and the gripper clear of the table.
HOME = np.array([0.0, -3.0, 3.0, 1.0, -1.57, 1.5])
JAW_OPEN, JAW_CLOSED = 1.5, 0.05
HOVER_HEIGHT = 0.105     # metres above the target's top surface
GRASP_HEIGHT = 0.032


class MuJoCoRuntime:
    name = "MuJoCo + dual SO-ARM100"
    connected = True

    def __init__(self, items: list[Item] | None = None,
                 width: int | None = None, height: int | None = None):
        import mujoco

        self.mj = mujoco
        self.items = items if items is not None else default_items()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.items))
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        # Each rendered frame costs real memory, and a small free-tier host has
        # very little of it. COACT_CAM_W/H let a deployment trade resolution for
        # headroom without touching the code.
        self._wh = (int(os.getenv("COACT_CAM_W", width or 640)),
                    int(os.getenv("COACT_CAM_H", height or 480)))
        self.frames: list[np.ndarray] = []

        self.arm_dofs = {
            arm: [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_{j}")
                for j in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
            ]
            for arm in ("A", "B")
        }
        # The arm's position servos have a 3.5 Nm force range, which is not
        # enough to hold an extended posture against gravity, so the tracked
        # pose drifts ~0.1 m below the commanded one. Enabling MuJoCo's gravity
        # compensation on the arm links models a servo that holds its own
        # weight — which is what the real SO-ARM100 controller does — instead of
        # inflating the torque limits past what the hardware can do.
        for b in range(self.model.nbody):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if nm.startswith(("A_", "B_")):
                self.model.body_gravcomp[b] = 1.0

        # The Menagerie arm ships in white plastic, which is the same colour as
        # the plates and merges with them under colour segmentation. Recolour
        # the links graphite so the cameras have an honest signal to separate
        # robot from crockery — the alternative is a detector that only works
        # because it was told where the robot is.
        for g in range(self.model.ngeom):
            b = self.model.geom_bodyid[g]
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if nm.startswith(("A_", "B_")):
                self.model.geom_rgba[g] = (0.17, 0.18, 0.20, 1.0)
                self.model.geom_matid[g] = -1

        self.gripper_body = {
            arm: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_Moving_Jaw")
            for arm in ("A", "B")
        }
        self.reset()

    # ------------------------------------------------------------------ basics
    def reset(self) -> None:
        self.mj.mj_resetData(self.model, self.data)
        for arm in ("A", "B"):
            for k, act in enumerate(self.arm_dofs[arm]):
                self.data.ctrl[act] = HOME[k]
                jnt = self.model.actuator_trnid[act, 0]
                self.data.qpos[self.model.jnt_qposadr[jnt]] = HOME[k]
        self.data.ctrl[self.arm_dofs["A"][5]] = JAW_OPEN
        self.data.ctrl[self.arm_dofs["B"][5]] = JAW_OPEN
        self.mj.mj_forward(self.model, self.data)
        self.settle(300)
        self.frames.clear()

    def settle(self, n: int = 200) -> None:
        for _ in range(n):
            self.mj.mj_step(self.model, self.data)

    def item_pos(self, item_id: str) -> np.ndarray:
        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, item_id)
        return self.data.xpos[bid].copy()

    def gripper_pos(self, arm: str) -> np.ndarray:
        return self.data.xpos[self.gripper_body[arm]].copy()

    # ---------------------------------------------------------------------- IK
    def solve_ik(self, arm: str, goal: np.ndarray, iters: int = 400,
                 damping: float = 0.03) -> tuple[np.ndarray, float]:
        """Damped-least-squares IK solved KINEMATICALLY on a scratch state.

        Running DLS directly on the dynamic state fights the position
        controller's tracking lag and stalls well short of the target. Solving
        kinematically first and then tracking the solution is both faster and
        far more accurate.
        """
        scratch = self.mj.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        self.mj.mj_forward(self.model, scratch)

        cols = [self.model.actuator_trnid[a, 0] for a in self.arm_dofs[arm][:5]]
        qadr = [self.model.jnt_qposadr[c] for c in cols]
        dofs = [self.model.jnt_dofadr[c] for c in cols]
        lims = [self.model.actuator_ctrlrange[a] for a in self.arm_dofs[arm][:5]]
        bid = self.gripper_body[arm]
        jacp = np.zeros((3, self.model.nv))

        best_q, best_err = np.array([scratch.qpos[a] for a in qadr]), 1e9
        for _ in range(iters):
            self.mj.mj_jacBody(self.model, scratch, jacp, None, bid)
            err = goal - scratch.xpos[bid]
            e = float(np.linalg.norm(err))
            if e < best_err:
                best_err, best_q = e, np.array([scratch.qpos[a] for a in qadr])
            if e < 5e-4:
                break
            J = jacp[:, dofs]
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err)
            dq = np.clip(dq, -0.05, 0.05)
            for k, a in enumerate(qadr):
                scratch.qpos[a] = float(np.clip(scratch.qpos[a] + dq[k], *lims[k]))
            self.mj.mj_forward(self.model, scratch)
        return best_q, best_err

    def move_to(self, arm: str, goal: np.ndarray, steps: int = 400,
                record: bool = False, every: int = 12) -> float:
        """Solve for a joint target, then ramp the controller onto it."""
        q_goal, _ = self.solve_ik(arm, goal)
        acts = self.arm_dofs[arm][:5]
        q_start = np.array([self.data.ctrl[a] for a in acts])
        for i in range(steps):
            alpha = min(1.0, (i + 1) / max(1, steps * 0.6))
            for k, a in enumerate(acts):
                self.data.ctrl[a] = float(q_start[k] + alpha * (q_goal[k] - q_start[k]))
            self.mj.mj_step(self.model, self.data)
            if record and i % every == 0:
                self.frames.append(self.render("cam_obs"))
        return float(np.linalg.norm(goal - self.gripper_pos(arm)))

    # ------------------------------------------------------------------ actions
    def hover(self, arm: str, target_id: str, record: bool = False) -> dict:
        """Approach and stop above the target. Never descends, never closes."""
        goal = self.item_pos(target_id) + np.array([0, 0, HOVER_HEIGHT])
        err = self.move_to(arm, goal, 700, record)
        return {"arm": arm, "action": "hover", "target": target_id, "simulated": True,
                "position_error_m": round(err, 4), "jaw": "open",
                "sim_time_s": round(float(self.data.time), 3)}

    def commit(self, arm: str, source_id: str, target_id: str, record: bool = False) -> dict:
        """The only path that touches anything: descend, close, lift, place."""
        jaw = self.arm_dofs[arm][5]
        self.move_to(arm, self.item_pos(source_id) + np.array([0, 0, HOVER_HEIGHT]), 300, record)
        self.move_to(arm, self.item_pos(source_id) + np.array([0, 0, GRASP_HEIGHT]), 300, record)
        self.data.ctrl[jaw] = JAW_CLOSED
        self.settle(120)
        self.move_to(arm, self.item_pos(source_id) + np.array([0, 0, HOVER_HEIGHT + 0.05]), 250, record)
        dest = self.item_pos(target_id) + np.array([0, 0, HOVER_HEIGHT])
        err = self.move_to(arm, dest, 420, record)
        self.data.ctrl[jaw] = JAW_OPEN
        self.settle(200)
        if record:
            self.frames.append(self.render("cam_obs"))
        moved = float(np.linalg.norm(self.item_pos(source_id)[:2] - self.item_pos(target_id)[:2]))
        return {"arm": arm, "action": "commit", "source": source_id, "target": target_id,
                "simulated": True, "position_error_m": round(err, 4),
                "final_xy_offset_m": round(moved, 4),
                "sim_time_s": round(float(self.data.time), 3)}

    def retract(self, arm: str, record: bool = False) -> dict:
        for k, act in enumerate(self.arm_dofs[arm][:5]):
            self.data.ctrl[act] = HOME[k]
        self.data.ctrl[self.arm_dofs[arm][5]] = JAW_OPEN
        for i in range(260):
            self.mj.mj_step(self.model, self.data)
            if record and i % 12 == 0:
                self.frames.append(self.render("cam_obs"))
        return {"arm": arm, "action": "retract", "simulated": True,
                "sim_time_s": round(float(self.data.time), 3)}

    # ----------------------------------------------------------------- cameras
    def render(self, camera: str = "cam_obs") -> np.ndarray:
        if self._renderer is None:
            self._renderer = self.mj.Renderer(self.model, self._wh[1], self._wh[0])
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def viewpoints(self) -> dict[str, np.ndarray]:
        """The two embodied views the agents actually ground against."""
        return {"A": self.render("cam_A"), "B": self.render("cam_B")}

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
