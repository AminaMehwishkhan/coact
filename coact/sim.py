"""Which simulation runtime this process gets.

CoAct's contribution sits above the motion stack, so the simulator is behind a
small interface. With MuJoCo installed you get real physics, real cameras and
real perception. Without it — which is the case on a small cloud dyno — you get
`NullRuntime`, `/api/health` reports MuJoCo as not connected, and the dashboard
says so rather than pretending.
"""

from __future__ import annotations

import os


class NullRuntime:
    """No physics. Reports itself as absent; never fakes a measurement."""

    name = "unavailable (MuJoCo not installed)"
    connected = False
    items: list = []

    def reset(self) -> None:
        return None

    def render(self, camera: str = "cam_obs"):
        raise RuntimeError("No physics runtime: nothing to render.")


def mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401
    except Exception:
        return False
    from .world import assets_present

    return assets_present()


def get_runtime(items=None):
    """Return a MuJoCoRuntime when one can be built, else NullRuntime."""
    if os.getenv("COACT_FORCE_NULL_RUNTIME") == "1" or not mujoco_available():
        return NullRuntime()
    try:
        from .mj_runtime import MuJoCoRuntime

        return MuJoCoRuntime(items=items)
    except Exception:
        # A missing GL backend on a headless host is the usual cause. Degrade
        # loudly in /api/health rather than crashing the server at import time.
        return NullRuntime()
