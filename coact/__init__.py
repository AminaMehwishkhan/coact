"""CoAct — Consensus-Gated Bimanual Manipulation with Live-Correctable Confirmation."""

import os

# MuJoCo chooses its GL backend the first time it is imported, so this has to
# happen before anything in the package pulls mujoco in. EGL is the headless
# backend; without this a container with no DISPLAY falls through to GLFW and
# dies with "gladLoadGL error" the first time a camera renders.
os.environ.setdefault("MUJOCO_GL", "egl")

__version__ = "2.0.0"
