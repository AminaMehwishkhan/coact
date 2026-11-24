"""Builds the CoAct dinner-table world as a real MuJoCo scene.

Two SO-ARM100 arms (the open-source Standard Open Arm, vendored from MuJoCo
Menagerie under Apache-2.0) face each other across a table. Each arm carries a
wrist-mounted camera; each also has a fixed "head" camera looking over its own
shoulder at the table. Those two head cameras are the two embodied viewpoints
the whole project is about.

The scene is generated rather than hand-written so the benchmark can sample
hundreds of randomised table layouts with known ground truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "so_arm100"

# Arms sit on opposite long sides of the table, facing each other.
# Yaw is pi for A because the SO-ARM100 base extends along its own -y.
# The arms face each other along the x axis and the two candidate objects are
# separated along y. That makes "left" cleanly opposite for the two agents
# while keeping every object inside the SO-ARM100's ~0.30 m reach envelope.
# Yaw is chosen so each base's own -y (the direction the arm extends) points
# into the shared workspace.
ARM_A = {"pos": (-0.20, 0.0, 0.73), "yaw": 1.5708}
ARM_B = {"pos": (0.20, 0.0, 0.73), "yaw": -1.5708}

COLORS = {
    "red": "0.78 0.16 0.22 1",
    "blue": "0.18 0.33 0.72 1",
    "green": "0.20 0.58 0.32 1",
    "white": "0.90 0.90 0.88 1",
    "grey": "0.45 0.47 0.50 1",
    # Dark enough that the brightness-based crockery mask cannot pick it up:
    # the carafe is there to occlude, not to be a detection puzzle.
    "slate": "0.19 0.21 0.25 1",
}


@dataclass(frozen=True)
class Item:
    """One object on the table. `size` is semantic; `radius` is physical."""

    id: str
    label: str          # plate | cup | bowl
    x: float
    y: float
    size: str           # small | medium | large
    color: str
    radius: float
    graspable: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


# Sized so every object sits inside the SO-ARM100's ~0.30 m reach envelope.
PLATE_R = {"small": 0.042, "large": 0.068}
CUP_R = 0.022
BOWL_R = 0.050
CARAFE_R = 0.058
CARAFE_HALF_H = 0.095


def _plate(i: Item) -> str:
    return f"""
    <body name="{i.id}" pos="{i.x} {i.y} 0.735">
      <freejoint/>
      <geom name="{i.id}_g" type="cylinder" size="{i.radius} 0.006"
            rgba="{COLORS[i.color]}" mass="0.25" friction="1 .02 .001"/>
      <geom name="{i.id}_rim" type="cylinder" size="{i.radius} 0.009" pos="0 0 0.002"
            rgba="{COLORS[i.color]}" mass="0.02" contype="0" conaffinity="0"/>
    </body>"""


def _cup(i: Item) -> str:
    return f"""
    <body name="{i.id}" pos="{i.x} {i.y} 0.765">
      <freejoint/>
      <geom name="{i.id}_g" type="cylinder" size="{i.radius} 0.030"
            rgba="{COLORS[i.color]}" mass="0.12" friction="1 .02 .001"/>
    </body>"""


def _bowl(i: Item) -> str:
    return f"""
    <body name="{i.id}" pos="{i.x} {i.y} 0.762">
      <freejoint/>
      <geom name="{i.id}_g" type="ellipsoid" size="{i.radius} {i.radius} 0.026"
            rgba="{COLORS[i.color]}" mass="0.30" friction="1 .02 .001"/>
    </body>"""


def _carafe(i: Item) -> str:
    """A tall opaque object. Grey, so the detector has no class for it — it is
    there to occlude, which is exactly what unmodelled clutter does in a real
    kitchen."""
    # Static: it is fixed clutter standing beside the table, not something the
    # arms interact with, and a free body this close to the edge would simply
    # topple off before the cameras ever saw it.
    return f"""
    <body name="{i.id}" pos="{i.x} {i.y} {0.737 + CARAFE_HALF_H}">
      <geom name="{i.id}_g" type="cylinder" size="{i.radius} {CARAFE_HALF_H}"
            rgba="{COLORS[i.color]}" contype="0" conaffinity="0"/>
    </body>"""


_BUILDERS = {"plate": _plate, "cup": _cup, "bowl": _bowl, "carafe": _carafe}


def build_mjcf(items: list[Item], asset_dir: Path | str = ASSET_DIR) -> str:
    """Return a complete MJCF string for a table laid out with `items`."""
    asset_dir = Path(asset_dir)
    arm_xml = (asset_dir / "so_arm100.xml").as_posix()
    bodies = "".join(_BUILDERS[i.label](i) for i in items)

    return f"""<mujoco model="coact_dinner_table">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" cone="elliptic" impratio="10" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.42 0.42 0.42" specular="0.1 0.1 0.1"/>
    <quality shadowsize="2048"/>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <model name="soarm" file="{arm_xml}"/>
    <texture type="skybox" builtin="gradient" rgb1="0.05 0.07 0.10" rgb2="0.01 0.02 0.03"
             width="256" height="1536"/>
    <texture type="2d" name="wood" builtin="checker" rgb1="0.33 0.26 0.19" rgb2="0.29 0.22 0.16"
             width="300" height="300"/>
    <material name="wood" texture="wood" texuniform="true" texrepeat="4 4" reflectance="0.05"/>
    <material name="floor" rgba="0.12 0.13 0.15 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 2.2" dir="0 0 -1" directional="true" diffuse="0.7 0.7 0.7"/>
    <light pos="-0.6 -0.6 1.8" dir="0.3 0.3 -1" diffuse="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor"/>

    <!-- table -->
    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" size="0.17 0.155 0.02" pos="0 0 0.71"
            material="wood" friction="1 .02 .001"/>
      <geom name="leg1" type="box" size="0.02 0.02 0.355" pos="0.13 0.12 0.355" material="wood"/>
      <geom name="leg2" type="box" size="0.02 0.02 0.355" pos="-0.13 0.12 0.355" material="wood"/>
      <geom name="leg3" type="box" size="0.02 0.02 0.355" pos="0.13 -0.12 0.355" material="wood"/>
      <geom name="leg4" type="box" size="0.02 0.02 0.355" pos="-0.13 -0.12 0.355" material="wood"/>
    </body>
{bodies}

    <!-- arm pedestals: the arms stand beside the table, not on it, so the
         base never collides with the workspace it is reaching into -->
    <geom name="ped_A" type="box" size="0.045 0.045 0.365" pos="-0.20 0 0.365" material="wood"/>
    <geom name="ped_B" type="box" size="0.045 0.045 0.365" pos="0.20 0 0.365" material="wood"/>

    <!-- Arm A: south side, looking north across the table -->
    <body name="A_mount" pos="{ARM_A['pos'][0]} {ARM_A['pos'][1]} {ARM_A['pos'][2]}" euler="0 0 {ARM_A['yaw']}">
      <attach model="soarm" body="Base" prefix="A_"/>
    </body>
    <camera name="cam_A" mode="fixed" pos="-0.44 0 1.52" xyaxes="0 -1 0 0.861 0 0.508" fovy="36"/>

    <!-- Arm B: north side, looking south across the table -->
    <body name="B_mount" pos="{ARM_B['pos'][0]} {ARM_B['pos'][1]} {ARM_B['pos'][2]}" euler="0 0 {ARM_B['yaw']}">
      <attach model="soarm" body="Base" prefix="B_"/>
    </body>
    <camera name="cam_B" mode="fixed" pos="0.44 0 1.52" xyaxes="0 1 0 -0.861 0 0.508" fovy="36"/>

    <!-- observer camera, for recordings only; never used for grounding -->
    <camera name="cam_obs" mode="fixed" pos="0.52 -0.62 1.22" xyaxes="0.68 0.74 0 -0.38 0.35 0.86"/>
  </worldbody>
</mujoco>
"""


def default_items() -> list[Item]:
    """The canonical demo table: two plates of different size, two cups, a bowl."""
    return [
        Item("plate_small", "plate", 0.0, -0.088, "small", "white", PLATE_R["small"]),
        Item("plate_large", "plate", 0.0, 0.088, "large", "white", PLATE_R["large"]),
        Item("cup_red", "cup", -0.072, 0.0, "medium", "red", CUP_R, graspable=True),
        Item("cup_blue", "cup", 0.072, 0.0, "medium", "blue", CUP_R, graspable=True),
    ]


def assets_present() -> bool:
    return (ASSET_DIR / "so_arm100.xml").exists() and os.path.isdir(ASSET_DIR / "assets")
