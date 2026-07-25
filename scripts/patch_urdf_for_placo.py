#!/usr/bin/env python3
"""Patch robot/robot.urdf with the fixed-joint frame aliases Placo's
HumanoidRobot requires by name (trunk/left_foot/right_foot/head), then write
the result into reference_motion_generator's robots/open_duck_mini_v2/ dir.

Why this exists: our OnShape-exported robot.urdf names its links after
OnShape part/assembly names (trunk_assembly, foot_assembly, foot_assembly_2,
head_pitch_assembly, ...), but placo.HumanoidRobot(urdf_path) looks up
specific canonical frame names ("left_foot", "right_foot", possibly "trunk",
"head") internally and raises `RuntimeError: Frame with name left_foot not
found in model` if they're missing. The original (now-replaced) bundled copy
of this URDF had these as extra zero-mass fixed joints; our raw OnShape
export doesn't, since onshape-to-robot has no concept of "Placo's expected
frame names."

This is a real file (not a symlink) written into
reference_motion_generator/.../open_duck_mini_v2/ for BOTH Mac and the
exFAT-formatted lab-PC SSD (which can't hold symlinks at all) — re-run this
script after any OnShape re-import instead of relying on the old
symlink-to-robot/robot.urdf approach.

The 4 fixed-joint offsets below are inherited from the old (2026-07-15,
pre-rebuild) bundled URDF as a best-effort approximation, NOT re-derived
from the current geometry — reasonable since the underlying kinematic
design hasn't changed, but flagged here as worth revisiting if reference
motions look physically off (especially foot contact height/orientation).
head_frame's parent changed from the old "head_assembly" (link no longer
exists) to "head_pitch_assembly" (closest current equivalent) — also
approximate, and lower-priority since gait planning cares mostly about
trunk/feet.

Usage:
  python3 scripts/patch_urdf_for_placo.py
"""

import os
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_URDF = os.path.join(_REPO_ROOT, "robot", "robot.urdf")
DST_URDF = os.path.join(
    _REPO_ROOT,
    "reference_motion_generator",
    "open_duck_reference_motion_generator",
    "robots",
    "open_duck_mini_v2",
    "open_duck_mini_v2.urdf",
)

# (joint_name, parent_link, child_link, xyz, rpy) — xyz/rpy inherited from the
# pre-rebuild bundled URDF (see module docstring).
FRAME_ALIASES = [
    (
        "trunk_frame",
        "trunk_assembly",
        "trunk",
        "-0.024 0 0.08819",
        "0 0 0",
    ),
    (
        "left_foot_frame",
        "foot_assembly",
        "left_foot",
        "0.0005 -0.036225 0.01955",
        "-1.570796326794896558 0 0",
    ),
    (
        "right_foot_frame",
        "foot_assembly_2",
        "right_foot",
        "0.0005 -0.036225 0.01955",
        "-1.570796326794896558 0 0",
    ),
    (
        "head_frame",
        "head_pitch_assembly",
        "head",
        "0.04245 0 0.03595",
        "0 1.570796326794896558 0",
    ),
]

ZERO_MASS_LINK_TEMPLATE = """    <link name="{name}">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1e-9"/>
            <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>
    </link>
"""

FIXED_JOINT_TEMPLATE = """    <joint name="{joint}" type="fixed">
        <origin xyz="{xyz}" rpy="{rpy}"/>
        <parent link="{parent}"/>
        <child link="{child}"/>
        <axis xyz="0 0 0"/>
    </joint>
"""


def main():
    with open(SRC_URDF) as f:
        urdf = f.read()

    existing_links = set(re.findall(r'<link name="([^"]+)"', urdf))

    additions = []
    for joint, parent, child, xyz, rpy in FRAME_ALIASES:
        if parent not in existing_links:
            raise SystemExit(
                f"patch_urdf_for_placo: parent link '{parent}' (needed for '{joint}') "
                f"not found in {SRC_URDF} — robot.urdf's link names have changed, update FRAME_ALIASES."
            )
        if child in existing_links:
            print(f"[skip] link '{child}' already exists in robot.urdf, not adding a duplicate")
            continue
        additions.append(ZERO_MASS_LINK_TEMPLATE.format(name=child))
        additions.append(FIXED_JOINT_TEMPLATE.format(joint=joint, parent=parent, child=child, xyz=xyz, rpy=rpy))

    patched = urdf.replace("</robot>", "".join(additions) + "</robot>")

    os.makedirs(os.path.dirname(DST_URDF), exist_ok=True)
    with open(DST_URDF, "w") as f:
        f.write(patched)
    print(f"Wrote Placo-compatible URDF: {DST_URDF}")
    print(f"Added frame aliases: {[a[0] for a in FRAME_ALIASES]}")


if __name__ == "__main__":
    main()
