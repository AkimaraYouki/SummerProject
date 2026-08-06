#!/usr/bin/env python3
"""Verify robot/robot.urdf has the fixed-joint frame aliases Placo's
HumanoidRobot requires by name (trunk/left_foot/right_foot/head), then copy
it into reference_motion_generator's robots/open_duck_mini_v2/ dir.

History: this used to INJECT these 4 frames as hardcoded fixed-joint offsets
copied from an old pre-rebuild URDF, because our OnShape export didn't have
them. That was a real source of bugs — the injected left/right foot offsets
were identical (not mirrored), which visibly asymmetric-ified every
generated gait even after the URDF's own mass/joint-limit asymmetries were
fixed. As of 2026-07-26 the user added real Fastened mates named
trunk_frame/left_foot_frame/right_foot_frame/head_frame directly in OnShape
(matching the upstream GitHub project's mate structure), so onshape-to-robot
now emits these natively with correct, properly-mirrored transforms — same
mechanism that already produced imu_frame natively. This script is now just
a safety-net verifier + copy step: it only injects the old hardcoded
FALLBACK_FRAME_ALIASES for any frame that's still missing (which should not
happen anymore), and always warns loudly if it has to.

This is a real file (not a symlink) written into
reference_motion_generator/.../open_duck_mini_v2/ for BOTH Mac and the
exFAT-formatted lab-PC SSD (which can't hold symlinks at all) — re-run this
script after any OnShape re-import instead of relying on the old
symlink-to-robot/robot.urdf approach.

Usage:
  python3 scripts/patch_urdf_for_placo.py
"""

import os
import re
import shutil

# scripts/setup/ 에 있으므로 세 단계 올라가야 레포 루트다. 두 단계였던 시절
# (스크립트가 scripts/ 에 있었을 때) 그대로 남아 있어서, 옮긴 뒤로는 실행하면
# scripts/robot/robot.urdf 를 찾다가 FileNotFoundError 로 죽었다. 그 결과
# 2026-07-26 이후 아무도 이걸 못 돌렸고, 생성기의 URDF 가 그날 형상(2.3388 kg)에
# 멈춰 있었다 — 새 CAD 로 레퍼런스를 만들려던 참에 발견했다 (2026-08-07).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_URDF = os.path.join(_REPO_ROOT, "robot", "robot.urdf")
DST_URDF = os.path.join(
    _REPO_ROOT,
    "reference_motion_generator",
    "open_duck_reference_motion_generator",
    "robots",
    "open_duck_mini_v2",
    "open_duck_mini_v2.urdf",
)

SRC_ASSETS = os.path.join(_REPO_ROOT, "robot", "assets")
DST_ASSETS = os.path.join(os.path.dirname(DST_URDF), "assets")

# Required child frame link names Placo's HumanoidRobot looks up by name.
REQUIRED_FRAME_LINKS = ["trunk", "left_foot", "right_foot", "head"]

# (joint_name, parent_link, child_link, xyz, rpy) — STALE fallback values
# inherited from the old (2026-07-15) pre-rebuild URDF. Only used if OnShape
# stops providing one of REQUIRED_FRAME_LINKS natively; do not trust these
# for anything but an emergency stopgap (see module docstring).
FALLBACK_FRAME_ALIASES = [
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

    missing = [name for name in REQUIRED_FRAME_LINKS if name not in existing_links]

    additions = []
    if not missing:
        print(f"All required frame links present natively from OnShape: {REQUIRED_FRAME_LINKS}")
    else:
        print(
            f"WARNING: {missing} missing from OnShape export — falling back to STALE "
            "hardcoded offsets (see FALLBACK_FRAME_ALIASES docstring warning). "
            "Add named Fastened mates for these in OnShape instead of relying on this fallback."
        )
        for joint, parent, child, xyz, rpy in FALLBACK_FRAME_ALIASES:
            if child not in missing:
                continue
            if parent not in existing_links:
                raise SystemExit(
                    f"patch_urdf_for_placo: fallback parent link '{parent}' (needed for '{joint}') "
                    f"not found in {SRC_URDF} either — update FALLBACK_FRAME_ALIASES."
                )
            additions.append(ZERO_MASS_LINK_TEMPLATE.format(name=child))
            additions.append(FIXED_JOINT_TEMPLATE.format(joint=joint, parent=parent, child=child, xyz=xyz, rpy=rpy))

    patched = urdf.replace("</robot>", "".join(additions) + "</robot>") if additions else urdf

    os.makedirs(os.path.dirname(DST_URDF), exist_ok=True)
    with open(DST_URDF, "w") as f:
        f.write(patched)
    print(f"Wrote Placo-compatible URDF: {DST_URDF}")

    # 메시도 같이 옮긴다. URDF 만 갱신하고 assets 를 두면 placo 가 옛 형상으로
    # 궤적을 만든다 — URDF 는 mesh 를 package://assets/*.stl 로 참조하므로 파일
    # 이름이 같으면 조용히 낡은 메시를 읽는다. 2026-08-07 에 실제로 URDF 는
    # 7/26 판, assets 는 46개(현재 56개)로 어긋나 있었다. 문서가 "수동 재복사" 로
    # 남겨둔 단계인데, 수동이라 빠졌다.
    if os.path.isdir(SRC_ASSETS):
        if os.path.isdir(DST_ASSETS):
            shutil.rmtree(DST_ASSETS)
        shutil.copytree(SRC_ASSETS, DST_ASSETS)
        n = len([f for f in os.listdir(DST_ASSETS) if f.endswith(".stl")])
        print(f"Copied meshes: {DST_ASSETS}  ({n} STL)")
    else:
        print(f"WARNING: {SRC_ASSETS} not found — meshes NOT refreshed.")


if __name__ == "__main__":
    main()
