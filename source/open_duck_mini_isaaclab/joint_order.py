"""Single source of truth for Open Duck Mini V2's joint ordering.

Every module in this package (robot_cfg, observations, rewards, ...) must
import these constants instead of re-listing joint names, so the ordering
can never silently drift between files the way it did in the original
Playground code (see docs/decisions.md's note on reward_imitation's slices).

Source of truth: Open_Duck_Playground's
playground/open_duck_mini_v2/xmls/open_duck_mini_v2.xml <actuator> block,
cross-checked against mini_bdx/robots/open_duck_mini_v2/robot.urdf joint
names (same OnShape export, confirmed matching).
"""

# The 14 actuated joints, in MJCF <actuator> block order. This is the order
# used for: robot_cfg.py's ArticulationCfg joint_pos dict iteration order
# assumptions, action vector order, joint_pos/joint_vel observation order,
# and motor_targets order. DO NOT reorder — it must match the exported ONNX
# policy's expected action layout for any future sim2real work.
ACTUATOR_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
NUM_ACTUATED_JOINTS = len(ACTUATOR_JOINT_NAMES)  # 14

# Playground's original hardware had 2 non-actuated antenna joints; this
# rebuild's OnShape export has none at all (confirmed 2026-07-26: zero
# "antenna" occurrences anywhere in robot/robot.urdf — no antenna hardware
# was modeled). No NON_ACTUATED_JOINT_NAMES constant needed as a result.

# Index into ACTUATOR_JOINT_NAMES (14-dim) of the 10 leg joints, i.e. every
# actuated joint EXCEPT the 4 head joints (indices 5-8: neck_pitch,
# head_pitch, head_yaw, head_roll). Left leg first, then right leg.
ACT_LEG_JOINT_IDX = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# The reference-motion pkl (playground/common/poly_reference_motion.py's
# header comment) originally encoded a 16-joint layout including 2 antenna
# joints this rebuild doesn't have; reference_motion_generator's placo
# configs were fixed to stop requesting them (2026-07-26, see
# docs/training_log.md), so the recorded/fitted reference frame is now
# 14-joint — same order as ACTUATOR_JOINT_NAMES:
#   0 left_hip_yaw, 1 left_hip_roll, 2 left_hip_pitch, 3 left_knee, 4 left_ankle,
#   5 neck_pitch, 6 head_pitch, 7 head_yaw, 8 head_roll,
#   9 right_hip_yaw, 10 right_hip_roll, 11 right_hip_pitch, 12 right_knee, 13 right_ankle
REF_JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
NUM_REF_JOINTS = len(REF_JOINT_NAMES)  # 14 — now numerically identical to ACTUATOR_JOINT_NAMES,
# kept as a separate constant anyway since it documents a distinct concept
# (reference-pkl layout vs. actuator/action-vector layout) that happened to
# converge only because antennas were the sole difference.

# Index into REF_JOINT_NAMES (14-dim) of the same 10 leg joints, same order
# as ACT_LEG_JOINT_IDX above (left leg first, then right leg). Now
# numerically identical to ACT_LEG_JOINT_IDX for the same reason as
# NUM_REF_JOINTS above. Indices 5-8 (head x4) are excluded.
REF_LEG_JOINT_IDX = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# Sanity: both index lists must select the same 10 physical joints in the
# same order when applied to their respective name lists.
assert [ACTUATOR_JOINT_NAMES[i] for i in ACT_LEG_JOINT_IDX] == [
    REF_JOINT_NAMES[i] for i in REF_LEG_JOINT_IDX
], "ACT_LEG_JOINT_IDX / REF_LEG_JOINT_IDX select different joints — fix before using either."

# Body / site names, verified directly against robot/robot.urdf and the
# converted USD (2026-07-25, OnShape-derived rebuild). These do NOT match
# the original project's MJCF names — e.g. there is no link literally named
# "base" (Playground's MJCF had one; this URDF's root/trunk link is named
# ROOT_BODY_NAME below instead). Re-verify after any future re-import, since
# onshape-to-robot derives these from OnShape mate/part names.
ROOT_BODY_NAME = "trunk_assembly"
LEFT_FOOT_BODY_NAME = "foot_assembly"
RIGHT_FOOT_BODY_NAME = "foot_assembly_2"
# Same left/"_2"-suffix-for-right convention as the rest of this URDF
# (foot_assembly/foot_assembly_2, hip_roll_assembly/hip_roll_assembly_2,
# etc.) — NOT "left_"/"right_"-prefixed like the original project's names.
LEFT_FOOT_COLLISION_GEOM = "foot_bottom_tpu"
RIGHT_FOOT_COLLISION_GEOM = "foot_bottom_tpu_2"

# "home" keyframe joint positions (rad), in ACTUATOR_JOINT_NAMES order.
# Source: playground xmls/scene_flat_terrain.xml <keyframe name="home">,
# EXCEPT right_hip_pitch and right_knee — see note below.
#
# This URDF's <joint><limit> ranges (checked 2026-07-25 against
# robot/robot.urdf) don't use the same left/right sign convention
# Playground's original URDF did:
#   right_hip_pitch: [-1.745, 0]  — SAME sign as left_hip_pitch's [-1.745, 0]
#                                    (Playground's was mirrored: left negative,
#                                    right positive)
#   right_knee:      [-3.142, 0]  — OPPOSITE sign from left_knee's [0, 3.142]
#                                    (Playground's was NOT mirrored: both
#                                    positive)
# Playground's original values (right_hip_pitch=0.635, right_knee=1.379) are
# out of range for this URDF (ValueError: default positions out of limits,
# hit during the 2026-07-25 smoke test) — negated below to land the same
# physical pose under this URDF's axis convention. NOT visually verified in
# Isaac Sim yet — if the robot's home stance looks asymmetric, check these
# two first.
HOME_JOINT_POS = {
    "left_hip_yaw": 0.002,
    "left_hip_roll": 0.053,
    "left_hip_pitch": -0.63,
    "left_knee": 1.368,
    "left_ankle": -0.784,
    "neck_pitch": 0.0,
    "head_pitch": 0.0,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": -0.003,
    "right_hip_roll": -0.065,
    "right_hip_pitch": -0.635,
    "right_knee": -1.379,
    "right_ankle": -0.796,
}
HOME_BASE_HEIGHT = 0.15  # m
