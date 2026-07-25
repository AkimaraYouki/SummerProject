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

# Present in the URDF/MJCF but not actuated (no <actuator> entry).
NON_ACTUATED_JOINT_NAMES = ["left_antenna", "right_antenna"]

# Index into ACTUATOR_JOINT_NAMES (14-dim) of the 10 leg joints, i.e. every
# actuated joint EXCEPT the 4 head joints (indices 5-8: neck_pitch,
# head_pitch, head_yaw, head_roll). Left leg first, then right leg.
ACT_LEG_JOINT_IDX = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# The reference-motion pkl (playground/common/poly_reference_motion.py's
# header comment) encodes a 16-joint layout that additionally includes the
# two (non-actuated) antenna joints, in this order:
#   0 left_hip_yaw, 1 left_hip_roll, 2 left_hip_pitch, 3 left_knee, 4 left_ankle,
#   5 neck_pitch, 6 head_pitch, 7 head_yaw, 8 head_roll,
#   9 left_antenna, 10 right_antenna,
#   11 right_hip_yaw, 12 right_hip_roll, 13 right_hip_pitch, 14 right_knee, 15 right_ankle
REF_JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "left_antenna", "right_antenna",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
NUM_REF_JOINTS = len(REF_JOINT_NAMES)  # 16

# Index into REF_JOINT_NAMES (16-dim) of the same 10 leg joints, same order
# as ACT_LEG_JOINT_IDX above (left leg first, then right leg). Indices 5-10
# (head x4 + antennas x2) are excluded.
REF_LEG_JOINT_IDX = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]

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
# Source: playground xmls/scene_flat_terrain.xml <keyframe name="home">.
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
    "right_hip_pitch": 0.635,
    "right_knee": 1.379,
    "right_ankle": -0.796,
}
HOME_BASE_HEIGHT = 0.15  # m
