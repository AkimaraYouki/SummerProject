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
# Updated 2026-07-26: briefly "frame"/"frame_3" during an intermediate
# OnShape restructure, but the user rebuilt the mate hierarchy to match the
# upstream GitHub project's structure (and added properly-named Fastened
# mates for trunk_frame/left_foot_frame/right_foot_frame/head_frame — see
# patch_urdf_for_placo.py, now a no-op verifier since these come natively
# from OnShape), which reverted the foot link names back to
# foot_assembly/foot_assembly_2. Re-verify after any future re-import.
LEFT_FOOT_COLLISION_GEOM = "foot_bottom_tpu"
RIGHT_FOOT_COLLISION_GEOM = "foot_bottom_tpu_2"

# "home" keyframe joint positions (rad), in ACTUATOR_JOINT_NAMES order.
#
# UPDATED 2026-07-26 after the OnShape re-import that fixed left/right
# mass/joint-axis asymmetry AND re-set the assembly's reference pose to be
# upright: all-zero now IS the standing pose (previously needed the
# Playground-derived offsets below, which went out-of-range after the
# re-import and are kept here only as a historical note). Verified in
# Isaac Sim via check_joint_stability.sh with zero-action PD hold:
# steady-state base height settled at 0.1917-0.1942m, worst-case upright
# -0.9925 (-1=perfect), 0 soft-limit violations, PASS. See
# docs/decisions.md's "좌우 비대칭 발견" section for the full story.
#
# Old (pre-2026-07-26) values, source: playground xmls/scene_flat_terrain.xml
# <keyframe name="home">, with right_hip_pitch/right_knee negated to match
# this URDF's then-asymmetric joint-limit sign convention (since fixed):
#   left_hip_yaw=0.002, left_hip_roll=0.053, left_hip_pitch=-0.63,
#   left_knee=1.368, left_ankle=-0.784, right_hip_yaw=-0.003,
#   right_hip_roll=-0.065, right_hip_pitch=-0.635, right_knee=-1.379,
#   right_ankle=-0.796
HOME_JOINT_POS = {
    # The robot's physical rest/home pose: legs straight, everything at zero.
    # This is what the real hardware powers on at, and what a "go home" command
    # would drive to. It is deliberately NOT the pose training runs from -- see
    # READY_JOINT_POS below.
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": 0.0,
    "left_knee": 0.0,
    "left_ankle": 0.0,
    "neck_pitch": 0.0,
    "head_pitch": 0.0,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.0,
    "right_knee": 0.0,
    "right_ankle": 0.0,
}

# READY pose (2026-07-28): the stance the robot adopts when it starts
# operating -- crouched, knees loaded, ready to step. Computed by
# scripts/calc_home.py as the reference gait's mean pose over 8 representative
# commands x a full gait cycle.
#
# This is what the RL articulation initializes to, which makes it
# `default_joint_pos`, which is what actions are applied around:
#     target = default_joint_pos + action * action_scale(0.25)
# Training previously initialized to HOME (straight legs) instead, so reaching
# the reference's crouch (knee ~2.03 rad) needed action ~8.1 -- 8 sigma out of
# the policy's init distribution, i.e. unreachable by exploration. Measured
# consequence (scripts/imit_internals.py on imitation_v9): the joint_pos
# imitation term sat at +0.012 of a possible 1.0 with a permanent ~79 deg
# per-joint error and therefore a flat gradient, which is why v1-v9 all failed
# the same way no matter how the reward weights were tuned. From READY the
# gait's own amplitude needs only |action| <= 1.30.
READY_JOINT_POS = {
    # neck_pitch/head_pitch form the "Z" neck: specifying either angle fixes
    # the other, since the head is levelled by counter-rotating the neck's
    # lean. Set to 50 deg (0.8727) per request -- but head_pitch's URDF limit
    # is +-0.785 (45 deg), so it cancels only 45 of the neck's 50 and the head
    # ends up 5 deg off level. 45/-45 would be exactly level if that matters
    # more than the neck angle. "Level" here means level in the TRUNK frame:
    # the head carries no IMU, so no other frame is measurable on hardware.
    # The reference motion holds both joints at 0 and the imitation reward
    # excludes the head entirely, so this pose costs nothing in tracking error.
    "left_hip_yaw": 0.0004,
    "left_hip_roll": 0.0213,
    "left_hip_pitch": 1.1069,
    "left_knee": -2.0143,
    "left_ankle": 0.9785,
    "neck_pitch": 0.785,    # +45 deg
    "head_pitch": 0.785,    # +45 deg (sign flipped per visual check -- the
                            # two joints level the head at matching signs, not
                            # opposite ones, in this URDF's convention)
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": -0.0010,
    "right_hip_roll": -0.0018,
    "right_hip_pitch": 1.1197,
    "right_knee": 2.0320,
    "right_ankle": -0.9832,
}
# Base height with legs straight (HOME_JOINT_POS), measured via
# check_joint_stability.sh steady-state settle (2026-07-26, post-reimport).
HOME_BASE_HEIGHT = 0.193  # m

# Base height in the READY crouch, measured via scripts/settle_pose.py
# (2026-07-28): dropped in READY_JOINT_POS with action=0 and left to settle,
# the base holds [0.1175, 0.1237]m, mean 0.1208, pitch fixed at -3.3 deg with
# sub-cm drift. This is the height training actually operates at, so it -- not
# HOME_BASE_HEIGHT -- is what the spawn height and the `collapsed` termination
# threshold (READY_BASE_HEIGHT * cfg.min_base_height_ratio) must use.
READY_BASE_HEIGHT = 0.121  # m

# 스폰 전용 높이 (2026-07-29). READY_BASE_HEIGHT(0.121)는 READY 자세에서
# 로봇이 실제로 안착하는 높이라 종료 판정 기준으로는 맞지만, 스폰 높이로
# 쓰면 RSI와 충돌한다: _reset_idx는 루트 z를 항상 이 값으로 두는데 RSI는
# 다리 관절만 랜덤 위상의 레퍼런스 자세로 덮어쓰기 때문이다. pinocchio 순
# 기구학으로 위상별 필요 높이를 재보니 116.7~126.4 mm로 9.7 mm 변동하고,
# 발이 가장 낮은 위상에서는 지면을 5.4 mm 파고든다. PhysX가 그 관통을
# 해소하며 로봇을 튕겨내는 게 사용자가 관찰한 "처음에 뿅하고 튀어오름"이다.
# 130 mm면 어떤 위상에서도 관통이 없다(최대 13 mm 낙하는 튕김보다 훨씬
# 덜 교란적이다).
SPAWN_BASE_HEIGHT = 0.130


# ── 좌우 거울 규칙 ────────────────────────────────────────────────────────
# 오른다리 관절각이 왼다리의 몇 배여야 **자세가 좌우 대칭**인가.
#
# 추측하지 않고 URDF 에서 구했다 (2026-08-09): 부호 조합 2^5 = 32 가지를 전부
# 넣어 보고, 무작위 자세 12 개에서 좌우 발 위치가 서로 거울이 되는 조합을 골랐다.
# 그 조합에서 남는 오차는 0.5 mm 이고 (좌우 링크 질량도 완전히 동일하다), 따라서
# **로봇 모델 자체는 대칭**이다. 이 규칙에서 벗어나는 것은 전부 레퍼런스 또는
# 정책 탓이다.
#
# hip_yaw 만 부호가 뒤집히고 hip_roll / hip_pitch 는 같은 부호라는 점이 직관과
# 어긋나 보이는데, 실기 캘리브(hardware_map.py)의 "hip_yaw 만 좌우 같은 부호"
# 와는 **다른 이야기**다. 저쪽은 모터 회전 방향 부호이고 이쪽은 URDF 관절각의
# 거울 관계다. 섞지 말 것.
LEG_MIRROR_SIGN = {
    "hip_yaw": -1.0,
    "hip_roll": +1.0,
    "hip_pitch": +1.0,
    "knee": -1.0,
    "ankle": -1.0,
}

# ACTUATOR_JOINT_NAMES 안에서의 (좌 인덱스, 우 인덱스, 거울부호) 쌍.
LEG_MIRROR_PAIRS = [
    (ACTUATOR_JOINT_NAMES.index("left_" + j),
     ACTUATOR_JOINT_NAMES.index("right_" + j),
     s)
    for j, s in LEG_MIRROR_SIGN.items()
]
assert len(LEG_MIRROR_PAIRS) == 5
