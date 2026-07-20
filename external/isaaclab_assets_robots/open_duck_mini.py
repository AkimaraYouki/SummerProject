"""Configuration for Open Duck Mini V2 robot with XM430-W350 actuators.

USD 구조 (OnShape Importer, 2026-05-23):
  USD 경로         : ~/Desktop/Open_Duck_Mini_v2.usd
  ArticulationRoot : /Root/Open_Duck_Mini_v2/Open_Duck_Mini_v2
  Base body        : body_assembly
  Left foot body   : Frame_04
  Right foot body  : Frame_03

  제어 관절 (dof_):
    왼쪽 다리 (5): dof_left_hip_yaw/roll/pitch, dof_left_knee, dof_left_ankle
    오른쪽 다리 (5): dof_right_hip_yaw/roll/pitch, dof_right_knee, dof_right_ankle
    머리 (4): dof_neck_pitch, dof_head_pitch, dof_head_yaw, dof_head_roll
  고정 관절 (fix_): fix_left_antenna, fix_right_antenna → 액추에이터 불필요

BAM m4 XM430 피팅 파라미터:
  armature         = 0.01432  kg·m²
  friction (쿨롱)  = 0.07610  Nm
  viscous_friction = 0.84708  Nm·s/rad
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

OPEN_DUCK_MINI_USD_PATH = "/home/parksuho/Desktop/Open_Duck_Mini_v2.usd"

##
# Actuator configs
##

XM430_LEG_CFG = DCMotorCfg(
    joint_names_expr=[
        "dof_left_hip_yaw", "dof_left_hip_roll", "dof_left_hip_pitch",
        "dof_left_knee", "dof_left_ankle",
        "dof_right_hip_yaw", "dof_right_hip_roll", "dof_right_hip_pitch",
        "dof_right_knee", "dof_right_ankle",
    ],
    saturation_effort=4.1,    # XM430 stall torque @ 12V (Nm)
    effort_limit=3.5,
    velocity_limit=4.82,      # 46 rpm → rad/s
    stiffness={".*": 30.0},   # PD Kp — 첫 학습 후 튜닝
    damping={".*": 1.5},      # PD Kd
    armature=0.01432,         # BAM m4
    friction=0.07610,         # BAM m4: friction_base
    viscous_friction=0.84708, # BAM m4: friction_viscous
)

XM430_HEAD_CFG = ImplicitActuatorCfg(
    joint_names_expr=["dof_neck_pitch", "dof_head_pitch", "dof_head_yaw", "dof_head_roll"],
    effort_limit_sim=1.0,
    stiffness={".*": 10.0},
    damping={".*": 0.5},
)

# fix_left_antenna / fix_right_antenna:
# OnShape에서 "fix_" 접두어이지만 실제로는 revolute joint (enabled, limits ±90°).
# 보행 정책에서 제어할 필요 없으므로 stiffness=0, damping=0 패시브 처리.
# → 14 != 16 actuator 경고 제거.
PASSIVE_ANTENNA_CFG = ImplicitActuatorCfg(
    joint_names_expr=["fix_left_antenna", "fix_right_antenna"],
    effort_limit_sim=0.0,
    stiffness={".*": 0.0},
    damping={".*": 0.0},
)

##
# Robot config
##

OPEN_DUCK_MINI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=OPEN_DUCK_MINI_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.19),
        joint_pos={
            "dof_left_hip_yaw":    0.0,
            "dof_left_hip_roll":   0.0,
            "dof_left_hip_pitch":  0.45,
            "dof_left_knee":      -0.9,
            "dof_left_ankle":      0.45,
            "dof_right_hip_yaw":   0.0,
            "dof_right_hip_roll":  0.0,
            "dof_right_hip_pitch": 0.45,
            "dof_right_knee":     -0.9,
            "dof_right_ankle":     0.45,
            "dof_neck_pitch":  0.0,
            "dof_head_pitch":  0.0,
            "dof_head_yaw":    0.0,
            "dof_head_roll":   0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": XM430_LEG_CFG,
        "head": XM430_HEAD_CFG,
        "antenna": PASSIVE_ANTENNA_CFG,
    },
)
