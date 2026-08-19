"""DirectRLEnvCfg port of Open_Duck_Playground's joystick.py::default_config().

Every numeric value here is copied from
Open_Duck_Playground/playground/open_duck_mini_v2/joystick.py's
`default_config()` unless a comment says otherwise. Where Isaac Lab requires
something Playground doesn't have an equivalent for (terrain, contact
sensor, IMU sensor configs), it's added using the same pattern as Isaac
Lab's own anymal_c_env_cfg.py (a first-party DirectRLEnv locomotion example
this port was modeled on).

Simplification vs. Playground (documented, not accidental): this v1 port
does NOT implement Playground's asymmetric-critic `privileged_state`
(state_space=0, like Isaac Lab's own Anymal example). The policy-visible
101-dim `state` observation is what determines behavior and sim2real
fidelity; the privileged critic input is a training-efficiency optimization
that can be added later without changing the task's semantics.
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from open_duck_mini_isaaclab.joint_order import (
    ACTUATOR_JOINT_NAMES,
    READY_BASE_HEIGHT,
    LEFT_FOOT_BODY_NAME,
    RIGHT_FOOT_BODY_NAME,
    ROOT_BODY_NAME,
)
from open_duck_mini_isaaclab.imu_map import MOUNT_POS as IMU_MOUNT_POS
from open_duck_mini_isaaclab.robot_cfg import (
    OPEN_DUCK_MINI_BIGFOOT_USD_PATH,
    OPEN_DUCK_MINI_V2_CFG,
    OPEN_DUCK_MINI_V2_DC_CFG,
)

from .events import randomize_default_joint_pos as randomize_default_joint_pos_event

# ── Observation dimension (computed, not hardcoded — see joint_order.py) ──
# gyro(3) + accel(3) + command(3 vel + 4 head = 7) + joint_pos_rel(14) +
# joint_vel(14) + last_act(14) + last_last_act(14) + last_last_last_act(14) +
# motor_targets(14) + contact(2) + imitation_phase(2) = 101
# (Playground's own joystick.py has stale "# 10" comments next to some of
# these terms, left over from an earlier no-head-actuation version — the
# real per-term width is len(ACTUATOR_JOINT_NAMES)==14 for every joint-sized
# term. Trust this computation, not those comments.)
NUM_JOINTS = len(ACTUATOR_JOINT_NAMES)  # 14
NUM_COMMANDS = 7  # lin_vel_x, lin_vel_y, ang_vel_yaw, neck_pitch, head_pitch, head_yaw, head_roll
OBS_STATE_DIM = (
    3  # gyro
    + 3  # accelerometer
    + NUM_COMMANDS
    + NUM_JOINTS  # joint_pos_rel
    + NUM_JOINTS  # joint_vel
    + NUM_JOINTS  # last_act
    + NUM_JOINTS  # last_last_act
    + NUM_JOINTS  # last_last_last_act
    + NUM_JOINTS  # motor_targets
    + 2  # feet contact
    + 2  # imitation phase (cos, sin)
)
assert OBS_STATE_DIM == 101, f"expected 101-dim state obs, computed {OBS_STATE_DIM} — check NUM_JOINTS/NUM_COMMANDS"

# Privileged (critic-only) observation — see joystick_env.py::_get_observations.
# Mirrors upstream's privileged_state: policy state + noiseless sensors + the
# true base velocity + torques + foot velocities + root height + the full
# reference frame. Computed, not hardcoded, so a joint-count change can't
# silently desync it from the tensor actually built.
# path frame이 켜지면 정책/크리틱 관측에 3차원이 붙는다
# (횡방향 오차, cos(방향오차), sin(방향오차)).
PATH_ERR_DIM = 3

# 중력 방향(projected_gravity, 3차원). 정책 관측에 넣을 때만 붙는다.
# 상류 Playground도 정책 state에는 안 넣는다(계산만 하고 버리는 죽은 코드) —
# 빠뜨린 게 아니라 원래 그렇다. v27은 여기서 의도적으로 이탈한다.
# noise_scale_gravity 는 그동안 정의만 돼 있고 쓰이지 않았는데, 이제 쓰인다.
GRAVITY_OBS_DIM = 3

REF_FRAME_WIDTH = 36  # poly_reference_motion.REF_FRAME_DIM
NUM_FEET = 2
OBS_CRITIC_DIM = (
    OBS_STATE_DIM
    + 3   # gyro (noiseless)
    + 3   # accelerometer (noiseless)
    + 3   # projected gravity
    + 3   # root_lin_vel_b — the policy never observes true velocity
    + 3   # root_ang_vel_w
    + NUM_JOINTS  # joint_pos_rel (noiseless)
    + NUM_JOINTS  # joint_vel (noiseless, unscaled)
    + 1   # root height
    + NUM_JOINTS  # applied torque
    + 2   # feet contact
    + NUM_FEET * 3  # feet linear velocity
    + REF_FRAME_WIDTH
    + 2   # imitation phase
)


@configclass
class EventCfg:
    """Domain randomization — see docs/decisions.md mapping table.

    6 of Playground's 7 randomization knobs map to IsaacLab builtins; the
    7th (qpos0/default-joint-pos jitter) has no builtin and is implemented
    in events.py, wired in here once Stage 4 lands.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk_assembly"),
            "mass_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )

    scale_all_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    randomize_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.9, 1.1),
            "armature_distribution_params": (1.0, 1.05),
            "operation": "scale",
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )

    # 7th knob: no IsaacLab builtin covers this (see events.py docstring).
    randomize_default_joint_pos = EventTerm(
        func=randomize_default_joint_pos_event,
        mode="reset",
        params={
            "position_range": (-0.03, 0.03),
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )


@configclass
class JoystickEnvCfg(DirectRLEnvCfg):
    # ── env ──────────────────────────────────────────────────────────────
    decimation = 10  # ctrl_dt(0.02) / sim_dt(0.002)
    episode_length_s = 20.0  # 1000 steps * ctrl_dt(0.02)
    action_scale = 0.25
    action_space = NUM_JOINTS  # 14 — policy directly commands all 14 actuators
    observation_space = OBS_STATE_DIM  # 101
    state_space = 0  # see module docstring — no asymmetric critic in v1

    dof_vel_scale = 0.05
    # rad/s, clamps the per-step motor-target delta. XM430-W350 no-load speed
    # @ 12.0V (confirmed operating voltage) = 46 rpm; must stay in sync with
    # robot_cfg.py's velocity_limit_sim (same datasheet source).
    max_motor_velocity = 4.82

    # 고관절 roll/yaw 가 레퍼런스에서 벗어날 수 있는 한계 (rad). None 이면 끈다.
    # 다리-몸통 접촉이 이 이탈에 딸려 온다 -- v25 는 접촉 56.7%, v28 은 이탈이
    # 절반으로 줄면서 접촉도 11.5% 로 떨어졌다 (leg_trunk_clearance.py).
    # 실기에서는 접촉이 곧 파손이라 0 이어야 한다.
    hip_dev_limit_yaw = None
    hip_dev_limit_roll = None

    # 고관절이 레퍼런스보다 **안쪽으로** 이 각(rad)을 넘게 돌면 벌점. None 이면 끈다.
    # v31 이 목표(_motor_targets)를 양방향으로 잘랐다가 실패했다 -- 실제 관절각은
    # 안 묶이고 자세만 무너졌다. 여기서는 실측 관절각에, 안쪽 한 방향만 건다.
    hip_inward_thresh = None
    #: 레퍼런스보다 이만큼(rad) 넘게 **바깥으로** 벌어지면 벌한다. None 이면 끔.
    #: hip_inward_thresh 와 같은 3 도. 같은 계수(hip_inward_scale)를 쓰되
    #: hip_outward_rel 로 비율을 조절한다.
    hip_outward_thresh = None
    #: 바깥 항이 안쪽 항 대비 갖는 비중.
    hip_outward_rel = 1.0
    hip_inward_scale = -25.0
    # hip_inward 를 보행 중에만 걸지. 기본 False = 종전대로 정지에서도 건다.
    # 이 항은 레퍼런스를 기준으로 삼는데 정지에서는 그 기준이 "걷는 중 한쪽 발을
    # 든 순간" 이라, 정지 좌우대칭을 요구하는 leg_symmetry 와 정면으로 싸운다.
    hip_inward_walking_only = False

    # 런타임 안전 필터 (safety_filter.py). 경로를 주면 켜진다.
    # 리워드는 위반을 줄일 뿐 0 을 보장하지 못한다 -- 액추에이터 파손이 걸린
    # 조건에서는 필터만이 보장을 준다. 정적 검증은 통과했다 (v28 위반 38% -> 0%).
    safety_filter_ckpt = None
    safety_margin_mm = 5.0

    # ── simulation ───────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=0.002,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # ── scene ────────────────────────────────────────────────────────────
    # num_envs deliberately small (see agents/rsl_rl_ppo_cfg.py) since the
    # target GPU is unknown-spec / VRAM-constrained (RTX-class, per the
    # abandoned WIP's own comments) — override with --num_envs at launch.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=2.0, replicate_physics=True)

    events: EventCfg = EventCfg()

    robot = OPEN_DUCK_MINI_V2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.0,  # every physics step
        track_air_time=True,
    )

    # IMU 는 ROOT_BODY_NAME ("trunk_assembly") 에 붙인다. Playground 의 원본 MJCF
    # 는 그 바디를 "base" 라 불렀지만 이 OnShape URDF 에는 그런 링크가 없다
    # (2026-07-25 확인). 하드코딩된 "base" 대신 상수를 쓰는 이유다.
    #
    # offset 은 **이 로봇의 CAD 값**이다 — robot/robot.urdf 의 `imu_frame` 고정관절
    # (부모 trunk_assembly 기준) 에서 읽었다. imu_map.MOUNT_POS 와 같은 값이다.
    #
    # 2026-08-06 정정: 그 전까지 (-0.08, 0.0, 0.05) 를 썼는데, 그건 **원본 Open Duck
    # Mini 의 MJCF <site name="imu" pos="-0.08 -0.0 0.05"/> 에서 복사한 값**이었다.
    # 이 로봇은 IMU 의 브랜드와 장착 위치가 바뀌었고 실제 CAD 값은 41 mm 씩 다르다:
    #     원본 MJCF  (-80.0, 0, +50.0) mm
    #     이 로봇 CAD (-38.8, 0, +91.4) mm
    # 위치만 다르고 방향은 같다 (URDF rpy=0, 실기 실측 축맵도 항등 — imu_map.py).
    #
    # ⚠️ 이 값을 바꾸면 관측이 바뀐다. ang_vel_b 는 위치와 무관하지만 lin_acc_b 에는
    # 원심·접선 항 (ω×(ω×r), α×r) 이 섞이고, r 이 58 mm 움직이면 보행 과도구간에서
    # 최대 1~2 m/s^2 차이가 난다 (9.81 대비 10~20 %). 즉 **이 커밋 이전에 학습된
    # 체크포인트는 이 offset 과 맞지 않는다** — 재학습이 필요하다.
    imu: ImuCfg = ImuCfg(
        prim_path=f"/World/envs/env_.*/Robot/{ROOT_BODY_NAME}",
        offset=ImuCfg.OffsetCfg(pos=IMU_MOUNT_POS),
        update_period=0.0,
    )

    left_foot_body_name: str = LEFT_FOOT_BODY_NAME
    right_foot_body_name: str = RIGHT_FOOT_BODY_NAME

    # ── noise (uniform, per-channel scale * global level — see
    # observations.py; NOT Isaac Lab's generic Gaussian noise wrapper,
    # since Playground's noise is uniform, not Gaussian) ────────────────
    noise_level = 1.0
    noise_scale_hip_pos = 0.03
    noise_scale_knee_pos = 0.05
    noise_scale_ankle_pos = 0.08
    noise_scale_head_pos = 0.03  # not specified separately in Playground; reuses hip_pos scale as a reasonable default — revisit if head jitter looks wrong on Ubuntu
    noise_scale_joint_vel = 2.5
    noise_scale_gravity = 0.1
    noise_scale_gyro = 0.1
    noise_scale_accelerometer = 0.05

    action_min_delay = 0  # env steps
    action_max_delay = 3

    # 액션 저역통과(EMA) 계수. 0 = 끔(기존 동작 그대로), 1 에 가까울수록 부드럽다.
    #   a_f[t] = alpha*a_f[t-1] + (1-alpha)*a[t]
    # 몸통 떨림을 줄이려고 넣었다. 기존 속도 제한(0.0964 rad/step)은 큰 도약만
    # 막고 그 안에서 매 스텝 부호가 바뀌는 떨림은 통과시킨다.
    # 50 Hz 기준 -3 dB 차단주파수: alpha 0.3 -> 11.0 Hz, 0.5 -> 5.8 Hz,
    #                              0.6 -> 4.2 Hz, 0.7 -> 2.9 Hz, 0.8 -> 1.8 Hz.
    # 보행 주기가 0.54 s(1.85 Hz)이므로 **0.8 은 이미 보행 주파수보다 낮다** —
    # 떨림만 걷어내려면 0.5~0.7 이 상한이다.
    action_lowpass_alpha = 0.0

    # 정지에서 쓸 alpha. None 이면 위 값을 그대로 쓴다(구분 없음).
    # 정지와 보행의 최적값이 다르다는 것이 스윕에서 나왔다 — 정지는 올릴수록
    # 대가 없이 좋아지고(떨림도 정지 속력 오차도 같이 줌), 보행은 추종이
    # 단조롭게 나빠진다. 명령을 아는데 하나로 타협할 이유가 없다.
    action_lowpass_alpha_standstill = None
    # cmd_norm 이 이 구간을 지나는 동안 두 alpha 를 smoothstep 으로 섞는다.
    # 딱딱하게 전환하면 문턱을 넘는 순간 필터 상태가 튄다.
    action_lowpass_blend = (0.01, 0.05)
    imu_min_delay = 0
    imu_max_delay = 3

    # ── commands ─────────────────────────────────────────────────────────
    lin_vel_x_range = (-0.15, 0.15)
    lin_vel_y_range = (-0.2, 0.2)
    ang_vel_yaw_range = (-1.0, 1.0)
    neck_pitch_range = (-0.34, 1.1)
    head_pitch_range = (-0.78, 0.78)
    head_yaw_range = (-1.5, 1.5)
    head_roll_range = (-0.5, 0.5)
    zero_command_prob = 0.1  # 10% chance every resample of an all-zero command
    #: 이동 3 축(vx/vy/wz) 중 하나만 남기고 나머지를 0 으로 두는 확률.
    #: 0 이면 예전 동작(축별 독립 균등)과 완전히 같다.
    pure_axis_prob = 0.0
    #: 순수축을 뽑을 때 (vx, vy, wz) 의 상대 가중치.
    #: 사용자 순위 (2026-08-14): 앞뒤 > 회전 > 완전 옆걸음.
    pure_axis_weights = (0.50, 0.15, 0.35)
    command_resample_steps = 500

    # ── push ─────────────────────────────────────────────────────────────
    push_interval_range_s = (5.0, 10.0)
    push_magnitude_range = (0.1, 1.0)

    # ── termination ──────────────────────────────────────────────────────
    # Fraction of HOME_BASE_HEIGHT below which an episode terminates as
    # "fallen", in addition to the flipped-over (projected_gravity_b.z > 0)
    # check. Added 2026-07-26: the flip-only check alone (ported faithfully
    # from Playground's `upvector_z < 0`) only catches >90 deg tips — it
    # never fires for a robot that has simply collapsed/splayed out without
    # inverting. Playground relies on its imitation reward to make that
    # posture unattractive; Stage 1 here runs with use_imitation=False, so
    # nothing else penalizes it, and a 3000-iter training run converged on
    # exactly this degenerate "collapse but stay upright-ish" policy (high
    # alive_scale reward, high episode length, robot visibly in a heap) —
    # confirmed both by check_joint_stability.sh (base height pinned near
    # the ground, never triggering `terminated`) and by watching the trained
    # policy over WebRTC. This threshold gives Stage 1 its own "has fallen"
    # signal independent of imitation.
    #
    # Raised 0.6->0.75 on 2026-07-27: contact_diagnostic.py measured
    # imitation_v4's converged policy holding a perfectly stable crouch at
    # base_z=0.133-0.134 (ratio ~0.69 of HOME_BASE_HEIGHT=0.193) with ZERO
    # trunk/head contact force for 180+ consecutive steps — a pose that
    # evades all three termination conditions (not flipped, above the old
    # 0.6 height ratio, no trunk/head contact) simultaneously. 0.75 puts
    # that specific crouch below the new threshold.
    # 붕괴 판정의 기준 높이. joint_order.READY_BASE_HEIGHT 를 기본으로 쓰되
    # 설정으로 뺀 이유: 레퍼런스 보행의 키를 바꾸면(walk_com_height) 안착 높이가
    # 달라지는데, 전역 상수를 고치면 v25~v27 체크포인트의 판정 기준까지 함께
    # 바뀐다. 버전마다 자기 높이를 갖게 한다 (2026-07-30).
    ready_base_height = READY_BASE_HEIGHT

    min_base_height_ratio = 0.75

    # ── reward scales ────────────────────────────────────────────────────
    tracking_lin_vel_scale = 2.5
    tracking_ang_vel_scale = 6.0
    tracking_sigma = 0.01
    torques_scale = -1.0e-3
    action_rate_scale = -0.5
    # 액션 2차차분(진동) 벌점. 기본 0 = 꺼짐 = 기존 태스크 리워드 불변.
    # 실기 실측 2차차분^2 이 0.441 이므로 -0.25 면 벌점 -0.11 로, action_rate 의
    # 기여(-0.5 x 0.148 = -0.074)와 같은 자릿수다.
    action_jerk_scale = 0.0
    #: 스윙 발이 접지 발보다 이만큼(m) 뜨는 것까지는 공짜. 그 위를 제곱으로 벌한다.
    #: 다리 마디가 157 mm 이고 실측 발 들림이 22 mm(14 %) 였다.
    foot_clearance = 0.015
    #: 발 들림 벌점 계수. 0 = 꺼짐.
    foot_lift_scale = 0.0
    #: 발의 좌우(몸통 y) 속도 벌점 계수. 0 = 꺼짐.
    #: 실측 수직성(스윙 상승/좌우)이 0.47~0.78 로 1 을 못 넘는다 — 발을 위로
    #: 드는 것보다 옆으로 더 많이 던지고 있고, 그것이 roll ±10° 의 원인이다.
    foot_lateral_scale = 0.0
    #: True 면 명령한 좌우 속도를 뺀 나머지만 벌한다. False 는 v53·v55 재현용.
    foot_lateral_cmd_relative = False
    #: |cmd_vy| 가 이 값일 때 항이 `foot_lateral_gate_floor` 배로 줄어든다.
    #: 0 이면 게이트 없음. lin_vel_y_range 상한(0.2)에 맞추는 것이 자연스럽다.
    foot_lateral_gate_max = 0.0
    foot_lateral_gate_floor = 0.2
    #: 스윙 중 유지할 발 여유 (m). 실측 roll ±10° 가 발을 12 mm 내리므로
    #: 그보다 커야 끌리지 않는다.
    foot_clearance_target = 0.020
    #: 목표 높이 유지 벌점 계수. 0 = 꺼짐.
    foot_clearance_scale = 0.0
    #: 접지 중 수평 쓸림 벌점 계수. 0 = 꺼짐.
    foot_slip_scale = 0.0
    #: 착지 충격 벌점이 걸리기 시작하는 높이 (m).
    foot_impact_band = 0.010
    #: 착지 충격 벌점 계수. 0 = 꺼짐.
    foot_impact_scale = 0.0
    #: 몸통 roll/pitch 각속도 벌점 계수. 0 = 꺼짐.
    torso_ang_vel_scale = 0.0
    #: 관절 가속도 벌점 계수. 0 = 꺼짐.
    joint_accel_scale = 0.0
    stand_still_scale = -0.2

    # 정지했을 때 몸통 기울기(projected_gravity 의 x,y)를 직접 벌하는 계수.
    # 기본 0 = 꺼짐 — v34 이하의 리워드를 한 항도 안 바꾸고 그대로 재현한다.
    upright_standstill_scale = 0.0

    # 정지에서 좌우 다리 자세가 거울에서 벗어난 만큼(제곱합, rad^2) 벌하는 계수.
    # 기본 0 = 꺼짐. v35 실측 기준 어긋남 제곱합이 0.178 rad^2 이므로 -3 이면
    # 벌점 -0.53 — upright(8도에서 -0.41) 과 같은 자릿수다.
    leg_symmetry_scale = 0.0

    # 정지 명령(cmd_norm <= 0.01)에서 보행 위상을 0 에 묶는다.
    # 기본은 꺼짐 — v33n 이하를 그대로 재현할 수 있어야 한다.
    # 자세한 근거는 joystick_env.py 의 해당 지점 주석.
    standstill_hold = False

    # 정지했을 때 `cost_stand_still` 이 목표로 삼을 자세 (14 관절, rad).
    # None 이면 default_joint_pos(=보행 평균 자세)를 그대로 쓴다 — v34c10 까지의 동작.
    #
    # 왜 따로 두는가. default_joint_pos 는 **액션의 원점**이다
    # (target = default + action * action_scale). 그래서 보행 궤적의 평균에
    # 맞춰져 있어야 액션이 도달 가능한 범위에 들어온다 — v1~v9 를 아홉 번
    # 실패시킨 지점이다. 그런데 그 평균 자세는 placo 의 walk_trunk_pitch=-4 도가
    # 배어 있어서 **몸통이 4 도 앞으로 기운 자세**다. 정지 목표로는 부적절하다.
    # 둘은 목적이 다르므로 분리한다.
    standstill_joint_pos = None
    # alive_scale (2026-07-26): Playground's/Disney's original value is 20.0.
    # A whole night's worth of experiments (see docs/training_log.md) showed
    # this only works when `imitation` is active to counterbalance it —
    # reward_imitation's internal joint-pos-error term (w_joint_pos=15.0, see
    # rewards.py) swings sharply negative whenever the pose doesn't match the
    # reference gait, which is what keeps alive=20 from being worth farming
    # on its own. Tried a Stage-1-only fix instead (cutting alive_scale to
    # 2/5/10, no imitation) — all three still collapsed under
    # eval_policy_stability verification. A structural reward redesign
    # (teammate's leg-antiphase/alternating-contact terms) was considered and
    # rejected: those are straight-line-walking-specific heuristics that
    # would likely distort turning/lateral-command behavior since they treat
    # "both legs still" as reward-neutral regardless of whether the command
    # calls for standing still. Reverting to 20.0 now that use_imitation=True
    # below actually provides the counterweight it was always designed with.
    alive_scale = 20.0
    imitation_scale = 1.0
    # w_joint_pos (2026-07-27): was hardcoded inside reward_imitation() at
    # 15.0. Pulled into cfg for the alive_scale x w_joint_pos sweep below —
    # imitation_v2 (alive=20, w_joint_pos=15, this exact combo) FAILED just
    # like imitation_v1, with MORE violent twitching (43.4 foot-contact
    # toggles/10s vs v1's 11.7) despite the reference data itself being
    # fully fixed (knee ROM, symmetry, limits, drift) by then — so the
    # remaining suspects are these two reward weights, not the data.
    imitation_w_joint_pos = 15.0
    # bounded_joint_pos (2026-07-28): reward_breakdown_v2.py measured, on
    # imitation_v8's converged policy, that the imitation term averaged
    # -1.01/step while the largest positive term (alive) gave only +0.40 —
    # so the summed reward went negative on 81.6% of steps and `_get_rewards`
    # clamped every one of those to exactly 0, erasing the learning signal on
    # over four fifths of all experience. Root cause: joint_pos was the only
    # unbounded term in reward_imitation (raw -err^2 * w) while every other
    # tracking term there is exp(-err), bounded to [0,1]. Setting this True
    # switches joint_pos to the same exp form so it can't dominate. Left
    # False on the base cfg so v1-v8's numbers stay reproducible; see
    # JoystickEnvCfg_A20J5_Bounded.
    imitation_bounded_joint_pos = False
    # See rewards.py's swing_only_contact branch. Default False so v1-v10's
    # numbers stay reproducible.
    imitation_swing_only_contact = False
    # Freeze the 4 head DOFs at their READY pose (see joystick_env.py's
    # _pre_physics_step). Default False to keep older variants unchanged.
    lock_head_joints = False

    # Per-term sensitivities inside reward_imitation. Defaults reproduce
    # v1-v11 exactly; JoystickEnvCfg_Walk2 is where they actually change.
    # See rewards.py's comment block for the imit_internals2.py measurement
    # that motivated making them configurable.
    imitation_k_lin_vel_xy = 8.0
    imitation_w_lin_vel_z = 1.0
    imitation_w_ang_vel_xy = 0.5
    imitation_w_contact = 1.0

    # ── path frame (Disney BD-X) ────────────────────────────────────────
    # 명령 속도를 적분한 "의도한 궤적"을 유지하고, 로봇이 거기서 벗어난
    # 횡방향·방향 오차를 관측과 리워드에 넣는다. 끄면 관측 차원이 예전과
    # 같으므로 v1~v24 체크포인트가 그대로 로드된다.
    use_path_frame = False
    # 정책 관측에 중력 방향을 넣을지. 크리틱은 use_gravity_obs 와 무관하게
    # 항상 갖고 있다(비대칭 크리틱).
    use_gravity_obs = False
    path_error_clip = 0.5      # m, 횡방향 오차 클리핑 (초기 발산 방지)
    path_tracking_scale = 0.0  # 리워드 가중치
    path_k_lateral = 20.0      # exp 예민도: 0.22 m 벗어나면 exp(-1)
    path_k_yaw = 4.0           # exp 예민도: 0.5 rad(29도) 벗어나면 exp(-1)
    path_w_yaw = 1.0           # 방향 오차 항의 상대 비중
    # Penalty weight for lifting a foot the reference says should be planted.
    # 0.0 reproduces v11/v12 exactly; see rewards.py for the chatter it caused.
    imitation_w_stance_violation = 0.0
    imitation_w_joint_pos_amp = 1.0

    # ── reference motion (imitation) ────────────────────────────────────
    # Stage 2 (2026-07-26): polynomial_coefficients.pkl now exists for real
    # (240 swept gaits, see docs/training_log.md) after fixing the Placo
    # pipeline (antenna joints, placo version pin, auto_waddle.py's python->
    # python3 bug). Flipping this on for the first time — Stage 1
    # (use_imitation=False, pure RL) reward-hacked into a collapsed-but-
    # technically-alive pose across every alive_scale tried.
    use_imitation: bool = True
    # RSI (Reference State Initialization, DeepMimic/Peng et al. 2018) —
    # added 2026-07-27 for imitation_v6/v7, NOT part of Disney's BD-X paper
    # (confirmed 2026-07-28 by reading the actual paper: no mention of RSI
    # or random-phase episode initialization anywhere in it — Disney always
    # implicitly starts from a fixed reference state, same as our pre-RSI
    # behavior). Kept as a toggle rather than deleted so "RSI on" vs "RSI
    # off, Disney-literal" can be compared directly on the same codebase.
    # When False, _reset_idx() always resets to reference-motion phase 0,
    # matching v1-v5's original (pre-RSI) behavior.
    use_rsi: bool = True
    reference_motion_pkl = "source/open_duck_mini_isaaclab/reference_motion/data/polynomial_coefficients.pkl"
    # Gait-phase clock period (env steps), used for the imitation_phase
    # observation channel when use_imitation=False (no PolyReferenceMotion
    # loaded to supply nb_steps_in_period). Purely a periodic clock signal
    # in that case — has no effect on reward, only lets the policy sense
    # gait phase if it finds that useful on its own.
    gait_period_steps: int = 50


# ── 변형 설정 ────────────────────────────────────────────────────────────
# 2026-07-29 정리. imitation_v1~v20을 거치며 21개까지 늘어난 설정 클래스를
# 살아있는 4개로 줄였다. 삭제한 것들(Alive5/Alive10/A10J10/A5J15/A5J5/A30J25/
# A30J25Im2/A20J5/A20J5_NoRSI/A20J5_Bounded/Walk/Walk2/Walk4/Walk5/Walk7/Walk8)
# 은 전부 실패로 판정된 경로이고, 값과 실패 이유는 git 이력과
# docs/training_log.md에 남아 있다.
#
# 남긴 것들은 상속 사슬을 풀어 값을 직접 적었다. 예전에는 Walk6가
# Walk5←Walk4←Walk3←Walk2←Walk←A20J5_Bounded←A20J5_NoRSI←A20J5←base로
# 8단계를 타고 올라가야 실제 값을 알 수 있었다. 평탄화 전후로 모든 필드의
# 해석값이 동일함을 정적 분석으로 확인했다.


@configclass
class JoystickEnvCfg_LegsOnly(JoystickEnvCfg):
    """머리 4관절을 READY 자세에 고정하고 다리 10개만 학습한다.

    사용자 제안으로 도입. 머리에는 IMU가 없어 수평 유지를 측정할 수 없으니,
    목을 Z자로 접은 READY 자세(neck +45°, head +45° — 이 URDF에서는 같은
    부호가 상쇄 조합)에 고정하고 명령도 0으로 묶는다. 그래야 액션 노이즈와
    명령 채널이 보행 리워드가 무시하는 자유도에 낭비되지 않는다.
    """

    lock_head_joints = True
    neck_pitch_range = (0.0, 0.0)
    head_pitch_range = (0.0, 0.0)
    head_yaw_range = (0.0, 0.0)
    head_roll_range = (0.0, 0.0)


@configclass
class JoystickEnvCfg_Walk3(JoystickEnvCfg_LegsOnly):
    """imitation_v13 — 명령 추종이 가장 좋았던 설정 (전진 0.117 m/s).

    스탠스 위반 페널티가 들어간 첫 버전. 그 전(v12)은 스윙 크레딧만 있어
    발을 빠르게 깜빡일수록 스윙 구간과 우연히 겹쳐 점수를 벌 수 있었고,
    실제로 접지가 진동하듯 떨렸다(토글 144~319/10s → 122~134로 감소).
    육안으로는 걷긴 하지만 제자리 회전과 낙상이 관찰됐다.
    """

    alive_scale = 3.0
    tracking_lin_vel_scale = 10.0
    imitation_scale = 4.0
    imitation_bounded_joint_pos = True
    imitation_w_joint_pos = 4.0
    imitation_k_lin_vel_xy = 20.0
    imitation_w_lin_vel_z = 0.1
    imitation_w_ang_vel_xy = 0.1
    imitation_w_contact = 2.0
    imitation_swing_only_contact = True
    imitation_w_stance_violation = 1.0


@configclass
class JoystickEnvCfg_Walk6(JoystickEnvCfg_LegsOnly):
    """imitation_v17 — 육안 판정 최고 보행 ("실제로 제일 잘 걷는 것 같음").

    Walk3 대비 자세 항의 예민도와 비중만 다르다. w_joint_pos는 정책이 실제로
    내는 오차(0.684 rad²)에 맞춘 값이고(레퍼런스 자체 분산에서 유도한 4.0은
    포화 상태였다), amp 2.0은 1.0(v13)과 3.0(v16) 사이 절충이다. amp 3.0은
    관절 오차를 14.4°→9.3°로 줄였지만 전진이 0.117→0.059로 반토막 났는데,
    READY가 곧 보행의 평균 자세라 자세 비중을 올릴수록 "가만히 서있기"가
    유리해지기 때문이다. 남은 증상은 몸통 흔들림(뒤뚱거림)이다.
    """

    alive_scale = 3.0
    tracking_lin_vel_scale = 10.0
    imitation_scale = 4.0
    imitation_bounded_joint_pos = True
    imitation_w_joint_pos = 1.5
    imitation_w_joint_pos_amp = 2.0
    imitation_k_lin_vel_xy = 20.0
    imitation_w_lin_vel_z = 0.0
    imitation_w_ang_vel_xy = 0.1
    imitation_w_contact = 2.0
    imitation_swing_only_contact = True
    imitation_w_stance_violation = 1.0


@configclass
class JoystickEnvCfg_Walk9(JoystickEnvCfg_Walk6):
    """imitation_v20 — Walk6의 리워드 그대로, 학습 설정만 upstream 정렬.

    비대칭 크리틱을 켠다. upstream은 가치 네트워크에 privileged_state를 주는데
    (mujoco_playground가 value_obs_key="privileged_state"로 설정) 이 포팅은
    state_space=0이라 크리틱이 정책과 똑같은 노이즈 섞인 101차원만 봤다.
    가치 추정이 나쁘면 어드밴티지가 오염되고, 그건 지금까지 만진 모든 리워드
    수정보다 상류에 있다. 네트워크/PPO 파라미터는
    agents/rsl_rl_ppo_cfg.py::JoystickPPORunnerCfg_Upstream 참고.
    """

    state_space = OBS_CRITIC_DIM


@configclass
class JoystickEnvCfg_Path(JoystickEnvCfg_Walk9):
    """imitation_v25 — Disney BD-X의 path frame 도입. v24 설정 + 경로 추종.

    v24를 눈으로 보니 전진 명령만 주는데도 요가 순간적으로 ±1.5 rad/s까지
    요동치고, 좌우 명령에서는 옆으로 가는 대신 비틀거리며 회전했다. 평균은
    0에 가까워 측정치(좌 71% / 우 81%)로는 잘 안 드러났다.

    원인은 명령이 순수 rate라는 데 있다. yaw_rate=0은 "지금 회전하지 마라"이지
    "원래 방향으로 돌아와라"가 아니고, vy=0도 마찬가지다. 한 번 휘면 정책의
    관측 어디에도 그 사실이 없어서 되돌릴 수단이 없다.

    논문은 이를 path frame으로 푼다 -- "a path frame that integrates these
    velocity commands over time"를 유지하고 정책의 상태를 그 프레임 기준으로
    표현한다. 여기서는 적분된 경로 대비 횡방향 오차와 방향 오차를 관측(3차원)과
    리워드에 추가했다. 종방향 오차는 뺐다: path frame은 *명령* 속도를 적분하므로
    로봇이 조금이라도 느리면(0.148 vs 0.15) 무한히 쌓이고, 그걸 보정하라고 하면
    뒤처졌을 때 무리하게 가속하는 쪽으로 학습된다.
    """

    use_path_frame = True
    path_tracking_scale = 5.0
    observation_space = OBS_STATE_DIM + PATH_ERR_DIM
    state_space = OBS_CRITIC_DIM + PATH_ERR_DIM


@configclass
class JoystickEnvCfg_Grav(JoystickEnvCfg_Path):
    """imitation_v27 — v26(Path) 설정 그대로 + 정책 관측에 중력 방향 3차원.

    v26을 3000 iter까지 돌린 결과가 근거다. 명령 추종은 v25@1500보다 오히려
    내려갔는데(전진 89% -> 81%) 모방 리워드는 올랐고, 재보니 몸통이 안 잡혀
    있었다:

        몸통 roll RMS            7.20 deg   (보행 중 8.2, 정지 4.1)
        모방 ang_vel_xy 항       최댓값의 3.6%
        고관절 roll/yaw 이탈     3.9 / 4.5 deg  (무릎·발목 5.3 보다 작다)

    즉 고관절이 특별히 나쁜 게 아니라 **몸통 자세 자체가 안 잡힌다.** 그런데
    정책 관측에는 중력 방향이 없다 — 자이로(각속도)와 가속도계만으로 기울기를
    유추해야 하고, 가속도계는 중력과 실제 가속이 섞인다. 크리틱은 노이즈 없는
    중력을 이미 보고 있으니(비대칭 크리틱) 정책만 못 보는 비대칭이다.

    이 프로젝트에서 판을 바꾼 것은 매번 관측/가치추정 수정이었다(비대칭 크리틱
    +3배, path frame). 리워드 계수는 열 번 넘게 만져도 국소 개선에 그쳤다.
    그래서 리워드를 건드리지 않고 관측만 3차원 늘린다 — **한 번에 하나만.**

    실기 이식 관점에서도 정당하다. 중력 방향은 IMU로 추정 가능한 양이고,
    노이즈(noise_scale_gravity=0.1)를 실어 준다.
    """

    use_gravity_obs = True
    observation_space = OBS_STATE_DIM + PATH_ERR_DIM + GRAVITY_OBS_DIM
    # 크리틱 관측은 정책 state 를 그대로 앞에 붙이므로 같이 3차원 늘어난다.
    state_space = OBS_CRITIC_DIM + PATH_ERR_DIM + GRAVITY_OBS_DIM


# imitation_v27 의 레퍼런스로 계산한 READY 자세 (walk_com_height 0.175).
# scripts/diag/calc_home.py --pkl .../ref_h175.pkl 출력 그대로.
# 기존(0.16) 대비 무릎이 12.4도, 고관절 pitch 가 6.2도 펴졌다.
# scripts/diag/settle_height.py 로 실측 (2026-07-30, 16 envs x 400 steps).
# 기존 자세: 안착 121 mm / 스폰 130 mm  ->  새 자세: 안착 136 mm / 스폰 144 mm.
# 로봇이 15 mm(+12%) 커진다.
#
# 처음엔 스폰을 130 mm 그대로 두고 쟀다가 안착값이 92~188 mm 로 흩어졌다.
# 최댓값이 스폰보다 높았는데 중력만으로는 불가능한 값이다 — 키가 커진 자세라
# 발이 지면을 파고들어 PhysX 가 튕겨낸 것이다(joint_order.py 의 SPAWN_BASE_HEIGHT
# 주석이 기록한 그 현상). 200 mm 에서 떨어뜨려 다시 재니 표준편차가 6.9 -> 1.7 mm
# 로 떨어졌다. **스폰이 낮으면 안착 측정 자체가 오염된다.**
READY_BASE_HEIGHT_H175 = 0.1360
SPAWN_BASE_HEIGHT_H175 = 0.1437

READY_JOINT_POS_H175 = {
    "left_hip_yaw": 0.0003,
    "left_hip_roll": 0.0213,
    "left_hip_pitch": 0.9910,
    "left_knee": -1.7852,
    "left_ankle": 0.8647,
    # 머리는 0 으로 둔다. Z 자 자세(neck/head_pitch = 0.785)로 바꿔 봤다가
    # 되돌렸다 -- **머리 자세는 균형에 영향을 준다.**
    #
    # "머리는 모방 리워드에서 빠져 있으니 학습과 무관하다" 고 생각했는데
    # 틀렸다. 리워드에 안 들어가는 것과 동역학에 영향이 없는 것은 다르다.
    # pinocchio 로 재보면 +45 도로 세울 때 무게중심이 x 로 14.3 mm 옮겨간다
    # (-8.8 -> -23.1 mm). 발 길이가 60 mm 인 로봇에서 작지 않은 값이고,
    # 정책은 머리가 0 인 무게중심으로 학습됐다. 실제로 재생해 보니 앞으로
    # 기울다가 넘어졌다.
    #
    # Z 자 목을 쓰려면 그 자세로 **다시 학습해야 한다** -- 자세만 바꾸면 안 된다.
    "neck_pitch": 0.0,
    "head_pitch": 0.0,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": -0.0005,
    "right_hip_roll": -0.0092,
    "right_hip_pitch": 1.0114,
    "right_knee": 1.8163,
    "right_ankle": -0.8754,
}


@configclass
class JoystickEnvCfg_Tall(JoystickEnvCfg_Grav):
    """imitation_v28 — v27 그대로 + 로봇을 더 세운다 (walk_com_height 0.16 -> 0.175).

    사용자 요청. 기존 로봇은 121 mm 로 상당히 웅크리고 걷는다.

    **레퍼런스부터 다시 만들어야 한다.** 서는 높이는 임의 값이 아니라 레퍼런스
    보행의 평균 자세에서 나온다 (calc_home.py). READY 자세만 높이면 모방 리워드가
    매 스텝 웅크린 레퍼런스로 되돌리고, 기본 자세와 레퍼런스가 어긋나면 액션이
    도달 불가능해진다 — v1~v9 를 아홉 번 실패시킨 바로 그 상황이다.
    그래서 placo 프리셋의 walk_com_height 를 올려 레퍼런스를 새로 생성했고
    (scripts/setup/gen_reference_remote.sh, 랩PC), 그 레퍼런스에서 READY 자세를
    다시 뽑았다.

    바뀐 것:
        레퍼런스   walk_com_height 0.16 -> 0.175
        무릎       2.032 -> 1.816 rad  (12.4도 펴짐)
        고관절pitch 1.120 -> 1.011 rad (6.2도)
        발목       -0.983 -> -0.875 rad (6.2도)
        필요 액션  최대 1.30 -> 1.46   (v1~v9 를 죽인 값은 8.1 이었다. 안전)

    ready_base_height / 스폰 높이는 **실측해서** 넣는다. 추측하면 스폰 순간
    지면을 파고들거나 낙하한다 (joint_order.py 의 SPAWN_BASE_HEIGHT 주석 참고).
    """

    reference_motion_pkl = "source/open_duck_mini_isaaclab/reference_motion/data/ref_h175.pkl"

    robot = OPEN_DUCK_MINI_V2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=OPEN_DUCK_MINI_V2_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_H175),
            joint_pos=dict(READY_JOINT_POS_H175),
        ),
    )
    ready_base_height = READY_BASE_HEIGHT_H175


# ref_h190 (walk_com_height 0.19) 의 READY 자세. 실측 전 잠정 높이는
# scripts/diag/settle_height.py 로 재서 갱신한다.
# settle_height.py 실측 (16 envs x 400 steps, 200 mm 에서 낙하, 표준편차 1.5 mm).
# 키 변화:  121 mm (원본) -> 136 mm (h175) -> 154 mm (h190).  원래 대비 +27%.
READY_BASE_HEIGHT_H190 = 0.1539
SPAWN_BASE_HEIGHT_H190 = 0.1614

READY_JOINT_POS_H190 = {
    "left_hip_yaw": 0.0003,
    "left_hip_roll": 0.0208,
    "left_hip_pitch": 0.8252,
    "left_knee": -1.4750,
    "left_ankle": 0.7202,
    # 머리는 0 으로 둔다. Z 자 자세(neck/head_pitch = 0.785)로 바꿔 봤다가
    # 되돌렸다 -- **머리 자세는 균형에 영향을 준다.**
    #
    # "머리는 모방 리워드에서 빠져 있으니 학습과 무관하다" 고 생각했는데
    # 틀렸다. 리워드에 안 들어가는 것과 동역학에 영향이 없는 것은 다르다.
    # pinocchio 로 재보면 +45 도로 세울 때 무게중심이 x 로 14.3 mm 옮겨간다
    # (-8.8 -> -23.1 mm). 발 길이가 60 mm 인 로봇에서 작지 않은 값이고,
    # 정책은 머리가 0 인 무게중심으로 학습됐다. 실제로 재생해 보니 앞으로
    # 기울다가 넘어졌다.
    #
    # Z 자 목을 쓰려면 그 자세로 **다시 학습해야 한다** -- 자세만 바꾸면 안 된다.
    "neck_pitch": 0.0,
    "head_pitch": 0.0,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": -0.0003,
    "right_hip_roll": -0.0114,
    "right_hip_pitch": 0.8490,
    "right_knee": 1.5137,
    "right_ankle": -0.7350,
}


@configclass
class JoystickEnvCfg_Taller(JoystickEnvCfg_Grav):
    """imitation_v30 — v28 에서 확인된 방향(키)을 한 단계 더 (0.175 -> 0.19).

    v28 이 유일하게 명확한 개선이었다. 키를 121 -> 136 mm 로 올렸더니 이 프로젝트
    최대 결함이던 횡방향 오차가 절반이 됐다(좌 0.040 -> 0.017, 우 0.035 -> 0.018,
    각 4표본으로 구간이 겹치지 않는다). v29(좌우 미러 손실)는 같은 표적을 노렸으나
    이득이 없었다. 그래서 통한 쪽을 한 번 더 민다.

        무릎        2.032 -> 1.816 -> **1.514** rad
        필요 액션   1.30  -> 1.46  -> **1.87**

    필요 액션이 계속 오른다. v1~v9 를 죽인 값은 8.1 이라 아직 여유가 크지만,
    계속 올리면 언젠가 같은 벽에 부딪힌다. 이번에도 나아지면 그 다음은 액션
    도달성부터 확인할 것.

    ready_base_height / 스폰 높이는 settle_height.py 로 **실측해서** 넣는다.
    추측하면 발이 지면을 파고들어 PhysX 가 튕겨내고, 그러면 안착 측정 자체가
    오염된다 (v28 준비 때 실제로 겪었다 — 안착값이 92~188 mm 로 흩어졌다).
    """

    reference_motion_pkl = "source/open_duck_mini_isaaclab/reference_motion/data/ref_h190.pkl"

    robot = OPEN_DUCK_MINI_V2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=OPEN_DUCK_MINI_V2_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_H190),
            joint_pos=dict(READY_JOINT_POS_H190),
        ),
    )
    ready_base_height = READY_BASE_HEIGHT_H190



# ── 자기충돌: 조사했으나 보류 (2026-07-30) ───────────────────────────────
# 사용자가 재생에서 다리와 몸통이 겹치는 것을 보고 제기했고, Disney BD-X는
# 자기충돌을 켤 뿐 아니라 종료 조건으로도 쓴다("...also if we detect a
# self-collision between the head and torso"). 우리는 계속 꺼져 있었고,
# v4에서 머리를 자기 다리에 얹은 "플랭크" 자세가 접촉력 0.000 N으로 모든
# 종료 조건을 빠져나간 것도 이 때문이었다.
#
# 켜보니 에피소드가 1스텝 만에 끝났다. 원인을 pinocchio 정확 메시로 재보니
# 물리가 아니라 충돌 형상 근사의 문제였다 -- READY 자세에서 비인접 링크쌍
# 91종 중 관통 0개, 최소 간격 10.2 mm인데 PhysX는 수천 N을 만들었다.
# IsaacLab의 convert_urdf.py가 UrdfConverterCfg.collider_type을 설정하지 않아
# 기본값 convex_hull이 쓰인 탓이다.
#
# scripts/convert_urdf_cd.py로 convex_decomposition 재변환까지 해봤고
# neck_pitch<->neck_yaw(4340 N)는 해소됐지만 두 쌍이 남았다:
#   head_pitch <-> xm430          1847 N   (정확 메시로는 13.8 mm 여유)
#   trunk      <-> roll_to_pitch  1089 N   (11.3 mm 여유)
# 둘 다 운동학 체인에서 두 관절 떨어져 있어 PhysX가 자동 제외하지 않는데,
# 설계상 관절 하우징이 부모 부품 *안에* 끼워지는 구조라 어떤 볼록 근사로도
# 겹친다. 즉 전역 자기충돌은 이 기구 구조에서 쓸 수 없다. Disney가 머리<->몸통
# 한 쌍만 집어 쓴 것도 같은 이유로 보인다 -- 전역이 아니라 선별이다.
#
# 남은 길은 USD physics:filteredPairs로 끼워진 쌍만 제외하는 선별 필터링인데,
# instanceable 자산에 적용하기가 까다로워 보류했다.
#
# 보류하되 실측은 남긴다 (v25 model_1500, 정확 메시 기준):
#   레퍼런스 : 다리<->몸통 최소 7.1 mm, 접촉 0.0%
#   정책     : 최소 0.0 mm, 접촉 56.7%
# 레퍼런스는 한 번도 안 닿는데 정책은 절반 이상 닿는다 -- CAD나 레퍼런스가
# 아니라 정책 결함이다. 접촉 시점의 관절 오차를 보면 원인이 고관절이다:
#   right_hip_roll  접촉 -8.0도 / 비접촉 +0.9도
#   left_hip_yaw    접촉 -8.9도 / 비접촉 -1.5도
#   right_hip_yaw   메시 간격과 상관 -0.540 (가장 강함)
# 무릎/발목은 거의 무관하다. 실기 이식 전에는 반드시 닫아야 한다.


@configclass
class JoystickEnvCfg_HipLimit(JoystickEnvCfg_Tall):
    """imitation_v31 — v28 그대로 + 고관절 roll/yaw 이탈에 상한.

    실기 이식이 목표로 확정됐고, 다리-몸통 접촉은 성능 문제가 아니라 **파손**이다.

    접촉률은 정확 메시로 재야 한다. PhysX 는 못 쓴다 -- 관절 하우징이 부모 부품
    안에 끼워지는 구조라 어떤 볼록 근사도 겹친다 (아래 자기충돌 주석).
    `scripts/diag/leg_trunk_clearance.py` 로 v28 을 재면:

        레퍼런스  접촉  0.0 %   최소 간격 5.9 mm
        v28       접촉 11.5 %   최소 간격 0.0 mm   (좌 25% / 앞 19% / 정지·뒤 0%)

    v25 의 56.7% 에서 크게 내려왔는데, 아무 대책도 넣지 않았는데 내려왔다 --
    고관절 이탈이 같이 줄었기 때문이다. **접촉은 이탈에 딸려 온다.**

    그러면 이탈을 직접 묶는 것이 가장 짧은 길이다. **절대 관절 범위 제한은
    소용없다**: 레퍼런스가 쓰는 범위가 고관절 roll ±16.5° / yaw ±10.2° 인데
    정책의 절대각은 거기서 크게 안 벗어난다. 문제는 절대각이 아니라 매 순간
    레퍼런스 대비 얼마나 어긋나 있느냐다.

    한계값은 v28 자신의 이탈 분포 p90 에서 잡았다:

        left_hip_yaw   |이탈| 평균  4.3°  p90  8.4°
        right_hip_yaw               3.9°  p90  7.8°
        left_hip_roll              10.4°  p90 20.0°
        right_hip_roll              9.5°  p90 17.9°

    roll 이 yaw 보다 두 배 이상 벗어나므로 한계도 따로 준다. p90 을 고른 것은
    **상위 10% 만 자르기 위해서**다 -- 접촉률 11.5% 와 거의 겹치므로, 접촉이
    일어나는 꼬리만 막고 나머지 90% 의 동작에는 손대지 않는다는 뜻이다.
    고관절 roll/yaw 는 이족보행에서 몸통을 지지발 위에 세우는 관절이라
    (torso_vs_hip.py 는 몸통과의 상관을 |r| = 0.404, "부분적으로 얽혀 있음"
    으로 판정했다) 넉넉히 자르면 균형 권한을 뺏어 넘어질 위험이 있다.

    **감시할 것**: 넘어짐 빈도, 추종 오차(v28 = 0.0158), 그리고 재측정한 접촉률.
    접촉이 0 이 안 되면 한계를 조이고, 넘어지기 시작하면 되돌린다.
    """

    hip_dev_limit_yaw = 0.1466    # 8.4°  (좌우 p90 중 큰 쪽)
    hip_dev_limit_roll = 0.3491   # 20.0°


@configclass
class JoystickEnvCfg_HipInward(JoystickEnvCfg_Tall):
    """imitation_v32 — v28 + 고관절이 **안쪽으로** 벗어나는 것만 벌점.

    v31 의 재설계다. v31 은 모터 목표를 레퍼런스 ±p90 으로 양방향 클램프했다가
    접촉을 11.5% -> 30.2% 로 세 배 만들었다. 두 가지가 틀렸다:

      1. 목표를 묶어도 실제 관절각은 안 묶인다 (실측 이탈 7.03° -> 9.44°).
      2. 꼬리만 잘리지 않았다. 간격 중앙값이 7.5 -> 1.6 mm 로 밀렸다 --
         양방향으로 자르면 균형에 쓰는 바깥쪽 여유까지 없어진다.

    그래서 **어느 관절이 어느 방향으로 갈 때 간격이 줄어드는지** 를 먼저 쟀다
    (`leg_trunk_clearance.py --dump`, v28 포즈 300개, 접촉 13%):

        관절              상관 r    접촉 시    비접촉 시
        right_hip_yaw    -0.472    +4.8°      -0.3°
        left_hip_yaw     +0.412    -4.4°      -2.6°
        right_hip_roll   +0.352    -5.0°      -1.0°
        left_hip_roll    -0.064    -2.1°      -1.0°

    부호가 물리적으로 일관된다. 검증된 미러 맵에서 hip_yaw 는 부호 -1 이므로
    **왼쪽 음 / 오른쪽 양은 같은 물리 방향** -- 양다리가 안쪽으로 도는 것이다.
    hip_roll 은 부호 +1 이라 양쪽 모두 음(내전)이 안쪽이다. 즉 접촉은
    **다리가 몸통 밑으로 파고들 때** 난다. 바깥쪽으로 벌어지는 것은 무해하다.

    임계 3° 는 접촉 시(4.4~5.0°)와 비접촉 시(1.0~2.6°) 사이다. 평상시에는 항이
    0 이고 꼬리에서만 켜진다 -- v31 이 못 한 "꼬리만 건드리기" 를 여기서 한다.

    벌점은 초과분에 **선형**이다. 제곱은 임계 근처에서 거의 0 이라 정작 필요한
    구간에서 안 듣고, 큰 이탈에서만 급격해져 가중치에 민감하다.
    -25 는 접촉 수준(두 관절이 5°)에서 스텝당 약 -1.8, 전체 리워드 27 의 7% 다.
    균형을 포기할 만큼 크지 않고 무시할 만큼 작지도 않게 잡았다.

    **감시**: 접촉률(v28 11.5%, 목표 0), 추종 오차(v28 0.0158), 넘어짐.
    """

    hip_inward_thresh = 0.0524   # 3°
    hip_inward_scale = -25.0


# Z 자 목으로 학습할 때 쓰는 READY 자세. 다리 값은 H175 그대로이고 머리만 세운다.
# (2026-08-20 복구: 죽은 cfg 를 지우면서 이 표까지 같이 날아가 모듈 import 가
#  깨져 있었다. 아래 ZNeck 이 쓴다.)
READY_JOINT_POS_H175_ZNECK = dict(READY_JOINT_POS_H175)
READY_JOINT_POS_H175_ZNECK.update({"neck_pitch": 0.785, "head_pitch": 0.785})


@configclass
class JoystickEnvCfg_ZNeck(JoystickEnvCfg_HipInward):
    """imitation_v33 — v32 + BD-X 특유의 Z 자 목 자세로 **처음부터 학습**.

    자세만 바꾸면 안 된다. v32 정책에 머리만 세워 재생해 봤더니 앞으로 기울다가
    넘어졌다. pinocchio 로 재보면 머리를 +45 도로 세울 때 무게중심이 x 로
    14.3 mm 옮겨간다 (-8.8 -> -23.1 mm). 발 길이가 60 mm 인 로봇이라 작지 않고,
    정책은 머리가 0 인 무게중심으로 학습돼 있었으니 겪어본 적 없는 외란이다.

    "머리는 모방 리워드에서 빠져 있으니 무관하다" 는 생각이 틀린 지점이 여기다 --
    리워드에 없는 것과 동역학에 없는 것은 다르다. 머리는 질량을 갖는다.

    그래서 이 설정은 **그 무게중심으로 학습**한다. 바뀌는 것은 머리 자세 하나뿐이고
    (lock_head_joints 가 그대로라 정책은 여전히 다리 10 관절만 쓴다), 나머지는
    v32 와 같다 -- 안쪽 고관절 이탈 벌점, ref_h175, 큰 네트워크, gamma 0.97.

    비교 기준: v32 는 추종 0.0200 · 5 mm 위반 1.0%. 무게중심이 뒤로 가는 만큼
    균형이 어려워질 수 있으니 **넘어짐 빈도(에피소드 길이)를 같이 볼 것.**
    """

    robot = OPEN_DUCK_MINI_V2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=OPEN_DUCK_MINI_V2_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_H175),
            joint_pos=dict(READY_JOINT_POS_H175_ZNECK),
        ),
    )
    ready_base_height = READY_BASE_HEIGHT_H175


# ──────────────────────────────────────────────────────────────────────────
# imitation_v33n — 새 CAD 모델(2.7140 kg) + 새 레퍼런스(ref_g115)
# ──────────────────────────────────────────────────────────────────────────
# scripts/diag/calc_home.py --pkl .../ref_g115.pkl 출력 그대로.
# 필요 최대 액션 1.54 (v1~v9 를 죽인 값은 8.1 이었다. 안전)
#
# ref_g115 는 walk_com_height 0.1873 으로 생성했다. 이 값이 "몸통 바닥 지상고
# 115 mm" 를 노린 역산이고, 실제로 114.9 mm 가 나왔다 (목표 대비 0.1 mm).
#
# ⚠️ walk_com_height 는 **CoM 높이**지 몸통 높이가 아니다. 새 CAD 는 질량이
# 몸통·머리로 늘어 CoM 이 몸통 원점 위 35.2 -> 53.1 mm 로 올라갔고, placo 는
# URDF 링크 질량에서 CoM 을 계산하므로(placo_walk_engine.py 의 robot.com_world())
# **같은 walk_com_height 가 다른 자세를 만든다.** 옛 ref_h175 의 0.175 를 그대로
# 쓰면 안 되는 이유다. 지렛대도 약하다 — CoM 목표 +1 mm 당 몸통 +0.47 mm.
READY_JOINT_POS_G115 = {
    "left_hip_yaw": -0.0031,
    "left_hip_roll": 0.0208,
    "left_hip_pitch": 0.9915,
    "left_knee": -1.7483,
    "left_ankle": 0.8273,
    "neck_pitch": 0.0,
    "head_pitch": 0.0,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": -0.0038,
    "right_hip_roll": -0.0088,
    "right_hip_pitch": 1.0269,
    "right_knee": 1.7744,
    "right_ankle": -0.8180,
}

# Z 자 목 = 30도. neck 은 위로, head 는 아래로 도는 축이라 같은 값을 주면 서로
# 상쇄되고 **얼굴은 수평**을 유지한다 (FK 로 코 기울기 0.0도 확인). 머리에 카메라를
# 달 것이므로 시선이 수평인 게 중요하다.
#
# head_pitch 의 URDF 한계는 원래 정확히 45도(0.7854)여서 45도가 리밋 위에
# 얹혔다. 사용자 확인 결과 **실제 설계 가동범위는 ±60도** 이고 45도는 CAD 상
# 클램프였을 뿐이다. robot/robot.urdf 의 head_pitch limit 을 ±0.872665(±50도) 로
# 넓혀 두었다. 자세는 30도라 여유가 20도다
# (45도로 돌려봤다가 30도로 낮췄다 — 2026-08-07 사용자 판단).
#
# ⚠️ 다만 시뮬은 USD 에서 관절 한계를 읽는다. USD 를 재변환하지 않으면 시뮬 안에서는
# 여전히 45도가 상한이라 관절이 리밋에 얹힌 채로 돈다 — v33 이 그 상태로 663 iter
# 를 돌았고 곡선은 건강했다(경고 없음). 60도를 시뮬에 반영하려면
# `ISAACLAB_PATH=... ./scripts/setup/convert_urdf.sh --headless` 를 한 번 돌리면 된다.
#
# ⚠️ `odm import` 로 CAD 를 다시 받으면 이 한계가 45도로 되돌아간다. 영구히 하려면
# OnShape 어셈블리의 메이트 한계를 고쳐야 한다.
READY_JOINT_POS_G115_ZNECK = dict(READY_JOINT_POS_G115)
READY_JOINT_POS_G115_ZNECK.update({"neck_pitch": 0.5236, "head_pitch": 0.5236})

# settle_height.py 실측 (2026-08-07, 16 envs x 400 steps, 200 mm 에서 낙하).
#   안착 138.3 mm (표준편차 1.6) · 최소/최대 135.0 / 154.1
#   스폰은 최대 안착 + 4 mm 여유. 레퍼런스 위상마다 다리 높이가 달라서
#   평균이 아니라 **최댓값** 기준으로 잡아야 어떤 위상에서도 관통이 없다.
# 키 변화: 136 mm (h175, 구 모델) -> 138.3 mm (g115, 신 모델).
READY_BASE_HEIGHT_G115 = 0.1383
SPAWN_BASE_HEIGHT_G115 = 0.1581


@configclass
class JoystickEnvCfg_V33N(JoystickEnvCfg_ZNeck):
    """imitation_v33n — v33 에서 **CAD 모델과 보행 궤적만** 새것으로 바꾼 것.

    v33 은 iter 663 에서 멈췄지만 곡선은 건강했다 (경고 없음, 최근 500 iter
    +24.8% 상승, 같은 시점 v32 대비 리워드 +7% / 에피소드 +12%). 즉 Z 자 목
    자세 자체는 문제가 아니었다. 다만 그 뒤로 로봇이 바뀌어서 이어 돌릴 수 없다:

        총질량      2.3388 -> 2.7140 kg  (+16%)
        IMU 위치    (-80, 0, +50) -> (-38.5, 0, +89.6) mm   ← 관측이 바뀐다
        레퍼런스    ref_h175 -> ref_g115

    IMU offset 은 원본 Open Duck Mini 의 MJCF 에서 복사돼 있던 값이었고 이 로봇의
    CAD 값으로 고쳤다 (imu_map.MOUNT_POS). lin_acc_b 에 원심·접선 항이 섞이므로
    **그 이전 체크포인트는 이 설정과 호환되지 않는다.**

    v33 대비 바뀐 것은 위 셋 + head_pitch 를 하드 리밋에서 떼어낸 것뿐이다.
    나머지(안쪽 고관절 이탈 벌점, lock_head_joints, 큰 네트워크, gamma 0.97)는 같다.

    비교 기준: v32 는 추종 0.0200 · 5 mm 위반 1.0% · 에피소드 674.
    질량이 16% 늘었으니 **에피소드 길이(넘어짐 빈도)를 같이 볼 것.**
    """

    reference_motion_pkl = "source/open_duck_mini_isaaclab/reference_motion/data/ref_g115.pkl"

    robot = OPEN_DUCK_MINI_V2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=OPEN_DUCK_MINI_V2_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_G115),
            joint_pos=dict(READY_JOINT_POS_G115_ZNECK),
        ),
    )
    ready_base_height = READY_BASE_HEIGHT_G115


@configclass
class JoystickEnvCfg_V34C(JoystickEnvCfg_V33N):
    """imitation_v34c — v33n 그대로 + **정지에서 보행 위상 고정** 하나만.

    실기 배포에서 "정지하면 가만히 서 있을 것" 이 요구사항인데, v33n 은 정지
    명령에서도 몸통이 움찔거린다. 2026-08-07 실측으로 원인을 좁혔다:

      * 정지 명령 실측 속도가 (-0.019, +0.003) m/s — 뒤로 1.9 cm/s 흐른다.
      * `reward_imitation` 은 정지에서 이미 완전히 꺼진다 (`* (cmd_norm > 0.01)`).
      * 그런데 관측의 `imitation_phase` 는 명령과 무관하게 계속 돈다.
      * 붙잡아 줄 것은 `cost_stand_still`(-0.2) 뿐이고 imitation(+0.19)의 1/30 이다.

    즉 정책은 "정지" 라는 상태를 나타내는 안정된 입력을 받지 못한 채 회전하는
    위상 신호를 계속 받는다. 여기서는 **그 구조만** 고친다 — 정지면 위상을 0 으로
    묶어 (cos,sin)=(1,0) 한 점으로 만든다.

    `stand_still_scale` 은 일부러 -0.2 그대로 둔다. 이 프로젝트의 교훈이
    "리워드 계수보다 구조가 먼저" 이고, 계수까지 같이 바꾸면 어느 쪽이 들었는지
    알 수 없다. 위상 고정만으로 부족하면 그때 계수를 올린다.

    비교 기준 (v33n, 2026-08-07 실측):
        정지 실측 속도   (-0.019, +0.003) m/s
        추종 평균        0.0271 m/s
        리워드 / 에피소드  399.49 / 729.1
        짧은 접촉        27.4 %
    **판정은 정지 속도가 0 에 얼마나 가까워지는가로 한다.**
    """

    standstill_hold = True


# ── 레퍼런스 높이 +10 mm / +20 mm 변형 ──────────────────────────────────
# v34c 로 정지가 잡히면, 같은 정지 수정을 유지한 채 몸통 높이만 올려 추종을
# 되찾는 실험이다. v28(추종 최고 0.0171)의 실제 보행 높이가 141 mm 였고
# v33n 은 130 mm 라, 높이가 추종에 유리할 가능성이 있다.
#
# READY 자세는 각 레퍼런스에서 calc_home.py 로 뽑았다. 필요 최대 액션은
# g125 1.78 / g135 1.96 — v1~v9 를 죽인 8.1 대비 여유가 있다.
# 안착/스폰 높이는 settle_height.py 로 **실측해서** 채운다 (추측 금지).

READY_JOINT_POS_G125 = {
    "left_hip_yaw": -0.0031, "left_hip_roll": 0.0207, "left_hip_pitch": 0.8952,
    "left_knee": -1.5693, "left_ankle": 0.7444,
    "neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": -0.0037, "right_hip_roll": -0.0104, "right_hip_pitch": 0.9315,
    "right_knee": 1.6000, "right_ankle": -0.7388,
}
READY_JOINT_POS_G125_ZNECK = dict(READY_JOINT_POS_G125)
READY_JOINT_POS_G125_ZNECK.update({"neck_pitch": 0.5236, "head_pitch": 0.5236})

READY_JOINT_POS_G135 = {
    "left_hip_yaw": -0.0031, "left_hip_roll": 0.0204, "left_hip_pitch": 0.7897,
    "left_knee": -1.3732, "left_ankle": 0.6538,
    "neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": -0.0036, "right_hip_roll": -0.0111, "right_hip_pitch": 0.8264,
    "right_knee": 1.4084, "right_ankle": -0.6522,
}
#: ref_g125sym (2026-08-11) 기준 홈 자세. `scripts/diag/calc_home.py` 로 뽑았다.
#:
#: 기존 G115 자세는 그 자체가 좌우 비대칭이었다 — hip_roll 이 좌 +0.0207 / 우 -0.0104
#: 인데 거울 부호가 +1 이라 대칭이면 우도 +0.0207 이어야 한다 (**1.78도 어긋남**).
#: 관측이 `joint_pos_rel = pos - READY` 라 이 어긋남이 정책 입력에 그대로 실린다.
#: 대칭 레퍼런스로 다시 뽑으니 좌 +0.0074 / 우 +0.0043 로 **0.18도**가 됐다.
READY_JOINT_POS_G125SYM = {
    "left_hip_yaw": -0.0036,
    "left_hip_roll": 0.0074,
    "left_hip_pitch": 0.9052,
    "left_knee": -1.5647,
    "left_ankle": 0.7298,
    "neck_pitch": 0.0,
    "head_pitch": 0.0,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": -0.0039,
    "right_hip_roll": 0.0043,
    "right_hip_pitch": 0.9341,
    "right_knee": 1.5859,
    "right_ankle": -0.7220,
}
#: 학습은 목·머리를 30도 숙인 자세로 한다 (사용자 지시, v33n 이후 계속).
#: settle_height.py 실측 (2026-08-11, 16 env x 400 스텝, 175 mm 스폰).
#:   안착 149.2 mm (표준편차 1.9) · 최소/최대 146.1 / 152.0
READY_BASE_HEIGHT_G125SYM = 0.1492
SPAWN_BASE_HEIGHT_G125SYM = 0.1560

READY_JOINT_POS_G125SYM_ZNECK = dict(READY_JOINT_POS_G125SYM)
READY_JOINT_POS_G125SYM_ZNECK.update({"neck_pitch": 0.5236, "head_pitch": 0.5236})

READY_JOINT_POS_G135_ZNECK = dict(READY_JOINT_POS_G135)
READY_JOINT_POS_G135_ZNECK.update({"neck_pitch": 0.5236, "head_pitch": 0.5236})

# settle_height.py 실측 (2026-08-07, 16 envs x 400 steps, 200 mm 낙하).
#   g125: 안착 148.5 mm (표준편차 1.3) · 최소/최대 146.5 / 150.8
READY_BASE_HEIGHT_G125 = 0.1485
SPAWN_BASE_HEIGHT_G125 = 0.1548
#   g135: 안착 158.4 mm (표준편차 1.4) · 최소/최대 156.8 / 160.9
# 처음 스폰 210 mm 로 쟀을 때는 한 환경이 안착하지 못해 최대값이 스폰 높이
# 그대로 210 으로 찍혔다. 그 값을 스폰 기준(최대+4mm)으로 쓰면 214 mm 에서
# 떨어뜨리게 된다 — 스폰을 175 mm 로 낮춰 다시 쟀고 표준편차가 2.2 -> 1.4 로
# 조여졌다. **안착 최대값이 스폰 높이와 같으면 오염된 측정이다.**
READY_BASE_HEIGHT_G135 = 0.1584
SPAWN_BASE_HEIGHT_G135 = 0.1649


@configclass
class JoystickEnvCfg_V34C10(JoystickEnvCfg_V34C):
    """imitation_v34c10 — v34c(정지 위상 고정) + 레퍼런스 높이 +10 mm (ref_g125)."""

    reference_motion_pkl = "source/open_duck_mini_isaaclab/reference_motion/data/ref_g125.pkl"
    robot = OPEN_DUCK_MINI_V2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=OPEN_DUCK_MINI_V2_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_G125),
            joint_pos=dict(READY_JOINT_POS_G125_ZNECK),
        ),
    )
    ready_base_height = READY_BASE_HEIGHT_G125


@configclass
class JoystickEnvCfg_V35(JoystickEnvCfg_V34C10):
    """imitation_v35 — v34c10 + **정지 시 몸통 수직**을 중력벡터로 직접 요구.

    목표는 사용자가 한 문장으로 준 것 그대로다: "정지시 몸통 수직".

    왜 v34u 방식이 아닌가. v34u 는 같은 목표를 **정지 목표 관절각**으로 우회해서
    풀었는데(FK 로 몸통이 수직이 되는 관절각을 계산해 cost_stand_still 의 목표로
    넣음), 실측 몸통 피치가 +8.19 -> **+20.93 도로 2.5배 나빠졌다**. 관절각은
    간접 손잡이라서다 — 발바닥이 지면에 눕도록 물리가 몸통을 돌려버리면 목표
    관절각을 맞춰도 몸통은 안 선다. 그래서 이번엔 `projected_gravity` 의 x,y 를
    직접 벌한다 (rewards.cost_upright_standstill).

    계수 -20 의 근거: tilt = sin^2(theta) 이므로 8.19 도에서 0.41, 20.93 도에서
    2.55 의 벌점이 된다. tracking(2.5/6.0) 과 같은 자릿수이고 alive(20) 보다는
    작다 — 자세를 잡되 서 있는 것 자체를 포기하게 만들지는 않는 크기.
    무릎은 건드리지 않는다 (사용자: 지금도 무릎 토크가 높다).

    ⚠️ 로봇도 새 CAD 다. body tail 실측 113 g 을 밀도로 반영해 총질량
    2.7140 -> 2.7430 kg (+1.07%), 전체 CoM 이 뒤로 1.06 mm. USD 를 백업본과
    대조해 그 외에는 아무것도 안 바뀌었음을 확인했다(head_pitch 한계는 구 USD 도
    이미 ±50도). 질량은 도메인 랜덤화 범위 안이지만 v34c10 과의 비교에는
    섞여 있으니, 판정이 애매하면 계수만 0 으로 둔 순수 새-CAD 기준선을 따로 잰다.

    비교 기준 (v34c10 @2999): 정지 0.0058 · 추종 0.0166 · 리워드 426.2 · 에피 778.7
                    (@1500): 정지 0.0041 · 추종 0.0165
    몸통 피치 @1500:  v34c10 +8.19 도 · v34u +20.93 도  <- **이걸 0 에 가깝게**
    """

    upright_standstill_scale = -20.0


@configclass
class JoystickEnvCfg_V36(JoystickEnvCfg_V35):
    """imitation_v36 — v35 + **정지 좌우 대칭** + **액션 저역통과를 학습에 포함**.

    ⚠️ 한 번에 둘을 바꾼다 (사용자 지시). 원칙에는 어긋나지만 **읽는 지표가
    서로 달라서** 사후 귀속이 된다: 대칭 벌점은 leg_symmetry.py 의 좌우 어긋남
    으로, 필터는 몸통 각속도 RMS 로 읽는다.

    1) 정지 좌우 대칭 (`leg_symmetry_scale = -3.0`)
       v35 실측: **보행은 이미 거의 완벽히 대칭**(모든 관절 1.03도 이내)인데
       **정지만 크게 어긋난다** — 무릎 15.2도, hip_roll 12.3도, ankle 11.4도.
       env 별 표준편차가 2.0~2.3도로 좁아 무작위가 아니라 32개 환경이 전부 같은
       쪽으로 치우친 일관된 편향이다(한쪽 다리에 기대어 선다).
       정지에서는 imitation 이 꺼지고 위상도 0 에 묶여, 좌우 대칭을 요구하는 항이
       하나도 없던 것이 원인이다. 몸통 수직을 projected_gravity 로 직접 벌해
       해결한 것과 같은 모양의 처방이다.

    2) 액션 저역통과 (`action_lowpass_alpha = 0.5`, 차단 5.8 Hz, 지연 20 ms)
       **배포에만 걸지 않고 학습에 넣는다.** 배포에서만 걸면 정책이 겪어본 적
       없는 위상 지연이 생기고, 그건 우리가 줄이려는 sim2real 격차를 스스로
       만드는 것이다. v35 로 잰 사후 스윕에서 추종 오차가 alpha 에 따라 단조롭게
       나빠진 것(+2% @0.5, +26% @0.7)이 바로 그 대가였다. 플랜트의 일부로 넣으면
       정책이 지연을 미리 보상하는 것을 배운다.
       0.5 를 고른 근거: 그 스윕의 무릎점이다 — 정지 떨림 -44%, 보행 -14% 를
       추종 +2% 로 얻는 지점. 0.7 이상은 추종 대가가 급히 커지고, 0.8 은 차단
       1.8 Hz 로 보행 주파수(1.85 Hz)보다 낮아 보행 자체를 깎는다.

    비교 기준 (v35 @2999): 정지 0.0030 · 몸통피치 +0.63도 · 추종 0.0206
                            · 리워드 375.3 · 에피 683.8
    정지 좌우 어긋남 (v35): 무릎 -15.24도 · hip_roll +12.26도 · ankle +11.35도
    **판정: 몸통 수직(+0.63도)과 정지 속력을 지키면서 좌우 어긋남이 줄어드는가.**
    """

    leg_symmetry_scale = -3.0
    action_lowpass_alpha = 0.5


@configclass
class JoystickEnvCfg_V37(JoystickEnvCfg_V36):
    """imitation_v37 — v36 에서 **저역통과를 정지에만** 건다. 한 가지만 바뀐다.

    v36 은 alpha 0.5 를 명령과 무관하게 걸었는데, 스윕이 그게 타협이었음을
    보여줬다:

    | | 정지 몸통각속도 | 정지 속력오차 | 보행 추종오차 |
    |---|---|---|---|
    | alpha 0.0 | 0.2408 | 0.0193 | 0.0828 |
    | alpha 0.5 | 0.1357 | 0.0122 | 0.0803 |
    | alpha 0.7 | 0.0934 | 0.0091 | 0.0963 |

    **정지는 올릴수록 대가 없이 좋아진다** — 떨림도 줄고 정지 속력 오차까지 같이
    준다. **보행만 0.7 에서 추종이 꺾인다**(0.0803 -> 0.0963). 명령을 아는데
    하나로 타협할 이유가 없어서 갈랐다: 정지 0.7, 보행 0.

    보행에서 0 으로 두는 근거: v35 사후 스윕에서 보행 떨림 감소가 alpha 0.5 에서
    14%뿐이었고 추종은 그만큼 나빠졌다. 보행 쪽 떨림은 필터가 아니라 다른
    수단(접촉 채터링, action_rate)으로 다뤄야 한다.

    정지 0.7 의 위험: 군지연 47 ms 라 외란이 왔을 때 반응이 그만큼 늦는다.
    사후 스윕은 push_robot 을 꺼서 이걸 시험하지 못했지만, **학습에는 외란이
    켜져 있으므로 이 값이 균형 회복을 망치면 학습 중에 드러난다** — 넘어짐 비율과
    에피소드 길이로 본다.

    비교 기준 (v36 @2999): 정지 속력 0.0020 · 몸통피치 +0.52도(표준편차 6.05)
      · 추종 0.0210 · 정지 좌우 어긋남 무릎 -2.57도 · 리워드 355.5 · 에피 644.3
    v36 의 몸통피치 표준편차(6.05도)가 v35(3.32도)보다 나빴다 — 여기가 개선되는지도 본다.
    """

    action_lowpass_alpha = 0.0              # 보행 중에는 끈다
    action_lowpass_alpha_standstill = 0.7   # 정지에서만 건다


@configclass
class JoystickEnvCfg_V38(JoystickEnvCfg_V37):
    """imitation_v38 — v37 에서 **hip_inward 를 보행 중에만** 건다. 한 가지만 바뀐다.

    v36 에 남은 정지 좌우 비대칭(hip_roll +4.66도)의 정체를 숫자까지 맞췄다.
    `hip_inward`(-25.0)는 레퍼런스를 기준으로 고관절 4축(L/R yaw·roll)이 안쪽으로
    3도 넘게 벗어나는 것을 벌하는데, **cmd_norm 게이트가 없어 정지에서도 산다.**
    그리고 정지에서는 standstill_hold 가 위상을 0 에 묶으므로 그 기준이 "걷는 중
    한쪽 발을 든 순간" 의 자세다 — 위상 0 의 hip_roll 이 좌 -7.69 / 우 +5.50 으로
    13.19도 벌어져 있다.

    그래서 정지에서 두 항이 정면으로 싸운다. leg_symmetry(-3.0)는 R_roll 을 L_roll
    (+1.60도)과 같게 하라 하고, hip_inward(-25.0)는 R_roll 이 +2.50도 아래로 가면
    벌한다. 계수가 8배라 hip_inward 가 이기고, 정책은 +6.26도에서 멈춘다.
    **어긋남 6.26 - 1.60 = 4.66도, 측정값과 정확히 일치한다.**

    정지에서 꺼도 되는 이유: 이 항의 목적은 보행 중 다리-몸통 자가충돌 방지인데
    (접촉 = 액추에이터 파손), 정지에서는 양발이 땅에 붙어 거의 기본 자세이고
    재생 실측 최소 간격이 62 mm 다 (위험선 5 mm).

    비교 기준 (v36 @2999): 정지 hip_roll 어긋남 +4.66도 · hip_yaw -3.15도
      · 몸통피치 +0.52도(표준편차 6.05) · 정지속력 0.0020 · 추종 0.0210
    **판정: hip_roll 어긋남이 줄되 보행 대칭과 자가충돌 여유가 안 나빠지는가.**
    (`scripts/diag/leg_symmetry.py` 와 `leg_trunk_clearance.py` 로 본다.)
    """

    hip_inward_walking_only = True


@configclass
class JoystickEnvCfg_V39(JoystickEnvCfg_V38):
    """imitation_v39 — **저역통과 필터를 완전히 빼고, 진동을 리워드로 억제**한다.

    필터는 증상을 가리는 도구였고 실제로 정책을 게으르게 만들었다. 필터로 학습한
    v36 의 **원시 액션 요동이 0.0638 로 무필터 v35(0.0537)보다 컸다** — 필터가
    결과를 흡수해 주니 정책이 거칠어져도 대가를 안 치른 것이다. 게다가 위상 지연
    (alpha 0.5 에서 20 ms, 0.7 에서 47 ms)이 추종을 깎고 외란 극복의 여유를 먹는다.

    **왜 기존 action_rate 로는 안 되는가.** 그건 1차차분이라 "빠른 움직임" 을
    벌한다 — 진동이든 정상 보행이든 똑같이. 진동만 골라 누르지 못한다.
    2차차분은 방향이 뒤집힐 때만 커지고 매끄러운 궤적에서는 정확히 0 이다:

        매끄러운 램프        1차차분^2 0.0001   2차차분^2 0.0000
        매 스텝 부호 반전    1차차분^2 1.0000   2차차분^2 4.0000

    실기 로그(정지 10 초) 실측: 1차차분^2 0.148 / **2차차분^2 0.441**. 진동 성분이
    지배적이고 이것이 액션 방향 반전 68.3% 의 정체다.

    계수 -0.25 의 근거: 실측 2차차분^2 0.441 에 곱하면 -0.11 로, action_rate 의
    기여(-0.5 x 0.148 = -0.074)와 같은 자릿수다. 첫 추정치이므로 결과를 보고
    조정한다.

    v38 에서 두 가지가 바뀐다 (읽는 지표가 달라 사후 귀속이 된다):
      필터 제거 + 진동 벌점  ->  액션 방향 반전 %, 몸통 피치 표준편차, 추종
      hip_inward 정지 게이트 ->  정지 hip_roll 좌우 어긋남 (v38 에서 물려받음)

    비교 기준 (v37 @2000): 몸통피치 +2.38도(표준편차 3.36) · 정지 hip_roll 어긋남
      +4.67도 · 추종 평균 0.0130 · 관절한계 포화 0.0% · 외란 1.0 m/s 넘어짐 28.1%
    **판정: 필터 없이도 떨림이 안 늘고(액션 반전·피치 표준편차), 추종과 외란이
    나아지는가.** 위상 지연이 사라지므로 그 둘은 좋아져야 앞뒤가 맞는다.
    """

    action_lowpass_alpha = 0.0
    action_lowpass_alpha_standstill = 0.0   # 필터 완전히 끔
    action_jerk_scale = -0.25


@configclass
class JoystickEnvCfg_V40(JoystickEnvCfg_V39):
    """imitation_v40 — v39 에서 **action_rate 를 절반으로** 낮춘다. 한 가지만.

    v39 는 떨림을 잡았지만(action_rate 비용 v37 대비 -45%, 원시 액션요동 v36 대비
    -39%) 추종을 0.0120 -> 0.0197 로 내줬다. 원인은 취향이 아니라 **예산**이다:

        매끄러움에 쓴 리워드   v37  -0.0304  (action_rate 만)
                              v39  -0.0371  (rate -0.0167 + jerk -0.0204)

    22% 더 쓴 만큼 추종에서 빠졌다. 그런데 두 항의 **역할이 겹치지 않는다** —
    1차차분은 "빠른 움직임"을 벌하고 2차차분은 "진동만" 벌한다 (매끄러운 램프는
    2차차분이 정확히 0). 진동 억제를 jerk 가 맡은 이상 action_rate 가 남아서 하는
    일은 추종에 필요한 빠른 관절 이동을 이중과금하는 것뿐이다.

    그래서 -0.5 -> -0.25. jerk(-0.25)는 건드리지 않는다. 예상 매끄러움 예산은
    -0.008 - 0.020 = -0.028 로 v37(-0.0304)보다 낮아지고, 그 여유가 추종으로 간다.

    비교 기준 (v39 @2600): 추종 0.0197 · 정지 원시 액션요동 0.0390 · 정지 몸통
      각속도 0.1457 · 원시 2차차분 0.0816 · 정지 피치 +2.93도(표준편차 4.56)
      · hip_roll 어긋남 +1.85도 · 외란 1.0 m/s 넘어짐 37.5%
    (v37 @2000 참고: 추종 0.0120 · 피치 +2.38도(3.36) · 외란 28.1%)

    **판정: 추종이 v37 수준(0.0120)으로 돌아오면서 원시 2차차분과 액션요동이
    v39 수준을 지키는가.** 요동이 같이 늘면 진동 억제를 실제로 하고 있던 건
    jerk 가 아니라 rate 였다는 뜻이므로, 되돌리고 jerk 계수 쪽을 만진다.
    """

    action_rate_scale = -0.25


@configclass
class JoystickEnvCfg_V41(JoystickEnvCfg_V40):
    """imitation_v41 — 액추에이터를 **DCMotor 로**. 토크-속도 결합 하나만 바뀐다.

    지금까지의 `ImplicitActuatorCfg` 는 effort_limit(4.1)과 velocity_limit(4.82)을
    **서로 독립으로** 건다. 심의 다리는 "4.1 N·m 를 내면서 동시에 4.82 rad/s 로"
    움직일 수 있는데 실제 DC 모터는 그 조합을 못 낸다. **정책은 존재하지 않는
    액추에이터를 전제로 걸음을 배웠다.**

    2026-08-10 실기가 이걸 정확히 짚었다 (docs/reports/hw_v40_followup...md §3).
    전류 상한을 700 틱(1.88 A)으로 올려 **토크 포화를 0 % 로 없앤 뒤에도**
    무릎·고관절이 지령 속도를 못 따라갔다:

        관절             지령 p95   실측 최대      토크-속도 직선 사용률
        left_knee          4.84       3.53              81 %
        right_knee         5.17       3.98               —
        left_hip_pitch     4.66       3.72              93 %
        right_ankle          —          —               99 %
        left/right hip_yaw  (오차 거의 없음)          51 / 57 %

    같은 무릎이 **무부하** 계단시험에서는 4.41 rad/s (한계의 92 %) 를 낸다.
    부하가 걸릴 때만 막히고, **직선에 붙은 축이 곧 뒤처지는 축이다.**

    DCMotor 는 그 직선을 그대로 구현한다 (actuator_pd.py `_clip_effort`):
        torque_speed_top = saturation_effort * (1 - joint_vel / velocity_limit)

    값 (robot_cfg.py 참조): 스톨 4.1 · 무부하속도 4.82 · 연속토크 3.16
    (= 실기 전류 상한 700 틱). stiffness/damping/armature/friction 은 그대로 —
    37.65 는 실기 P 게인 800 을 넣어 BAM 실측으로 유도한 값이고, 실기에서 읽은
    P/I/D 도 다리 전 축 800/0/4700 이다. **게인은 문제가 아니었다.**

    비교 기준 (v40 @2000): 추종 0.0168 · 정지 피치 +3.26도(표준편차 2.77)
      · 정지 hip_roll 어긋남 +0.40도 · 액션요동(정지) 0.0445 · 관절한계 0.0 %
      · 외란 1.0 m/s 넘어짐 50.0 % · 보행 토크 제곱합 7.983

    **판정: 실기가 낼 수 있는 걸음으로 바뀌는가.** 심 지표는 나빠질 수 있다 —
    이번엔 그게 정상이다. 정책이 이제 못 내는 토크·속도를 못 쓰기 때문이다.
    심 성능이 아니라 **실기 추종오차**(v40 무릎 28.10도)로 판정한다.
    보행 중 관절 속도가 토크-속도 직선 안에 들어오는지도 같이 본다.

    ⚠️ DCMotor 는 명시적 액추에이터라 PD 가 PhysX 에서 파이썬으로 옮겨온다.
    같은 게인·같은 dt(0.002)라 발산할 이유는 없지만, 초반 리워드가 이전 곡선과
    많이 다르면 액추에이터 모델 교체의 영향이므로 계수 탓으로 오독하지 말 것.
    """

    robot = OPEN_DUCK_MINI_V2_DC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=OPEN_DUCK_MINI_V2_DC_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_G115),
            joint_pos=dict(READY_JOINT_POS_G115_ZNECK),
        ),
    )


@configclass
class JoystickEnvCfg_V42(JoystickEnvCfg_V41):
    """imitation_v42 — **대칭으로 다시 뽑은 레퍼런스**(ref_g125sym). 하나만 바뀐다.

    v35 이후 모든 정책이 좌우 비대칭 레퍼런스를 모방해 왔다. 2026-08-11 에
    원인을 셋 찾았고 전부 고쳤다 (docs/reports/reference_gaps_2026-08-11.md):

      1. `fit_poly.py` 가 시간축을 **float32** 로 캐스팅했다. 15 차를 t∈[0,1) 에서
         맞추면 조건수가 나빠 정밀도가 그대로 결과에 나온다 —
         float32 재현오차 6.883도 / 무릎 좌우차 +16.4 %,
         float64 재현오차 0.786도 / 무릎 좌우차 -3.1 % (녹화 원본 -1.8 %).
      2. `poly_reference_motion.py` 가 계수를 float32 텐서에 담았다. 계수가 1e8 대라
         Horner 평가에서 자릿수 소거가 나 무릎 좌우차가 +176 % 까지 튀었다.
         평가를 float64 로 바꿨다.
      3. `auto_waddle.py` 속도 필터가 격자의 45 % 를 삭제했다. 하필 직진 슬라이스의
         dx = 0.0/0.148/0.222 가 전부 빠져 **전진 속도를 바꿔도 레퍼런스가 안 바뀌었다**
         (저속 제자리걸음의 원인). `--no-speed-filter` 로 껐다.

    추가로 직진 녹화 6 개를 반주기 거울로 시계열에서 대칭화했다.

        지표 (vx=+0.222 직진)      ref_g125 -> ref_g125sym
        격자 충전율                  55 %   ->  100 %
        dy=0 존재                    없음   ->  있음
        hip_pitch 진폭 좌우차      +41.0 %  ->  -2.9 %
        knee                       +11.1 %  ->  +2.5 %
        ankle                       -8.0 %  ->  +1.2 %

    **READY 자세도 함께 바뀐다** — 레퍼런스에서 유도되는 값이라 분리할 수 없다.
    기존 G115 자세는 그 자체가 hip_roll 1.78도 비대칭이었고 (관측이
    `pos - READY` 라 정책 입력에 그대로 실린다) 새 자세는 0.18도다.

    비교 기준 (v41 @800): 정지 피치 +0.59도(표준편차 4.82) · 정지 hip_yaw 어긋남
      -3.89도 · hip_roll +0.73도 · 추종 6방향 · 실기 앞쏠림 +4.87도

    **판정: 정지·보행 좌우 어긋남이 줄고, 저속에서 제자리걸음이 사라지는가.**
    실기 앞쏠림(+4.87도)이 더 줄면 레퍼런스 비대칭이 그 일부였다는 뜻이다.
    """

    reference_motion_pkl = "source/open_duck_mini_isaaclab/reference_motion/data/ref_g125sym.pkl"

    robot = OPEN_DUCK_MINI_V2_DC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=OPEN_DUCK_MINI_V2_DC_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_G125SYM),
            joint_pos=dict(READY_JOINT_POS_G125SYM_ZNECK),
        ),
    )
    ready_base_height = READY_BASE_HEIGHT_G125SYM


@configclass
class JoystickEnvCfg_V43(JoystickEnvCfg_V42):
    """imitation_v43 — 액션 지연을 **실측 고정값**으로. 하나만 바뀐다.

    2026-08-12 에 실기 14축 주파수 응답을 처음 쟀다 (`scripts/hw/full_check.py`).
    하드웨어는 완전히 정상이었다 — 다리 10축 전부 이득 0.95~1.00, 좌우차 2 % 이내,
    전류는 상한의 30 % 이하. 그동안 의심하던 무릎도 0.99 였다.

    남은 것은 **위상 지연**이고, 두 주파수에서 같은 시간지연으로 환산된다:

        1.85 Hz  +20.2도  ->  30.3 ms
        3.00 Hz  +36.8도  ->  34.1 ms

    두 값이 거의 같으므로 대역폭 부족이 아니라 **순수 시간지연**이다 (서보 내부
    제어 주기 + 버스 왕복). 제어 주기 20 ms 기준 약 1.5 스텝이다.

    지금까지는 `action_min_delay=0, action_max_delay=3` 으로 **0~60 ms 를 매
    리셋마다 무작위**로 줬다. 평균은 맞지만 정책은 "지연이 얼마인지 모르는 채
    평균적으로 버티기" 를 배운다. 실기 지연은 **항상 30 ms 로 일정**하므로,
    그걸 전제로 미리 내다보는 여지를 정책이 못 쓰고 있었다.

    실기 증상이 이걸로 설명된다: 목표를 내면 20도 늦게 따라오고, 정책이 그
    오차를 더 큰 목표로 보상하다 URDF 한계(hip_pitch +70도)에 부딪힌다.
    심 보행 목표 최대는 65.2도로 여유 4.8도인데, 실기에서는 같은 정책이
    78.7도까지 갔다.

    1~2 스텝(20~40 ms)으로 좁혀 실측 30 ms 를 감싸되 약간의 변동만 남긴다.
    0 을 없애는 것이 핵심이다 — 지연 없는 세상은 실기에 존재하지 않는다.

    비교 기준 (v42 @2000): 추종 0.0153 · 정지 피치 +0.86도(표준편차 6.53)
      · 보행 hip_roll 여유 +1.3도 · hip_pitch 여유 +4.8도
      · 실기 앞쏠림 +4.3도 (vx=0.05, 손 받침)

    **판정: 보행 중 한계 여유가 늘고, 실기 앞쏠림이 줄어드는가.**
    지연을 전제로 배우면 정책이 앞서 움직여 오차 보상이 줄고, 그러면 한계에
    덜 붙는다.
    """

    action_min_delay = 1      # env step (20 ms)
    action_max_delay = 2      # env step (40 ms) — 실측 30 ms 를 감싼다


#: ref_g135sym (2026-08-12) 기준 홈 자세. calc_home.py 로 뽑았다.
#: v42(g125sym) 대비 무릎이 13도 더 펴진다 — 실기 왼무릎이 -110도 아래에서
#: 힘을 못 내는 것을 피하려는 것이다 (아래 V46 docstring 참고).
READY_JOINT_POS_G135SYM = {
    "left_hip_yaw": -0.0036,
    "left_hip_roll": 0.0076,
    "left_hip_pitch": 0.7827,
    "left_knee": -1.3391,
    "left_ankle": 0.6266,
    "neck_pitch": 0.0,
    "head_pitch": 0.0,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": -0.0037,
    "right_hip_roll": 0.0027,
    "right_hip_pitch": 0.8125,
    "right_knee": 1.3654,
    "right_ankle": -0.6230,
}
#: settle_height.py 실측 (2026-08-12, 16 env x 400 스텝, 185 mm 스폰).
#:   안착 161.2 mm (표준편차 1.5) — v42(g125sym) 의 149.2 mm 보다 12 mm 더 선다.
READY_BASE_HEIGHT_G135SYM = 0.1612
SPAWN_BASE_HEIGHT_G135SYM = 0.1687

READY_JOINT_POS_G135SYM_ZNECK = dict(READY_JOINT_POS_G135SYM)
READY_JOINT_POS_G135SYM_ZNECK.update({"neck_pitch": 0.5236, "head_pitch": 0.5236})

# ref_g135fh20 — g135sym 에서 walk_foot_height 만 0.04 -> 0.02 로 낮춘 레퍼런스.
# calc_home.py --pkl ref_g135fh20.pkl (2026-08-13). 최대 |action| 1.05.
READY_JOINT_POS_G135FH20 = {
    "left_hip_yaw": -0.0039,
    "left_hip_roll": 0.0044,
    "left_hip_pitch": 0.7447,
    "left_knee": -1.2694,
    "left_ankle": 0.5948,
    "neck_pitch": 0.0000,
    "head_pitch": 0.0000,
    "head_yaw": 0.0000,
    "head_roll": 0.0000,
    "right_hip_yaw": -0.0035,
    "right_hip_roll": -0.0007,
    "right_hip_pitch": 0.7677,
    "right_knee": 1.2841,
    "right_ankle": -0.5866,
}
READY_JOINT_POS_G135FH20_ZNECK = dict(READY_JOINT_POS_G135FH20)
READY_JOINT_POS_G135FH20_ZNECK.update({"neck_pitch": 0.5236, "head_pitch": 0.5236})

# 이 두 높이는 settle_height.py 실측이 아니라 **FK + 계통보정** 이다.
# GPU 가 학습으로 차 있어 실측을 못 돌렸다.
#
#   leg_fk 로 READY 자세의 몸통-발 거리를 구하면
#       G135SYM   0.1568 m   (settle_height 실측 0.1612 → -4.4 mm)
#       G135FH20  0.1604 m
#   FK 체인이 발바닥이 아니라 발 프레임 원점에서 끝나 생기는 일정 오차이므로
#   같은 -4.4 mm 를 보정했다: 0.1604 + 0.0044 = 0.1648.
#   SPAWN 은 READY 와의 차이(0.1687-0.1612 = 7.5 mm)를 그대로 얹었다.
#
# ⚠️ GPU 가 비면 `scripts/diag/settle_height.py` 로 확인할 것. 어긋나면 리셋
#    직후 로봇이 지면을 파고들거나 떠서 시작한다.
READY_BASE_HEIGHT_G135FH20 = 0.1648
SPAWN_BASE_HEIGHT_G135FH20 = 0.1723

# ref_g135fs14 — fh20 에서 feet_spacing 만 0.18 -> 0.14 로 좁힌 레퍼런스.
# calc_home.py (2026-08-14). 최대 |action| 0.96.
# hip_roll 이 -0.12 rad(-6.9도) 로 크게 바뀐 것이 보간격이 좁아진 결과다.
READY_JOINT_POS_G135FS14 = {
    "left_hip_yaw": -0.0126,
    "left_hip_roll": -0.1191,
    "left_hip_pitch": 0.8050,
    "left_knee": -1.3775,
    "left_ankle": 0.6430,
    "neck_pitch": 0.0000,
    "head_pitch": 0.0000,
    "head_yaw": 0.0000,
    "head_roll": 0.0000,
    "right_hip_yaw": 0.0052,
    "right_hip_roll": -0.1243,
    "right_hip_pitch": 0.8241,
    "right_knee": 1.3903,
    "right_ankle": -0.6367,
}
READY_JOINT_POS_G135FS14_ZNECK = dict(READY_JOINT_POS_G135FS14)
READY_JOINT_POS_G135FS14_ZNECK.update({"neck_pitch": 0.5236, "head_pitch": 0.5236})
# G135FH20 과 같은 방법 (FK 0.1609 + 계통보정 4.4 mm). settle_height 미실측.
READY_BASE_HEIGHT_G135FS14 = 0.1653

#: ref_g135fs22 (2026-08-17) 기준 홈 자세. calc_home.py 로 뽑았다.
#: g135sym 에서 feet_spacing 만 0.18 -> 0.22 로 **넓힌** 레퍼런스다.
#: 부수 효과 하나가 유리하다 — 다리를 벌리면 같은 몸통 높이에서 무릎이 펴진다.
#: 무릎 -76.7 -> -66.7 도로 10 도 더 펴져 **실기 왼무릎 사각지대(-110 도)에서
#: 더 멀어진다** (V46 docstring 참고).
READY_JOINT_POS_G135FS22 = {
    "left_hip_yaw": 0.0051,
    "left_hip_roll": 0.1291,
    "left_hip_pitch": 0.6880,
    "left_knee": -1.1638,
    "left_ankle": 0.5469,
    "neck_pitch": 0.0,
    "head_pitch": 0.0,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": -0.0125,
    "right_hip_roll": 0.1249,
    "right_hip_pitch": 0.7227,
    "right_knee": 1.1956,
    "right_ankle": -0.5439,
}
READY_JOINT_POS_G135FS22_ZNECK = dict(READY_JOINT_POS_G135FS22)
READY_JOINT_POS_G135FS22_ZNECK.update({"neck_pitch": 0.5236, "head_pitch": 0.5236})
#: ⚠️ settle_height.py 로 아직 재지 않았다. leg_fk 로 보면 발 z 가 g135sym 과
#: 같으므로(-156.8 mm 동일) 같은 값을 쓴다. GPU 가 나면 실측해서 고칠 것.
#: 같은 FK 로 보간격은 184.2 -> 224.3 mm (+22 %) 로 의도대로 넓어졌다.
READY_BASE_HEIGHT_G135FS22 = READY_BASE_HEIGHT_G135SYM
SPAWN_BASE_HEIGHT_G135FS22 = SPAWN_BASE_HEIGHT_G135SYM

SPAWN_BASE_HEIGHT_G135FS14 = 0.1728


@configclass
class _LowFrictionEventCfg(EventCfg):
    """마찰 무작위화를 **아래로** 넓힌다. 0.5~1.0 -> 0.3~1.0.

    2026-08-15, 사용자가 실기에서 "발이 밀린다, 슬라이딩하는 것 같다" 고 했다.
    지면은 1.0 고정이고 결합이 multiply 이므로 실효 마찰이 곧 이 범위다 —
    지금은 0.5 아래를 한 번도 겪지 않는다. 실기 바닥이 그보다 미끄러우면
    정책은 있지도 않은 접지력을 믿고 걷는다.

    아래로만 넓히는 이유: 위쪽(1.0)은 이미 덮고 있고, 미끄러운 쪽만 미지수다.
    바닥 마찰을 실제로 잰 적이 없으므로(경사 시험 미실시) 범위로 학습한다 —
    v44 의 질량과 같은 방침이되, 그때 배운 대로 **필요 이상으로 넓히지
    않는다** (0.85~1.25 로 넓힌 질량이 앞뒤 추종을 망쳤다).
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class _ComBackEventCfg(EventCfg):
    """몸통 CoM 을 **뒤로** 밀어 놓고 학습한다. 오프셋은 하위 클래스가 정한다.

    2026-08-18, 교수님 지적: "무게중심을 못 잡는다." 사용자가 뒤로 10/15/20 mm
    를 시험해 보라고 했다.

    근거는 v44 때 이미 측정돼 있다 — 심 모델의 CoM 은 발 중심보다 **16.7 mm
    뒤**인데 **실기는 앞으로 넘어진다.** 두 사실이 같이 서려면 실기의 실제
    CoM 이 심 모델보다 앞에 있어야 한다. 그러면 정책은 있지도 않은 뒤쪽
    여유를 믿고 걷는 셈이다.

    `randomize_rigid_body_com` 은 오프셋을 **더한다.** 범위의 상하한을 같은
    값으로 주면 무작위가 아니라 고정 이동이 된다. -x 가 뒤다 (cmd_vx > 0 이
    전진이고 `root_lin_vel_b[:,0]` 이 그것과 부호가 맞는 것을 실기 로그로
    확인했다).

    ⚠️ 이 함수는 CPU 텐서로 CoM 을 쓴다. 그래서 mode="startup" 으로만 건다.
    """

    com_back = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk_assembly"),
            "com_range": {"x": (0.0, 0.0)},   # 하위 클래스가 덮어쓴다
        },
    )


@configclass
class _WideMassEventCfg(EventCfg):
    """EventCfg 에서 **질량 배율 범위만** 넓힌 것. 나머지 무작위화는 그대로."""

    scale_all_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # 0.9~1.1 -> 0.85~1.25. USD 2.6404 kg 기준 2.24~3.30 kg.
            # 위쪽으로 더 넓힌 이유는 클래스 docstring 참고.
            "mass_distribution_params": (0.85, 1.25),
            "operation": "scale",
        },
    )


@configclass
class JoystickEnvCfg_V44(JoystickEnvCfg_V42):
    """imitation_v44 — **질량 무작위화 범위를 넓힌다**. 하나만 바뀐다.

    2026-08-12 에 심 로봇의 실제 질량을 처음 읽어 봤다: **2.6404 kg**.
    그런데 README 와 training_log 는 **2.7430 kg** 으로 적고 있다 (body tail
    113 g 을 밀도로 반영했다는 기록). **103 g (3.7 %) 이 어긋나 있다** — 그
    반영이 USD 까지 안 갔거나 중간에 되돌아갔다.

    실기 무게는 아직 안 쟀다. 하나에 베팅하는 대신 **범위로 학습**한다 —
    무게를 재고 나서도 같은 정책을 그대로 쓸 수 있다.

        scale_all_mass  0.90~1.10  ->  0.85~1.25
        실효 질량       2.38~2.90  ->  2.24~3.30 kg
        (add_base_mass ±0.1 kg 은 그대로라 실효 범위는 그만큼 더 넓다)

    위쪽으로 더 넓힌 이유: 실기가 심보다 **무거울** 정황이 있다. 같은 READY
    자세에서 심은 무릎에 1.14 N·m 를 쓰는데 실기는 전류가 무부하 수준이라
    환산 토크가 0.2 N·m 다 — 질량 분포가 다르다는 뜻이고, 총 질량도 의심된다.

    함께 확인된 것 (이번 실험 대상은 아니지만 기록):
      * 무게중심이 **y 로 +26.1 mm** 치우쳐 있다. 좌우 대칭이어야 할 로봇인데
        2.6 cm 다. 실기 몸통 롤이 +4.60도로 한쪽으로 기운 것과 방향이 맞는다.
      * CoM 은 발 중심보다 16.7 mm **뒤**다. 앞으로 넘어질 자세가 아닌데
        실기는 앞으로 넘어진다.

    비교 기준 (v42 @2000): 추종 0.0153 · 정지 피치 +0.86도 · 보행 hip_roll
      여유 +1.3도 · hip_pitch 여유 +4.8도

    **판정: 넓은 질량 범위에서도 보행이 유지되는가.** 유지되면 그 정책은
    실기 무게가 얼마로 나오든 쓸 수 있다. 무너지면 질량 민감도가 크다는
    뜻이고, 그때는 무게를 먼저 재서 좁혀야 한다.
    """

    events: EventCfg = _WideMassEventCfg()


@configclass
class JoystickEnvCfg_V46(JoystickEnvCfg_V44):
    """imitation_v46 — **몸통을 10 mm 더 세운다** (ref_g135sym). 하나만 바뀐다.

    2026-08-12 실기에서 앞으로 넘어지는 원인을 관절 단위로 좁혔다.

    **발목·고관절은 정상이다.** 앞으로 쏠릴 때 오차가 오히려 버티는 방향으로
    움직였고 (left_ankle -4.96도, right_hip_pitch -5.88도) 목표 자체도
    되돌리는 쪽으로 갔다 (right_ankle 목표가 -30.6 -> -41.2도). 정책은 일하고
    있었다.

    **하중에 밀린 것은 무릎뿐이다.** left_knee 는 앞쏠림 구간에서 오차가
    13.94도 더 벌어졌고, 시작 1초 만에 -119도(URDF 한계 -120도에서 1.1도)에
    처박혀 15초 내내 안 나왔다. 목표는 -75~-104도를 오가는데 실제는 -118도
    고정, **전류 0.07 A** 였다. 결국 서보가 Overload(err=33)로 토크를 끊었다.

    그런데 **하드웨어는 정상이다.** 두 번 확인했다:
      * 주파수 응답 (READY -89.65도 중심 ±10도): 이득 0.99, 좌우차 1 %
      * 손으로 몸통을 들고 보행: 무릎 -95~-70도 정상 추종, Overload 없음

    즉 **-110도 아래에서만** 힘을 못 낸다. 그 구간은 한 번도 시험한 적이 없다.
    원인(기어·혼·간섭)은 아직 모르지만, **그 구간을 안 지나가면 된다.**

        walk_com_height  0.1973 -> 0.2073  (+10 mm)
        READY 무릎       -89.7도 -> **-76.7도**  (13도 더 펴짐)
        보행 목표 최저   -105.9도 -> 약 -93도
        실기 14도 밀림   -119.9도 (한계) -> 약 -107도  (문제 구간 회피)

    레퍼런스는 오늘 고친 파이프라인으로 새로 뽑았다 (격자 100 %, 직진 녹화
    대칭화, float64 적합). 대칭성 유지: 진폭차 ±2 %, 중립 어긋남 <1도.

    **주의: 심에서는 더 높은 자세가 나빴던 전례가 있다** (v34c20, +20 mm).
    그래도 시도하는 이유는 실기 무릎이 특정 각도에서 죽기 때문이고, 심 점수를
    조금 내주더라도 실기가 걷는 쪽이 낫기 때문이다.

    비교 기준 (v44 @1000): 스텝당 0.5344 · (v42 @2000 추종 0.0153)
    **판정: 실기에서 무릎이 -110도 아래로 안 가고 발이 떨어지는가.**
    """

    reference_motion_pkl = "source/open_duck_mini_isaaclab/reference_motion/data/ref_g135sym.pkl"

    robot = OPEN_DUCK_MINI_V2_DC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=OPEN_DUCK_MINI_V2_DC_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_G135SYM),
            joint_pos=dict(READY_JOINT_POS_G135SYM_ZNECK),
        ),
    )
    ready_base_height = READY_BASE_HEIGHT_G135SYM


@configclass
class JoystickEnvCfg_V47(JoystickEnvCfg_V46):
    """imitation_v47 — **액션 지연을 실측에 맞춘다** (2~3 스텝). 하나만 바뀐다.

    2026-08-13, 실기가 처음으로 자기 발로 걷고 나서 심과 실기를 **같은 자로**
    처음 비교했다 (`scripts/diag/track_stats.py`, vx=+0.15 구간).

                  이득          지연        오프셋
        관절    심     실기     심   실기    심     실기
        L.knee   0.69  0.76    40ms  64ms  -2.48  -3.06
        R.knee   0.71  0.83    20ms  64ms  +1.19  +1.15
        L.hip_roll 0.70 0.86   20ms  43ms  +1.21  +1.15
        L.hip_yaw  0.82 1.02   20ms  43ms  -0.09  +0.11

    두 가지가 나왔다.

    **강성은 이제 맞다.** 실기가 오히려 심보다 잘 따라간다 (무릎 이득 심
    0.69~0.71, 실기 0.76~0.83). 중력에 눌려 처지는 양(오프셋)까지 재현된다.
    그러니 P 게인을 더 올리면 안 된다 — 실기가 심보다 뻣뻣해져 반대 방향
    갭이 생긴다. 오늘 맞춘 전류 700 틱 / P 1402 조합이 제대로 들어맞았다.

    **남은 갭은 지연 하나다.** 실기가 심보다 정확히 **한 스텝(약 21 ms)**
    느리다. 전 축이 일관되게 그렇다. 그런데 지금 설정은 그걸 못 담는다:

        v46 (지금)   0~3 스텝  (0~60 ms, 평균 30 ms)
        v43          1~2 스텝  (20~40 ms, 평균 30 ms)
        실측         43 ms · 무릎 64 ms

    v46 은 범위는 넓지만 절반이 실측보다 빠르고, v43 은 평균이 같을 뿐
    상한이 실측 무릎에 못 미친다. **2~3 스텝(40~60 ms)** 이 실측을 감싼다.

    지연을 늘리면 정책이 더 보수적으로 걷게 되어 심 점수는 내려갈 것이다.
    그래도 하는 이유는, 지연은 **실기에서 P 로 못 고치는 유일한 항목**이라서
    심 쪽을 실기에 맞추는 것 말고는 방법이 없기 때문이다 (버스·제어주기가
    만드는 값이다. Return Delay Time 을 0 으로 낮추면 5 ms 쯤 줄지만 그뿐이다).

    비교 기준 (v46 @999): 보상 293.4 · 에피소드 556.6
    **판정: 실기 추종 RMS 가 줄고 접지 전환이 유지되는가.**
    """

    action_min_delay = 2      # env step (40 ms)
    action_max_delay = 3      # env step (60 ms) — 실측 43~64 ms 를 감싼다


@configclass
class JoystickEnvCfg_V48(JoystickEnvCfg_V47):
    """imitation_v48 — **action_rate 를 v40 이전으로 되돌린다** (-0.25 -> -0.5). 하나만.

    2026-08-13, 실기가 걷기 시작한 뒤 사용자가 "좌우로 덜컹거리고 발이 팍팍
    찍힌다" 고 했다. 심과 실기를 같은 자로 재서 원인을 갈랐다.

    **배포 문제가 아니다.** 몸통 흔들림이 심과 소수점까지 같다:

        항목                    심      실기    배율
        roll 진폭 (p-p)       20.98°   20.39°  0.97x
        각속도 roll RMS        1.22     1.22   0.99x
        각속도 pitch RMS       0.79     0.76   0.96x

    그러니 P 게인이나 필터로 고칠 것이 아니다 — 심에서도 똑같이 흔들린다.

    **떨림도 아니다.** 목표각의 방향전환 비율이 실기 21.4 %, 심 29.7 % 로
    둘 다 백색잡음(50 %) 아래다. 명령은 매끈하다.

    **빠른 움직임이다.** 정책이 속도 클램프를 상시 물고 있다:

        목표각 변화 속도 p95   심 4.82 rad/s  (= max_motor_velocity 그 값)
                              실기 3.86~5.34
        관절 실제 속도 p95     심 3.5~3.9 · 실기 3.7~4.4   (한계 4.82)
        실기 착지 피크 가속도  중앙값 2.39 g · 최대 4.81 g

    무부하 속도의 76~86 % 로 다리를 휘두르면 토크-속도 직선상 남는 토크가
    거의 없다. 착지 직전에 감속할 힘이 없으니 발이 찍힌다.

    그런데 **그게 v40 이 깎은 항목이다.** v40 독스트링:

        1차차분(action_rate)은 "빠른 움직임"을 벌하고 2차차분(jerk)은
        "진동만" 벌한다 (매끄러운 램프는 2차차분이 정확히 0). 진동 억제를
        jerk 가 맡은 이상 action_rate 가 남아서 하는 일은 추종에 필요한
        빠른 관절 이동을 이중과금하는 것뿐이다.

    논리는 맞았고 심 점수도 올랐다. 다만 그때는 실기가 걷질 못해서 "빠른
    관절 이동" 의 실기 대가를 볼 수 없었다. 이제 보인다 — 그 이중과금이
    실은 필요했다. jerk(-0.25)는 그대로 둔다. 떨림은 문제가 아니었으므로.

    대가는 추종이다. v40 이 -0.5 -> -0.25 로 얻은 것이 추종이었으니
    되돌리면 그만큼 내준다 (v39 계열에서 0.0197 수준). 지금은 정확도보다
    실기가 부드럽게 걷는 쪽이 낫다.

    비교 기준 (v47 @1999): 보상 304.3 · 길이 558.3
    **판정: 심에서 목표각 변화 속도 p95 가 4.82(클램프)에서 내려오는가.**
    거기서 안 내려오면 리워드로는 부족한 것이고, 그때는
    `max_motor_velocity` 를 직접 낮추는 쪽으로 간다.
    """

    action_rate_scale = -0.5


@configclass
class JoystickEnvCfg_V49(JoystickEnvCfg_V48):
    """imitation_v49 — v48 + **속도 한계를 직접 낮춘다** (4.82 -> 3.50 rad/s).

    v48 은 리워드로 "천천히 움직이라" 고 말한다. 이건 **말 대신 벽을 세운다.**
    둘을 한 설정에 합치지 않고 나눈 이유는, 합치면 어느 쪽이 들었는지 못
    가리기 때문이다.

    왜 이 값인가. `max_motor_velocity` 는 목표각이 한 스텝에 움직일 수 있는
    양의 상한이다. 지금 정책은 이 값을 **상시 물고 있다** (목표각 변화 속도
    p95 가 심에서 정확히 4.82 = 클램프 값). 즉 상한이 곧 정책의 동작점이다.

        4.82 rad/s = 무부하 속도 그 자체. 여기서는 토크가 0 이다.
        3.50 rad/s = 무부하의 73 %. 토크-속도 직선상 스톨 토크의 27 % 가
                     남는다 — 착지 직전에 감속할 힘이 그만큼 생긴다.

    실측 근거: 실기 무릎·발목이 3.5~4.2 rad/s 로 휘둘리고 (한계의 76~86 %)
    착지 피크가 중력의 2.39 배(중앙값)·4.81 배(최대) 였다.

    **주의: 이 값은 배포 계약이다.** `rl_walk.py` 가 policy.meta.json 에서
    `max_motor_velocity` 를 읽어 그대로 쓰므로 (MAX_MOTOR_VEL) 실기에도
    자동으로 따라간다. 심에서만 낮추고 실기에서 안 낮추는 사고는 안 난다.

    대가는 최대 보행 속도다. 다리를 느리게 흔들면 같은 보폭을 같은 시간에
    못 낸다. 명령 범위(vx ±0.15)를 못 따라갈 수 있다.

    비교 기준 (v48): 목표각 변화 속도 p95 · 접지 전환 · 추종
    **판정: v48 만으로 p95 가 충분히 내려오면 v49 는 버린다.** 벽보다 말이
    낫다 — 벽은 정책이 그 앞에서 여전히 bang-bang 으로 붙어 있게 만든다.
    """

    max_motor_velocity = 3.50


@configclass
class JoystickEnvCfg_V50(JoystickEnvCfg_V47):
    """imitation_v50 — **지면 마찰을 1.0 -> 0.5 로 낮춘다**. 하나만.

    v48/v49(부드러움)와 섞이지 않게 v47 에서 따로 갈라 나온다.

    ## 왜

    2026-08-13 실기 로그에서 **발이 미끄러진다** 는 증거가 나왔다. 접지 상태로
    나눠 본 몸통 yaw 각속도:

        상태        심 평균    실기 평균
        양발 접지   -0.069     +0.672 rad/s
        왼발만      -0.129     +2.253
        오른발만    +0.202     -1.000

    **양발이 다 땅에 있는데 몸통이 0.67 rad/s(38°/s)로 돈다.** 두 발이 지면에
    고정된 폐쇄 체인에서는 마찰이 충분하면 물리적으로 돌 수 없다. 심에서는
    실제로 못 돈다(-0.069). 단발 지지에서는 실기가 심의 10 배로 돌고, 좌우가
    비대칭이라(+2.25 / -1.00) 순 드리프트가 왼쪽으로 남는다 — 명령 없이 8 초에
    191° 돌던 것의 절반은 이것이다 (나머지 절반은 path_error 고정).

    그리고 걸음이 요구하는 마찰이 심이 학습한 범위를 넘는다. IMU 비력에서
    중력을 빼 `|수평가속도| / |수직항력|` 로 계산한 값:

        구간      중앙값   p90    p95
        전진 중    0.47   1.07   1.46
        회전 중    0.30   0.97   1.27
        정지       0.13   0.62   0.85

    (p95 이상은 착지 충격 스파이크가 섞여 있어 그대로 믿을 값이 아니다.
     중앙값과 p90 만 본다. 그리고 이건 병진 마찰이고 위의 yaw 미끄러짐은
     비틀림 마찰이라 이 계산에 안 잡힌다 — 실제 요구는 이보다 크다.)

    걷는 시간의 10 % 가 μ 1.07 을 요구하는데 심의 상한이 1.00 이다.

    ## 무엇을 바꾸나

    지면의 `static/dynamic_friction` 을 1.0 -> 0.5 로. 결합이 multiply 이고
    로봇 링크는 0.5~1.0 으로 무작위되므로:

        지금 (v47)   실효 μ = (0.5~1.0) × 1.0 = 0.50 ~ 1.00
        v50          실효 μ = (0.5~1.0) × 0.5 = 0.25 ~ 0.50

    실기 바닥은 **아직 안 쟀다.** 딱딱한 발바닥에 랩 타일·장판이면 μ 0.3~0.4
    가 흔한데 그건 지금 심 하한(0.50)에 못 미친다. 즉 심은 실기보다 잘 붙는
    바닥만 보고 배웠을 수 있다. v50 은 그 반대쪽 — **실기보다 미끄러운 쪽까지**
    보게 한다.

    ## 대가

    0.25 는 꽤 미끄럽다. 정책이 아예 못 걷게 될 수도 있다. 그때는 지면을 0.67
    로 두어 실효 0.33~0.67 (실기 추정 대역의 한가운데)로 좁히는 쪽이 다음 수다.

    미끄러운 바닥에서 걷는 법은 보통 **발을 덜 끌고 수직으로 내려놓는** 것이라,
    부수 효과로 v48/v49 가 노리는 부드러움이 같이 올 수도 있다.

    비교 기준 (v47 @1999): 보상 304.3 · 길이 558.3
    **판정: 실기에서 양발 접지 중 yaw 각속도가 0.67 rad/s 에서 내려오는가.**
    거기가 안 내려오면 미끄러짐이 아니라 path_error 쪽이 주범이다.
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.5,
            dynamic_friction=0.5,
        ),
        debug_vis=False,
    )


@configclass
class JoystickEnvCfg_V51(JoystickEnvCfg_V47):
    """imitation_v51 — **발 들림을 절반으로** (ref_g135fh20). 레퍼런스만 바뀐다.

    v48/v49(리워드·속도한계), v50(마찰)과 섞이지 않게 v47 에서 따로 갈라 나온다.

    ## 왜 — 발을 다리 길이의 23 % 나 들고 있었다

    2026-08-13, 사용자가 "보폭이 너무 크거나 발을 너무 위로 들어서 그런 걸
    수도 있다" 고 했다. URDF FK 로 몸통 기준 발 궤적을 재 보니 맞았다
    (`scripts/diag/foot_traj.py`).

                    발 들림          보폭(x)
        심 목표     31.3 / 39.9 mm   79 / 84 mm
        심 실제     20.5 / 29.4 mm   63 / 73 mm
        실기 실제   32.7 / 36.2 mm   76 / 71 mm

    다리 마디가 78.65 mm × 2 = 157 mm 다. 발 들림 36 mm 는 **다리 길이의 23 %**
    이고, 사람 보행의 발 여유는 1~2 % 다. 걷기가 아니라 행진이다.

    **보폭은 줄일 수 없다.** 76 mm × (1/0.54 s) = 0.14 m/s 로 명령 속도 0.15 와
    맞는다 — 속도가 보폭을 정한다. **발 들림만 자유 변수다.**

    이것이 세 증상의 공통 원인이다. 스윙 0.18 초에 36 mm 를 올렸다 내리면
    평균 0.4 m/s, 피크 약 0.63 m/s 로 착지한다:

      * 착지 충격이 중력의 2.39 배(중앙값)·4.81 배(최대) — "발이 팍팍 찍힌다"
      * 그 속도를 내려면 모터 속도 한계(4.82 rad/s)를 물어야 한다 — 여유가 0
      * 발을 높이 들려면 무게를 한쪽 다리로 완전히 옮겨야 한다 — roll ±10°

    그리고 `imitation_scale` 이 4.0 으로 지배적이라 **이걸 벌점으로 이기려는
    것은 헛수고다.** 레퍼런스를 고치는 쪽이 맞다.

    원본 리포 비교: v1(open_duck_mini) 은 `walk_foot_height` 0.03 /
    `feet_spacing` 0.14, v2 는 0.04 / 0.18 이다. 우리 `gen_reference_local.sh`
    는 `walk_com_height` 만 덮고 나머지는 medium 프리셋 그대로라 0.04 가
    그대로 들어가 있었다.

    ## 무엇을 바꾸나

    `walk_foot_height` 0.04 -> 0.02 로 레퍼런스를 다시 뽑았다. 나머지는
    ref_g135sym 과 완전히 같은 레시피다 (`--height 0.2073 --yaw-sweep 0.28`).

    생성 후 실측 (레퍼런스 자체를 FK 로):

        ref_g135sym    발 들림 44.8 / 44.0 mm   보폭 56.9 / 56.7 mm
        ref_g135fh20   발 들림 23.5 / 23.0 mm   보폭 56.5 / 56.6 mm

    **발 들림만 47 % 줄고 보폭은 그대로다.** 다리 길이 대비 28 % -> 15 %.
    좌우 대칭도 유지된다 (직진 녹화 대칭화 후 무릎 진폭차 ±2.5 % 이내).

    READY 자세는 calc_home.py 로 다시 뽑았다 (최대 |action| 1.05). 무릎이
    -76.7° -> -72.7° 로 4° 더 펴진다.

    ## 대가

    발 여유가 23 mm 로 줄어 바닥이 고르지 않거나 몸통이 기울면 발끝이 걸릴 수
    있다. 다만 지금 roll 이 ±10° 라 오히려 그 흔들림이 줄면 여유가 더 생긴다.

    비교 기준 (v47 @1999): 보상 304.3 · 길이 558.3
    판정: 심에서 목표각 변화 속도 p95 가 4.82(클램프)에서 내려오고 roll
    진폭이 21도 에서 줄어드는가.

    ## 결과 — 목표 미달, 그리고 **전제가 틀렸다** (2026-08-14)

    레퍼런스 발 들림을 44.8 -> 23.2 mm 로 반토막냈는데 실제 걸음은 거의 안
    바뀌었다. roll 은 20.7 -> 18.6 도로 이 계열 중 가장 좋았지만 10 % 다.

    그런데 이 실험의 근거였던 "발을 다리 길이의 23 % 나 든다" 가 **잘못 잰
    값이었다.** 몸통 기준 좌표에서 발 하나의 z 범위를 썼는데 거기엔 몸통이
    위아래로 출렁이는 양(표준편차 4.6~5.3 mm)과 몸통이 기울며 발이 상대적
    으로 움직이는 양이 섞인다. **스윙 발이 접지 발보다** 얼마나 뜨는지로
    다시 재면:

        v48  중앙값 5.2 mm (다리의 3.3 %) · 스윙당 최고 14.2 mm
        v51  중앙값 4.5 mm · 18.3 mm
        v52  중앙값 4.1 mm · 19.4 mm

    사람 보행(1~2 %)에 가깝다. **발 높이는 애초에 문제가 아니었고** 이 실험은
    없는 문제를 고치려던 것이다. 레퍼런스를 반토막내도 실제 걸음이 안 바뀐
    것이 당연하다 — 바꿀 여지가 없었다.

    진짜 문제는 좌우다. 스윙 한 번에 세로로 14 mm 뜨는 동안 옆으로 28 mm
    간다 (수직성 0.50). v53 이 그쪽을 직접 벌한다.
    """

    reference_motion_pkl = "source/open_duck_mini_isaaclab/reference_motion/data/ref_g135fh20.pkl"

    robot = OPEN_DUCK_MINI_V2_DC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=OPEN_DUCK_MINI_V2_DC_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_G135FH20),
            joint_pos=dict(READY_JOINT_POS_G135FH20_ZNECK),
        ),
    )
    ready_base_height = READY_BASE_HEIGHT_G135FH20


@configclass
class JoystickEnvCfg_V52(JoystickEnvCfg_V51):
    """imitation_v52 — **보간격을 좁힌다** (feet_spacing 0.18 -> 0.14). 레퍼런스만.

    ## 왜

    v48~v51 을 같은 마찰(μ 1.0)로 놓고 걸음 품질을 재니 **덜컹거림이 거의 안
    줄었다** (`scripts/diag/gait_quality.py`):

        런                     roll 진폭   발 들림   목표각 p95
        v48 action_rate -0.5    20.7도    22.5 mm   4.82 (클램프)
        v49 속도한계 3.50       21.9도    22.0 mm   3.50 (클램프)
        v50 마찰 0.5            21.8도    28.0 mm   4.82 (클램프)
        v51 발 들림 절반        18.6도    21.4 mm   4.82 (클램프)

    세 가지가 나왔다.

    **하나. v50 은 착시였다.** 자기 환경(μ 0.5)에서는 roll 이 11.9도 로 압도적
    이었지만 **그 정책을 원래 마찰에 넣으니 21.8도 로 돌아왔다.** 미끄러운
    바닥에서는 좌우로 흔들 지면반력이 없어 못 흔든 것이지 잘 걷게 된 것이
    아니다. 보상 351.0 도 같은 착시다 — 안 흔들리니 안 넘어져 에피소드가
    길어진 것이고, 원래 바닥에서는 접지 전환이 25.2/s 로 오히려 채터링한다.

    **둘. 아무도 속도 클램프를 못 벗어났다.** 리워드로도(v48), 레퍼런스 발
    들림을 반으로 줄여도(v51) 목표각 변화 속도 p95 가 정확히 클램프 값에
    붙어 있다.

    **셋. v51 은 목표 미달이었다.** 레퍼런스 발 들림을 44.8 -> 23.2 mm 로
    반토막냈는데 **실제 걸음의 발 들림은 22.5 -> 21.4 mm** 로 거의 안 변했다.
    정책이 그 부분에서 레퍼런스를 따라가지 않는다. 다만 roll 은 20.7 ->
    18.6 도로 이 중 가장 좋았다.

    그러니 덜컹거림의 원인은 발을 드는 높이가 아니라 **좌우로 무게를 옮기는
    거리**다. 아직 한 번도 안 건드린 값이 거기 있다 — 원본 v1 로봇은
    `feet_spacing` 0.14 인데 v2 는 0.18 이고, 우리 생성 스크립트는
    `walk_com_height` 만 덮으므로 0.18 이 그대로 들어가 있었다.

    ## 검증 (레퍼런스 자체를 FK 로)

        레퍼런스                 발들림   보폭   발간격   좌우폭
        g135sym  fh.04 fs.18     44.4    56.8   180.5    41.5 mm
        g135fh20 fh.02 fs.18     23.2    56.5   180.5    41.7
        g135fs14 fh.02 fs.14     23.2    56.5   **140.4**  **28.1**

    발간격이 정확히 설정대로 좁아졌고, **발의 좌우 흔들림이 41.7 -> 28.1 mm
    (-33 %)** 로 같이 줄었다. 발 들림과 보폭은 그대로라 v51 대비 단일 변수다.

    READY 의 `hip_roll` 이 -0.004 -> **-0.12 rad(-6.9도)** 로 크게 바뀌는데,
    좁은 보간격으로 서려면 고관절을 안쪽으로 말아야 하기 때문이다.

    ## 대가

    지지 다각형이 좁아져 좌우 외란에 약해진다. 그리고 다리가 서로 가까워지므로
    **자기충돌 여유가 준다** — 이 프로젝트의 5 mm 기준을 다시 확인해야 한다
    (`scripts/diag/leg_trunk_clearance.py`).

    비교 기준 (v51): roll 진폭 18.6도 · 발 들림 21.4 mm · 접지 11.9/s
    판정: roll 진폭이 18.6도 에서 눈에 띄게 내려가는가.

    ## 결과 — **실패** (2026-08-14)

                        v48     v51     v52
        roll 진폭      20.7    18.6    25.1 도
        스윙 좌우      29.5    35.5    41.1 mm
        수직성         0.78    0.65    0.47
        발 들림        22.5    21.4    19.2 mm
        보상(최종)    318.8   296.0   305.5

    roll 이 가장 크고 수직성이 가장 낮다. 지지 다각형이 좁아져 좌우 여유가
    준 만큼 정책이 균형을 잡느라 발을 더 크게 휘두른 것으로 보인다. 발 들림만
    19.2 mm 로 가장 낮은데, 그건 노린 것이 아니라 곁다리다.

    **더 중요한 것은 v51 과 v52 가 같은 방향으로 어긋났다는 점이다.**
    레퍼런스에서는 발의 좌우 흔들림을 41.7 -> 28.1 mm 로 줄였는데 실제
    걸음에서는 29.5 -> 35.5 -> 41.1 mm 로 **오히려 늘었다.** 두 번 다
    레퍼런스와 반대로 갔다.

    즉 **정책이 레퍼런스의 발 궤적을 따라가지 않는다.** 모방 리워드가 비교
    하는 것은 관절각이지 발 위치가 아니고 (`reward_imitation` 은 joint_pos /
    joint_vel / contact / lin_vel / ang_vel 을 본다), 실제 발 궤적은 균형
    요구가 지배한다. 관절각을 대충 맞추면서도 발은 다른 데로 갈 수 있다.

    **그러므로 레퍼런스(walk_foot_height, feet_spacing)를 만지는 방향은
    접는다.** 남는 길은 둘이다:

      * 발의 좌우 속도나 착지 위치에 **직접** 벌점을 거는 항목 — 지금 없다.
      * roll 자체를 벌하는 항목 — `upright_standstill` 이 있지만 정지에만
        걸려 있어 보행 중에는 아무 규제가 없다.

    같은 마찰 기준 최선은 여전히 v48 (roll 20.7 · 수직성 0.78 · 보상 318.8)
    이고, roll 만 보면 v51 (18.6) 이다.
    """

    reference_motion_pkl = "source/open_duck_mini_isaaclab/reference_motion/data/ref_g135fs14.pkl"

    robot = OPEN_DUCK_MINI_V2_DC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=OPEN_DUCK_MINI_V2_DC_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_G135FS14),
            joint_pos=dict(READY_JOINT_POS_G135FS14_ZNECK),
        ),
    )
    ready_base_height = READY_BASE_HEIGHT_G135FS14


@configclass
class JoystickEnvCfg_V53(JoystickEnvCfg_V48):
    """imitation_v53 — **발의 좌우 움직임을 직접 벌한다**. 리워드 두 항.

    사용자가 낸 네 가지 목표에서 출발했다:
      1. 보행 시 발을 높이 들지 말 것
      2. 발을 최소한만 움직일 것
      3. 동작이 효율적일 것
      4. 진동이 없을 것

    재 보니 **1 과 4 는 이미 달성돼 있었다.**

        발 높이   스윙 발이 접지 발보다 뜨는 양 중앙값 5.2 mm = 다리(157 mm)의
                  3.3 %. 사람 보행이 1~2 % 다. (며칠 동안 "23 % 를 든다" 고
                  판단했는데 그건 몸통 출렁임이 섞인 잘못 잰 값이었다.)
        진동      목표각 방향전환 비율 33~36 % 로 백색잡음(50 %) 아래다.
                  action_jerk(-0.25)가 v39 에서 이미 잡았다.

    남은 것이 2 다. **세로로 14 mm 뜨는 동안 옆으로 28 mm 간다** (수직성 0.50).
    발을 옆으로 던지려면 몸통이 무게를 그만큼 옮겨야 하므로 이것이 roll 이
    ±10 도로 흔들리는 직접 원인이다.

    ## 왜 리워드인가 — 레퍼런스로는 두 번 실패했다

        v51  레퍼런스 발 들림 44.8 -> 23.2 mm   실제 걸음 좌우 29.5 -> 35.5 mm
        v52  레퍼런스 발 좌우폭 41.7 -> 28.1 mm  실제 걸음 좌우 35.5 -> 41.2 mm

    두 번 다 레퍼런스와 **반대로** 갔다. `reward_imitation` 이 비교하는 것은
    관절각이지 발 위치가 아니고 (joint_pos/vel/contact/lin_vel/ang_vel), 실제
    발 궤적은 균형 요구가 지배한다. 그러니 발 위치를 **직접** 벌해야 한다.

    ## 무엇을 거나

        foot_lateral_scale = -10     발의 몸통기준 y 속도 제곱합
        foot_lift_scale    = -3000   접지 발 기준 10 mm 를 넘는 들림
        foot_clearance     = 0.010

    계수는 실측 궤적에서 각 항의 기여가 -0.03 쯤 되도록 잡았다 —
    `action_rate` 가 -0.008~-0.03 대역이므로 같은 무게다.
    전후(x) 속도는 벌하지 않는다. 보폭은 명령 속도가 정하는 값이라 줄이면
    명령을 못 따라간다 (76 mm × 1/0.54 s = 0.14 m/s ~ 명령 0.15).

    발 들림 항을 함께 켜는 이유는 낮추려는 것이 아니라 **좌우를 누를 때 위로
    도망가지 못하게** 막으려는 것이다. 10 mm 까지는 공짜다.

    기준선은 v48 이다 — 같은 마찰에서 수직성 0.50, roll 20.7 도, 보상 318.8 로
    v49~v52 중 가장 낫고 레퍼런스도 안 건드린 원본(g135sym)이다.

    비교 기준 (v48): 수직성 0.50 · 스윙 좌우 28.4 mm · roll 20.7 도 · 보상 318.8
    판정: 스윙 좌우가 28 mm 에서 내려가고 roll 이 따라 줄어드는가.
    발 들림이 5 mm 아래로 더 내려가면 발을 끄는 것이니 접지 전환도 같이 본다.

    ## 1 차 시도 실패와 계수 재조정 (2026-08-14)

    처음 계수는 `foot_lift -300 / foot_lateral -0.25` 였고 **학습이 무너졌다** —
    보상 318.8 -> 100.4, 에피소드 561 -> 320. 원인은 계수가 아니라
    `cost_foot_lift` 의 **버그**였다.

    `body_pos_w` 는 발 링크 원점이지 발바닥이 아니다. 이 로봇은 좌우 발 링크의
    z 오프셋이 서로 반대라 **양발이 나란히 땅에 있어도 39 mm 차이난다.** 그걸
    안 빼고 벌하니 정지 명령에서도 -0.2415 (리워드 예산의 45 %) 가 걸렸고,
    정책은 줄일 방법이 없어 걸음을 망가뜨리는 쪽으로 갔다.

    **서 있는데 벌점이 0 이 아니면 그 항은 틀린 것이다.** 리워드 항을 새로
    넣으면 `scripts/diag/reward_terms.py` 의 stop 열을 먼저 볼 것.

    오프셋을 빼도록 고친 뒤 v48 정책으로 다시 재니:

        foot_lift     -0.0004  (정지 0.0000)   <- 상수 성분 사라짐
        foot_lateral  -0.0008

    이번엔 너무 약하다. 관측값으로 역산해 목표 기여에 맞췄다:

        foot_lateral  -0.25  -> -10     (-0.0008 -> 약 -0.032, action_rate 대역)
        foot_lift     -300   -> -3000   (-0.0004 -> 약 -0.004, 가벼운 가드)

    처음 추정이 37 배 빗나간 이유는 몸통 기준 발 위치를 수치미분한 값과
    실제 발 속도를 몸통 축으로 돌린 값이 다른 양이기 때문이다. 계수는
    추정하지 말고 **`reward_terms.py` 로 재고 정할 것.**
    """

    foot_clearance = 0.010
    foot_lift_scale = -3000.0
    foot_lateral_scale = -10.0


@configclass
class JoystickEnvCfg_V55(JoystickEnvCfg_V48):
    """imitation_v55 — **부드러움 패키지**. 발은 직선으로, 몸통·다리는 감쇠.

    2026-08-14 사용자가 우선순위를 못박았다: **1 추종 · 2 부드러움 · 3 진동.**
    추종을 깎으면 실패다. 그래서 새 항의 기여를 각 2.5 % 로 눌러 잡았다.

    ## 무엇이 바뀌었나 — 관측이 먼저 뒤집혔다

    v53 은 `foot_lift_scale = -3000` 으로 발 들림을 **낮추는** 쪽에 걸었다.
    그 전제가 그 뒤 무너졌다. 발에 3 mm 고무패드를 붙이자 사용자가 **발이
    바닥을 끈다**고 했고, 재 보니:

        스윙 여유   중앙값 5.8 mm · p90 10.6 mm
        roll 진폭   ±10 도 -> 반스탠스 70 mm 에서 발 모서리가 12 mm 내려감

    즉 **roll 만으로 스윙 발이 바닥에 닿는다.** 패드 전에는 조용히 미끄러지던
    것이 패드가 물면서 끌림으로 드러난 것이다. 낮출 게 아니라 **붙잡아야**
    한다. 그래서 `foot_lift_scale = 0` 으로 끄고 방향을 뒤집는다.

    ## 네 항, 그리고 각각이 사용자의 어느 말에 대응하는가

        foot_clearance -8000 (목표 20 mm)   "발이 미끄러진다" / "직선으로"
            스윙 중 발을 목표 높이로 유지. 수평 속도로 가중해 이·착지 순간은
            빠진다. 낮아도 높아도 벌하므로 20 mm 근처 우물이 된다.
        foot_slip      -18                  "발이 미끄러진다"
            딛고 있는 발의 수평 속도. 마찰이 아무리 좋아도 수평 속도를 가진
            채 착지하면 그만큼 쓸린다.
        torso_ang_vel  -0.7                 "토르소를 최대한 부드럽게"
            몸통 roll·pitch 각속도. yaw 는 뺀다 — 회전 명령이 요구하니까.
        joint_accel    -2.2e-5              "다리 움직임도"
            관절 가속도. action_rate/jerk 는 **정책 출력**의 매끄러움이라
            PD 와 부하 뒤의 실제 다리 움직임과 다르다.

    v53 의 `foot_lateral = -10` 은 유지한다. 스윙 중 좌우 흔들림이 roll 을
    만든다는 근거는 그대로고, v53 이 그 계수로 보상 305.8 / 에피소드 593.0 을
    내 v48(318.8 / 581.8) 대비 대가가 작음을 확인했다.

    ## 계수는 재서 정했다

    `scripts/diag/probe_terms.py` 로 v53 정책에서 원시값을 재고 역산했다.
    v53 1 차 시도가 무너진 것이 계수를 **추정**해서였기 때문이다.

        항목            원시 평균   정지     3 % 계수
        foot_clearance   0.0001   0.0000    -9971
        foot_lateral     0.0725   0.0007    -13.2
        foot_slip        0.0447   0.0014    -21.4
        torso_ang_vel    1.1242   0.0251    -0.85
        joint_accel     35689.5   1681.3    -2.68e-5

    **정지 열이 모두 0 근처다** — v53 을 무너뜨린 상시 세금이 없다는 뜻이다.
    위 계수를 2.5 % 로 낮춰 쓴다 (추종이 1 순위이므로).

    ## 한 번에 넷을 거는 이유

    보통은 하나씩 바꾼다. 여기서는 넷이 **하나의 가설**을 이룬다 — "발을
    수직으로 들고, 딛는 동안 붙잡고, 몸통과 다리를 감쇠하면 부드러워진다".
    각각 2.5 % 라 하나만으로는 신호가 안 나온다. 이기면 v56 부터 하나씩 빼서
    무엇이 들었는지 가린다. 지면 같은 방법으로 범인을 찾는다.

    비교 기준 (v48): roll 진폭 20.7 도 · 수직성 0.50 · 스윙 좌우 28.4 mm ·
                     보상 318.8 · 에피소드 581.8
    **판정: 추종(tracking_lin_vel/ang_vel)이 v48 수준을 지키면서 roll 진폭과
    스윙 좌우가 내려가는가.** 추종이 깎이면 계수를 반으로 줄여 다시 건다.
    """

    # v53 의 발 좌우 억제는 유지, 발 들림 억제는 방향이 반대라 끈다.
    foot_lateral_scale = -10.0
    foot_lift_scale = 0.0

    # 스윙 발을 20 mm 에 붙잡는다. roll 이 발을 12 mm 내리므로 그보다 커야 한다.
    foot_clearance_target = 0.020
    foot_clearance_scale = -8000.0

    # 딛고 있는 발은 정지해 있어야 한다.
    foot_slip_scale = -18.0

    # 몸통 roll/pitch 각속도 감쇠.
    torso_ang_vel_scale = -0.7

    # 관절 가속도 감쇠.
    joint_accel_scale = -2.2e-5


@configclass
class JoystickEnvCfg_V56(JoystickEnvCfg_V55):
    """imitation_v56 — v55 + **부드러운 착지**. "퉁퉁거림" 의 정체를 잡았다.

    ## 앞의 판단을 철회한다

    이 클래스는 원래 `imitation_w_stance_violation` 을 1.0 -> 3.0 으로 올리는
    실험이었다. 전제가 틀려서 폐기한다.

    접지 전환만 세면 v53 이 주기 195 ms, v55 가 210 ms 로 레퍼런스 540 ms 의
    4 분의 1 이라, **"발을 7 Hz 로 떨고 있다"** 고 판단했다. 접지 신호를
    디바운스해서 다시 재니 결론이 뒤집혔다:

        디바운스     없음   20 ms   40 ms   60 ms
        v48           480     540     540     540
        v53           195     385     490     540
        v55@2k        210     480     500     540
        v55@3k        340     430     490     540

    **전부 540 ms 로 수렴한다.** 걸음 주기는 처음부터 옳았다. 달랐던 것은 한
    스텝 안에서 발이 **착지하며 튀는 정도**다. 걸음이 가짜였던 게 아니라
    착지가 거칠었던 것이고, 그게 사용자가 계속 말한 "퉁퉁거린다" 다.

    따라서 `stance_violation`(걸음 주기를 교정하는 항)은 애초에 고칠 게 없는
    것을 고치려던 것이었다. 되돌리고 착지 속도를 직접 벌한다.

    ## 무엇을 바꾸나

        foot_impact_scale   착지 충격 (지면 10 mm 안에서의 하강 속도 제곱)

    접지 순간에만 벌하지 않고 **지면 근처** 에서 하강 속도를 벌한다. 접지
    순간은 이미 늦어서 정책이 고칠 방법이 없다 — 미리 감속할 기울기를 줘야
    한다. 상태(이전 스텝 접지)를 안 들어도 되는 것은 덤이다.

    ## v55 가 이미 이룬 것 — 여기를 깎으면 안 된다

        전진 추종      0.93 -> 1.05 (3k)      관절 추종 이득  0.77 -> 0.86
        roll 진폭      20.7 -> 6.9 도          roll RMS       5.86 -> 1.01 도
        수직성         0.78 -> 1.04            스윙 좌우      29.5 -> 20.8 mm

    roll 은 v48 대비 **83 % 줄었다.** 남은 것이 착지 바운스뿐이다.

    3000 iter 로 돈다. v55 에서 리워드는 1500 부터 평평했는데(+0.31 %) 행동은
    2000 -> 3000 사이에 계속 좋아졌다 — 주기 215 -> 340 ms, 추종 0.93 -> 1.05.
    **리워드 수렴을 행동 수렴으로 읽으면 안 된다**는 것을 이 런에서 배웠다.

    비교 기준: v55@3k
    **판정: 디바운스 없는 걸음 주기가 340 -> 480 ms 쪽으로 올라오고 100 ms
    미만 접지가 38 % 아래로 내려가는가.** 그러면서 roll 과 추종이 유지되는가.
    """

    foot_impact_scale = -90.64


@configclass
class JoystickEnvCfg_V57(JoystickEnvCfg_V47):
    """imitation_v57 — **순수 단일축 명령을 학습에 넣는다.** 1 순위를 직접 겨냥.

    ## 왜 v47 에서 분기하나

    2026-08-14 에 사용자가 순위를 못박았다: 1 6 방향 추종 · 2 보행 안정성 ·
    3 보행 효율. 그 자로 25 개 버전을 시간순으로 다시 재니 최고점이 v40/v42
    였고 v48 에서 무너졌다:

        버전   전진     후진     좌      우      회전    최악     평균
        v40   0.0078  0.0051  0.0286  0.0302  0.0114  0.0302  0.0166
        v42   0.0087  0.0110  0.0264  0.0371  0.0040  0.0371  0.0174
        v48   0.0231  0.0191  0.0298  0.0219  0.0644  0.0644  0.0317
        v55   0.0234  0.0249  0.0878  0.0735  0.0782  0.0878  0.0576

    v48 은 그 손해를 **알고** 냈다. 독스트링에 "대가는 추종이다 (...) 지금은
    정확도보다 실기가 부드럽게 걷는 쪽이 낫다" 고 적혀 있다. 근거는 정책이
    속도 클램프를 물어 착지 직전에 감속할 토크가 없다는 것이었다.

    **그 근거가 그 뒤 하드웨어로 해결됐다.** 전원 어댑터를 바꿔 전압 붕괴를
    없앴고(넘어짐의 진짜 원인이었다), 발에 3 mm 고무패드를 붙여 수직 가속도
    RMS 6.46 -> 3.92, 착지 피크 2.39 -> 2.12 g 가 됐다. v48 이 추종을 팔아서
    산 것을 하드웨어가 공짜로 줬으므로 그 거래를 유지할 이유가 없다.

    `action_rate_scale` 은 v40 에서 -0.25 로 내린 뒤 v41→v42→v44→v46→v47 까지
    상속된다. **v48 이 -0.5 로 되돌린 유일한 클래스다.** 그러니 v47 이 곧
    "v48 에서 그 거래만 취소한 것" 이면서 실기에 필요한 것을 다 갖고 있다:
    몸통 +10 mm(v46) · 액션지연 2~3 스텝(v47, 실측 지연과 일치) ·
    질량 무작위화 확대(v44).

    ## 무엇을 바꾸나 — 리워드가 아니라 명령 분포

        pure_axis_prob = 0.35

    명령은 축별 **독립 균등**으로 뽑혀 왔다. 그러면 vx·vy·wz 가 항상 섞여
    나오고 "옆으로만", "제자리 회전만" 같은 명령은 확률적으로 거의 안 나온다.
    그런데 시험은 순수 단일축으로 본다 — `gait_compare` 의 6 방향이 그렇고,
    사람이 조이스틱을 한 방향으로 밀 때도 그렇다.

    측정한 25 개 버전 **전부** 좌/우가 전진보다 2~4 배 나빴다. 리워드 계수를
    아무리 만져도 안 고쳐진 이유가 여기 있다 — **학습에서 거의 안 본 분포로
    시험을 봤다.** 이 프로젝트의 교훈대로 계수보다 구조를 먼저 고친다.

    0.35 를 고른 이유: 10 % 정지 + 약 31.5 % 순수축 + 약 58.5 % 혼합.
    6 방향이 충분히 나오면서 혼합 명령도 다수로 남는다.

    ## 축은 균등하게 안 뽑는다

    사용자가 6 방향 안에서도 순위를 줬다 (2026-08-14): **앞뒤 > 회전 > 완전
    옆걸음.** 균등하게 뽑으면 가장 덜 중요한 옆걸음에 순수축 예산의 3 분의 1 이
    간다. 그래서 `pure_axis_weights = (0.50, 0.15, 0.35)` — vx 절반, wz 그 다음,
    vy 가장 적게.

    그 순위로 지금까지의 25 개 버전을 다시 매기면 (앞뒤 3 · 회전 2 · 옆 1):

        1  v40   앞뒤 0.0065  회전 0.0114  옆 0.0294   0.0121
        2  v42   앞뒤 0.0098  회전 0.0040  옆 0.0318   0.0131
        3  v25   앞뒤 0.0114  회전 0.0006  옆 0.0360   0.0141
        4  v37   앞뒤 0.0039  회전 0.0340  옆 0.0269   0.0145

    v48 과 v55 는 12 위 안에도 못 든다. 재미있게도 v48 은 **옆걸음만** 제일
    잘한다 (0.0219~0.0298) — 우선순위와 정확히 반대로 최적화돼 있었다.

    비교 기준 (v47 @1999, 6 방향)
    **판정: 최악 방향이 v40 의 0.0302 아래로 내려가는가.** 평균보다 최악을
    본다 — 여섯 중 하나만 못 가도 조이스틱으로는 못 쓴다.

    안정성(2 순위)·효율(3 순위) 항은 여기 넣지 않는다. 1 순위가 회복되는지
    먼저 확인하고 v58 에서 얹는다 — 같이 걸면 무엇이 들었는지 못 가린다.
    """

    pure_axis_prob = 0.35


@configclass
class JoystickEnvCfg_V58(JoystickEnvCfg_V47):
    """imitation_v58 — **액션 지연을 다시 넓힌다** (2~3 -> 0~3). 하나만 바뀐다.

    ## v57 이 가설을 기각했다

    회전이 v42 0.0040 -> v47 0.0672 로 17 배 나빠진 원인을 "순수 회전 명령이
    학습 분포에 거의 없다" 로 봤다. v57 에서 순수축 명령을 35 % 넣고 축을
    앞뒤 3 : 회전 2 : 옆 1 로 가중 추첨했다. 결과:

        회전   v47 0.0672 -> v57 0.0666    거의 그대로
        전진   v47 0.0150 -> v57 0.0338    나빠짐
        정지   v47 0.0018 -> v57 0.0159    나빠짐
        점수   v47 0.0241 -> v57 0.0310    나빠짐

    **명령 분포는 원인이 아니었다.** 회전은 학습량이 모자라서 못 하는 게
    아니라 물리적으로 못 하고 있다.

    ## 남은 용의자와 이 실험

    v42 -> v47 사이 변경은 셋이다:

        v44   질량 무작위화 범위 확대              (v44@999 회전 0.0715)
        v46   몸통 +10 mm, ref g125sym -> g135sym  (v46@999 회전 0.0317)
        v47   액션 지연 0~3 -> 2~3 스텝

    지연을 먼저 본다. **v42 의 0~3 은 넓은 무작위화이고 v47 의 2~3 은 좁은
    데다 전부 느린 쪽이다.** 도메인 무작위화에서 범위를 좁히면 강건해지는 게
    아니라 "항상 느리다" 를 배운다. 회전은 요 피드백 지연에 가장 민감하니 그
    차이가 여기서 가장 크게 나타날 수 있다.

    v47 이 좁힌 근거는 실측이었다 — 실기 지연 43 ms, 무릎 64 ms 이고 0~3 은
    절반이 실측보다 빠르다. 그 지적은 옳지만 **상한 3 스텝(60 ms)이 실측을
    이미 덮으므로 0~3 도 최악을 경험한다.** 차이는 빠른 쪽도 함께 보느냐다.

    2026-08-14 에 버스 타이밍을 직접 재서 지연을 줄일 수 없음을 확인했다
    (`bus_timing.py`: 매 루프 통신 약 13 ms / 20 ms 예산, Return Delay 를
    낮추면 반이중 버스가 깨진다). 그러니 지연은 사실로 두고 **그 지연에
    강건해지는** 쪽을 시험한다.

    비교 기준 (v47 @1999): 점수 0.0241 · 앞뒤 0.0115 · 회전 0.0672 · 옆 0.0190
    **판정: 회전이 0.0672 에서 유의하게 내려가는가.** 안 내려가면 지연은
    범인이 아니고, 다음은 v44 의 질량 무작위화다.
    """

    action_min_delay = 0      # env step. v42 와 같은 넓은 범위로 되돌린다
    action_max_delay = 3      # 상한 유지 (60 ms — 실측 43~64 ms 를 덮는다)


@configclass
class JoystickEnvCfg_V59(JoystickEnvCfg_V58):
    """imitation_v59 — v58 + **질량 무작위화를 원래 범위로**. 하나만 바뀐다.

    ## 지금까지 좁혀진 논리

        구성                                  앞뒤     회전
        v42  지연 0~3 · 질량 원래 · 몸통 원래   0.0098  0.0040   둘 다 좋음
        v47  지연 2~3 · 질량 확대 · 몸통 +10mm  0.0115  0.0672   회전만 나쁨
        v58  지연 0~3 · 질량 확대 · 몸통 +10mm  0.0377  0.0342   앞뒤만 나쁨

    v58 이 회전을 절반으로 되돌린 것(0.0672 -> 0.0342)으로 **지연 폭이 회전에
    영향을 준다**는 것은 확인됐다. 그런데 같은 변경이 후진을 6.7 배 망쳤다
    (0.0079 -> 0.0532).

    v58 과 v42 의 차이는 **질량 무작위화(v44)** 와 **몸통 +10 mm / 레퍼런스
    (v46)** 둘뿐이다. 그러니 앞뒤를 망친 것은 그 둘 중 하나이고, v47 의 좁은
    지연이 그것을 우연히 가려 주고 있었다.

    ## 왜 질량부터 보나

    v44 자신이 판정 기준을 이렇게 적어 뒀다: "넓은 질량 범위에서도 보행이
    유지되는가. **무너지면 질량 민감도가 크다는 뜻이고, 그때는 무게를 먼저
    재서 좁혀야 한다.**" 지금 그 답이 나왔다 — 유지되지 않았다.

        scale_all_mass  0.85~1.25  ->  0.90~1.10   (기본 EventCfg 로 복귀)
        실효 질량       2.24~3.30  ->  2.38~2.90 kg

    47 % 폭을 견디게 하려면 정책이 보수적으로 걸어야 하고, 그 대가가 추종이다.
    실기 무게를 재서 실제 값 근처로 좁히는 것이 정공법이다 (아직 안 쟀다).

    비교 기준 (v58 @2999): 점수 0.0349 · 앞뒤 0.0377 · 회전 0.0342 · 옆 0.0272
    **판정: 앞뒤가 v47 수준(0.0115)으로 돌아오면서 회전이 0.0342 를
    유지하는가.** 둘 다 되면 v42 를 넘어선다. 앞뒤만 돌아오고 회전이 다시
    나빠지면 지연과 질량이 상호작용하는 것이고, 앞뒤가 안 돌아오면 범인은
    v46 의 몸통 높이/레퍼런스다.

    ⚠️ 측정 잡음: v48 을 같은 날 두 번 쟀더니 단일 방향이 ±25 % 흔들렸다
    (전진 0.0231 vs 0.0287). 그보다 작은 차이는 읽지 말 것.
    """

    events: EventCfg = EventCfg()


@configclass
class JoystickEnvCfg_V60(JoystickEnvCfg_V59):
    """imitation_v60 — v59 + **몸통 각속도 감쇠**. 2 순위(안정성)를 정공법으로.

    2026-08-15, 사용자가 v59 녹화를 보고 "몸통이 너무 흔들린다" 고 했다.
    측정이 같은 말을 한다:

        roll RMS 평균     v47 5.80  ·  v57 5.56  ·  v59 6.66 도
        roll 진폭 최악    v47 85.0  ·  v57 75.3  ·  v59 102.7 도
        방향별 roll RMS   우 8.69 · 후진 7.22 · 좌 6.83 · 전진 5.44 · 정지 1.28

    v59 는 1 순위(추종)를 크게 얻으면서 2 순위를 내줬다. 회전 0.0672 -> 0.0106
    을 지키면서 흔들림만 되찾는 것이 이 실험이다.

    ## 왜 이 항인가

    `cost_torso_ang_vel` 은 몸통 roll·pitch 각속도를 직접 벌한다. yaw 는
    빼므로 회전 명령과 싸우지 않고, **발 인덱스 버그와도 무관하다**
    (`root_ang_vel_b` 를 읽지 발 링크를 읽지 않는다).

    v55 가 roll RMS 2.64 도로 좋았던 것은 `torso_ang_vel -0.7` 과 **머리 좌우
    속도 억제**(발 인덱스 버그로 `foot_lateral` 이 실제로 하던 일) 둘의
    합작이었다. 머리 억제가 사라진 지금은 정공법 항 하나가 그만큼 더 져야
    한다.

    ## 계수

    `probe_terms.py` 로 v59 정책에서 잰 원시값:

        torso_ang_vel   평균 1.7930  (정지 0.0212)   4 % 계수 -0.7184

    v53 때 같은 항의 원시값이 1.1242 였으므로 **v59 가 60 % 더 흔들린다.**
    4 % 권장값보다 세게 -1.2 (약 6.7 %) 를 건다 — 위 두 가지 이유 때문이고,
    그래도 `action_rate`(-0.5)·`action_jerk`(-0.25) 와 같은 대역이다.

    비교 기준 (v59 @2999): 점수 0.0165 · 앞뒤 0.0153 · 회전 0.0106 ·
                           roll RMS 6.66 · roll 진폭 최악 102.7
    **판정: roll RMS 가 v47 수준(5.80) 아래로 내려가면서 점수가 0.0165 를
    유지하는가.** 점수가 무너지면 계수를 반으로 줄인다. roll 이 안 내려가면
    항 하나로는 부족한 것이고, 다음은 `foot_lateral` 을 **명령 상대**
    (`(v_y - cmd_vy)^2`) 로 고쳐 함께 거는 것이다 — 지금 형태는 옆걸음
    명령을 그대로 막아서 쓸 수 없다.
    """

    torso_ang_vel_scale = -1.2


@configclass
class JoystickEnvCfg_V61(JoystickEnvCfg_V59):
    """imitation_v61 — 흔들림의 **원인**을 친다. 명령 상대 발 좌우 억제.

    ## v60 이 보여 준 트레이드

        구성                  점수     roll RMS
        v59  torso 0        0.0165     6.66
        v60  torso -1.2     0.0304     3.73

    두 점으로 기울기를 재면 계수 c 에 대해

        roll  = 6.66 - 2.44 x (c/1.2)   ->  5.50 이 되려면 c ~ 0.57
        점수  = 0.0165 + 0.0139 x (c/1.2) ->  0.0180 을 넘는 c ~ 0.13

    **`torso_ang_vel` 하나로는 v59 기반에서 양쪽을 만족할 수 없다.** 각속도는
    흔들림의 *증상*이고, 그것을 직접 감쇠하면 정책이 명령을 따라가는 데 쓰던
    자유도까지 같이 묶인다.

    ## 원인을 친다

    흔들림의 원인은 **좌우 무게 이동**이다. 발을 옆으로 던지면 몸통이 그만큼
    무게를 옮겨야 하고 그것이 roll 이 된다. 실측으로도 스윙 좌우가 큰 정책이
    roll 이 컸다.

    v55 가 roll RMS 2.64 도를 낸 적이 있는데, 그것은 발 인덱스 버그로
    `foot_lateral` 이 실제로는 **머리의 좌우 속도**를 벌하고 있었기 때문이다.
    효과 자체는 진짜였다 — 다만 그 대가로 옆걸음이 명령의 56~64 % 로
    무너졌다. 명령과 무관하게 좌우 움직임을 벌했으니 당연하다.

    이번에는 발을 제대로 보고, **명령한 좌우 속도는 면제한다**:

        cost = sum((v_foot_y - cmd_vy)^2)

    `cmd_vy = 0` 인 전진·후진·회전·정지에서는 예전 식과 완전히 같아서 안정성
    효과가 그대로 남고, 옆걸음 명령에서만 풀린다.

    ## 계수

    `probe_terms.py` 로 v59 정책에서 잰 값 (명령 상대형 포함):

        foot_lateral      평균 0.1578   (좌 0.3008 · 우 0.2705)
        foot_lat_rel      평균 0.1396   (좌 0.2456 · 우 0.2163)   <- 명령분 면제 확인
        5 % 계수          -11.54        정지 기여 0.0000

    -12 를 쓴다.

    비교 기준 (v59 @3000): 점수 0.0165 · roll RMS 6.66 · 넘어짐 0.8 %
    **판정: roll RMS 가 5.50 아래로 내려가면서 점수가 0.0180 을 넘지 않는가.**
    (합격선은 `scripts/diag/scoreboard.py` 에 박혀 있다.)

    roll 이 내려가되 부족하면 `torso_ang_vel` 을 **작게** 얹어 마무리한다 —
    원인을 먼저 줄이면 증상 감쇠에 필요한 계수도 작아지므로 추종 손해가 준다.
    """

    foot_lateral_cmd_relative = True
    foot_lateral_scale = -12.0


@configclass
class JoystickEnvCfg_V62(JoystickEnvCfg_V61):
    """imitation_v62 — v61 + **옆 명령일 때 발 좌우 억제를 푼다**. 하나만.

    v61 은 실패했지만 **어디서 실패했는지가 하나로 특정된다.**

        항목        v59      v61      판정
        점수      0.0165   0.0221   (합격선 0.0180)
        앞뒤      0.0153   0.0155   그대로 지킴
        회전      0.0106   0.0104   그대로 지킴
        옆걸음    0.0263   0.0534   << 유일하게 두 배
        roll RMS   6.66     4.97    합격 (<= 5.50)

    점수를 분해하면 **옆걸음만 v59 수준으로 돌아오면 0.0166 이 되어 양쪽 다
    합격**이다. 나머지는 이미 다 맞았다.

    ## 왜 명령 상대형만으로 부족했나 — 기하 때문이다

    옆으로 걸으려면 **스윙 발이 몸통보다 더 빨리** 옆으로 가서 다음 디딤
    자리를 잡아야 한다. 그러니 `v_foot_y - cmd_vy` 에서 명령분을 빼도 남는
    성분이 크고, 그걸 벌하면 옆걸음이 그만큼 느려진다. 실측이 그대로 보여
    준다 — 좌 0.3008 -> 0.2456 로 18 % 만 면제됐다.

    빼는 것으로는 안 되고 **옆 명령 구간에서는 항 자체를 낮춰야** 한다.

        foot_lateral_gate_max   = 0.2   (lin_vel_y_range 상한)
        foot_lateral_gate_floor = 0.2

    |cmd_vy| = 0 에서 1 배, 0.2 에서 0.2 배. 옆 명령이 없는 전진·후진·회전·
    정지에서는 억제가 그대로라 roll 이득을 지킨다.

    ## 예상

    v61 의 옆걸음 악화분(+0.0271)이 계수에 대략 비례한다고 보면, 0.2 배에서
    +0.0054 -> 옆걸음 약 0.0317. 그러면

        점수 = (3x0.0236 + 3x0.0074 + 2x0.0104 + 0.0317 + 0.0317) / 10 = 0.0177

    로 합격선 안이다. roll 은 좌/우 두 방향에서 억제가 약해지므로 4.97 에서
    5.2~5.5 로 오를 것이다 — 합격선 경계다. 넘으면 `torso_ang_vel` 을 작게
    (-0.3) 얹어 마무리한다. 원인을 이미 줄였으므로 그때 필요한 감쇠는 작다.

    비교 기준: v61 · v59
    **판정: 점수 <= 0.0180 이면서 roll RMS <= 5.50.** 둘 다면 종료하고 배포한다.
    """

    foot_lateral_gate_max = 0.2
    foot_lateral_gate_floor = 0.2


@configclass
class JoystickEnvCfg_V63(JoystickEnvCfg_V59):
    """imitation_v63 — v59 + **미끄러운 바닥도 겪게 한다**. 하나만 바뀐다.

    2026-08-15, 사용자: "실기에서 발이 밀린다, 슬라이딩하는 것 같다."

    지금 마찰 무작위화는 로봇 바디 0.5~1.0 이고 지면은 1.0 고정이다. 결합이
    multiply 라 **실효 마찰이 0.5~1.0** — 0.5 아래를 한 번도 겪지 않는다.
    실기 바닥이 그보다 미끄러우면 정책은 있지도 않은 접지력을 믿고 걷는다.

        static/dynamic_friction_range  (0.5, 1.0) -> (0.3, 1.0)

    아래로만 넓힌다. 위쪽은 이미 덮고 있고 미지수는 미끄러운 쪽뿐이다.
    바닥 마찰을 실제로 잰 적이 없어서(경사 시험 미실시) 범위로 학습하되,
    v44 에서 배운 대로 **필요 이상으로 넓히지 않는다** — 질량을 0.85~1.25 로
    넓혔다가 앞뒤 추종을 망쳤다.

    ## 이미 있는 대비책

    발에 3 mm 고무패드를 붙여 이중지지 요 각속도가 0.672 -> 0.167 rad/s,
    드리프트 24 -> 약 1 도/s 로 줄었다. 물리적 접지력은 그때 확보했고, 이번은
    **정책이 접지력에 덜 의존하게** 만드는 쪽이다.

    `cost_foot_slip`(접지 중 발의 월드 수평 속도 벌점)도 준비돼 있고 v59 에서
    5 % 계수 -135.2 로 실측해 뒀다. 여기 같이 걸지 않는 이유는 마찰 범위만으로
    충분한지 먼저 보기 위해서다 — 부족하면 v64 에서 얹는다.

    ## 기반은 v62 가 아니라 v59 다

    v61(-12)과 v62(+게이트) 가 둘 다 실패했다 — 점수 0.0221 / 0.0223 으로
    합격선(0.0180)을 못 넘겼다. `foot_lateral -12` 는 전반적으로 과했다.
    실패한 기반 위에 마찰을 쌓으면 무엇 때문인지 못 가리므로, 측정된 최고
    배포본인 v59 에서 분기한다.

    비교 기준: v59 (점수 0.0165 · roll 6.66)
    **판정: 점수와 roll 이 유지되는가.** 유지되면 실기 미끄러짐에 대한 대비를
    공짜로 얻은 것이다. 나빠지면 0.4 로 덜 넓힌다.
    """

    events: EventCfg = _LowFrictionEventCfg()


@configclass
class JoystickEnvCfg_V64(JoystickEnvCfg_V59):
    """imitation_v64 — **보간격을 넓힌다** (0.18 -> 0.22). 레퍼런스만 바뀐다.

    ## 왜 리워드가 아니라 기하인가

    2 순위(흔들림)를 리워드로 세 번 시도했고 세 번 다 1 순위를 깎았다:

        구성                        점수     roll RMS
        v59  (기준)               0.0165     6.66
        v60  torso_ang_vel -1.2   0.0304     3.73
        v61  foot_lateral -12     0.0221     4.97
        v62  v61 + 옆명령 게이트   0.0223     5.39

    v59↔v60 두 점의 기울기로 보면 **어떤 계수를 써도 두 합격선을 동시에 만족
    할 수 없다** (roll 5.50 이 되려면 c~0.57, 점수 0.0180 을 넘는 c~0.13).
    벌점은 흔들림을 줄이는 만큼 정책의 자유도를 묶고, 그 대가가 추종이다.

    **기하는 세금을 물리지 않는다.** 스탠스를 넓히면 좌우 지지 기반이 커져
    같은 무게 이동에도 몸통이 덜 기운다. 리워드 항이 하나도 안 늘어나므로
    추종에 낼 대가가 없다.

    ## 방향이 맞다는 근거

    v52 가 반대로 **좁혔다** (0.18 -> 0.14). 결과는 roll 진폭 20.7 -> 25.1 도로
    **나빠졌다.** 같은 축을 반대로 밀면 반대 결과가 나올 것이다.

    ## 부수 효과 하나가 유리하다

    다리를 벌리면 같은 몸통 높이에서 무릎이 펴진다. calc_home 으로 뽑은 READY
    무릎이 -76.7 -> **-66.7 도**로 10 도 더 펴졌다. 실기 왼무릎이 -110 도
    아래에서 힘을 못 내는 문제(V46 참고)에서 **여유가 10 도 더 생긴다.**

    leg_fk 검증: 보간격 184.2 -> 224.3 mm (+22 %), 발 z 는 -156.8 mm 로 동일
    하므로 몸통 높이는 그대로다.

    ## 예상되는 대가

    스탠스가 넓으면 옆으로 걸을 때 발을 더 멀리 옮겨야 하고, 좌우 명령 추종이
    나빠질 수 있다. 옆걸음은 사용자 순위에서 가장 낮으므로(가중 1) 그 정도는
    감수한다. 앞뒤·회전이 나빠지면 0.20 으로 덜 넓힌다.

    비교 기준 (v59 @3000): 점수 0.0165 · roll RMS 6.66 · 넘어짐 0.8 %
    **판정: roll RMS <= 5.50 이면서 점수 <= 0.0180.** 둘 다면 종료·배포한다.

    ⚠️ `READY_BASE_HEIGHT_G135FS22` 는 아직 `settle_height.py` 실측이 아니라
    FK 근거의 승계값이다. GPU 가 나면 재서 고칠 것.
    """

    reference_motion_pkl = "source/open_duck_mini_isaaclab/reference_motion/data/ref_g135fs22.pkl"

    robot = OPEN_DUCK_MINI_V2_DC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=OPEN_DUCK_MINI_V2_DC_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_G135FS22),
            joint_pos=dict(READY_JOINT_POS_G135FS22_ZNECK),
        ),
    )
    ready_base_height = READY_BASE_HEIGHT_G135FS22


@configclass
class JoystickEnvCfg_V65(JoystickEnvCfg_V59):
    """imitation_v65 — **레퍼런스는 몸통을 수평으로 유지하라고 이미 말하고 있다.**

    벌점 세 개와 무작위화 두 개가 모두 실패한 뒤(v60~v64) 레퍼런스를 직접
    열어 봤다. placo 가 만든 궤적의 **roll 각속도가 정확히 0 이다**:

        명령        roll rate RMS   pitch rate RMS
        정지            0.00 deg/s      0.00
        전진            0.00 deg/s      0.00
        옆              0.00 deg/s      0.00
        회전            0.08 deg/s      0.00

    즉 "흔들리지 마라" 는 **이미 모방 리워드 안에 있다** — `reward_imitation`
    의 `ang_vel_xy` 항이 그것이다. 그런데 그 가중치가 **0.1** 이다.
    기본값 0.5 의 5 분의 1 이고, Walk6(v17)에서 낮춘 뒤 40 개 버전 동안
    그대로 상속됐다. 정작 그 Walk6 독스트링의 마지막 줄이

        "남은 증상은 몸통 흔들림(뒤뚱거림)이다."

    다. 증상을 적어 놓고 그것을 고칠 항을 눌러 둔 채로 40 판을 돌렸다.

    ## 왜 이번엔 다를 수 있나

    v60~v64 는 전부 **벌점**이거나 **무작위화 확대**였다. 둘 다 정책의
    자유도를 빼앗아 추종에서 대가를 받아 갔다 — 여섯 번 반복 관측했다.

    이 항은 **양의 보상**이다. 가중치를 올리면 세금이 느는 게 아니라 **얻을
    수 있는 상금이 는다.** 예산을 다투는 구조가 아니다.

        지금       exp(-2 x err^2) ~ 0.018  -> 가능한 값의 2 % 만 받는다
        w 0.1->1.0 roll 을 0 으로 만들면 스텝 보상의 약 12 % 를 더 받는다

    ## 위험

    Walk6 이 자세 항 비중을 올렸을 때 "가만히 서 있기" 가 유리해져 전진이
    0.117 -> 0.059 로 반토막 난 전례가 있다. 정지 상태도 roll rate 가 0 이라
    같은 함정이 있을 수 있다. 판정에서 그것부터 본다.

    비교 기준 (v59 @3000): 점수 0.0165 · 앞뒤 0.0153 · 회전 0.0106 · roll 6.66
    **판정: roll RMS <= 5.50 이면서 점수 <= 0.0180.** 전진 속도가 눈에 띄게
    줄면 "서 있기" 함정이므로 0.5 로 낮춰 다시 건다.
    """

    imitation_w_ang_vel_xy = 1.0


@configclass
class JoystickEnvCfg_V66(JoystickEnvCfg_V65):
    """imitation_v66 — **딛는 발은 정지해 있어라**. 미끄러짐의 진짜 처방.

    ## 마찰은 문제가 아니었다 (2026-08-18 실측)

    사용자가 실기에서 "발이 밀린다" 고 해서 v63 으로 마찰 무작위화를
    0.5~1.0 -> 0.3~1.0 으로 넓혔다. **회전 추종이 5 배 나빠졌다**
    (0.0106 -> 0.0543).

    그 뒤 저울로 실제 정지마찰을 쟀다. 로봇(약 2.6 kg)을 수평으로 당겼는데
    **1 kgf 에서 꿈쩍도 하지 않았다.** 즉 μ > 0.38 이고, 근처도 안 갔으니
    실제로는 0.6~1.0 대다. **심의 0.5~1.0 범위가 이미 맞았고**, v63 은 있지도
    않은 문제를 대비시키느라 회전을 내준 것이다.

    ## 그러면 왜 미끄러지나 — 정적이 아니라 동적이다

    정지마찰이 충분한데 미끄러진다면 원인은 둘 중 하나다:

      1. **착지 바운스로 수직항력이 사라진다.** μ 가 아무리 높아도 N=0 이면
         마찰력도 0 이다. 실측 접지 바운스가 v59 1.22 · v61 1.68 이다.
      2. **발이 수평 속도를 가진 채 닿는다.** 그러면 μ 와 무관하게 그 속도만큼
         쓸린다.

    `cost_foot_slip` 은 2 번을 직접 가르친다 — 접지 중 발의 **월드** 수평
    속도를 벌한다. 월드 프레임이라 명령과 무관하게 "딛은 발은 지면에 대해
    정지해 있어야 한다" 는 물리적으로 옳은 목표이고, 그래서 추종과 덜
    싸울 것으로 본다 (`foot_lateral` 이 옆걸음과 싸운 것과 다르다).

    ## 계수

    `probe_terms.py` 로 v59 정책에서 실측 (2026-08-15):

        foot_slip   평균 0.0119   정지 0.0001   5 % 계수 -135.2

    정지 기여가 0 에 가깝다 — 서 있을 때 상시로 깎지 않는다.

    ## 기반을 v61 로 바꾼다 (2026-08-18)

    실기에서 사용자가 확인했다: **"지금 실기에서 잘 되는 건 v61 이다. 전 것들은
    전부 잘 안 되거나 넘어진다."** 심 점수는 v59(0.0165)를 v61(0.0221)보다
    위로 매겼는데 실기는 반대다.

    이유는 점수의 구성에 있다. 둘은 앞뒤(0.0153/0.0155)와 회전(0.0106/0.0104)이
    같고 옆걸음만 갈린다(0.0263/0.0534). 그런데 v61 은 흔들림이 25 % 적다
    (roll RMS 6.66 -> 4.97). 실기에서는 **넘어지지 않는 것이 추종보다 먼저**라
    그 차이가 뒤집힌다.

    ## 기반을 v65 로 바꾼다 (2026-08-18)

    v65(모방 `ang_vel_xy` 0.1 -> 1.0)가 측정에서 **추종과 흔들림 둘 다 1 등**을
    했다 — 처음이다:

        점수 0.0138 (v59 0.0165) · 앞뒤 0.0054 (v59 0.0153, 3 배) ·
        roll RMS 4.89 (v61 4.97) · 실사용 속도 0.0073

    실패한 것은 **낙상률 1.9 %** 하나다 (v59 0.8 · v61 0.7). 그러니 v61 이
    아니라 v65 위에 쌓는 것이 맞다.

    비교 기준 (v59 @3000): 점수 0.0165 · roll 6.66 · 접지 바운스 1.22
    **판정: 점수 <= 0.0180 을 지키면서 접지 바운스와 착지 하강속도가 내려가는가.**
    실기 미끄러짐은 심 지표로 직접 못 보므로, 그 둘을 대리 지표로 삼고
    최종 확인은 실기에서 한다.
    """

    foot_slip_scale = -135.0


@configclass
class JoystickEnvCfg_V67(JoystickEnvCfg_V65):
    """imitation_v67 — **쩍벌을 막는다.** 벌어지는 쪽엔 제동이 없었다.

    2026-08-18, 사용자: "다리가 쩍벌인데 이게 맞아?" v64(보간격 0.22)를 보고
    한 말인데, 재 보니 **v64 만의 문제가 아니었다.**

    v59 를 레퍼런스와 관절별로 비교했다 (전 명령 평균, 도):

        left_hip_roll   +7.8   +6.2   +3.6   +9.1   +4.6
        right_hip_roll  +6.0   +6.7   +6.6   +4.7   +4.9

    양쪽 다 같은 부호로 **바깥** — 레퍼런스보다 스탠스가 약 38 mm 넓다.
    함께 나온 것 두 개도 "정상 보행" 과 어긋난다:

        hip_yaw 진폭   레퍼런스의 **6 배** (2.5 도 vs 0.4 도) — 골반을 비튼다
        무릎 진폭      레퍼런스의 **절반** (5.5 도 vs 11.0 도) — 안 굽힌다

    ## 왜 벌어졌나 — 아무도 안 막았다

    `hip_inward` 는 **안쪽**만 벌한다 (자가충돌 방지용, 임계 3 도):

        inward = dir x (실제각 - 레퍼런스각)
        over   = clamp(inward - 0.0524, min=0)

    우리는 바깥으로 +6~9 도이므로 `inward` 가 음수라 `over = 0` — 벌점이
    정확히 0 이다. 그리고 넓은 지지 기반은 안 넘어지는 데 유리하니, 막는 것이
    없으면 정책이 계속 벌리는 것이 합리적이다. **버그가 아니라 빈 자리다.**

    ## 무엇을 바꾸나

        hip_outward_thresh = 0.0524   (3 도, 안쪽과 대칭)
        hip_outward_rel    = 0.28     (hip_inward_scale -25 x 0.28 = 실효 -7)

    실측으로 계수를 잡았다 (v59 gait npz, 보행 5 방향):

        바깥 초과분 raw 평균 0.1890   ->  4 % 기여 계수 -6.81  ->  -7 을 쓴다

    `hip_inward_walking_only = True` 를 그대로 물려받으므로 **정지에서는 꺼진다.**
    정지 위상에서는 레퍼런스 자체가 "한 발 든 순간" 자세라 기준이 어긋나고,
    좌우 대칭 리워드와 싸운다 (v36 에서 그 균형점이 실측된 바 있다).

    ## 기대와 위험

    스탠스가 레퍼런스대로 좁아지면 보기에 정상이 되고, hip_yaw·무릎도 레퍼런스
    쪽으로 끌려올 가능성이 있다 (같은 원인 — 레퍼런스 미추종).

    위험은 **흔들림이 늘 수 있다**는 것이다. 좁은 지지 기반은 좌우로 덜
    안정하다. 다만 레퍼런스 자체는 roll 각속도가 정확히 0 인 ZMP 궤적이므로,
    v52 가 레퍼런스를 좁혔을 때와는 다르다 — 이번엔 레퍼런스 **쪽으로** 간다.

    ## 기반을 v61 로 바꾼다 (2026-08-18)

    실기에서 사용자가 확인했다: **"지금 실기에서 잘 되는 건 v61 이다. 전 것들은
    전부 잘 안 되거나 넘어진다."** 심 점수는 v59(0.0165)를 v61(0.0221)보다
    위로 매겼는데 실기는 반대다.

    이유는 점수의 구성에 있다. 둘은 앞뒤(0.0153/0.0155)와 회전(0.0106/0.0104)이
    같고 옆걸음만 갈린다(0.0263/0.0534). 그런데 v61 은 흔들림이 25 % 적다
    (roll RMS 6.66 -> 4.97). 실기에서는 **넘어지지 않는 것이 추종보다 먼저**라
    그 차이가 뒤집힌다.

    ## 기반을 v65 로 바꾼다 (2026-08-18)

    v65(모방 `ang_vel_xy` 0.1 -> 1.0)가 측정에서 **추종과 흔들림 둘 다 1 등**을
    했다 — 처음이다:

        점수 0.0138 (v59 0.0165) · 앞뒤 0.0054 (v59 0.0153, 3 배) ·
        roll RMS 4.89 (v61 4.97) · 실사용 속도 0.0073

    실패한 것은 **낙상률 1.9 %** 하나다 (v59 0.8 · v61 0.7). 그러니 v61 이
    아니라 v65 위에 쌓는 것이 맞다.

    비교 기준 (v59 @3000): 점수 0.0165 · roll 6.66 · hip_roll 편차 +6~9 도
    **판정: hip_roll 바깥 편차가 3 도 아래로 내려가면서 점수 <= 0.0180.**
    흔들림이 크게 늘면 `hip_outward_rel` 을 0.14 로 반감한다.
    """

    hip_outward_thresh = 0.0524
    hip_outward_rel = 0.28


@configclass
class JoystickEnvCfg_V68(JoystickEnvCfg_V65):
    """imitation_v68 — **낙상은 보수적으로, 토크는 세게.** 탐색은 넓혀서 벌충.

    2026-08-18, 사용자: "낙상은 매우 보수적으로 잡아야 하고 토크 패널티도 많이
    줘야 할 것 같음. 왜냐면 피크 토크가 매우 큼. 이를 위해서 탐색 시그마를
    조금 올려 달라."

    기반은 v61 이다 — 실기에서 실제로 잘 걷는 것이 그것이기 때문이다
    (심 점수는 v59 가 위지만 실기는 반대였다).

    ## 세 가지를 함께 건다

        min_base_height_ratio  0.75 -> 0.85
        torques_scale          -1.0e-3 -> -0.2
        init_noise_std         1.0 -> 1.3  (JoystickPPORunnerCfg_Explore13)

    앞의 둘은 정책을 조이고, 셋째는 그 반대 힘이다. 하나만 떼서 보면 의미가
    없어서 한 번에 건다 — 사용자 요청도 그 조합이다.

    ### 낙상 (종료 조건)

    `min_base_height_ratio` 는 READY 높이 대비 이 비율 아래로 주저앉으면
    에피소드를 끝낸다. 0.75 -> 0.85 로 올리면 살짝만 무너져도 종료되므로
    정책이 그 근처에 가지 않으려 한다.

    **상시 벌점이 아니라 종료로 거는 이유**: 벌점은 "가만히 서 있기" 를
    유리하게 만든다 (Walk6 에서 자세 비중을 올렸다가 전진이 반토막 난 전례).
    종료는 미래 보상을 통째로 잃게 하므로 그 편향이 없다.

    ### 토크

    실측 τ² 합이 **7.70 N·m²** (v61, verdict.py)인데 `torques_scale` 이
    -1.0e-3 이라 기여가 **-0.00015 (스텝당 총 0.644 의 0.02 %%)** 다 — 사실상
    규제가 없었다. v54 가 -1.0e-2 로 올리려 했지만 그래도 0.24 %% 다.

        -0.2 x 7.70 x dt(0.02) = -0.031  ->  약 4.8 %%

    `action_rate`(-0.5) 대역을 넘어서는 실효 규제가 된다. 사용자가 지적한
    "피크 토크가 매우 크다" 는 τ² 이 제곱이라 피크에 더 크게 걸린다.

    ### 탐색

    종료를 조이고 벌점을 키우면 초기 정책이 살아남기 어려워 탐색이 일찍
    죽는다. `init_noise_std` 를 1.3 으로 올려 반대 힘을 준다.

    ## 기반을 v65 로 바꾼다 (2026-08-18)

    v65(모방 `ang_vel_xy` 0.1 -> 1.0)가 측정에서 **추종과 흔들림 둘 다 1 등**을
    했다 — 처음이다:

        점수 0.0138 (v59 0.0165) · 앞뒤 0.0054 (v59 0.0153, 3 배) ·
        roll RMS 4.89 (v61 4.97) · 실사용 속도 0.0073

    실패한 것은 **낙상률 1.9 %** 하나다 (v59 0.8 · v61 0.7). 그러니 v61 이
    아니라 v65 위에 쌓는 것이 맞다.

    비교 기준 (v61 @3000): 점수 0.0221 · roll 4.97 · 낙상 0.7 %% · τ² 7.70
    **판정: 낙상률이 0.5 %% 아래로 내려가고 τ² 이 유의하게 줄면서, 점수가
    0.0250 을 넘지 않는가.** 실기 기준이 낙상이므로 점수는 조금 내줘도 된다.
    무너지면(에피소드 길이가 크게 짧아지면) 토크를 -0.05 로 완화한다.
    """

    min_base_height_ratio = 0.85
    torques_scale = -0.2


@configclass
class JoystickEnvCfg_V69(JoystickEnvCfg_V68):
    """imitation_v69 — v68 + **토크 벌점을 절반으로**. 후진을 되찾는다.

    v68 이 낸 것과 잃은 것 (기준 v65):

        낙상        1.9 -> 1.1 %       종료 조건 0.85 가 먹혔다
        토크 제곱합  8.24 -> 5.91      벌점 -0.2 가 먹혔다 (-28 %)
        roll RMS    4.89 -> 4.05      역대 최고
        회전        0.0229 -> 0.0036  6 배, 역대 최고 (예상 밖 보너스)
        ---
        후진        0.0039 -> 0.0315  **8 배 악화** <- 점수를 0.0204 로 밀었다
        점수        0.0138 -> 0.0204

    의도한 셋이 전부 먹혔고 대가는 후진 하나다. 뒤로 걸을 때 토크가 더 드는
    것은 자연스러우므로 -0.2 가 그쪽에 과했다고 본다. 후진은 앞뒤 가중 3 이라
    점수에 크게 걸린다.

    v68 독스트링은 "무너지면 -0.05 로 완화한다" 고 적었는데 무너지지는 않았다
    (에피소드 505 유지). 그래서 중간값을 쓴다:

        torques_scale  -0.2 -> -0.1   (실측 τ² 5.91 기준 약 1.8 % 기여)

    낙상 개선(1.9 -> 1.1)과 회전 개선(6 배)을 지키면서 후진만 되찾는 것이
    목표다. 토크가 절반이면 τ² 은 5.91 과 8.24 사이 어딘가로 돌아올 것이고,
    그 정도는 감수한다 — 3 순위이기 때문이다.

    비교 기준 (v68 @3000): 점수 0.0204 · 후진 0.0315 · roll 4.05 · 낙상 1.1 %
    **판정: 후진이 0.015 아래로 돌아오면서 낙상 1.1 %% 이하, roll 4.5 이하를
    지키는가.** 후진이 안 돌아오면 토크가 아니라 종료 조건(0.85)이 범인이므로
    다음은 그것을 0.80 으로 완화한다.
    """

    torques_scale = -0.1


@configclass
class _ComBack10EventCfg(_ComBackEventCfg):
    """몸통 CoM 을 뒤로 10 mm."""

    com_back = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk_assembly"),
            "com_range": {"x": (-0.010, -0.010)},
        },
    )


@configclass
class JoystickEnvCfg_V70(JoystickEnvCfg_V65):
    """imitation_v70 — **몸통 CoM 을 뒤로 10 mm.**

    2026-08-18, 교수님: "무게중심을 못 잡는다." 사용자가 뒤로 10/15/20 mm 를
    시험해 보라고 해서 v70/v71/v72 로 나눠 돌린다. 세 개가 같은 축의 서로 다른
    값이므로 한 번에 걸어도 무엇이 들었는지 가릴 수 있다.

    근거 (v44 때 실측): 심 모델의 CoM 은 발 중심보다 **16.7 mm 뒤**인데
    **실기는 앞으로 넘어진다.** 둘이 같이 서려면 실기의 실제 CoM 이 심보다
    앞에 있어야 하고, 그러면 정책은 있지도 않은 뒤쪽 여유를 믿고 걷는다.

    기반은 v65 다 — 점수(0.0138)와 흔들림(4.89) 둘 다 1 등인데 **낙상만
    1.9 %%로 최악**이라, 낙상을 겨냥한 이 실험의 기준선으로 맞다.

    비교 기준 (v65 @3000): 점수 0.0138 · roll 4.89 · **낙상 1.9 %%**
    **판정: 낙상률이 내려가는가.** 세 값 중 최소를 고르고, 20 mm 에서도 계속
    내려가면 더 뒤까지 시험한다. 점수가 0.0200 을 넘으면 그만큼은 대가다.
    """

    events: EventCfg = _ComBack10EventCfg()


@configclass
class _ComBack15EventCfg(_ComBackEventCfg):
    """몸통 CoM 을 뒤로 15 mm."""

    com_back = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk_assembly"),
            "com_range": {"x": (-0.015, -0.015)},
        },
    )


@configclass
class JoystickEnvCfg_V71(JoystickEnvCfg_V65):
    """imitation_v71 — **몸통 CoM 을 뒤로 15 mm.**

    2026-08-18, 교수님: "무게중심을 못 잡는다." 사용자가 뒤로 10/15/20 mm 를
    시험해 보라고 해서 v70/v71/v72 로 나눠 돌린다. 세 개가 같은 축의 서로 다른
    값이므로 한 번에 걸어도 무엇이 들었는지 가릴 수 있다.

    근거 (v44 때 실측): 심 모델의 CoM 은 발 중심보다 **16.7 mm 뒤**인데
    **실기는 앞으로 넘어진다.** 둘이 같이 서려면 실기의 실제 CoM 이 심보다
    앞에 있어야 하고, 그러면 정책은 있지도 않은 뒤쪽 여유를 믿고 걷는다.

    기반은 v65 다 — 점수(0.0138)와 흔들림(4.89) 둘 다 1 등인데 **낙상만
    1.9 %%로 최악**이라, 낙상을 겨냥한 이 실험의 기준선으로 맞다.

    비교 기준 (v65 @3000): 점수 0.0138 · roll 4.89 · **낙상 1.9 %%**
    **판정: 낙상률이 내려가는가.** 세 값 중 최소를 고르고, 20 mm 에서도 계속
    내려가면 더 뒤까지 시험한다. 점수가 0.0200 을 넘으면 그만큼은 대가다.
    """

    events: EventCfg = _ComBack15EventCfg()


@configclass
class _ComBack20EventCfg(_ComBackEventCfg):
    """몸통 CoM 을 뒤로 20 mm."""

    com_back = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk_assembly"),
            "com_range": {"x": (-0.020, -0.020)},
        },
    )


@configclass
class JoystickEnvCfg_V72(JoystickEnvCfg_V65):
    """imitation_v72 — **몸통 CoM 을 뒤로 20 mm.**

    2026-08-18, 교수님: "무게중심을 못 잡는다." 사용자가 뒤로 10/15/20 mm 를
    시험해 보라고 해서 v70/v71/v72 로 나눠 돌린다. 세 개가 같은 축의 서로 다른
    값이므로 한 번에 걸어도 무엇이 들었는지 가릴 수 있다.

    근거 (v44 때 실측): 심 모델의 CoM 은 발 중심보다 **16.7 mm 뒤**인데
    **실기는 앞으로 넘어진다.** 둘이 같이 서려면 실기의 실제 CoM 이 심보다
    앞에 있어야 하고, 그러면 정책은 있지도 않은 뒤쪽 여유를 믿고 걷는다.

    기반은 v65 다 — 점수(0.0138)와 흔들림(4.89) 둘 다 1 등인데 **낙상만
    1.9 %%로 최악**이라, 낙상을 겨냥한 이 실험의 기준선으로 맞다.

    비교 기준 (v65 @3000): 점수 0.0138 · roll 4.89 · **낙상 1.9 %%**
    **판정: 낙상률이 내려가는가.** 세 값 중 최소를 고르고, 20 mm 에서도 계속
    내려가면 더 뒤까지 시험한다. 점수가 0.0200 을 넘으면 그만큼은 대가다.
    """

    events: EventCfg = _ComBack20EventCfg()


@configclass
class JoystickEnvCfg_V73(JoystickEnvCfg_V70):
    """imitation_v73 — v70 + **발바닥 접지면을 키운다**. 발 하나만 바뀐다.

    2026-08-20. roll 흔들림을 제어로 여섯 번 시도해 전부 실패한 뒤 발 형상을
    실측했다. `foot_bottom_tpu.stl` 의 **실제 접지 패치**(최저면 ±0.05 mm):

        기존      1252 mm^2   전후  85.1 mm   **좌우 16.3 mm**
        big_foot  4226 mm^2   전후 108.0 mm   **좌우 40.0 mm**

    **좌우 지지폭이 16 mm 다.** 거의 칼날 위에 서 있는 셈이고, CoM 이 조금만
    옆으로 가도 발 모서리를 넘는다. 앞서 계산한 "roll ±10 도면 반스탠스
    70 mm 에서 발 모서리가 12 mm 내려간다" 와 맞물리면 16 mm 폭의 절반 넘게
    먹는다. **제어로 못 고치는 기구 한계였을 가능성이 크다.**

    전후는 85 mm 로 이미 과잉이라 (CoM 이 -32 mm 뒤여도 여유롭게 들어온다)
    문제는 좌우뿐이었다.

    ## 기반이 v70 인 이유

    v70(CoM 뒤로 10 mm)이 낙상을 1.9 -> 0.9 %% 로 절반으로 줄였다. 그 위에
    발만 바꾸면 **발 효과만 분리**해서 볼 수 있다 — v70 이 같은 CoM 으로
    이미 측정돼 있으므로 직접 비교가 된다.

    ## 바뀌는 것과 안 바뀌는 것

    운동학이 **완전 동일**하다 (14 축 부모·자식·축·rpy·xyz·한계 전부 일치).
    캘리브·영점·READY·정책 관절 규약이 그대로 유효하다. 바뀌는 것은 발
    형상과 질량뿐이다 (편측 69.5 -> 79.9 g).

    ## 예상되는 대가

    발 질량이 늘어 스윙 관성이 커지고, 발이 넓어져 기울었을 때 모서리가 먼저
    닿는다. 지금 스윙 여유가 9~14 mm 라 접지 바운스가 늘 수 있다.

    비교 기준 (v70 @3000): 점수 0.0307 · roll RMS 4.32 · **낙상 0.9 %%**
    **판정: 낙상률과 roll RMS 가 내려가는가.** 좌우 지지가 2.5 배인데도 안
    내려가면 흔들림의 원인은 발이 아니다.
    """

    robot = OPEN_DUCK_MINI_V2_DC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=OPEN_DUCK_MINI_V2_DC_CFG.spawn.replace(
            usd_path=OPEN_DUCK_MINI_BIGFOOT_USD_PATH,
        ),
        init_state=OPEN_DUCK_MINI_V2_DC_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_G135SYM),
            joint_pos=dict(READY_JOINT_POS_G135SYM_ZNECK),
        ),
    )

















@configclass
class JoystickEnvCfg_V34C20(JoystickEnvCfg_V34C):
    """imitation_v34c20 — v34c(정지 위상 고정) + 레퍼런스 높이 +20 mm (ref_g135)."""

    reference_motion_pkl = "source/open_duck_mini_isaaclab/reference_motion/data/ref_g135.pkl"
    robot = OPEN_DUCK_MINI_V2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=OPEN_DUCK_MINI_V2_CFG.init_state.replace(
            pos=(0.0, 0.0, SPAWN_BASE_HEIGHT_G135),
            joint_pos=dict(READY_JOINT_POS_G135_ZNECK),
        ),
    )
    ready_base_height = READY_BASE_HEIGHT_G135


@configclass
class JoystickEnvCfg_V34C2(JoystickEnvCfg_V34C):
    """imitation_v34c2 — v34c 로 부족할 때 쓸 후속: 정지 벌점을 5배로.

    구조(위상 고정)만으로 정지가 안 잡히면 그때 계수를 올린다. 순서를 지키는
    이유는 둘을 같이 바꾸면 어느 쪽이 들었는지 알 수 없기 때문이다.
    """

    stand_still_scale = -1.0


# ── 정지 전용 자세: 몸통 수직 ──────────────────────────────────────────────
# 보행 평균 자세(READY_JOINT_POS_G125)를 몸통 수직으로 놓고 FK 로 재면 **발이
# 4.03 도 기울어 있다.** placo 프리셋의 walk_trunk_pitch=-4 도가 관절각에 배어
# 있기 때문이다. 시뮬에서는 발바닥이 지면에 눕도록 물리가 몸통을 그만큼 돌리므로,
# 서 있을 때 몸통이 4 도 앞으로 기운다.
#
# "발 피치" 는 관절이 아니라 발 링크의 기울기다 — hip_pitch + knee + ankle 의
# 합으로 정해진다. 그 합이 0 이면 발바닥이 지면과 평행하고 몸통이 수직으로 선다.
#
# 그래서 **무릎은 그대로 두고 hip_pitch 에서 ankle 로 6.3 도만 옮긴다.**
# hip_pitch = ankle = knee/2 인 완전 대칭 자세가 된다:
#
#                     무릎    hip    ankle   발피치   높이     무릎토크
#   현재 READY_G125   89.9   51.3    42.6   +4.03   145.4    0.871 N·m
#   이 자세           89.9   45.0    45.0    0.00   146.1    0.840 N·m
#
# 무릎 각도는 그대로인데 토크가 3.5 % **줄어든다**. 높이도 0.7 mm 올라간다.
# CoM 은 발 지지면 안에서 앞여유 56.7 / 뒤여유 47.2 mm 로 거의 정중앙이다
# (현재는 45.4 / 58.5 로 앞으로 치우쳐 있다). 좌우도 완전 대칭이라 레퍼런스의
# 좌우 비대칭(0.03 rad)도 함께 사라진다.
#
# ⚠️ 처음엔 "CoM 을 지지면 정중앙에" 라는 조건까지 넣어 풀었다가 무릎이
# 89.9 -> 101.7 도로 늘고 토크가 0.925 N·m 로 올랐다. 목표 높이를 133.8 mm
# (ref_g115 값) 로 잘못 넣은 탓이었다 — ref_g125 의 실제 높이는 145.4 다.
# 무릎은 이 로봇에서 토크가 가장 큰 관절이라(평균 1.6 N·m, 스톨의 39 %)
# 더 굽히는 방향은 피해야 한다.
STANDSTILL_JOINT_POS_UP = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0,
    "left_hip_pitch": 0.7846, "left_knee": -1.5693, "left_ankle": 0.7846,
    "neck_pitch": 0.5236, "head_pitch": 0.5236, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0,
    "right_hip_pitch": 0.7846, "right_knee": 1.5693, "right_ankle": -0.7846,
}


@configclass
class JoystickEnvCfg_V34U(JoystickEnvCfg_V34C10):
    """imitation_v34u — v34c10 + **정지 목표 자세를 몸통 수직으로**.

    v34c10 은 정지 속력 0.0058 m/s 로 잘 서지만, 서 있는 **자세**가 몸통이 4 도
    앞으로 기운 채다. `cost_stand_still` 이 default_joint_pos(보행 평균)를 목표로
    삼는데 그 자세에 placo 의 walk_trunk_pitch=-4 도가 배어 있기 때문이다.

    무릎은 건드리지 않는다 — hip_pitch 에서 ankle 로 6.3 도를 옮길 뿐이고,
    그 결과 무릎 토크는 0.871 -> 0.840 N·m 로 오히려 준다.

    액션 원점(default_joint_pos)은 그대로 둔다 — 그건 보행 궤적 평균에 맞춰져
    있어야 액션이 도달 가능하고, 건드리면 v1~v9 를 아홉 번 실패시킨 상황이 된다.
    바뀌는 것은 **정지했을 때 무엇을 목표로 삼는가** 하나뿐이다.

    비교 기준 (v34c10): 정지 0.0058 · 추종 0.0166 · 리워드 426.2 · 에피 778.7.
    **판정은 정지 속력이 유지되면서 몸통이 실제로 수직으로 서는가로 한다.**
    """

    standstill_joint_pos = STANDSTILL_JOINT_POS_UP
