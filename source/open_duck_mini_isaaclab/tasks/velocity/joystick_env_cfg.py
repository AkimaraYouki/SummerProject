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
from open_duck_mini_isaaclab.robot_cfg import OPEN_DUCK_MINI_V2_CFG

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
    hip_inward_scale = -25.0

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
    stand_still_scale = -0.2
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
class JoystickEnvCfg_Upstream(JoystickEnvCfg_LegsOnly):
    """imitation_v14 — Open_Duck_Playground의 리워드 계수 그대로 (기준선).

    베이스 클래스의 값이 이미 upstream과 동일하므로 여기서는 RSI만 끈다
    (upstream은 RSI를 쓰지 않는다). 이 조합은 iter ~900에서 중단했다 —
    reward_at_ready 실측으로 imitation이 스텝당 +0.004, alive(+0.400)의 1%밖에
    안 됐기 때문이다. 상한 없는 관절각 페널티(−0.2023 × 15 = −3.03)가 양의
    추종 항들(~+3.2)을 거의 정확히 상쇄해 imitation이 0으로 붕괴한다.
    사실상 alive 하나로 학습되는 상태라 구조적으로 리워드 해킹이다.
    """

    use_rsi = False


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


@configclass
class JoystickEnvCfg_TallSafe(JoystickEnvCfg_Tall):
    """v28 + 런타임 안전 필터. 정책은 그대로 두고 안전만 얹는 경로의 검증용."""

    safety_filter_ckpt = "source/open_duck_mini_isaaclab/barrier_h5d.pt"


@configclass
class JoystickEnvCfg_HipInwardSafe(JoystickEnvCfg_HipInward):
    """v32 + 런타임 안전 필터. 실기 후보 조합이다.

    v32 는 이미 5 mm 위반 1.0% 라 필터가 개입할 일이 드물고, 그만큼 추종을 덜
    해친다. 필터 단독(v28 위에)과 비교하면 "정책을 안전하게 학습시킨 것" 과
    "필터로 막은 것" 이 각각 얼마를 기여하는지 갈린다.
    """

    safety_filter_ckpt = "source/open_duck_mini_isaaclab/barrier_h5d.pt"


# Z 자 목으로 학습할 때 쓰는 READY 자세. 다리 값은 H175 그대로이고 머리만 세운다.
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
