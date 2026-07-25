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
    LEFT_FOOT_BODY_NAME,
    RIGHT_FOOT_BODY_NAME,
    ROOT_BODY_NAME,
)
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

    # IMU offset matches the MJCF <site name="imu" pos="-0.08 -0.0 0.05"/>
    # relative to the root body. Playground's original MJCF called that body
    # "base"; this OnShape-derived URDF calls it ROOT_BODY_NAME
    # ("trunk_assembly") instead — confirmed via `robot/robot.urdf` and the
    # converted USD (2026-07-25), no link literally named "base" exists.
    # Mount the IMU on ROOT_BODY_NAME, not a hardcoded "base" string, so this
    # can't silently drift out of sync with joint_order.py again.
    imu: ImuCfg = ImuCfg(
        prim_path=f"/World/envs/env_.*/Robot/{ROOT_BODY_NAME}",
        offset=ImuCfg.OffsetCfg(pos=(-0.08, 0.0, 0.05)),
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
    min_base_height_ratio = 0.6

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

    # ── reference motion (imitation) ────────────────────────────────────
    # Stage 2 (2026-07-26): polynomial_coefficients.pkl now exists for real
    # (240 swept gaits, see docs/training_log.md) after fixing the Placo
    # pipeline (antenna joints, placo version pin, auto_waddle.py's python->
    # python3 bug). Flipping this on for the first time — Stage 1
    # (use_imitation=False, pure RL) reward-hacked into a collapsed-but-
    # technically-alive pose across every alive_scale tried.
    use_imitation: bool = True
    reference_motion_pkl = "source/open_duck_mini_isaaclab/reference_motion/data/polynomial_coefficients.pkl"
    # Gait-phase clock period (env steps), used for the imitation_phase
    # observation channel when use_imitation=False (no PolyReferenceMotion
    # loaded to supply nb_steps_in_period). Purely a periodic clock signal
    # in that case — has no effect on reward, only lets the policy sense
    # gait phase if it finds that useful on its own.
    gait_period_steps: int = 50


# ── alive_scale sweep variants (2026-07-26) ─────────────────────────────
# See the base class's alive_scale comment: 20.0 (Playground's original,
# imitation-counterbalanced value) reward-hacked into a not-quite-fallen
# contortion instead of walking once ported to Stage 1 (no imitation). The
# base class above now defaults to 2.0 (a 10x cut); these two add more
# points along the same axis so the sweep can compare 2 / 5 / 10 (20 already
# has a completed data point from the pre-sweep run at
# logs/rsl_rl/open_duck_mini_v2_joystick/2026-07-26_03-50-34) rather than
# guessing a single value. Registered as separate gym tasks in __init__.py
# so all three can train concurrently on one GPU.
@configclass
class JoystickEnvCfg_Alive5(JoystickEnvCfg):
    alive_scale = 5.0


@configclass
class JoystickEnvCfg_Alive10(JoystickEnvCfg):
    alive_scale = 10.0
