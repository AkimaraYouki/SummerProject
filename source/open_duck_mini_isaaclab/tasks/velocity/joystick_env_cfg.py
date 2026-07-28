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
    #
    # Raised 0.6->0.75 on 2026-07-27: contact_diagnostic.py measured
    # imitation_v4's converged policy holding a perfectly stable crouch at
    # base_z=0.133-0.134 (ratio ~0.69 of HOME_BASE_HEIGHT=0.193) with ZERO
    # trunk/head contact force for 180+ consecutive steps — a pose that
    # evades all three termination conditions (not flipped, above the old
    # 0.6 height ratio, no trunk/head contact) simultaneously. 0.75 puts
    # that specific crouch below the new threshold.
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


# ── alive_scale x w_joint_pos sweep (2026-07-27) ────────────────────────
# imitation_v2 (alive_scale=20, w_joint_pos=15 — Playground's original
# combo) FAILED with MORE violent twitching than imitation_v1 (43.4
# foot-contact toggles/10s vs 11.7), even though the reference-motion data
# itself was fully fixed by then (knee ROM, symmetry, joint limits, drift
# bias) — see docs/training_log.md Run 7. That rules the data out, leaving
# these two reward weights as the remaining suspects: alive_scale may still
# dominate the per-step reward ceiling enough to make "twitch but don't
# fall" cheap, and/or w_joint_pos=15 may be too harsh against the new
# reference's wider knee ROM (up to 120 deg), pushing the policy into
# high-frequency chatter instead of smooth tracking. Four combos, chosen to
# isolate each factor plus a jointly-reduced point, rather than the full
# 4x3 cross product (too many runs for one GPU in reasonable time):
#   A10J10 — both moderately cut, the "compromise" guess
#   A5J15  — alive cut hard, w_joint_pos untouched (isolates alive_scale)
#   A20J5  — alive untouched, w_joint_pos cut hard (isolates w_joint_pos)
#   A5J5   — both cut hard
@configclass
class JoystickEnvCfg_A10J10(JoystickEnvCfg):
    alive_scale = 10.0
    imitation_w_joint_pos = 10.0


@configclass
class JoystickEnvCfg_A5J15(JoystickEnvCfg):
    alive_scale = 5.0
    imitation_w_joint_pos = 15.0


@configclass
class JoystickEnvCfg_A20J5(JoystickEnvCfg):
    alive_scale = 20.0
    imitation_w_joint_pos = 5.0


# Disney's BD-X paper doesn't use RSI (confirmed 2026-07-28 against the
# actual paper — see joystick_env_cfg.py's use_rsi docstring). This variant
# is A20J5 (already Disney's own alive_scale=20) with RSI turned off, for a
# direct on/off comparison against imitation_v6/v7 on identical everything
# else (contact-termination + crouch-fix stay on — those match/extend
# Disney's own paper's contact-based termination, not a deviation from it).
@configclass
class JoystickEnvCfg_A20J5_NoRSI(JoystickEnvCfg_A20J5):
    use_rsi = False


# Bounded (2026-07-28) — the reward-shaping fix derived from
# reward_breakdown_v2.py's measurement on imitation_v8 (imitation -1.01/step
# vs alive +0.40/step => 81.6% of steps clamped to 0, no gradient). Three
# coupled changes, all following from that one measurement:
#   1. imitation_bounded_joint_pos=True — joint_pos becomes exp(-w*err^2),
#      bounded [0,1] like every other tracking term, so it can no longer
#      drive the sum negative on its own.
#   2. imitation_w_joint_pos 5.0 -> 0.25 — under exp(), w is a *sharpness*,
#      not a linear weight. At v8's measured error (sum err^2 ~ 10.6 rad^2)
#      w=5 would saturate exp() to ~0.0000 and hand back a flat gradient,
#      which is a different way of learning nothing. w=0.25 gives 0.07 there
#      and rises smoothly as tracking improves (0.61 at err^2=2, 0.88 at 0.5).
#   3. imitation_scale 1.0 -> 4.0 — bounding joint_pos caps the whole
#      imitation term near 6 raw, against alive's 20, which would re-create
#      the alive-dominated stand-still reward hacking that killed v1-v5.
#      x4 puts imitation's ceiling (~24) back on par with alive.
# RSI stays off (v7 vs v8 showed no real difference; user wants Disney's
# simpler recipe), so this isolates the reward-shaping change alone.
@configclass
class JoystickEnvCfg_A20J5_Bounded(JoystickEnvCfg_A20J5_NoRSI):
    imitation_bounded_joint_pos = True
    # 0.25 (what imitation_v9 ran with) was derived from imitation_v8's error
    # scale, back when the robot initialized straight-legged and sat ~79 deg
    # per joint from the reference. Now that it initializes into READY the
    # error is ~8.7 deg, and scripts/reward_at_ready.py measured joint_pos_rew
    # pinned at 0.944/1.0 there -- near-maximum for simply holding the neutral
    # pose, so the term barely discriminates and gives little reason to
    # actually step. scripts/ref_stats.py puts the reference's own joint_pos
    # spread at 0.234 rad^2, so k=1/0.234=4.28 is the value that lands a
    # typical error near exp(-1)~0.37, i.e. in the responsive middle of the
    # curve. Rounded to 4.0: holding READY now scores ~0.40 while tracking the
    # gait properly scores ~1.0.
    imitation_w_joint_pos = 4.0
    imitation_scale = 4.0


# Walk-incentive fix (2026-07-28). imitation_v10 reproduced the reward hack
# predicted from the term-by-term audit: the policy stood at READY, kept both
# feet planted, and trembled just enough not to fall. Measured freebies for
# doing exactly that: alive 100%, contact 67%, joint_pos 51%, lin_vel_z 82%,
# ang_vel_xy 100% -- while action_rate/torques/joint_vel all actively punish
# moving. Standing collected 0.734/step against a perfect walk's 1.050, i.e.
# 70%, for none of the risk. Three coupled changes, each aimed at one measured
# cause:
#   1. swing_only_contact — standing scores 0 on the contact term instead of
#      1.34/2.0, so the term can only be earned by actually lifting feet.
#   2. alive_scale 20 -> 10 — this is the single largest term and it is
#      identical whether the robot walks or stands, so it dilutes every signal
#      that does discriminate. (The earlier sweep showed 5 collapses training
#      and 10 trains stably, so 10 is the safe floor.)
#   3. tracking_lin_vel_scale 2.5 -> 10.0 — command following capped at
#      0.05/step against alive's 0.40/step, 8x smaller, so "go where the
#      joystick says" was nearly invisible in the objective. This is the whole
#      point of the task; it should not be the smallest term.
@configclass
class JoystickEnvCfg_Walk(JoystickEnvCfg_A20J5_Bounded):
    imitation_swing_only_contact = True
    alive_scale = 10.0
    tracking_lin_vel_scale = 10.0
    # Legs-only learning: freeze the head at READY and stop issuing random
    # head-pose commands, so neither the action noise nor the command channel
    # spends capacity on a subsystem the gait reward ignores anyway.
    lock_head_joints = True
    neck_pitch_range = (0.0, 0.0)
    head_pitch_range = (0.0, 0.0)
    head_yaw_range = (0.0, 0.0)
    head_roll_range = (0.0, 0.0)


@configclass
class JoystickEnvCfg_A5J5(JoystickEnvCfg):
    alive_scale = 5.0
    imitation_w_joint_pos = 5.0


# A30J25 — 2026-07-27, user-directed reversal of the A20J5 direction after
# imitation_v3 (A20J5 full run) came back WORSE than imitation_v2 (foot
# toggle 79.1 vs 43.4). User's read: both alive_scale and w_joint_pos
# should go UP, not down — opposite of what the sweep evidence pointed to
# (alive_scale=5 collapsed training; w_joint_pos=15 was imitation_v2's
# suspected culprit). Untested territory above alive_scale=20 — logged
# here, not silently assumed correct.
@configclass
class JoystickEnvCfg_A30J25(JoystickEnvCfg):
    alive_scale = 30.0
    imitation_w_joint_pos = 25.0


# A30J25Im2 — pre-approved fallback (imitation_v6) if imitation_v5 (A30J25)
# also fails. User's next lever, in order: raise imitation-related weights
# further (imitation_scale 1.0->2.0, doubling reward_imitation()'s overall
# contribution on top of A30J25's already-raised w_joint_pos) and raise PPO
# initial exploration noise (see rsl_rl_ppo_cfg.py's JoystickPPORunnerCfg_N2,
# init_noise_std 1.0->2.0) to try to escape the observed local-minimum
# ("joint locks near a limit angle then pops/flings outward") failure mode.
@configclass
class JoystickEnvCfg_A30J25Im2(JoystickEnvCfg_A30J25):
    imitation_scale = 2.0


# Walk2 / imitation_v12 (2026-07-28). imitation_v11 failed identically to
# v10 -- user watched it over WebRTC: "또 부들부들거림 발 안뜸... 걸을려고 안함".
# scripts/imit_internals2.py on v11's model_300 measured WHY, and it was not
# where the v11 changes had aimed:
#
#   base speed 0.064 m/s   (reference: 0.265 m/s)  <- barely moving
#   lin_vel_z  +0.954/1.0    ang_vel_xy +0.220/0.5   lin_vel_xy +0.556/1.0
#   joint_pos  +0.379/1.0    ang_vel_z  +0.258/0.5   contact    +0.205/~0.63
#   joint_vel  -0.056        (the term I had suspected -- 2% of the total)
#
# ~92% of the raw imitation reward was collectable while standing still. The
# decisive one is lin_vel_xy, the term whose entire job is to price walking:
# at sharpness k=8, missing the reference speed by 100% (err^2 = 0.265^2 =
# 0.070) still pays exp(-0.56) = 0.57. Standing and walking were nearly
# indistinguishable to the reward.
#
# Changes, each tied to one measured number above:
#   1. k_lin_vel_xy 8 -> 20 — standing now scores exp(-20*0.070) = 0.25
#      instead of 0.56, while walking at the reference speed still scores
#      ~1.0. (Not the "exact" k~43 that would drive standing to 0.05: that
#      also flattens the gradient everywhere below half speed, which is the
#      failure mode the w_joint_pos=5 experiment already demonstrated.)
#   2. w_lin_vel_z 1.0 -> 0.1 — 0.954/1.0 free, the single largest freebie,
#      and vertical velocity carries almost no gait information anyway.
#   3. w_ang_vel_xy 0.5 -> 0.1 — scripts/ref_stats.py puts the reference's
#      own roll/pitch-rate spread at 0.0000, i.e. the term has no
#      discriminating power by construction.
#   4. w_contact 1.0 -> 2.0 — with swing_only_contact this is now the ONLY
#      term that standing cannot earn, so it should not also be the smallest.
#   5. alive_scale 10 -> 3 — same reasoning as v11's 20->10, one step
#      further now that termination (not alive reward) is what prevents
#      falling.
#   6. use_rsi back ON — the v7-vs-v8 RSI comparison that concluded "no real
#      difference" ran while default_joint_pos was still HOME, i.e. while the
#      reference pose sat 8 sigma outside the policy's action distribution;
#      RSI was initializing into poses the policy could not hold under ANY
#      action, so that test measured nothing. Now that READY makes the
#      reference reachable, RSI does what it is for: starting episodes
#      mid-stride is the standard escape from exactly the local optimum
#      observed here (sitting at the gait's mean pose, which is what READY
#      is, because partial out-of-phase tracking scores worse than not
#      moving at all).
#
# Predicted effect, applying the new coefficients to v11's measured behavior:
# standing 2.52 -> 1.38 raw, perfect walk 4.58 -> 3.90, so the standing-to-
# walking ratio drops 55% -> 35%. Verify with reward_at_ready.py before
# trusting the run.
@configclass
class JoystickEnvCfg_Walk2(JoystickEnvCfg_Walk):
    imitation_k_lin_vel_xy = 20.0
    imitation_w_lin_vel_z = 0.1
    imitation_w_ang_vel_xy = 0.1
    imitation_w_contact = 2.0
    alive_scale = 3.0
    use_rsi = True


# Walk3 / imitation_v13 (2026-07-28). v12 plateaued from iter ~350 to ~1066
# (per-step 0.117 -> 0.112, episode length 214 -> 219, both flat) and the user
# watched it: forward motion too slow, feet chattering against the ground.
# Measured at iter 400: forward 0.097 m/s against a 0.15 command (65%), contact
# toggles 144-319/10s against v6's 29.4 best. The chatter traces to a flaw in
# v11's own swing_only_contact: it pays for lifting a foot the reference wants
# lifted but costs nothing for lifting one it wants planted, so flickering both
# feet buys overlap with the swing phase. w_stance_violation closes that.
@configclass
class JoystickEnvCfg_Walk3(JoystickEnvCfg_Walk2):
    imitation_w_stance_violation = 1.0


# Upstream / imitation_v14 (2026-07-28). Open_Duck_Playground's joystick.py
# default_config() reward scales, verbatim, on top of the READY-pose fix.
#
# This combination has never actually been run. v1-v9 used upstream's rewards
# but could not reach the reference pose at all (the 8-sigma action-space bug),
# so their failure said nothing about the rewards; v10-v13 fixed the pose but
# each carried an accumulating stack of my own reward modifications (bounded
# joint_pos, swing-only contact, stance penalty, sharpened k, cut alive, ...),
# so none of them isolate upstream's recipe on a robot that can physically hold
# the gait. User's instruction was to work the way Disney / the Open Duck repo
# do rather than keep layering custom terms.
#
# Verbatim from playground/open_duck_mini_v2/joystick.py:
#   tracking_lin_vel=2.5  tracking_ang_vel=6.0  torques=-1.0e-3
#   action_rate=-0.5  stand_still=-0.2  alive=20.0  imitation=1.0
# and reward_imitation's own internal defaults (w_joint_pos=15 unbounded,
# k_lin_vel_xy=8, w_lin_vel_z=1.0, w_ang_vel_xy=0.5, plain contact agreement).
#
# Two deliberate non-upstream keeps, both justified independently of reward
# shaping: READY as default_joint_pos (without it the task is impossible), and
# lock_head_joints (the user's own request -- head held level in the trunk
# frame, legs-only learning). RSI off, matching upstream.
@configclass
class JoystickEnvCfg_Upstream(JoystickEnvCfg):
    use_rsi = False
    lock_head_joints = True
    neck_pitch_range = (0.0, 0.0)
    head_pitch_range = (0.0, 0.0)
    head_yaw_range = (0.0, 0.0)
    head_roll_range = (0.0, 0.0)


# Walk4 / imitation_v15 (2026-07-29). Two measurements drive this.
#
# 1. imitation_v14 (upstream-exact rewards) was stopped at iter ~900 after its
#    curve turned out to overlay imitation_v11's almost exactly -- the user
#    spotted it. reward_at_ready.py on all three configs at the READY pose
#    explains why: upstream's imitation term nets +0.004/step against alive's
#    +0.400, i.e. 1%. The unbounded joint-position penalty (-0.2023 * 15 =
#    -3.03 raw) cancels the positive tracking terms (~+3.2 raw) almost exactly,
#    so imitation collapses to ~0 and the policy trains on `alive` alone. That
#    is the reward-hack regime by construction, and it means upstream's recipe
#    cannot work in this setup regardless of the pose fix -- a second,
#    independent reason v1-v9 failed.
#
# 2. v13's own joint_pos sensitivity is now too SHARP, in the mirror-image of
#    the mistake that produced it. k=4.0 was derived from the reference's
#    intrinsic spread (0.234 rad^2) measured while the robot was standing. The
#    trained policy's actual in-motion error is 0.684 rad^2 (imit_internals2 on
#    model_2999), where exp(-4 * 0.684) = 0.065 -- saturated near zero with a
#    flat gradient, so the term cannot pull the pose back. Matching k to the
#    error that actually occurs, 1/0.684 = 1.46, puts a typical error at
#    exp(-1) ~ 0.37, in the responsive part of the curve. Rounded to 1.5.
#    Consistent with v13's other symptoms: joint RMS rose 9.9deg -> 16.5deg and
#    the joint amplitude ratio overshot to 1.31x the reference.
@configclass
class JoystickEnvCfg_Walk4(JoystickEnvCfg_Walk3):
    imitation_w_joint_pos = 1.5


# Walk5 / imitation_v16 (2026-07-29). v15's negative result, measured at
# matched model_900 checkpoints against v13:
#   actual joint error  14.0deg (v13) -> 14.3deg (v15)   [unchanged]
#   joint_pos_rew        0.148  -> 0.425                 [pure exp() rescaling]
# So sharpness alone cannot move behavior. joint_pos is one of seven roughly
# equal terms inside imitation and is bounded to [0,1], so halving the pose
# error buys only ~+0.02/step -- less than the fall risk of moving precisely.
# Pose tracking is underpriced, not mis-scaled. w_joint_pos_amp=3.0 makes it
# worth ~3x any other imitation term; sharpness stays at v15's measurement-
# matched 1.5. w_lin_vel_z drops to 0 because it measured 0.86/1.0 across every
# checkpoint of every run -- constant income carrying no gait information.
@configclass
class JoystickEnvCfg_Walk5(JoystickEnvCfg_Walk4):
    imitation_w_joint_pos_amp = 3.0
    imitation_w_lin_vel_z = 0.0


# Walk6 / imitation_v17 (2026-07-29). amp=3.0 (v16) overshot: measured at
# model_3100 against v13's model_2999,
#   joint RMS      13.0-16.5deg -> 8.2-10.4deg   (won)
#   forward        0.117 -> 0.059 m/s            (lost, halved)
#   backward       -0.066 -> -0.029              (lost)
#   left / right   0.091/-0.061 -> 0.066/-0.038  (lost)
# and the periodicity check found the policy's dominant frequency at 3.70 Hz,
# exactly 2x the 1.85 Hz gait fundamental -- mincing at double cadence with
# half the stride rather than reproducing the reference gait. Consistent with
# the slower travel: with pose tracking worth 3x every other imitation term,
# hovering near the reference pose beats actually covering ground.
# amp=2.0 splits v13 (1.0, good command tracking / poor pose) and v16 (3.0,
# good pose / poor command tracking).
@configclass
class JoystickEnvCfg_Walk6(JoystickEnvCfg_Walk5):
    imitation_w_joint_pos_amp = 2.0


# Walk7 / imitation_v18 (2026-07-29). Term-by-term audit of what standing at
# READY earns versus what the trained policy earns, using measured values
# (READY joint error 0.186 rad^2 from reward_at_ready; v16's in-motion error
# 0.3099 rad^2 from imit_internals2 at model_3100):
#
#   term          standing            walking (v16)      winner
#   joint_pos     exp(-1.5*.186)*A    exp(-1.5*.310)*A   STANDING
#   contact       0.000               +0.305             walking
#   lin_vel_xy    ~0.247              +0.300             walking
#
# READY *is* the reference gait's mean pose, so holding it scores a LOWER
# average pose error than actually traversing the gait. joint_pos therefore
# pays the policy to stand at the mean, and raising w_joint_pos_amp widens that
# gap rather than closing it -- which is why v16 (amp=3) walked worse than v13
# (amp=1), the opposite of what I intended when I raised it. At amp=2 the net
# margin for walking is only ~+0.10 raw (~0.008/step): the policy is nearly
# indifferent between walking and standing.
#
# w_contact is the one term standing cannot earn by construction (swing-only
# credit plus a stance-violation penalty, both zero when both feet stay
# planted). Doubling it 2.0 -> 4.0 leaves standing's score untouched and widens
# the walking margin from +0.305 to +0.610. amp goes back to 1.0 (v13's value,
# the best command tracking measured so far) since the amp sweep showed higher
# values actively favor standing.
@configclass
class JoystickEnvCfg_Walk7(JoystickEnvCfg_Walk6):
    imitation_w_joint_pos_amp = 1.0
    imitation_w_contact = 4.0
    # Restores what v12 wrongly removed. ang_vel_xy is
    #   exp(-2 * ||trunk roll/pitch rate - reference||^2) * w
    # and the reference's roll/pitch rate is ~0, so the term is really "keep the
    # trunk from rocking". v12 cut w 0.5 -> 0.1 on the grounds that the
    # reference's own spread measured 0.0000 and therefore carried no
    # information -- but zero spread only means it cannot tell one GAIT from
    # another; it still cleanly separates a steady execution from a lurching
    # one, which is exactly the axis that matters here. imit_internals2 on v16
    # measured 0.0137 against the 0.1 ceiling (14%), i.e. the trunk really was
    # rotating hard and the weight was too small for the policy to care. The
    # user watching v17 over WebRTC: "앞으로 뒤뚱뒤뚱 걷는데, 몸체 흔들림이 큼."
    imitation_w_ang_vel_xy = 0.5


# Walk8 / imitation_v19 (2026-07-29). User watched v13 / v16 / v17 / v18 back to
# back over WebRTC with the command pinned forward and judged v17 the best
# walker ("17이 실제로 제일 잘 걷는거같음"), then asked to push imitation
# harder from there. v13 span in place and fell; v16 minced; v17 waddles
# forward with trunk sway.
#
# So this is v17 (Walk6) with imitation_scale 4.0 -> 8.0 and nothing else.
# imitation_scale multiplies the WHOLE imitation term against alive (3.0) and
# the tracking terms, which is the right knob here: the audit put walking ahead
# of standing by only ~+0.10 raw *inside* imitation, and scaling the term
# amplifies exactly that margin while alive stays put. Contrast with
# w_joint_pos_amp, which scales only the pose sub-term and therefore favors
# holding READY -- the mistake v16 made.
@configclass
class JoystickEnvCfg_Walk8(JoystickEnvCfg_Walk6):
    imitation_scale = 8.0
