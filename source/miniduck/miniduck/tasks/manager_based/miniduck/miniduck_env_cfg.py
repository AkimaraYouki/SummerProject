from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.open_duck_mini import OPEN_DUCK_MINI_CFG
from . import mdp


@configclass
class MiniDuckSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(300.0, 300.0)),
    )
    robot: ArticulationCfg = OPEN_DUCK_MINI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=600.0),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "dof_left_knee_0", "dof_left_ankle_0",
            "dof_right_knee_0", "dof_right_ankle_0",
        ],
        scale=0.5,
        use_default_offset=True,
    )


CONTROLLED_JOINTS = [
    "dof_left_knee_0", "dof_left_ankle_0",
    "dof_right_knee_0", "dof_right_ankle_0",
]


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 제어 관절만 관측 (링크장 37개 전부 넣으면 학습 노이즈)
        joint_pos    = ObsTerm(func=mdp.joint_pos_rel,      params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)})
        joint_vel    = ObsTerm(func=mdp.joint_vel_rel,      params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)})
        root_up      = ObsTerm(func=mdp.root_up_vector_obs, params={"asset_cfg": SceneEntityCfg("robot")})
        root_ang_vel = ObsTerm(func=mdp.root_ang_vel_obs,   params={"asset_cfg": SceneEntityCfg("robot")})
        root_lin_vel = ObsTerm(func=mdp.root_lin_vel_obs,   params={"asset_cfg": SceneEntityCfg("robot")})
        base_height  = ObsTerm(func=mdp.base_pos_z,         params={"asset_cfg": SceneEntityCfg("robot")})
        proj_gravity = ObsTerm(func=mdp.projected_gravity,  params={"asset_cfg": SceneEntityCfg("robot")})
        actions      = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=mdp.reset_default_state,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RewardsCfg:
    # 선형 속도 보상: v=0이면 0점, 빠를수록 증가 (제자리 서있기 유인 제거)
    forward_vel  = RewTerm(func=mdp.forward_velocity,        weight=8.0, params={"asset_cfg": SceneEntityCfg("robot")})

    # 생존
    alive        = RewTerm(func=mdp.is_alive,               weight=0.5)

    # 자세 유지 — 뒤로 기울기 강하게 억제
    upright      = RewTerm(func=mdp.root_upright_l2,        weight=-8.0,  params={"asset_cfg": SceneEntityCfg("robot")})

    # 베이스 높이 유지
    base_height  = RewTerm(func=mdp.base_height_l2,         weight=-2.0,  params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.16})

    # 옆으로 새는 것 방지 — 뒤뚱 억제
    lateral_vel  = RewTerm(func=mdp.root_axis_vel_l2,       weight=-1.5,  params={"asset_cfg": SceneEntityCfg("robot"), "axis": 1})

    # 수직 속도 패널티 — 점프식 보행 억제
    vertical_vel = RewTerm(func=mdp.root_axis_vel_l2,       weight=-2.0,  params={"asset_cfg": SceneEntityCfg("robot"), "axis": 2})

    # 전체 각속도 패널티 — 뒤뚱 억제
    ang_vel      = RewTerm(func=mdp.root_ang_vel_l2,        weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot")})

    # yaw 회전 패널티 — 한쪽으로 틀어지기 방지
    yaw_rate     = RewTerm(func=mdp.root_axis_ang_vel_l2,   weight=-1.5,  params={"asset_cfg": SceneEntityCfg("robot"), "axis": 2})

    # pitch 패널티 — 앞뒤 흔들림 억제
    pitch_rate   = RewTerm(func=mdp.root_axis_ang_vel_l2,   weight=-2.0,  params={"asset_cfg": SceneEntityCfg("robot"), "axis": 1})

    # 좌우 보폭 대칭 — 비대칭 보행 방지
    stride_sym   = RewTerm(func=mdp.stride_symmetry,        weight=-2.0,  params={"asset_cfg": SceneEntityCfg("robot")})

    # 관절 속도 패널티
    joint_vel    = RewTerm(func=mdp.joint_vel_l1,           weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot")})

    # 관절 기본자세 이탈 패널티 — 자세 안정화
    joint_pos    = RewTerm(func=mdp.joint_pos_l2,           weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot")})

    # 한 발만 접지 — 실제 걷기 패턴 직접 보상
    no_fly        = RewTerm(func=mdp.no_fly,                weight=1.0,   params={"sensor_cfg": SceneEntityCfg("foot_contact", body_names=["foot_bottom_pla", "foot_bottom_pla_1"])})

    # 두 발 다 뜨면 패널티 — 점프 억제
    no_contact    = RewTerm(func=mdp.no_contact,            weight=-1.0,  params={"sensor_cfg": SceneEntityCfg("foot_contact", body_names=["foot_bottom_pla", "foot_bottom_pla_1"])})

    # 관절 파워 패널티 — 에너지 효율 유도
    joint_powers  = RewTerm(func=mdp.joint_powers_l1,       weight=-2e-4, params={"asset_cfg": SceneEntityCfg("robot")})

    # 좌우 다리 반대 위상 강제 — 핵심 보행 신호
    leg_antiphase = RewTerm(func=mdp.leg_antiphase,         weight=4.0,   params={"asset_cfg": SceneEntityCfg("robot")})

    # 액션 패널티
    action_l2    = RewTerm(func=mdp.action_l2,              weight=-0.001)

    # 액션 변화율 패널티
    action_rate  = RewTerm(func=mdp.action_rate_l2,         weight=-0.01)

    # 터미네이션 패널티
    terminated   = RewTerm(func=mdp.is_terminated,          weight=-10.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen   = DoneTerm(
        func=mdp.root_tilt_exceeded,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_tilt_deg": 30.0},
    )


@configclass
class MiniDuckEnvCfg(ManagerBasedRLEnvCfg):
    scene: MiniDuckSceneCfg      = MiniDuckSceneCfg(num_envs=256, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg          = ActionsCfg()
    events: EventCfg             = EventCfg()
    rewards: RewardsCfg          = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1.0 / 120.0
        self.episode_length_s = 10.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.14)
        # 센서 업데이트 주기를 물리 시뮬레이션과 동기화
        self.scene.foot_contact.update_period = self.sim.dt
