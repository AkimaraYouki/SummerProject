"""play.py with the velocity command PINNED instead of randomly resampled.

rsl_rl's play.py drives the env's own command sampler, so the robot is
constantly switching between forward/backward/lateral/turn targets and you
cannot tell from the video which one it is currently reacting to. For judging
"does it walk forward", pin the command.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--cmd_x", type=float, default=0.15)
parser.add_argument("--cmd_y", type=float, default=0.0)
parser.add_argument("--cmd_yaw", type=float, default=0.0)
parser.add_argument("--seconds", type=float, default=1e9)
parser.add_argument("--cycle", action="store_true",
                    help="정지/전진/후진/좌/우/회전을 --hold 초씩 순환한다")
parser.add_argument("--hold", type=float, default=8.0, help="--cycle에서 명령 하나를 유지할 초")
parser.add_argument("--overlay", action="store_true", help="path frame·레퍼런스 접지·명령·자기충돌을 3D로 겹쳐 그린다")
parser.add_argument("--ghost", action="store_true",
                    help="레퍼런스 관절각을 입힌 로봇을 옆에 함께 띄운다 (meshcat 방식)")
parser.add_argument("--ghost_offset", type=float, default=0.30, help="고스트를 y로 띄울 거리 (m). 0이면 겹쳐 그린다")
parser.add_argument("--ghost_opacity", type=float, default=0.35, help="고스트 불투명도 (1.0이면 불투명)")
parser.add_argument("--hud", action="store_true", help="뷰포트 우상단에 명령·실제속도·관절오차 HUD를 띄운다")
parser.add_argument("--selfcol_thresh", type=float, default=0.085,
                    help="다리-몸통 링크 원점 거리가 이 값 아래면 빨간 구 표시 (m)")
parser.add_argument("--joystick", nargs="?", const="/dev/input/js0", default=None,
                    metavar="DEV",
                    help="Xbox 패드로 실시간 조종한다 (기본 장치 /dev/input/js0). "
                         "--cycle 보다 우선한다")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from open_duck_mini_isaaclab.agents.rsl_rl_compat import (  # noqa: E402
    build_runner,
    load_checkpoint,
)
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    JoystickPPORunnerCfg,
    JoystickPPORunnerCfg_Gamma097,
)

# The runner cfg must match the one the checkpoint was TRAINED with, not just
# the env cfg: Walk9 trains with the upstream network (512,256,128) while every
# other variant uses (256,128,64), and loading across them fails with a bare
# size-mismatch on actor.0.weight.
_TASK_TO_RUNNER = {
    "Isaac-OpenDuckMini-Joystick-Walk9-v0": JoystickPPORunnerCfg_Gamma097,
    "Isaac-OpenDuckMini-Joystick-Walk9G97-v0": JoystickPPORunnerCfg_Gamma097,
    "Isaac-OpenDuckMini-Joystick-Path-v0": JoystickPPORunnerCfg_Gamma097,
}

from open_duck_mini_isaaclab.tasks.velocity import joystick_env_cfg as _cm  # noqa: E402
from open_duck_mini_isaaclab.joint_order import (  # noqa: E402
    ACTUATOR_JOINT_NAMES, ACT_LEG_JOINT_IDX, REF_LEG_JOINT_IDX, READY_BASE_HEIGHT,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

_MAP = {
    "Isaac-OpenDuckMini-Joystick-v0": "JoystickEnvCfg",
    "Isaac-OpenDuckMini-Joystick-Walk3-v0": "JoystickEnvCfg_Walk3",
    "Isaac-OpenDuckMini-Joystick-Walk6-v0": "JoystickEnvCfg_Walk6",
    "Isaac-OpenDuckMini-Joystick-Walk9-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Walk9G97-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Path-v0": "JoystickEnvCfg_Path",
    "Isaac-OpenDuckMini-Joystick-Upstream-v0": "JoystickEnvCfg_Upstream",
}
env_cfg = getattr(_cm, _MAP[args_cli.task])()
env_cfg.scene.num_envs = args_cli.num_envs
# 재생에는 외란을 항상 끈다. push_robot은 5~10초마다 ±1 m/s로 몸통을 밀어서
# 학습 때는 강건성을 주지만, 눈으로 보행을 판단할 때는 정책의 문제인지 외력
# 때문인지 구분할 수 없게 만든다.
env_cfg.events.push_robot = None
print("[play] 외란(push_robot) 비활성화", flush=True)
# ── 레퍼런스 고스트 ─────────────────────────────────────────────────────
# 구/정육면체 마커로는 "레퍼런스가 어떤 자세를 원하는지"가 안 보인다.
# reference_motion_generator의 meshcat 재생처럼, 레퍼런스 관절각을 그대로
# 입힌 로봇을 한 대 더 띄워 옆에 나란히 세운다. 겹쳐 놓으면 z-fighting으로
# 지저분해지므로 y로 --ghost_offset 만큼 띄운다.
if args_cli.ghost:
    from open_duck_mini_isaaclab.robot_cfg import OPEN_DUCK_MINI_V2_CFG
    _g = OPEN_DUCK_MINI_V2_CFG.replace(prim_path="/World/envs/env_.*/RefGhost")
    _g.spawn.articulation_props.enabled_self_collisions = False
    # 표시용이므로 물리 개입을 최대한 뺀다. 중력만 껐을 때는 지면 충돌이
    # 남아 PhysX가 매 스텝 밀어내면서 고스트가 퉁퉁 튀었다. 충돌을 끄고
    # 액추에이터도 비워야 PD 제어가 우리가 써 넣는 자세와 싸우지 않는다.
    # (kinematic_enabled는 articulation 루트에 쓸 수 없어 생성이 실패한다.)
    _g.spawn.rigid_props.disable_gravity = True
    if _g.spawn.collision_props is None:
        _g.spawn.collision_props = sim_utils.CollisionPropertiesCfg()
    _g.spawn.collision_props.collision_enabled = False
    # 액추에이터를 비우면 관절이 완전히 자유가 되어 더 심하게 흔들린다.
    # 자세를 써 넣는 주기는 0.02 s인데 물리는 그 사이 10번 적분하므로,
    # 그 구간을 붙잡아 줄 구동이 없으면 관절이 제멋대로 낭창거린다.
    # 대신 아주 뻣뻣한 PD로 레퍼런스를 따라가게 한다 -- 충돌도 중력도 없으니
    # 사실상 정확히 추종하면서 움직임은 부드럽다.
    import copy as _copy
    for _k in list(_g.actuators.keys()):
        _a = _copy.deepcopy(_g.actuators[_k])
        _a.stiffness = 2000.0
        _a.damping = 100.0
        if hasattr(_a, "effort_limit_sim"):
            _a.effort_limit_sim = 1.0e6
        if hasattr(_a, "velocity_limit_sim"):
            _a.velocity_limit_sim = 1.0e6
        _g.actuators[_k] = _a
    # 반투명 머티리얼은 spawn 단계에서 지정한다. 생성 후 스테이지를 훑어
    # 바인딩하려 했더니 0개가 잡혔는데, 로봇 USD가 instanceable 참조라
    # Traverse()가 내부 메시로 들어가지 못하기 때문이다.
    if args_cli.ghost_opacity < 1.0:
        _g.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.35, 0.75, 1.0),
            opacity=args_cli.ghost_opacity,
            roughness=0.6,
        )
    env_cfg.scene.ref_ghost = _g

env = gym.make(args_cli.task, cfg=env_cfg)

agent_cfg = _TASK_TO_RUNNER.get(args_cli.task, JoystickPPORunnerCfg)()
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
dt = u.step_dt
print(f"[play] cmd pinned to ({args_cli.cmd_x:+.2f}, {args_cli.cmd_y:+.2f}, {args_cli.cmd_yaw:+.2f})", flush=True)

# 순환 모드용 명령 목록. gait_compare.py의 조건과 같은 값이라 영상과 측정치를
# 직접 대응시켜 볼 수 있다.
CYCLE = [
    ("STOP",  0.00,  0.00, 0.0),
    ("FWD ",  0.15,  0.00, 0.0),
    ("BACK", -0.15,  0.00, 0.0),
    ("LEFT",    0.00,  0.20, 0.0),
    ("RIGHT",    0.00, -0.20, 0.0),
    ("TURN",  0.00,  0.00, 1.0),
]

# ── 오버레이 마커 ────────────────────────────────────────────────────────
# path frame은 명령 속도를 적분한 "가야 할 곳"이라 숫자로만 보면 감이 안 온다.
# 로봇과 나란히 그려두면 얼마나 벌어졌는지가 바로 보인다.
markers = None
_HAS_PATH = bool(getattr(env_cfg, "use_path_frame", False))
if args_cli.overlay:
    markers = {
        # 레퍼런스가 "딛으라"는 발 = 초록 원반(발자국 패드)
        "ref_stance": VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/ref_stance",
            markers={"m": sim_utils.CylinderCfg(
                radius=0.035, height=0.004, axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.9, 0.3)))},
        )),
        # 레퍼런스가 "들라"는 발 = 빨간 원뿔(위로 들어올리라는 뜻)
        "ref_swing": VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/ref_swing",
            markers={"m": sim_utils.ConeCfg(
                radius=0.022, height=0.05, axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.2)))},
        )),
        # 명령 방향 = 파란 정육면체. 위치로만 표현하므로 회전 오류가 없다
        # (처음엔 arrow_x.usd를 썼는데 에셋 기본 축이 달라 몸통을 관통하는
        #  지느러미처럼 그려졌다).
        "cmd": VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/cmd_dir",
            markers={"m": sim_utils.CuboidCfg(
                size=(0.035, 0.035, 0.035),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 1.0)))},
        )),
        # 레퍼런스가 발을 놓아야 할 위치 = 노란 원반. 실제 발과 얼마나
        # 어긋나는지가 곧 관절 추종 오차의 공간적 표현이다.
        "ref_foot": VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/ref_foot",
            markers={"m": sim_utils.CylinderCfg(
                radius=0.030, height=0.003, axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1)))},
        )),
    }
    # path frame 마커는 그 기능을 쓰는 정책일 때만 만든다. 만들어두고
    # visualize()를 안 하면 원점에 그대로 렌더돼 "좌표축이 grid에 붙어있는"
    # 것처럼 보인다.
    if _HAS_PATH:
        markers["path"] = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/path_frame",
            markers={"frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.22, 0.22, 0.22))},
        ))
    print(f"[play] 오버레이: 초록원반=레퍼런스 딛기 · 빨간원뿔=레퍼런스 들기 · "
          f"파란정육면체=명령방향 · 노란원반=레퍼런스 발 목표위치"
          f"{' · 좌표축=path frame' if _HAS_PATH else ' (path frame 없음: 이 정책은 미사용)'}", flush=True)

# 다리-몸통 근접 판정용 인덱스. 접촉력이 아니라 링크 원점 간 거리를 쓰므로
# 임계값은 형상에 따라 보정이 필요하다 -- 그래서 실측 최소거리를 주기적으로
# 출력해서 --selfcol_thresh를 정할 수 있게 했다.
_TRUNK_ID, _ = env.unwrapped._robot.find_bodies(["trunk_assembly"], preserve_order=True)
_LEG_IDS, _LEG_NAMES = env.unwrapped._robot.find_bodies(
    ["knee_and_ankle_assembly.*", "hip_roll_assembly.*"], preserve_order=True)
_selfcol_min = 9.9

# 레퍼런스 발자국 = 레퍼런스 관절각을 순기구학으로 풀어 얻은 발 위치.
# 지금까지의 오버레이는 *실제* 발 위치에 색만 칠해 "언제 드는지"만 보여줬을
# 뿐, "어디에 내딛어야 하는지"는 보여주지 못했다. 베이스 프레임 기준으로
# 계산한 뒤 로봇의 실제 몸통 자세로 옮기면 목표 발 위치가 그대로 나온다.
_fk = None
if args_cli.overlay:
    try:
        import os, sys as _sys
        _sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
        from leg_fk import foot_in_trunk as _fk
        print("[play] 레퍼런스 발자국 FK 활성 (leg_fk, pinocchio 대비 0.01 mm)", flush=True)
    except Exception as e:
        print(f"[play] FK 비활성: {type(e).__name__} {e}", flush=True)


def _ref_foot_world():
    """레퍼런스 관절각 -> 로봇 루트 기준 발 위치 -> 월드 좌표. [1,2,3] 또는 None.

    Isaac Sim의 파이썬에서는 pinocchio 임포트가 실패하므로(libhpp-fcl가 번들
    Assimp와 심볼 충돌) URDF 변환을 박아둔 scripts/leg_fk.py를 쓴다.
    """
    if _fk is None:
        return None
    import math as _mt
    ref = u._current_reference_motion[0, 0:14].detach().cpu().numpy()
    ja = {nm: float(ref[i]) for i, nm in enumerate(ACTUATOR_JOINT_NAMES)}
    rel = _fk(ja)
    yaw = math_utils.euler_xyz_from_quat(u._robot.data.root_quat_w)[2][0].item()
    c, sn = _mt.cos(yaw), _mt.sin(yaw)
    base = u._robot.data.root_pos_w[0].detach().cpu().numpy()
    out = []
    for side in ("left", "right"):
        r = rel[side]
        out.append([base[0] + c * r[0] - sn * r[1], base[1] + sn * r[0] + c * r[1], base[2] + r[2]])
    return torch.tensor([out], device=u.device, dtype=torch.float32)


def _draw_overlay():
    if markers is None:
        return
    global _selfcol_min
    org = u._terrain.env_origins
    # path frame
    if _HAS_PATH:
        pp = torch.cat([u._path_pos + org[:, :2], org[:, 2:3] + 0.02], dim=-1)
        pq = math_utils.quat_from_euler_xyz(
            torch.zeros_like(u._path_yaw), torch.zeros_like(u._path_yaw), u._path_yaw)
        markers["path"].visualize(translations=pp, orientations=pq)
    # 레퍼런스 접지 상태를 실제 발 위치 위에 띄운다
    fp = u._robot.data.body_pos_w[:, u._feet_ids].clone()
    fp[:, :, 2] += 0.07
    ref_c = (u._current_reference_motion[:, 28:30] > 0.5)
    st, sw = fp[ref_c], fp[~ref_c]
    rf = _ref_foot_world()
    if rf is not None:
        markers["ref_foot"].visualize(translations=rf.reshape(-1, 3))
    markers["ref_stance"].visualize(translations=st if st.numel() else fp[:1, 0] * 0 - 100.0)
    markers["ref_swing"].visualize(translations=sw if sw.numel() else fp[:1, 0] * 0 - 100.0)
    # 명령 방향: 로봇 몸통 기준 앞쪽 25 cm. 명령이 0이면 몸통 바로 위에 붙어
    # 있으므로 "정지"도 한눈에 구분된다.
    yaw = math_utils.euler_xyz_from_quat(u._robot.data.root_quat_w)[2]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cx_b, cy_b = u._command[:, 0], u._command[:, 1]
    nrm = torch.sqrt(cx_b**2 + cy_b**2).clamp(min=1e-6)
    scale = 0.25 * torch.tanh(nrm * 6.0)
    dx = (cy * cx_b - sy * cy_b) / nrm * scale
    dy = (sy * cx_b + cy * cy_b) / nrm * scale
    tip = u._robot.data.root_pos_w.clone()
    tip[:, 0] += dx
    tip[:, 1] += dy
    tip[:, 2] += 0.10
    markers["cmd"].visualize(translations=tip)

    # 다리-몸통 최소 거리는 로그로만 남긴다. 3D 마커로 띄웠더니 화면을 가릴
    # 만큼 크고, 임계값(링크 원점 간 거리)에 근거도 없었다.
    tp = u._robot.data.body_pos_w[:, _TRUNK_ID]
    lp = u._robot.data.body_pos_w[:, _LEG_IDS]
    _selfcol_min = min(_selfcol_min, float(torch.norm(lp - tp, dim=-1).min()))


# ── 뷰포트 HUD ──────────────────────────────────────────────────────────
# 3D 마커는 "어디가 어긋났는지"를 보여주지만 "얼마나"는 숫자로 봐야 한다.
# 뷰포트 오버레이는 렌더 결과에 포함되므로 WebRTC 스트림에도 그대로 나온다.
_hud = None
_hud_plots = {}
_HIST = 150  # 최근 3초 (0.02 s/step)
_hist = {"vx": [], "vy": [], "yaw": [], "err": []}
if args_cli.hud:
    try:
        import omni.ui as ui
        import omni.kit.viewport.utility as vp_utils
        # 기본 폰트에는 한글 글리프가 없어 네모로 깨진다. 랩PC에 나눔고딕이
        # 설치돼 있으므로 스타일에서 직접 지정한다.
        _FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        _fst = {"font_size": 15, "color": 0xFFFFFFFF, "font": _FONT}
        _vp = vp_utils.get_active_viewport_window()
        _frame = _vp.get_frame("odm_hud")
        with _frame:
            with ui.HStack():
                ui.Spacer()
                with ui.VStack(width=ui.Pixel(360)):
                    ui.Spacer(height=8)
                    with ui.ZStack():
                        ui.Rectangle(style={"background_color": 0xB0000000, "border_radius": 6})
                        with ui.VStack(spacing=2):
                            ui.Spacer(height=8)
                            _hud = ui.Label("", style=_fst, alignment=ui.Alignment.LEFT_TOP)
                            ui.Spacer(height=6)
                            for key, lab, col in (
                                ("vx", "vx", 0xFF44AAFF),
                                ("vy", "vy", 0xFF44FF88),
                                ("yaw", "yaw", 0xFFFF8844),
                                ("err", "jnt err", 0xFF8888FF),
                            ):
                                with ui.HStack(height=ui.Pixel(26)):
                                    ui.Spacer(width=8)
                                    ui.Label(lab, width=ui.Pixel(70), style=_fst)
                                    _hud_plots[key] = ui.Plot(
                                        ui.Type.LINE, -1.0, 1.0, *([0.0] * _HIST),
                                        width=ui.Pixel(250), height=ui.Pixel(24),
                                        style={"color": col, "background_color": 0x30FFFFFF},
                                    )
                            ui.Spacer(height=8)
                    ui.Spacer()
                ui.Spacer(width=12)
        print("[play] HUD 활성 (우상단, 실시간 그래프 포함)", flush=True)
    except Exception as e:
        print(f"[play] HUD 비활성: {type(e).__name__} {e}", flush=True)


_ghost = None
if args_cli.ghost:
    _ghost = env.unwrapped.scene["ref_ghost"]
    _g_ids, _ = _ghost.find_joints(ACTUATOR_JOINT_NAMES, preserve_order=True)
    print(f"[play] 레퍼런스 고스트 활성 (y+{args_cli.ghost_offset:.2f} m)", flush=True)


@torch.inference_mode()
def _update_ghost():
    """고스트에 레퍼런스 관절각을 그대로 입히고 실제 로봇 옆에 세운다.

    inference_mode 데코레이터가 필요하다. env.step()이 inference_mode 안에서
    돌면서 Isaac Lab의 버퍼가 inference 텐서가 되는데, 그런 텐서는 같은
    컨텍스트 안에서만 in-place 수정이 허용된다.
    """
    if _ghost is None:
        return
    ref = u._current_reference_motion[:, 0:14]
    jp = _ghost.data.default_joint_pos.clone()
    jp[:, _g_ids] = ref
    ids = torch.arange(_ghost.num_instances, device=u.device)
    # 뻣뻣한 PD 목표로 준다 (텔레포트가 아니라 구동)
    _ghost.set_joint_position_target(jp, env_ids=ids)
    _ghost.write_data_to_sim()
    # 몸통은 수평·일정 높이로 고정한다. 처음엔 실제 로봇의 root_state를 그대로
    # 복사해 y로만 옮겼는데, 그러면 실제 로봇이 흔들리는 만큼 고스트도 같이
    # 흔들려 "레퍼런스가 원하는 자세"가 아니라 "실제 로봇의 흔들림 + 레퍼런스
    # 관절각"이 보인다. 레퍼런스 관절각은 애초에 수평인 몸통 기준으로 생성된
    # 값이므로, meshcat 재생처럼 몸통을 붙잡아 두고 다리만 움직이는 것이 맞다.
    root = torch.zeros(_ghost.num_instances, 7, device=u.device)
    root[:, 0] = u._robot.data.root_pos_w[:, 0]
    root[:, 1] = u._robot.data.root_pos_w[:, 1] + args_cli.ghost_offset
    root[:, 2] = u._terrain.env_origins[:, 2] + READY_BASE_HEIGHT
    yaw = u._path_yaw if _HAS_PATH else math_utils.euler_xyz_from_quat(u._robot.data.root_quat_w)[2]
    z = torch.zeros_like(yaw)
    root[:, 3:7] = math_utils.quat_from_euler_xyz(z, z, yaw)
    _ghost.write_root_pose_to_sim(root, env_ids=ids)
    _ghost.write_root_velocity_to_sim(torch.zeros(_ghost.num_instances, 6, device=u.device), env_ids=ids)


def _update_hud(tag, cx, cy, cw):
    if _hud is None:
        return
    v = u._robot.data.root_lin_vel_b[0]
    w = u._robot.data.root_ang_vel_b[0, 2]
    rf = u._current_reference_motion
    jp = u._robot.data.joint_pos[:, u._joint_ids][:, ACT_LEG_JOINT_IDX]
    ref_jp = rf[:, 0:14][:, REF_LEG_JOINT_IDX]
    err = float(torch.sum((jp - ref_jp) ** 2, dim=-1)[0])
    deg = (err / 10.0) ** 0.5 * 57.2958
    contact = u._get_foot_contact()[0]
    ref_c = (rf[0, 28:30] > 0.5).float()
    lines = [
        f"CMD       {tag}",
        f"          vx {cx:+.2f}  vy {cy:+.2f}  yaw {cw:+.2f}",
        "",
        f"ACTUAL    vx {v[0]:+.3f}  vy {v[1]:+.3f}  yaw {w:+.3f}",
        f"TRACKING  {100*abs(v[0])/max(abs(cx),1e-9):>5.0f}%" if abs(cx) > 1e-6 else "TRACKING     --",
        "",
        f"JOINT ERR {deg:.1f} deg  (sum {err:.3f})",
        f"CONTACT   actual {int(contact.sum())}/2   ref {int(ref_c.sum())}/2",
    ]
    if _HAS_PATH:
        pe = u._path_error()[0]
        yaw_err = float(torch.atan2(pe[2], pe[1])) * 57.2958
        lines += ["", f"PATH ERR  lat {float(pe[0])*1000:+.0f} mm   yaw {yaw_err:+.1f} deg"]
    _hud.text = "\n".join(lines)

    # 최근 3초 궤적. 숫자만으로는 "순간적으로 얼마나 튀는지"가 안 보이는데,
    # 좌우 명령에서 요가 ±1.5 rad/s로 요동치면서 평균은 0에 가까웠던 것이
    # 바로 그 사각지대였다.
    for k, val, rng in (("vx", float(v[0]), 0.3), ("vy", float(v[1]), 0.3),
                        ("yaw", float(w), 2.0), ("err", deg / 20.0, 1.0)):
        h = _hist[k]
        h.append(max(-1.0, min(1.0, val / rng)))
        if len(h) > _HIST:
            del h[0]
        if k in _hud_plots and len(h) == _HIST:
            _hud_plots[k].set_data(*h)


# ── 조이스틱 ────────────────────────────────────────────────────────────
# 명령 범위는 env_cfg에서 그대로 가져온다. 스틱을 끝까지 밀었을 때 학습에서 본
# 상한에 정확히 닿아야 하고, 넘어가면 정책이 본 적 없는 명령이 된다.
_pad = None
if args_cli.joystick:
    from open_duck_mini_isaaclab.joystick_input import (  # noqa: E402
        Gamepad,
        GamepadUnavailable,
        command_from_gamepad,
    )

    try:
        _pad = Gamepad(args_cli.joystick)
        print(f"[play] 조이스틱: {args_cli.joystick}  "
              f"왼쪽스틱=전후·좌우, 오른쪽스틱 X=회전, A=비상정지", flush=True)
        print(f"[play] 명령 범위: vx {env_cfg.lin_vel_x_range}  "
              f"vy {env_cfg.lin_vel_y_range}  yaw {env_cfg.ang_vel_yaw_range}", flush=True)
    except GamepadUnavailable as exc:
        print(f"[play] !! {exc}", flush=True)
        print("[play] 조이스틱 없이 계속합니다.", flush=True)

obs = env.get_observations()
t_end = time.time() + args_cli.seconds
step = 0
hold_steps = max(1, int(args_cli.hold / dt))
cur_idx = -1
while simulation_app.is_running() and time.time() < t_end:
    t0 = time.time()
    if _pad is not None:
        # 조이스틱이 최우선. poll()은 밀린 이벤트만 훑고 즉시 돌아오므로
        # 60Hz 루프를 붙잡지 않는다.
        _pad.poll()
        cx, cy, cw = command_from_gamepad(
            _pad, env_cfg.lin_vel_x_range, env_cfg.lin_vel_y_range, env_cfg.ang_vel_yaw_range
        )
    elif args_cli.cycle:
        idx = (step // hold_steps) % len(CYCLE)
        if idx != cur_idx:
            cur_idx = idx
            nm, cx, cy, cw = CYCLE[idx]
            print(f"\n[play] ▶ {nm}  cmd=({cx:+.2f}, {cy:+.2f}, {cw:+.2f})", flush=True)
        _, cx, cy, cw = CYCLE[idx]
    else:
        cx, cy, cw = args_cli.cmd_x, args_cli.cmd_y, args_cli.cmd_yaw
    u._command[:, 0] = cx
    u._command[:, 1] = cy
    u._command[:, 2] = cw
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    _draw_overlay()
    _update_ghost()
    # 조이스틱일 때 cur_idx 는 -1 이라 CYCLE[cur_idx] 를 쓰면 엉뚱하게 TURN 이 뜬다.
    mode_tag = "JOY " if _pad is not None else (CYCLE[cur_idx][0] if args_cli.cycle else "고정")
    if args_cli.hud:
        _update_hud(mode_tag, cx, cy, cw)
    step += 1
    if step % 100 == 0:
        v = u._robot.data.root_lin_vel_b[0, :2]
        w = u._robot.data.root_ang_vel_b[0, 2]
        tag = mode_tag
        extra = f"  다리-몸통 최소 {_selfcol_min*1000:.0f}mm" if args_cli.overlay else ""
        print(f"[play] {tag:4} vx={v[0]:+.3f} vy={v[1]:+.3f} yaw={w:+.3f}  (cmd {cx:+.2f},{cy:+.2f},{cw:+.2f}){extra}", flush=True)
    sleep = dt - (time.time() - t0)
    if sleep > 0:
        time.sleep(sleep)
env.close()
simulation_app.close()
