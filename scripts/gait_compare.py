"""Rolls out a policy under fixed directional commands and logs everything
needed to compare it against the reference gait, offline.

Built because the user cannot watch WebRTC — graphs are the only window into
what the policy actually does. One rollout per command direction (forward /
backward / left / right / turn), each with the command PINNED so the reference
asks for one consistent gait instead of the training-time random walk.

Logs joint position, joint velocity, base velocity (both frames), and foot
contacts, plus the reference's own value for each, so tracking error is a
subtraction rather than an estimate.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--out", type=str, default="/home/do/gait_compare.npz")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from open_duck_mini_isaaclab.agents.rsl_rl_compat import (  # noqa: E402
    build_runner,
    load_checkpoint,
)
from open_duck_mini_isaaclab.tasks.task_registry import (  # noqa: E402
    env_cfg_for,
    runner_cfg_for,
)
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401


from open_duck_mini_isaaclab.joint_order import (  # noqa: E402
    ACT_LEG_JOINT_IDX, REF_LEG_JOINT_IDX, ACTUATOR_JOINT_NAMES,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

env_cfg = env_cfg_for(args_cli.task)
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = runner_cfg_for(args_cli.task)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
# 이 스크립트는 명령을 직접 고정한다. heading 명령이 그것을 덮어쓰지
# 않도록 끈다 (2026-08-25, 이것 때문에 v78/v81 측정이 오염됐다).
if hasattr(u, "pin_commands"):
    u.pin_commands()
# Commands pinned inside the ranges the reference polynomial was actually fit
# over (lin_vel_x +-0.15, lin_vel_y +-0.2, ang_vel_yaw +-1.0) — outside them
# PolyReferenceMotion falls back to a nearest grid point and the "reference"
# being compared against would not be the one for this command.
CONDS = [
    # 정지 명령. reward_imitation은 cmd_norm <= 0.01에서 0으로 게이트되고
    # cost_stand_still만 남으므로, 학습 신호의 성격이 나머지 다섯과 완전히
    # 다르다. 조이스틱 사용에서 가장 기본인데 그동안 측정에서 빠져 있었다.
    ("stop",     0.0,   0.0,  0.0),
    ("forward",  0.15,  0.0,  0.0),
    ("backward", -0.15, 0.0,  0.0),
    ("left",     0.0,   0.2,  0.0),
    ("right",    0.0,  -0.2,  0.0),
    ("turn",     0.0,   0.0,  1.0),
    # 실사용 속도. 2026-08-18 에 사용자가 "끝까지 밀면 잘못하면 넘어져서
    # 살살 민다" 고 했고, 실기 로그에서도 |cmd_vx| 최대가 0.07 이었다.
    # 그동안 우리는 **쓰지도 못하는 0.15 에서** 최적화하고 판정해 왔다.
    # 심 자체도 0.15 에서 v59 가 0.8 % 확률로 넘어진다 — 실기 지연과 노이즈가
    # 얹히면 그게 실제 낙상이 된다.
    ("fwd_half", 0.07,  0.0,  0.0),
    ("turn_half", 0.0,  0.0,  0.5),
]
leg_names = [ACTUATOR_JOINT_NAMES[i] for i in ACT_LEG_JOINT_IDX]
print(f"[info] gait_period_steps={u._gait_period_steps} dt={u.step_dt}", flush=True)

store = {}
for name, cx, cy, cw in CONDS:
    obs, _ = env.reset()
    q, qr, dq, dqr, vb, vw, vr, ft, wz, wr, pe = [], [], [], [], [], [], [], [], [], [], []
    # 2·3 순위(안정성·효율)를 같은 롤아웃에서 재려고 2026-08-14 추가.
    # 그전에는 자세도 토크도 안 남아서, 6 방향 롤아웃이 있는데도 판정은
    # 전진 한 방향짜리 gait_quality.py 로 했다 — 1 순위를 놓치는 구조였다.
    gv, tq, fz, fv = [], [], [], []   # 투영중력 / 관절토크 / 발 z / 발 속도(몸통)
    bh, fr = [], []   # 몸통 높이 / 레퍼런스 접지
    for step in range(args_cli.num_steps):
        u._command[:, 0] = cx
        u._command[:, 1] = cy
        u._command[:, 2] = cw
        with torch.inference_mode():
            actions = policy(obs)
        obs, rew, dones, infos = env.step(actions)
        rf = u._current_reference_motion
        q.append(u._robot.data.joint_pos[:, u._joint_ids][:, ACT_LEG_JOINT_IDX].cpu().numpy())
        dq.append(u._robot.data.joint_vel[:, u._joint_ids][:, ACT_LEG_JOINT_IDX].cpu().numpy())
        qr.append(rf[:, 0:14][:, REF_LEG_JOINT_IDX].cpu().numpy())
        dqr.append(rf[:, 14:28][:, REF_LEG_JOINT_IDX].cpu().numpy())
        ft.append(u._get_foot_contact().float().cpu().numpy())
        # 몸통 높이. v28(ref_h175) vs v30(ref_h190)처럼 레퍼런스 높이를 바꾼
        # 실험은 "실제로 더 크게 서서 걷는가"로 판정해야 하는데, 그동안
        # 어느 측정에도 이 값이 남지 않아 판정 자체가 불가능했다.
        bh.append((u._robot.data.root_pos_w[:, 2]
                   - u._terrain.env_origins[:, 2]).cpu().numpy())
        # 레퍼런스 접지 — 위상만으로 복원 가능하지만 npz를 자족적으로 둔다.
        fr.append((rf[:, 28:30] > 0.5).float().cpu().numpy())
        # path frame 오차 (v25 이후). 순수 rate 명령에서는 "휘었다"는 사실이
        # 어디에도 안 남으므로, 이 값이 직진성을 재는 유일한 직접 지표다.
        if getattr(u.cfg, "use_path_frame", False):
            pe.append(u._path_error().cpu().numpy())
        gv.append(u._robot.data.projected_gravity_b.cpu().numpy())
        tq.append(u._robot.data.applied_torque[:, u._joint_ids].cpu().numpy())
        fz.append(u._robot.data.body_pos_w[:, u._foot_body_ids, 2].cpu().numpy())
        fv.append(u._feet_vel_b().cpu().numpy())
        vb.append(u._robot.data.root_lin_vel_b[:, :2].cpu().numpy())   # command frame
        vw.append(u._robot.data.root_lin_vel_w[:, :2].cpu().numpy())   # reference frame
        vr.append(rf[:, 30:32].cpu().numpy())
        wz.append(u._robot.data.root_ang_vel_b[:, 2].cpu().numpy())
        wr.append(rf[:, 35].cpu().numpy())
    store[name] = dict(
        q=np.asarray(q), qr=np.asarray(qr), dq=np.asarray(dq), dqr=np.asarray(dqr),
        feet=np.asarray(ft), v_base=np.asarray(vb), v_world=np.asarray(vw),
        v_ref=np.asarray(vr), w_base=np.asarray(wz), w_ref=np.asarray(wr),
        cmd=np.array([cx, cy, cw]),
        base_h=np.asarray(bh), feet_ref=np.asarray(fr),
        path_err=np.asarray(pe) if pe else np.zeros((0, 0, 3)),
        grav=np.asarray(gv), tau=np.asarray(tq),
        foot_z=np.asarray(fz), foot_v=np.asarray(fv),
    )
    err = np.linalg.norm(np.asarray(vb)[100:].mean(axis=(0, 1)) - np.array([cx, cy]))
    print(f"[ok] {name:9s} cmd=({cx:+.2f},{cy:+.2f},{cw:+.2f})  "
          f"achieved=({np.asarray(vb)[100:, :, 0].mean():+.3f},{np.asarray(vb)[100:, :, 1].mean():+.3f})  "
          f"err={err:.3f}", flush=True)

flat = {}
for name, d in store.items():
    for k, v in d.items():
        flat[f"{name}__{k}"] = v
np.savez_compressed(
    args_cli.out, leg_names=np.array(leg_names),
    gait_period_steps=u._gait_period_steps, ctrl_dt=u.step_dt,
    conds=np.array([c[0] for c in CONDS]), checkpoint=args_cli.checkpoint, **flat,
)
print(f"[done] wrote {args_cli.out}", flush=True)
env.close()
simulation_app.close()
