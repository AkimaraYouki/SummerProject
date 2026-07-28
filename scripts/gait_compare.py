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
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.agents.rsl_rl_ppo_cfg import JoystickPPORunnerCfg  # noqa: E402
from open_duck_mini_isaaclab.tasks.velocity import joystick_env_cfg as _cm  # noqa: E402
from open_duck_mini_isaaclab.joint_order import (  # noqa: E402
    ACT_LEG_JOINT_IDX, REF_LEG_JOINT_IDX, ACTUATOR_JOINT_NAMES,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

_MAP = {
    "Isaac-OpenDuckMini-Joystick-A20J5Bounded-v0": "JoystickEnvCfg_A20J5_Bounded",
    "Isaac-OpenDuckMini-Joystick-Walk-v0": "JoystickEnvCfg_Walk",
    "Isaac-OpenDuckMini-Joystick-Walk2-v0": "JoystickEnvCfg_Walk2",
}
env_cfg = getattr(_cm, _MAP[args_cli.task])()
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = JoystickPPORunnerCfg()
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped

# Commands pinned inside the ranges the reference polynomial was actually fit
# over (lin_vel_x +-0.15, lin_vel_y +-0.2, ang_vel_yaw +-1.0) — outside them
# PolyReferenceMotion falls back to a nearest grid point and the "reference"
# being compared against would not be the one for this command.
CONDS = [
    ("forward",  0.15,  0.0,  0.0),
    ("backward", -0.15, 0.0,  0.0),
    ("left",     0.0,   0.2,  0.0),
    ("right",    0.0,  -0.2,  0.0),
    ("turn",     0.0,   0.0,  1.0),
]
leg_names = [ACTUATOR_JOINT_NAMES[i] for i in ACT_LEG_JOINT_IDX]
print(f"[info] gait_period_steps={u._gait_period_steps} dt={u.step_dt}", flush=True)

store = {}
for name, cx, cy, cw in CONDS:
    obs, _ = env.reset()
    q, qr, dq, dqr, vb, vw, vr, ft, wz, wr = [], [], [], [], [], [], [], [], [], []
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
