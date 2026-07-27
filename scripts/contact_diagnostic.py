"""Logs actual trunk/head/foot contact-sensor force magnitudes per step for
a real rollout, to see whether the "plank pose" the user watched over WebRTC
is genuinely resting the head/torso on the ground (should trigger
termination) or merely visually overlapping its own legs with zero real
contact force (self-collision is off, so that alone generates no force) --
this determines whether lowering the termination threshold alone is enough.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-OpenDuckMini-Joystick-A20J5-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.agents.rsl_rl_ppo_cfg import JoystickPPORunnerCfg  # noqa: E402
from open_duck_mini_isaaclab.tasks.velocity.joystick_env_cfg import JoystickEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

env_cfg = JoystickEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = JoystickPPORunnerCfg()
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

print(f"[info] checkpoint={args_cli.checkpoint}", flush=True)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

unwrapped = env.unwrapped
print(f"[info] trunk_head_ids={unwrapped._trunk_head_ids} feet_ids={unwrapped._feet_ids}", flush=True)

obs, _ = env.get_observations()

n_trunk_over_05 = 0
n_trunk_over_10 = 0
n_trunk_nonzero = 0
max_trunk_force = 0.0
n_terminated_by_contact = 0
n_dones = 0

for step in range(args_cli.num_steps):
    with torch.inference_mode():
        actions = policy(obs)
    obs, rew, dones, infos = env.step(actions)

    forces = unwrapped._contact_sensor.data.net_forces_w_history[:, 0, unwrapped._trunk_head_ids, :]
    force_mag = torch.norm(forces, dim=-1)  # [N, 2] (trunk, head)
    per_env_max = force_mag.max(dim=-1).values  # [N]

    n_trunk_nonzero += (per_env_max > 1e-6).sum().item()
    n_trunk_over_05 += (per_env_max > 0.5).sum().item()
    n_trunk_over_10 += (per_env_max > 1.0).sum().item()
    max_trunk_force = max(max_trunk_force, per_env_max.max().item())

    n_dones += dones.sum().item()

    if step % 20 == 0:
        print(
            f"[step {step:3d}] trunk/head force per-env: "
            f"{[f'{v:.3f}' for v in per_env_max.tolist()]}  "
            f"base_z: {[f'{v:.3f}' for v in unwrapped._robot.data.root_pos_w[:, 2].tolist()]}",
            flush=True,
        )

print("\n" + "=" * 70)
print(f"CONTACT FORCE SUMMARY over {args_cli.num_steps} steps x {args_cli.num_envs} envs = {args_cli.num_steps * args_cli.num_envs} env-steps")
print("=" * 70)
print(f"  env-steps with trunk/head force > 0      : {n_trunk_nonzero}")
print(f"  env-steps with trunk/head force > 0.5N    : {n_trunk_over_05}  (would terminate under NEW 0.5N threshold)")
print(f"  env-steps with trunk/head force > 1.0N    : {n_trunk_over_10}  (would terminate under OLD 1.0N threshold)")
print(f"  max trunk/head force observed             : {max_trunk_force:.4f} N")
print(f"  total episode terminations (any cause)     : {n_dones}")
print("=" * 70)

env.close()
simulation_app.close()
