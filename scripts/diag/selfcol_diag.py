"""자기충돌을 켰을 때 어떤 종료 조건이 발동하는지 직접 본다."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-OpenDuckMini-Joystick-SelfCol-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import torch, gymnasium as gym  # noqa: E402
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.tasks.velocity import joystick_env_cfg as _cm  # noqa: E402
from open_duck_mini_isaaclab.joint_order import READY_BASE_HEIGHT  # noqa: E402

cfg = _cm.JoystickEnvCfg_SelfCol()
cfg.scene.num_envs = 4
env = gym.make(args_cli.task, cfg=cfg)
u = env.unwrapped
u.reset()
names = u._contact_sensor.body_names
print(f"[diag] self_collisions={cfg.robot.spawn.articulation_props.enabled_self_collisions}", flush=True)
for step in range(6):
    with torch.inference_mode():
        u.step(torch.zeros(4, 14, device=u.device))
    f = torch.norm(u._contact_sensor.data.net_forces_w_history[:, 0, :, :], dim=-1)[0]
    top = torch.topk(f, min(6, f.numel()))
    print(f"\n[step {step}] 접촉력 상위:", flush=True)
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"    {names[i]:34} {v:9.3f} N", flush=True)
    h = u.root_pos_w[:, 2] if hasattr(u, "root_pos_w") else u._robot.data.root_pos_w[:, 2]
    print(f"    base_z {h[0]:.4f} (임계 {READY_BASE_HEIGHT*cfg.min_base_height_ratio:.4f})"
          f"  trunk_head_contact={bool(u._get_trunk_head_contact()[0])}", flush=True)
env.close(); simulation_app.close()
