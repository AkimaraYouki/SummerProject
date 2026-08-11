#!/usr/bin/env python3
"""심 액추에이터의 주파수 응답을 잰다 — 실기 `scripts/hw/full_check.py` 와 같은 방식.

    isaaclab.sh -p scripts/_isaaclab_launch.py scripts/diag/sim_freq_response.py \\
        --task Isaac-OpenDuckMini-Joystick-V42-v0 --headless
    ... --stiffness 26.1 --damping 0.851 --armature 0.045     # 조합을 바꿔 훑기

## 왜

2026-08-12 에 실기 14축 주파수 응답을 처음 쟀다. 하드웨어는 정상이었고
(이득 0.95~1.00, 좌우차 2 % 이내) 남은 것은 **위상 지연**이었다:

    1.85 Hz  이득 0.99  위상 +20.2도
    3.00 Hz  이득 0.98  위상 +36.8도

심이 이 응답을 내는지는 한 번도 확인한 적이 없다. stiffness / damping /
armature 세 값은 각각 47 % / 59 % / 3 배의 불확실성을 안고 있는데
(BAM vs ROBOTIS 규격 vs 문헌), 개별 값을 다투는 것보다 **그 셋이 만드는
결과**를 실측에 맞추는 편이 강하다. 이 스크립트가 그 결과를 재 준다.

## 무엇을 하는가

정책을 쓰지 않는다. 관절에 직접 사인 목표를 넣고 (다른 관절은 기본 자세로
고정) 이득과 위상을 상관 적분으로 뽑는다 — 실기와 같은 정의다.

`--stiffness/--damping/--armature` 를 주면 그 값으로 덮어쓰고 잰다. 여러
조합을 훑어 실측에 가장 가까운 것을 찾는 데 쓴다.
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--joints", default="left_knee,left_hip_pitch",
                    help="잴 관절, 쉼표 구분")
parser.add_argument("--amp", type=float, default=10.0, help="사인 진폭 (도)")
parser.add_argument("--freqs", default="1.0,1.85,3.0")
parser.add_argument("--cycles", type=float, default=6.0)
parser.add_argument("--stiffness", type=float, default=None)
parser.add_argument("--damping", type=float, default=None)
parser.add_argument("--armature", type=float, default=None)
parser.add_argument("--effort_limit", type=float, default=None)
parser.add_argument("--delay", type=int, default=None,
                    help="액션 지연을 이 값(스텝)으로 고정. 0 이면 지연 없음. "
                         "기본은 태스크 설정 그대로(0~3 무작위).")
parser.add_argument("--fix_base", action="store_true", default=True,
                    help="몸통을 공중에 고정한다 (기본 켜짐). 실기는 매달아 놓고 쟀으므로 "
                         "같은 조건이어야 한다. 안 고정하면 한 관절만 흔드는 동안 로봇이 "
                         "넘어져 에피소드가 리셋되고, 그 리셋이 상관적분에 섞여 이득·위상이 "
                         "물리적으로 말이 안 되는 값으로 나온다 (2026-08-12 에 겪음: 지연을 "
                         "늘렸는데 위상이 오히려 줄었다).")
parser.add_argument("--free_base", dest="fix_base", action="store_false")
parser.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from open_duck_mini_isaaclab.joint_order import ACTUATOR_JOINT_NAMES  # noqa: E402
from open_duck_mini_isaaclab.tasks.task_registry import env_cfg_for  # noqa: E402
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401

cfg = env_cfg_for(args_cli.task)
cfg.scene.num_envs = 1
cfg.events.push_robot = None
# 도메인 랜덤화를 끈다 — 파라미터를 훑는 중에 무작위가 섞이면 비교가 안 된다.
for ev in ("randomize_joint_params", "randomize_actuator_gains",
           "add_base_mass", "scale_all_mass", "randomize_default_joint_pos"):
    if hasattr(cfg.events, ev):
        setattr(cfg.events, ev, None)

if args_cli.fix_base:
    cfg.robot.spawn.articulation_props.fix_root_link = True
    # 넘어짐 종료도 끈다 (고정했으니 넘어질 일이 없지만, 접촉 판정이 튈 수 있다).
    cfg.episode_length_s = 1.0e9

if args_cli.delay is not None:
    cfg.action_min_delay = args_cli.delay
    cfg.action_max_delay = args_cli.delay

for grp in cfg.robot.actuators.values():
    if args_cli.stiffness is not None:
        grp.stiffness = args_cli.stiffness
    if args_cli.damping is not None:
        grp.damping = args_cli.damping
    if args_cli.armature is not None:
        grp.armature = args_cli.armature
    if args_cli.effort_limit is not None and hasattr(grp, "effort_limit"):
        grp.effort_limit = args_cli.effort_limit

env = gym.make(args_cli.task, cfg=cfg)
u = env.unwrapped
env.reset()

names = [n.strip() for n in args_cli.joints.split(",")]
idx = [ACTUATOR_JOINT_NAMES.index(n) for n in names]
freqs = [float(x) for x in args_cli.freqs.split(",")]
dt = u.step_dt
d = math.degrees
default = u._robot.data.default_joint_pos[:, u._joint_ids].clone()

lines = []


def w(s=""):
    print(s, flush=True)
    lines.append(s)


w("=" * 76)
grp0 = next(iter(cfg.robot.actuators.values()))
w(f"심 주파수 응답 — {args_cli.task}")
w(f"  stiffness {grp0.stiffness}  damping {grp0.damping}  armature {grp0.armature}"
  + (f"  effort_limit {getattr(grp0, 'effort_limit', '-')}" if hasattr(grp0, "effort_limit") else ""))
w(f"  사인 ±{args_cli.amp}°  ·  제어 {1/dt:.0f} Hz  ·  액션지연 {cfg.action_min_delay}~{cfg.action_max_delay} 스텝"
  f"  ·  베이스 {'고정' if args_cli.fix_base else '자유'}")
w("  실기 실측:  1.85 Hz 이득 0.99 위상 +20.2°   3.0 Hz 이득 0.98 위상 +36.8°")
w("-" * 76)
w(f"  {'관절':16}{'Hz':>6}{'이득':>8}{'위상':>9}{'실기 대비':>22}")

REAL = {1.0: (0.99, 9.9), 1.85: (0.99, 20.2), 3.0: (0.98, 36.8)}

for k, name in zip(idx, names):
    for f in freqs:
        wf = 2 * math.pi * f
        n_steps = int(args_cli.cycles / f / dt)
        s = c = gs = gc = 0.0
        cnt = 0
        for i in range(n_steps):
            t = i * dt
            tgt = default.clone()
            tgt[:, k] = default[:, k] + math.radians(args_cli.amp) * math.sin(wf * t)
            # 액션 -> 목표각은 default + action*scale 이므로 역산해서 넣는다.
            act = (tgt - default) / cfg.action_scale
            with torch.inference_mode():
                env.step(act)
            if float(u.episode_length_buf[0]) < 3:   # 리셋이 났다는 신호
                w('  !! 에피소드 리셋 발생 — 이 측정은 못 믿는다')
            if t > 1.0 / f:                      # 첫 주기는 과도구간
                pos = float(u._robot.data.joint_pos[0, u._joint_ids][k] - default[0, k])
                s += pos * math.sin(wf * t)
                c += pos * math.cos(wf * t)
                gs += math.radians(args_cli.amp) * math.sin(wf * t) ** 2
                gc += math.radians(args_cli.amp) * math.sin(wf * t) * math.cos(wf * t)
                cnt += 1
        if cnt == 0:
            continue
        As, Ac = 2 * s / cnt, 2 * c / cnt
        Gs, Gc = 2 * gs / cnt, 2 * gc / cnt
        gain = math.hypot(As, Ac) / max(math.hypot(Gs, Gc), 1e-9)
        ph = d(math.atan2(-Ac, As) - math.atan2(-Gc, Gs))
        while ph > 180:
            ph -= 360
        while ph < -180:
            ph += 360
        cmp_ = ""
        if f in REAL:
            rg, rp = REAL[f]
            cmp_ = f"이득 {gain - rg:+.2f}  위상 {ph - rp:+.1f}°"
        w(f"  {name:16}{f:6.2f}{gain:8.2f}{ph:+8.1f}°{cmp_:>22}")

w("=" * 76)
if args_cli.out:
    open(args_cli.out, "w").write("\n".join(lines) + "\n")
env.close()
app.close()
