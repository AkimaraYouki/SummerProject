#!/usr/bin/env python3
"""Mac-runnable sim2real sanity check: replay a trained (and ONNX-exported)
policy in plain CPU MuJoCo against Open_Duck_Playground's own MJCF
(assets/robot/open_duck_mini_v2/playground_mjcf/), independent of Isaac Sim.

This does not need to produce good walking — its purpose is to prove the
export path and the 101-dim observation-vector layout are consistent
end-to-end between the Isaac Lab env (tasks/velocity/joystick_env.py) and
this MuJoCo playback harness. If this script produces NaNs or wildly
saturated joint commands, that's a layout/scale mismatch to chase down
before trusting any Ubuntu training run's behavior.

Observation layout MUST exactly match
source/open_duck_mini_isaaclab/tasks/velocity/joystick_env_cfg.py's
OBS_STATE_DIM composition (101-dim): gyro(3), accel(3), command(7),
joint_pos_rel(14), joint_vel_scaled(14), last_act(14), last_last_act(14),
last_last_last_act(14), motor_targets(14), contact(2), imitation_phase(2).
"""

import argparse
import os
import sys

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from open_duck_mini_isaaclab.joint_order import ACTUATOR_JOINT_NAMES  # noqa: E402
from open_duck_mini_isaaclab.reference_motion.poly_reference_motion import REF_FRAME_DIM  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCENE = os.path.join(
    REPO_ROOT, "assets", "robot", "open_duck_mini_v2", "playground_mjcf", "scene_flat_terrain.xml"
)

ACTION_SCALE = 0.25
DOF_VEL_SCALE = 0.05
NUM_JOINTS = len(ACTUATOR_JOINT_NAMES)


def build_obs(model, data, command, last_acts, motor_targets, phase_i, period_steps, default_qpos):
    gyro = data.sensor("gyro").data.copy()
    accel = data.sensor("accelerometer").data.copy()

    joint_pos = np.array([data.joint(name).qpos[0] for name in ACTUATOR_JOINT_NAMES])
    joint_vel = np.array([data.joint(name).qvel[0] for name in ACTUATOR_JOINT_NAMES])
    joint_pos_rel = joint_pos - default_qpos
    joint_vel_scaled = joint_vel * DOF_VEL_SCALE

    contact = np.zeros(2)  # left/right foot contact — not wired for this sanity check (no reference/reward needed)

    phase = 2.0 * np.pi * phase_i / period_steps
    imitation_phase = np.array([np.cos(phase), np.sin(phase)])

    last_act, last_last_act, last_last_last_act = last_acts
    obs = np.concatenate(
        [
            gyro,
            accel,
            command,
            joint_pos_rel,
            joint_vel_scaled,
            last_act,
            last_last_act,
            last_last_last_act,
            motor_targets,
            contact,
            imitation_phase,
        ]
    ).astype(np.float32)
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to exported policy.onnx")
    parser.add_argument("--scene", type=str, default=DEFAULT_SCENE)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--period_steps", type=int, default=50, help="Only used for the imitation-phase obs channel")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    default_qpos = np.array([data.joint(name).qpos[0] for name in ACTUATOR_JOINT_NAMES])
    motor_targets = default_qpos.copy()

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_dim = session.get_inputs()[0].shape[-1]
    output_dim = session.get_outputs()[0].shape[-1]
    print(f"Loaded {args.onnx}: input_dim={input_dim}, output_dim={output_dim}")
    if input_dim not in (None, 101):
        print(f"[WARN] expected 101-dim input, ONNX reports {input_dim} — obs layout may not match this script.")
    if output_dim not in (None, NUM_JOINTS):
        print(f"[WARN] expected {NUM_JOINTS}-dim output, ONNX reports {output_dim}.")

    last_acts = [np.zeros(NUM_JOINTS) for _ in range(3)]
    command = np.zeros(7)
    command[0] = 0.1  # small forward-velocity command, matching a typical Playground training command

    max_abs_action = 0.0
    any_nan = False

    for step in range(args.steps):
        obs = build_obs(model, data, command, last_acts, motor_targets, step, args.period_steps, default_qpos)
        action = session.run(None, {input_name: obs[None, :].astype(np.float32)})[0][0]

        if np.isnan(action).any():
            any_nan = True
            print(f"[step {step}] NaN in policy output — aborting.")
            break
        max_abs_action = max(max_abs_action, float(np.abs(action).max()))

        motor_targets = default_qpos + action * ACTION_SCALE
        for i, name in enumerate(ACTUATOR_JOINT_NAMES):
            data.ctrl[data.actuator(name).id] = motor_targets[i]

        mujoco.mj_step(model, data)

        last_acts = [action.copy(), last_acts[0], last_acts[1]]

    print(f"Ran {args.steps} steps. any_nan={any_nan}, max|action|={max_abs_action:.3f}")
    if not any_nan and max_abs_action < 5.0:
        print("Sanity check PASSED: no NaNs, actions in a plausible range.")
    else:
        print("Sanity check FAILED: inspect obs layout / ONNX export before trusting Ubuntu training results.")


if __name__ == "__main__":
    main()
