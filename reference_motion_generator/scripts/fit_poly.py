import numpy as np
import json
from glob import glob
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--ref_motion", type=str, default="ref_motion")
args = parser.parse_args()

all_files = glob(f"{args.ref_motion}/*.json")


def fit_ref_motion(file, intended_vel=None):
    data = json.load(open(file))
    Y_all = np.array(data["Frames"])
    period = data["Placo"]["period"]
    fps = data["FPS"]
    frame_offsets = data["Frame_offset"][0]
    startend_double_support_ratio = data["Placo"]["startend_double_support_ratio"]
    start_offset = int(startend_double_support_ratio * fps)
    nb_steps_in_period = int(period * fps)
    _Y = Y_all[start_offset : start_offset + int(nb_steps_in_period)]
    joints_pos = _Y[:, frame_offsets["joints_pos"] : frame_offsets["left_toe_pos"]]
    joints_vel = _Y[:, frame_offsets["joints_vel"] : frame_offsets["left_toe_vel"]]
    foot_contacts = _Y[
        :, frame_offsets["foot_contacts"] : frame_offsets["foot_contacts"] + 2
    ]
    base_linear_vel = _Y[
        :, frame_offsets["world_linear_vel"] : frame_offsets["world_angular_vel"]
    ]
    base_angular_vel = _Y[
        :, frame_offsets["world_angular_vel"] : frame_offsets["joints_vel"]
    ]

    # Remove the Placo planner's systematic drift bias from the velocity
    # channels (2026-07-26). Even for a commanded-straight walk the planner
    # yaws ~0.03 rad/s (reproduced on the untouched upstream open_duck_mini
    # v1 assets too, so it's inherent to the vendored planner, not our
    # robot). The pkl is keyed by the intended (grid) velocities, and the
    # runtime snaps joystick commands to those keys — so the reference's
    # mean lin x/y and ang z velocity should MATCH the key, not the drifted
    # measurement, otherwise the imitation reward pulls the policy slightly
    # off-command. Joint trajectories/contacts are left untouched (the
    # "style" of the gait); only the mean of the velocity channels is
    # re-centered onto the intended command.
    if intended_vel is not None:
        vx_int, vy_int, vth_int = intended_vel
        base_linear_vel = base_linear_vel.copy()
        base_angular_vel = base_angular_vel.copy()
        base_linear_vel[:, 0] += vx_int - base_linear_vel[:, 0].mean()
        base_linear_vel[:, 1] += vy_int - base_linear_vel[:, 1].mean()
        base_angular_vel[:, 2] += vth_int - base_angular_vel[:, 2].mean()

    Y = np.concatenate(
        [joints_pos, joints_vel, foot_contacts, base_linear_vel, base_angular_vel],
        axis=1,
    ).astype(np.float32)

    # Generate time feature
    X = np.linspace(0, 1, Y.shape[0]).reshape(-1, 1).astype(np.float32)  # Time variable

    # Polynomial degree
    degree = 15

    # Store coefficients
    coefficients = {}

    # ====== Fit Polynomial Regression per Dimension ======
    for dim in range(Y.shape[1]):
        coeffs = np.polyfit(X.flatten(), Y[:, dim], degree)
        coefficients[f"dim_{dim}"] = list(np.flip(coeffs))

    ret_data = {
        "coefficients": coefficients,
        "period": period,
        "fps": fps,
        "frame_offsets": frame_offsets,
        "start_offset": start_offset,
        "nb_steps_in_period": nb_steps_in_period,
        "startend_double_support_ratio": startend_double_support_ratio,
    }

    return ret_data


all_coefficients = {}
for file in all_files:
    name = os.path.basename(file).strip(".json")
    tmp = name.split("_")
    name = f"{tmp[1]}_{tmp[2]}_{tmp[3]}"

    intended = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
    all_coefficients[name] = fit_ref_motion(file, intended_vel=intended)


pickle.dump(all_coefficients, open("polynomial_coefficients.pkl", "wb"))