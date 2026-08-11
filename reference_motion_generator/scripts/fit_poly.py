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
    #
    # ⚠️ 런타임과 **같은 위상 격자**여야 한다. poly_reference_motion.py 는
    #        t = (i % nb_steps_in_period) / nb_steps_in_period
    #    로 평가한다 — 즉 0, 1/N, ..., (N-1)/N (간격 1/N, 끝이 1 이 아니다).
    #    원래 코드는 np.linspace(0, 1, N) 이었는데 그건 간격이 1/(N-1) 이고
    #    마지막 프레임을 t=1.0 에 놓는다. 두 축이 어긋나면 재생 위상이 계통적으로
    #    밀리고, 좌우 다리는 반주기 떨어져 있으므로 **그 밀림이 좌우 다르게**
    #    나타난다 (2026-08-11: 녹화는 ±2 % 대칭인데 적합 후 무릎이 +15~18 %).
    #
    # ⚠️ **float32 로 캐스팅하면 안 된다.** 15 차 다항식을 t∈[0,1) 에서 맞추는 것은
    #    조건수가 매우 나빠서(polyfit 이 RankWarning 을 낸다) 입력 정밀도가 그대로
    #    결과에 나온다. 2026-08-11 실측, 같은 데이터·같은 차수인데:
    #        float32 X : 재현오차 6.883°, 무릎 좌우차 +16.4 %
    #        float64 X : 재현오차 0.786°, 무릎 좌우차  -3.1 %  (녹화 원본 -1.8 %)
    #    즉 v35 이후 모든 정책이 물려받은 레퍼런스 좌우 비대칭의 정체가 이 캐스팅이다.
    #    (Y 는 float32 로 둔다 — 녹화 자체의 정밀도이고, 문제는 X 축이다.)
    N = Y.shape[0]
    X = (np.arange(N) / N).reshape(-1, 1)

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