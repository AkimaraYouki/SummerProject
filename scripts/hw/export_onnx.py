#!/usr/bin/env python3
"""학습된 정책을 실기용 ONNX 로 뽑는다. **Isaac Sim 없이 CPU 로만** 돈다.

왜 별도 스크립트인가. IsaacLab 의 `play.py` 는 실행할 때마다
`exported/policy.onnx` 를 자동으로 만들어 준다(`docs/decisions.md` 참고).
그런데 이 레포는 재생을 `scripts/play_fixed_cmd.py`(= `odm play`)로 하고 그건
익스포트를 안 한다. 그래서 7월 런(v3~v16)에는 onnx 가 있고 v28 이후로는 하나도
없다. 게다가 `play.py` 를 쓰려면 Isaac Sim 이 떠야 해서 학습 중에는 못 돌린다
(GPU 에 두 개 동시 금지). 이 스크립트는 체크포인트만 읽으므로 학습과 병행된다.

**정규화를 그래프 안에 넣는다.** 체크포인트에는 `obs_normalizer`(경험적 정규화,
러너 설정의 `empirical_normalization = True`)의 mean/std 가 들어 있고, 이걸
빼먹으면 실기에서 정책이 전혀 다른 입력을 받게 된다 — sim2real 에서 제일 흔하게
조용히 터지는 지점이다. 그래서 ONNX 입력은 **원시 관측**이고, 정규화는 모델
안에서 일어난다.

출력은 결정적(평균) 액션이다. `distribution.std_param` 은 탐색용이라 안 쓴다.

    python3 scripts/hw/export_onnx.py --ver v34c10
    python3 scripts/hw/export_onnx.py --checkpoint <...>/model_2999.pt --out /tmp/p.onnx

같이 나오는 `<이름>.obs.json` 은 관측 벡터의 순서·구간표다. 실기 코드가 이
순서대로 채워야 한다 — 한 칸만 밀려도 정책은 조용히 이상하게 걷는다.
"""

import argparse
import glob
import json
import os

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGROOT = os.path.join(REPO, "logs/rsl_rl/open_duck_mini_v2_joystick")

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}


class Policy(nn.Module):
    """정규화 + MLP. rsl-rl 의 actor 를 추론 경로만 남겨 재구성한 것."""

    def __init__(self, mean, std, layers):
        super().__init__()
        self.register_buffer("mean", mean)
        # std 가 0 인 차원(한 번도 안 변한 관측)에서 0 나누기가 나지 않게 막는다.
        self.register_buffer("std", torch.clamp(std, min=1e-6))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs):
        return self.mlp((obs - self.mean) / self.std)


def build(ckpt_path: str, activation: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in ck:
        raise SystemExit(f"actor_state_dict 가 없다. 최상위 키: {list(ck.keys())}")
    sd = ck["actor_state_dict"]

    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    if not idx:
        raise SystemExit("mlp.*.weight 를 못 찾았다 — rsl-rl 버전이 바뀌었는지 확인할 것")
    act_cls = ACTIVATIONS[activation]
    layers, dims = [], []
    for n, i in enumerate(idx):
        w, b = sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        dims.append((w.shape[1], w.shape[0]))
        if n < len(idx) - 1:                      # 마지막 층 뒤에는 활성함수 없음
            layers.append(act_cls())

    obs_dim, act_dim = dims[0][0], dims[-1][1]
    if "obs_normalizer._mean" in sd:
        mean = sd["obs_normalizer._mean"].reshape(1, -1).float()
        std = sd["obs_normalizer._std"].reshape(1, -1).float()
        normalized = True
    else:
        # empirical_normalization = False 로 학습한 체크포인트도 그대로 통과시킨다.
        mean = torch.zeros(1, obs_dim)
        std = torch.ones(1, obs_dim)
        normalized = False
    if mean.shape[1] != obs_dim:
        raise SystemExit(f"정규화 차원 {mean.shape[1]} != 관측 차원 {obs_dim}")

    return Policy(mean, std, layers).eval(), obs_dim, act_dim, int(ck.get("iter", -1)), normalized, dims


def obs_layout(obs_dim: int):
    """관측 벡터 구간표. joystick_env.py 의 `state` 조립 순서와 같아야 한다."""
    NJ = 14
    spec = [
        ("gyro", 3, "IMU 각속도 (trunk 프레임 x,y,z rad/s)"),
        ("accel", 3, "IMU 가속도 (비력, 중력 포함. trunk 프레임 m/s^2)"),
        ("command", 7, "lin_x, lin_y, ang_yaw, neck_pitch, head_pitch, head_yaw, head_roll"),
        ("joint_pos_rel", NJ, "관절각 - default_joint_pos (rad, ACTUATOR_JOINT_NAMES 순)"),
        ("joint_vel_scaled", NJ, "관절속도 * dof_vel_scale(0.05)"),
        ("last_act", NJ, "직전 액션"),
        ("last_last_act", NJ, "2 스텝 전 액션"),
        ("last_last_last_act", NJ, "3 스텝 전 액션"),
        ("motor_targets", NJ, "직전에 실제로 보낸 목표각 (rad)"),
        ("foot_contact", 2, "좌/우 발 접지 (0/1)"),
        ("imitation_phase", 2, "cos(phase), sin(phase) — 정지 위상고정 태스크는 정지 시 0 으로 묶임"),
    ]
    if obs_dim >= 104:
        spec.append(("path_error", 3, "use_path_frame=True 일 때만"))
    if obs_dim >= 107:
        spec.append(("projected_gravity", 3, "직립이면 (0,0,-1). imu_map.projected_gravity() 로 계산"))
    out, i = [], 0
    for name, n, desc in spec:
        out.append({"name": name, "start": i, "size": n, "desc": desc})
        i += n
    if i != obs_dim:
        out.append({"name": "!! 합계 불일치", "start": i, "size": obs_dim - i,
                    "desc": "joystick_env.py 의 관측 조립을 다시 대조할 것"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", help="버전 이름 (예: v34c10). 최신 런의 최신 체크포인트를 쓴다")
    ap.add_argument("--checkpoint", help="체크포인트 .pt 직접 지정")
    ap.add_argument("--out", help="출력 .onnx 경로 (기본: <런>/exported/policy.onnx)")
    ap.add_argument("--activation", default="elu", choices=sorted(ACTIVATIONS))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ck = args.checkpoint
    if not ck:
        if not args.ver:
            raise SystemExit("--ver 나 --checkpoint 중 하나는 있어야 한다")
        runs = sorted(glob.glob(os.path.join(LOGROOT, f"*imitation_{args.ver}")),
                      key=os.path.getmtime, reverse=True)
        if not runs:
            raise SystemExit(f"런이 없다: {args.ver}")
        cks = sorted(glob.glob(os.path.join(runs[0], "model_*.pt")),
                     key=lambda p: int(os.path.basename(p)[6:-3]))
        if not cks:
            raise SystemExit(f"체크포인트가 없다: {runs[0]}")
        ck = cks[-1]
    print(f"체크포인트: {ck}")

    policy, obs_dim, act_dim, it, normalized, dims = build(ck, args.activation)
    print(f"  iter {it} · 관측 {obs_dim} -> " + " -> ".join(str(d[1]) for d in dims)
          + f" · 정규화 {'포함' if normalized else '없음(체크포인트에 없음)'}")

    out = args.out or os.path.join(os.path.dirname(ck), "exported", "policy.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    dummy = torch.zeros(1, obs_dim)
    torch.onnx.export(
        policy, dummy, out,
        input_names=["obs"], output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"  -> {out}  ({os.path.getsize(out)/1024:.1f} KB)")

    side = os.path.splitext(out)[0] + ".obs.json"
    json.dump({"checkpoint": os.path.abspath(ck), "iter": it,
               "obs_dim": obs_dim, "action_dim": act_dim,
               "normalization_baked_in": normalized,
               "note": "ONNX 입력은 정규화 전 원시 관측이다. 아래 순서대로 채울 것.",
               "layout": obs_layout(obs_dim)},
              open(side, "w"), ensure_ascii=False, indent=2)
    print(f"  -> {side}")

    # 뽑고 나서 실제로 같은 값을 내는지 확인한다. 안 하면 조용히 틀린 걸 들고
    # 실기에 올라간다 — 정규화를 빠뜨렸을 때가 딱 그렇게 된다.
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        # 입력은 **정규화기 자신의 분포**에서 뽑는다. 표준정규를 넣으면 std 가
        # 1e-6 인 차원(학습 내내 상수였던 관측)에서 값이 폭발해 출력이 ±1e4 가
        # 되고, 절대오차만 보면 멀쩡한 그래프도 틀린 것처럼 보인다 (실제로 처음에
        # 그렇게 보였다 — 상대오차는 4.6e-07 이었다).
        m = policy.mean.numpy()
        s = policy.std.numpy()
        x = (m + s * rng.standard_normal((64, obs_dim))).astype(np.float32)
        with torch.no_grad():
            ref = policy(torch.from_numpy(x)).numpy()
        got = sess.run(None, {"obs": x})[0]
        err = float(np.abs(ref - got).max())
        rel = err / max(float(np.abs(ref).max()), 1e-9)
        ok = rel < 1e-5
        print(f"  검증: torch vs onnxruntime 최대 절대차 {err:.3e} (상대 {rel:.3e})  {'OK' if ok else '!! 불일치'}")
        n_const = int((s.reshape(-1) <= 1.0000001e-6).sum())
        if n_const:
            print(f"  참고: 학습 내내 상수였던 관측 차원이 {n_const} 개 있다 (std=0 -> 1e-6 로 클램프).")
    except ImportError:
        print("  검증 건너뜀: onnxruntime 없음 (pip install onnxruntime)")


if __name__ == "__main__":
    main()
