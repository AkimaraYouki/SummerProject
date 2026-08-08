#!/usr/bin/env python3
"""RL 보행 정책 실기 배포 루프 (Jetson).

    ssh -t parksuho@192.168.137.7 \
        'python3 ~/rl_walk.py --onnx ~/policy/policy.onnx --dry-run'
    ssh -t parksuho@192.168.137.7 \
        'python3 ~/rl_walk.py --onnx ~/policy/policy.onnx --vx 0.10'

────────────────────────────────────────────────────────────────────────
⚠️  로봇을 **매달아 놓고** 처음 돌릴 것. 바닥에 세우면 넘어질 수 있다.
────────────────────────────────────────────────────────────────────────

이 스크립트는 데스크탑 `open_duck_mini_isaaclab` 리포의 joystick_env.py /
joystick_env_cfg.py / joint_order.py / imu_map.py 를 실기용으로 이식한 것이다.
2026-08-08 `docs/handoff/onnx_deploy_2026-08-08.md`(데스크탑, 배포 계약 문서)
+ 실제 소스(joystick_env.py/joystick_env_cfg.py) 대조로 검증 완료. 대상은
`imitation_v34c10`(JoystickEnvCfg_V34C10, iter 2999, 현재 최고 성능) — v34u 는
v34c10 을 상속하고 관측/액션 계약이 완전히 같으므로 v34u 로 바뀌어도 이 스크립트는
그대로 쓴다. 이 계열은 observation_space=107, state_space=211,
lock_head_joints=true, use_path_frame=true, use_gravity_obs=true,
gait_period_steps=27(ref_g125, 0.54s@50Hz), standstill_hold=true.
다른 체크포인트를 쓰면 이 값들이 안 맞을 수 있으니 params/env.yaml 을
다시 확인할 것.

관측벡터 (107차원, 이 순서 그대로):
    gyro(3) + accel(3, 중력 포함) + command(7: vx,vy,wz,neck_pitch,head_pitch,
    head_yaw,head_roll — 뒤 4개는 lock_head_joints=true 라 항상 0) +
    joint_pos_rel(14, present-READY) + joint_vel*0.05(14) +
    last_act(14) + last_last_act(14) + last_last_last_act(14) +
    motor_targets(14) + feet_contact(2, left/right) + imitation_phase(2, cos/sin) +
    path_error(3, lateral·cos(yaw_err)·sin(yaw_err)) + projected_gravity(3)

머리 4축(neck_pitch/head_pitch/head_yaw/head_roll)은 lock_head_joints=true 라
정책이 안 움직인다 — 항상 READY 자세로 고정해서 내보낸다.

⚠️ path_error 는 실기에 없는 값이다 — 로봇의 절대 위치/방향 오차인데 이
로봇엔 오도메트리/모션캡처가 없다. Disney BD-X 논문(arXiv 2501.05204)도
실기에서 이걸 정확히 어떻게 복원하는지 명시하지 않는다. 2026-08-08 결정
(사용자): **[0, 1, 0]으로 고정** — "명령대로 완벽히 가고 있다"는 뜻(옆으로
안 새고 방향도 안 틀어짐)으로 두고 일단 돌려본다. 실제로는 로봇이 새거나
틀어져도 정책이 그걸 관측으로 못 받으니 그만큼 보정을 못 한다 — 순수
전진/제자리 테스트로 시작하고, 옆으로 새는 정도를 보고 필요하면 나중에
실제 오도메트리를 추가할 것.

안전 설계 (기존 dxl_bridge.py/play_ref_gait.py 와 같은 원칙):
  * 전류 제한 (기본 350 unit = 0.94 A) — 다리가 실제 보행 부하를 못 버티고
    멈추면(주저앉으면) --current 를 올릴 것. 이 값은 걷기용으로 검증된 적이
    없다 — 처음엔 반드시 매달아 놓고 관절이 걸리는지 눈으로 볼 것.
  * 모터 목표 속도 제한 (max_motor_velocity=4.82 rad/s, 학습 설정과 동일) —
    정책 출력이 튀어도 급격한 점프를 못 만든다.
  * URDF 관절 한계 클램프 (rustypot_hwi.tick_of 안에서).
  * 현재 자세 -> READY 자세로 smoothstep 램프인 (급출발 없음).
  * Ctrl+C / SIGHUP / SIGTERM -> 즉시 현재 위치 홀드.
  * 실행 중 아무 때나 콘솔에 sss + 엔터 -> 비상정지 (Ctrl+C 와 동일하게 즉시 홀드).
  * TTY 가 아니면 거부.

2026-08-08: 매 루프 SyncRead 를 다리 10축만으로 줄였다(get_leg_pos_vel, 머리 4축
제외 — lock_head_joints=true 라 정책이 안 씀). 14축일 때 실측 ~28ms/루프 ·
28.8Hz 였는데, 10축으로 줄면 그 비율만큼(대략 20ms대) 빨라져 50Hz 예산에 더
가까워질 것으로 기대. 머리 4축은 항상 READY 라 가정하고 joint_pos_rel/vel 을
0으로 채운다 — 실제로 손으로 건드리는 등 READY 에서 벗어나 있어도 정책은 그걸
모른다(어차피 lock_head_joints=true 라 정책 출력도 무시되니 상관없음).

같이 배포해야 하는 파일: rustypot_hwi.py, feet_contacts.py (같은 홈 디렉터리).
"""

import argparse
import csv
import json
import math
import os
import signal
import sys
import threading
import time

import numpy as np
import onnxruntime as ort
from smbus2 import SMBus, i2c_msg

from feet_contacts import FeetContacts
from rustypot_hwi import HWI, NAMES, LEG_NAMES

# ── 관절 상수 (source: joystick_env_cfg.py READY_JOINT_POS_G125_ZNECK, 2026-08-08
# 데스크탑에서 직접 대조 확인 — joint_order.py 의 7/28 자 READY_JOINT_POS 는
# ref_g125/Z넥 이전 값이라 다리 쪽이 최대 26° 까지 어긋난다. 쓰지 말 것). ──
READY_JOINT_POS = {
    "left_hip_yaw": -0.0031, "left_hip_roll": 0.0207, "left_hip_pitch": 0.8952,
    "left_knee": -1.5693, "left_ankle": 0.7444,
    "neck_pitch": 0.5236, "head_pitch": 0.5236, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": -0.0037, "right_hip_roll": -0.0104, "right_hip_pitch": 0.9315,
    "right_knee": 1.6000, "right_ankle": -0.7388,
}
READY_ARR = np.array([READY_JOINT_POS[n] for n in NAMES], dtype=np.float32)
HEAD_IDX = [NAMES.index(n) for n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]
LEG_IDX = [NAMES.index(n) for n in LEG_NAMES]

ACTION_SCALE = 0.25
DOF_VEL_SCALE = 0.05
MAX_MOTOR_VEL = 4.82        # rad/s, joystick_env_cfg.py max_motor_velocity (전 계열 공통)
GAIT_PERIOD_STEPS = 27      # ref_g125 레퍼런스 주기, 0.54s @ 50Hz (joystick_env.py._gait_period_steps)
STANDSTILL_HOLD_THRESH = 0.01  # ‖command[:3]‖ 이하면 위상을 0(=cos,sin=1,0)에 고정
CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ
NUM_COMMANDS = 7             # vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll (뒤 4개 항상 0)

# path_error 고정값 (2026-08-08 결정, docstring 참고): [lateral=0, cos(yaw_err)=1, sin(yaw_err)=0]
PATH_ERROR_FIXED = np.array([0.0, 1.0, 0.0], dtype=np.float32)

# 2026-08-09: 오른다리(특히 hip_roll/yaw)가 실기에서 왼다리와 다르게 움직이는
# 문제를 사후분석하려고 매 스텝을 CSV로 남긴다. 매번 덮어쓴다 — 직전 실행만 본다.
LOG_PATH = os.path.expanduser("~/rl_walk_log.csv")

# ── IMU (BNO055, imu_check.py 와 같은 레지스터) ─────────────────────────────
IMU_BUS = 7
IMU_ADDR = 0x28
REG_ACC_DATA = 0x08
REG_GYR_DATA = 0x14
REG_UNIT_SEL = 0x3B
REG_OPR_MODE = 0x3D
REG_PWR_MODE = 0x3E
MODE_CONFIG = 0x00
MODE_NDOF = 0x0C
ACC_LSB = 100.0
GYR_LSB = 16.0
IMU_CAL_JSON = os.path.expanduser("~/imu_calibration.json")

_hold = {"stop": False}


class Imu:
    def __init__(self, bus_num=IMU_BUS, addr=IMU_ADDR):
        self.bus = SMBus(bus_num)
        self.addr = addr
        self.axis_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if os.path.exists(IMU_CAL_JSON):
            with open(IMU_CAL_JSON) as f:
                self.axis_matrix = json.load(f)["matrix"]
        self._wake()

    def _rd(self, reg, n):
        w = i2c_msg.write(self.addr, [reg])
        r = i2c_msg.read(self.addr, n)
        self.bus.i2c_rdwr(w, r)
        return list(r)

    def _wr(self, reg, val):
        self.bus.write_byte_data(self.addr, reg, val)

    def _wake(self):
        self._wr(REG_PWR_MODE, 0x00)
        time.sleep(0.02)
        self._wr(REG_OPR_MODE, MODE_CONFIG)
        time.sleep(0.03)
        self._wr(REG_UNIT_SEL, 0x00)
        time.sleep(0.01)
        self._wr(REG_OPR_MODE, MODE_NDOF)
        time.sleep(0.05)

    def _s16(self, lo, hi):
        v = lo | (hi << 8)
        return v - 65536 if v & 0x8000 else v

    def _vec(self, reg, lsb):
        d = self._rd(reg, 6)
        return tuple(self._s16(d[i], d[i + 1]) / lsb for i in (0, 2, 4))

    def read(self):
        """(gyro_trunk[rad/s], accel_trunk[m/s^2, 중력 포함]) — 몸통 프레임으로 리맵."""
        a = self._vec(REG_ACC_DATA, ACC_LSB)
        g = self._vec(REG_GYR_DATA, GYR_LSB)
        g = tuple(v * math.pi / 180.0 for v in g)
        m = self.axis_matrix
        a_t = tuple(sum(m[i][j] * a[j] for j in range(3)) for i in range(3))
        g_t = tuple(sum(m[i][j] * g[j] for j in range(3)) for i in range(3))
        return g_t, a_t


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None,
                     help="policy.onnx 경로 (같은 폴더에 policy.onnx.data 도 있어야 함). "
                          "생략하면 정책 추론 없이 action=0(READY 유지)으로 나머지 파이프라인만 돈다 "
                          "(IMU/모터/GPIO/타이밍 검증용, --zero-action 과 동일)")
    ap.add_argument("--zero-action", action="store_true",
                     help="--onnx 를 줬어도 정책 추론을 건너뛰고 action=0 으로 강제 (파이프라인만 검증)")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--current", type=int, default=350,
                     help="Goal Current 상한 (1 unit=2.69mA). 다리가 부하로 멈추면 올릴 것")
    ap.add_argument("--rampin", type=float, default=3.0, help="현재 자세->READY 램프인 시간(초)")
    ap.add_argument("--vx", type=float, default=0.0, help="전후 속도 명령 [-0.15, 0.15]")
    ap.add_argument("--vy", type=float, default=0.0, help="좌우 속도 명령 [-0.2, 0.2]")
    ap.add_argument("--wz", type=float, default=0.0, help="회전 명령 [-1.0, 1.0]")
    ap.add_argument("--action-lpf-alpha", type=float, default=None,
                     help="액션 1차 저역필터 계수 a_f=α·a_f_prev+(1-α)·a. 생략하면 "
                          "정책 옆의 policy.meta.json 에 적힌 값(=그 정책이 학습된 값)을 "
                          "쓰고, 그것도 없으면 0(끔). **학습 때 쓴 값과 같아야 한다** — "
                          "v36 처럼 필터를 학습 환경에 넣은 정책은 반드시 켜야 하고"
                          "(α=0.5), v35 처럼 무필터로 학습된 정책에 켜면 정지에서만 "
                          "이득이고 보행에서는 추종이 나빠진다 "
                          "(docs/reports/lowpass_2026-08-09.md 실험 A).")
    ap.add_argument("--seconds", type=float, default=20.0, help="정책 루프 실행 시간(초)")
    ap.add_argument("--dry-run", action="store_true", help="모터/IMU 안 건드리고 로드만 확인")
    args = ap.parse_args()

    zero_action = args.zero_action or args.onnx is None
    sess = None
    if args.onnx is not None:
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        print(f"[rl_walk] 정책 로드: {args.onnx}")

    # 저역필터 계수는 **정책마다 다르다**(학습 환경에 넣었는지에 따라). 손으로
    # 매번 맞춰 주면 언젠가 깜빡하고 조용히 train/test 불일치가 나므로, 정책 옆
    # policy.meta.json 에 적어두고 그걸 기본값으로 읽는다. CLI 로 주면 그게 이긴다.
    meta = {}
    if args.onnx is not None:
        meta_path = os.path.join(os.path.dirname(os.path.abspath(args.onnx)), "policy.meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
    # v37 부터 알파가 **명령 크기에 따라 갈린다** (정지 0.7 / 보행 0.0). 시뮬의
    # joystick_env._pre_physics_step 과 같은 공식으로 섞는다:
    #   t = smoothstep(clamp((‖cmd‖ - lo) / (hi - lo), 0, 1))
    #   alpha = a_still + (a_move - a_still) * t
    # 이 스크립트는 실행 중 명령이 고정이라 한 번만 계산하면 된다.
    a_move = float(meta.get("action_lowpass_alpha", 0.0))
    a_still = float(meta.get("action_lowpass_alpha_standstill", a_move))
    blend_lo, blend_hi = meta.get("action_lowpass_blend", [0.01, 0.05])
    cmd_norm = float(np.linalg.norm([args.vx, args.vy, args.wz]))
    _t = min(1.0, max(0.0, (cmd_norm - blend_lo) / max(blend_hi - blend_lo, 1e-6)))
    _t = _t * _t * (3.0 - 2.0 * _t)
    alpha_auto = a_still + (a_move - a_still) * _t

    if args.action_lpf_alpha is None:
        args.action_lpf_alpha = alpha_auto
        if meta:
            src = (f"policy.meta.json ({meta.get('run', '?')}: 정지 {a_still} / 보행 "
                   f"{a_move}, ‖cmd‖={cmd_norm:.3f})")
        else:
            src = "meta 없음 -> 0"
    else:
        src = "CLI"
        if meta and abs(alpha_auto - args.action_lpf_alpha) > 1e-9:
            print(f"[rl_walk] !! 경고: 이 명령에서 학습 시 알파는 {alpha_auto:.2f} 인데 "
                  f"{args.action_lpf_alpha} 로 돌린다 — 학습/배포 불일치다.")
    print(f"[rl_walk] 액션 저역필터 α={args.action_lpf_alpha:.3f} ({src})")
    if zero_action:
        print("[rl_walk] !! zero-action 모드 — 정책 추론 없음. action=0 (READY 유지). "
              "IMU/모터/GPIO/타이밍 파이프라인만 검증한다.")

    if args.dry_run:
        print("[dry-run] 모터/IMU/GPIO 를 열지 않았다. ONNX 로드만 확인했다.")
        return

    if not sys.stdin.isatty():
        raise SystemExit(
            "TTY 가 아니다. 로봇이 걷는 동작이라 사람이 보고 있어야 한다.\n"
            "  ssh -t parksuho@192.168.137.7 'python3 ~/rl_walk.py ...'")

    print("\n⚠️  로봇이 매달려 있는지(발이 바닥에 안 닿는지) 확인할 것. "
          "바닥에 세우면 넘어진다.")
    if input("   매달려 있으면 'go' 입력: ").strip().lower() != "go":
        print("취소했다.")
        return

    hwi = HWI(port=args.port, current_limit=args.current)
    imu = Imu()
    feet = FeetContacts()

    def stop(*_):
        _hold["stop"] = True
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, stop)

    def estop_watcher():
        """콘솔에 sss + 엔터 -> 비상정지. Ctrl+C 못 먹거나 다른 터미널에서 보고 있을 때 대비."""
        while not _hold["stop"]:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            if line.strip().lower() == "sss":
                print("\n[rl_walk] !! 비상정지 (sss) — 즉시 홀드한다", flush=True)
                _hold["stop"] = True
                return
    threading.Thread(target=estop_watcher, daemon=True).start()

    try:
        start_pos = np.array(hwi.get_present_positions(), dtype=np.float32)
        hwi.arm()
        print(f"[rl_walk] 토크 켬 14축 · 전류 상한 {args.current * 2.69 / 1000:.2f} A")

        # 1) 램프인: 현재 자세 -> READY.
        print(f"[rl_walk] 램프인 {args.rampin:.1f}s -> READY 자세")
        t0 = time.time()
        while not _hold["stop"]:
            u = (time.time() - t0) / max(args.rampin, 1e-6)
            if u >= 1.0:
                break
            s = smoothstep(u)
            pos = start_pos + (READY_ARR - start_pos) * s
            hwi.set_position_vec(pos)
            time.sleep(DT)
        if not _hold["stop"]:
            hwi.set_position_vec(READY_ARR)
            time.sleep(0.3)

        # 2) 정책 루프 상태 초기화 (env reset과 동일).
        last_act = np.zeros(14, dtype=np.float32)
        last_last_act = np.zeros(14, dtype=np.float32)
        last_last_last_act = np.zeros(14, dtype=np.float32)
        motor_targets = READY_ARR.copy()
        imitation_i = 0
        command = np.array([args.vx, args.vy, args.wz, 0, 0, 0, 0], dtype=np.float32)
        action_filt = np.zeros(14, dtype=np.float32)  # EMA 저역필터 상태 (--action-lpf-alpha)

        # 무필터로 학습된 정책(v35 등)에 필터를 켠 채 보행 명령을 주면 추종이
        # 나빠진다 (실험 A). 학습 때부터 필터가 있던 정책(v36)에는 해당 없다.
        if (args.action_lpf_alpha > 0 and meta.get("action_lowpass_alpha") is None
                and np.linalg.norm(command[:3]) > STANDSTILL_HOLD_THRESH):
            print(f"[rl_walk] !! 경고: 이 정책이 필터로 학습됐는지 알 수 없는데(meta 없음) "
                  f"α={args.action_lpf_alpha} 로 보행 명령을 준다 — 무필터 학습 정책이면 "
                  f"추종이 나빠진다 (lowpass_2026-08-09.md 실험 A).")

        print(f"[rl_walk] 정책 시작 — cmd=({args.vx:+.2f},{args.vy:+.2f},{args.wz:+.2f}) "
              f"· Ctrl+C 로 즉시 정지")
        log_f = open(LOG_PATH, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(
            ["t", "step"] + [f"pos_{n}" for n in NAMES] + [f"vel_{n}" for n in NAMES]
            + [f"action_{n}" for n in NAMES] + [f"target_{n}" for n in NAMES]
            + ["contact_l", "contact_r", "gyro_x", "gyro_y", "gyro_z",
               "accel_x", "accel_y", "accel_z", "proj_grav_x", "proj_grav_y", "proj_grav_z",
               "phase_cos", "phase_sin"])
        t_start = time.time()
        step = 0
        over_budget_count = 0
        over_budget_worst = 0.0
        t_window0 = time.time()
        max_delta = MAX_MOTOR_VEL * DT
        nonfatal_seen = set()
        HW_ERR_PERIOD = 10  # 이 버스에서 SyncRead 하나가 ~16-28ms 라 매 루프 3번 읽으면 예산을 못 맞춘다.
        # 위치+속도는 매 루프 필수(관측벡터에 직접 들어감) — 한 번의 SyncRead로 묶는다.
        # hw_error 는 안전 확인용이라 몇 루프에 한 번이면 충분하다.
        while not _hold["stop"] and (time.time() - t_start) < args.seconds:
            t0 = time.time()

            gyro, accel = imu.read()
            leg_pos, leg_vel = hwi.get_leg_pos_vel()
            # 머리 4축은 lock_head_joints=true 라 안 읽는다(SyncRead 14->10축,
            # 버스 시간 절약) — 항상 READY 로 고정 명령 나가니 rel=0/vel=0 으로 채운다.
            pos = READY_ARR.copy()
            vel = np.zeros(14, dtype=np.float32)
            pos[LEG_IDX] = leg_pos
            vel[LEG_IDX] = leg_vel
            contact = np.array(feet.get(), dtype=np.float32)

            if step % HW_ERR_PERIOD == 0:
                # Shutdown 마스크에 걸린 에러(과열/전기충격/과부하)만 정지 사유다.
                # 마스크 밖 비트는 모터가 계속 도는 상태라 멈출 이유가 없다.
                #
                # 2026-08-09: 이 자리에서 Input Voltage(bit0) 를 보고 서보를
                # reboot 했었는데, 그 비트는 이 로봇 Shutdown 마스크에 없어서
                # 토크를 끊지도 않는 정보성 비트였다. 불필요한 복구가 보행 중
                # 1초에 한 번씩 400ms 씩 제어를 멈춰 로봇이 발을 구르며 주저앉는
                # 원인이 됐다. 이제 세지만 하고 넘어간다.
                fatal = hwi.get_fatal_errors()
                if any(fatal):
                    bad = [(n, e) for n, e in zip(NAMES, fatal) if e]
                    print(f"[rl_walk] !! 치명 하드웨어 에러 {bad} — 정지한다")
                    break
                nonfatal = [n for n, e in zip(NAMES, hwi.get_hw_errors()) if e]
                if nonfatal and not nonfatal_seen:
                    nonfatal_seen = set(nonfatal)
                    print(f"[rl_walk] (참고) 비치명 에러 비트 — {sorted(nonfatal_seen)}. "
                          f"토크는 안 끊긴다. 보행 부하로 전압이 처진 흔적일 수 있다.")

            joint_pos_rel = pos - READY_ARR
            joint_vel_scaled = vel * DOF_VEL_SCALE
            phase = 2.0 * math.pi * imitation_i / GAIT_PERIOD_STEPS
            imitation_phase = np.array([math.cos(phase), math.sin(phase)], dtype=np.float32)

            accel_arr = np.array(accel, dtype=np.float32)
            grav_norm = np.linalg.norm(accel_arr) or 1.0
            projected_gravity = -accel_arr / grav_norm

            obs = np.concatenate([
                np.array(gyro, dtype=np.float32),
                accel_arr,
                command,
                joint_pos_rel,
                joint_vel_scaled,
                last_act,
                last_last_act,
                last_last_last_act,
                motor_targets,
                contact,
                imitation_phase,
                PATH_ERROR_FIXED,
                projected_gravity,
            ]).astype(np.float32)
            assert obs.shape[0] == 107, f"obs 차원 {obs.shape[0]} != 107"

            if zero_action:
                action = np.zeros(14, dtype=np.float32)
            else:
                action = sess.run(None, {"obs": obs.reshape(1, 107)})[0].reshape(14)

            # 액션 저역필터 (docs/reports/lowpass_2026-08-09.md 실험 A). alpha=0이면
            # action_filt == action 그대로라 no-op. last_act(obs 이력)은 v35가 학습
            # 때 겪은 그대로 raw action 을 쓴다 — 필터는 target 계산에만 넣는다
            # (env 자체에 필터가 없는 v35 로는 이게 원 논문 실험 A 와 같은 구조).
            alpha = args.action_lpf_alpha
            action_filt = alpha * action_filt + (1.0 - alpha) * action

            target = READY_ARR + action_filt * ACTION_SCALE
            target[HEAD_IDX] = READY_ARR[HEAD_IDX]  # lock_head_joints=true

            motor_targets = np.clip(target, motor_targets - max_delta, motor_targets + max_delta)
            hwi.set_position_vec(motor_targets)

            log_w.writerow(
                [f"{time.time()-t_start:.4f}", step] + list(pos) + list(vel)
                + list(action) + list(motor_targets)
                + [contact[0], contact[1], gyro[0], gyro[1], gyro[2],
                   accel_arr[0], accel_arr[1], accel_arr[2],
                   projected_gravity[0], projected_gravity[1], projected_gravity[2],
                   imitation_phase[0], imitation_phase[1]])

            last_last_last_act = last_last_act
            last_last_act = last_act
            last_act = action.astype(np.float32)
            imitation_i = (imitation_i + 1) % GAIT_PERIOD_STEPS
            if np.linalg.norm(command[:3]) <= STANDSTILL_HOLD_THRESH:
                # 정지 명령에서는 위상을 0에 묶는다(joystick_env.py standstill_hold) —
                # 안 묶으면 정지 중에도 관측의 imitation_phase 가 계속 돌아, 학습 때
                # 없던 입력이 된다(정지 성능 2.7배 개선 조건, 실기서도 필수).
                imitation_i = 0

            step += 1
            took = time.time() - t0
            over_budget_count += 1 if took > DT else 0
            over_budget_worst = max(over_budget_worst, took - DT)
            if step % 50 == 0:
                hz = 50.0 / (time.time() - t_window0) if step > 0 else 0.0
                print(f"[rl_walk] t={time.time()-t_start:5.1f}s  {hz:4.1f}Hz  "
                      f"contact L{int(contact[0])}R{int(contact[1])}  "
                      f"gyro=({gyro[0]:+.2f},{gyro[1]:+.2f},{gyro[2]:+.2f})"
                      + (f"  예산초과 {over_budget_count}/50 (최대 +{over_budget_worst:.3f}s)"
                         if over_budget_count else ""),
                      flush=True)
                over_budget_count = 0
                over_budget_worst = 0.0
                t_window0 = time.time()
            time.sleep(max(0.0, DT - took))

        print(f"\n[rl_walk] 종료 — {step} 스텝 / {time.time()-t_start:.2f}s")
    finally:
        _hold["stop"] = True
        if "log_f" in locals() and not log_f.closed:
            log_f.close()
            print(f"[rl_walk] 스텝 로그 저장: {LOG_PATH}")
        try:
            hwi.hold_here()
            print("[rl_walk] 현재 위치 홀드. 토크는 켜둔 채로 둔다 — "
                  "풀려면 python3 ~/home_position.py --release")
        except Exception as e:
            print(f"[rl_walk] 종료 중 홀드 실패: {e}")
        feet.stop()


if __name__ == "__main__":
    main()
