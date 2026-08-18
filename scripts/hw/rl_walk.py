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
  * 전류 제한 (기본 700 unit = 1.88 A).

    **이 값은 학습 설정과 짝이다. 마음대로 낮추면 정책이 못 걷는다.**
    `robot_cfg.py` 의 `XM430_CONT_TORQUE_700 = 3.16 N·m` 이 바로 "실기가
    700 틱으로 돈다" 는 전제에서 나온 값이다 (τ = 1.96·(I − 0.27)).

    2026-08-12 까지 기본값이 350 (0.94 A → 1.32 N·m) 이었다. 즉 실기는
    심이 가정한 토크의 **42 %** 로 돌고 있었다. 심 정책은 무릎에 3.16 N·m
    를 쓸 수 있다고 배웠는데 실기는 1.32 N·m 에서 잘렸으니, 체중을 받는
    순간 무릎이 밀려 내려가고 앞으로 무너진다 — 그게 "앞으로 꼬구라진다"
    의 정체다. 위 문단은 원래 "주저앉으면 --current 를 올릴 것" 이라고
    적혀 있었는데, 정확히 그 증상이 났는데도 아무도 올리지 않았다.

    올릴 때의 대가: 700 틱은 스톨 전류 2.3 A 의 82 % 라 연속으로 물리면
    발열과 Overload 래치가 온다. 매 스텝 temp/hw_error 를 찍으니 60 °C
    를 넘으면 멈출 것. v42 에서 Overload 를 본 적이 있다.
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
from rustypot_hwi import (HWI, NAMES, LEG_NAMES, LEG_IDS, BY_NAME, TICK_RAD,
                          LEG_P_MATCHED, SIM_STIFFNESS, joint_stiffness)

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


def _ready_from_meta(meta: dict):
    """정책 메타의 ready_joint_pos 로 READY_ARR 을 덮는다.

    관측이 `joint_pos_rel = pos - READY` 라 READY 가 학습 설정과 다르면 정책이
    **전 관절에서 밀린 관측**을 받는다. 2026-08-12 에 실제로 겪었다 — 잿슨이
    v40 자세(left_knee -1.5693)를 든 채 v41 정책(-1.7483)을 돌려 무릎 관측이
    10.3도 어긋났고, 정책이 계속 무릎을 펴려 밀어 발이 안 뜨고 앞으로 고꾸라졌다.
    하드코딩을 남겨 두면 반드시 다시 어긋나므로 메타를 우선한다.
    """
    r = (meta or {}).get("ready_joint_pos") or {}
    if not r:
        return None
    missing = [n for n in NAMES if n not in r]
    if missing:
        print(f"[rl_walk] !! 메타 ready_joint_pos 에 빠진 관절 {missing} — 내장 표를 쓴다")
        return None
    arr = np.array([r[n] for n in NAMES], dtype=np.float32)
    diff = np.degrees(arr - READY_ARR)
    if np.abs(diff).max() > 0.05:
        print("[rl_walk] READY 를 정책 메타에서 가져왔다. 내장 표와 다른 축:")
        for n, dv in zip(NAMES, diff):
            if abs(dv) > 0.05:
                print(f"           {n:16} {dv:+7.2f}도")
    else:
        print("[rl_walk] READY 메타 = 내장 표 (차이 없음)")
    return arr
HEAD_IDX = [NAMES.index(n) for n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]
LEG_IDX = [NAMES.index(n) for n in LEG_NAMES]

ACTION_SCALE = 0.25
DOF_VEL_SCALE = 0.05
MAX_MOTOR_VEL = 4.82        # rad/s, joystick_env_cfg.py max_motor_velocity (전 계열 공통)
GAIT_PERIOD_STEPS = 27      # ref_g125 레퍼런스 주기, 0.54s @ 50Hz (joystick_env.py._gait_period_steps)
STANDSTILL_HOLD_THRESH = 0.01  # ‖command[:3]‖ 이하면 위상을 0(=cos,sin=1,0)에 고정
CONTROL_HZ = 50.0
#: 이 아래로 떨어지면 브라운아웃으로 보고 멈춘다 (V).
#: XM430 의 Min Voltage Limit 레지스터가 95 = 9.5 V 이고 그 아래는 규격 밖이라
#: 명령해도 토크가 안 난다. 여유를 조금 두고 10.0 으로 잡았다.
BROWNOUT_V = 10.0
#: 몇 번 연속으로 낮아야 멈출지. VOLT_PERIOD(25 스텝 = 0.5 초) × 이 값.
BROWNOUT_HOLD = 2
#: 심이 관절 하나에 허용하는 최대 토크 (N·m). robot_cfg.py 의
#: XM430_CONT_TORQUE_700 (DCMotor 계열) 과 같은 값. 메타에 effort_limit 이
#: 있으면 그쪽이 이긴다 — ImplicitActuator 계열(v42/v44/v46)은 4.1 이다.
#: 실기 --current 를 이 값과 맞추라고 시작할 때 비교해서 찍는다.
SIM_EFFORT_LIMIT = 3.16
DT = 1.0 / CONTROL_HZ
NUM_COMMANDS = 7             # vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll (뒤 4개 항상 0)

# path_error 고정값 (2026-08-08 결정, docstring 참고): [lateral=0, cos(yaw_err)=1, sin(yaw_err)=0]
PATH_ERROR_FIXED = np.array([0.0, 1.0, 0.0], dtype=np.float32)

# --path-imu: 방향 오차만 자이로로 복원한다 (2026-08-18).
#
# 사용자 증상: "회전이 잘 안 되고, 게걸음을 시키면 애가 회전을 한다. 그냥
# 앞으로 갈 때도 양옆으로 드리프팅한다."  셋 다 **방향** 문제다.
#
# path_error 는 [횡방향오차, cos(방향오차), sin(방향오차)] 인데 실기에서는
# [0,1,0] 상수 — 정책에게 "완벽히 잘 가고 있다" 고 계속 거짓말을 해 왔다.
# 그러니 틀어져도 관측으로 안 들어가서 보정할 방법이 없다.
#
# 횡방향 오차는 오도메트리가 필요해 못 만든다. 그러나 **방향 오차 두 성분은
# 자이로 z 적분으로 만들 수 있다.** 실기 로그로 부호도 확인했다 —
# cmd_wz > 0 일 때 gyro_z 평균 +0.186, cmd_wz < 0 일 때 -0.262 로 부호가
# 일치하므로 그대로 적분하면 된다.
#
# 무명령 구간 gyro_z 평균이 +0.0048 rad/s (0.27 도/s) 라 바이어스 보정이
# 필요하다. arm 직후 정지 상태에서 평균을 내어 뺀다.
# 자이로 z 바이어스. **추정하지 않는다.**
#
# 1 차 구현은 arm 직후 1 초 평균으로 재려 했는데, 그때 로봇은 rampin 으로
# READY 자세를 잡느라 **움직이고 있다.** 실제로 -0.0203 rad/s (-1.16 도/s) 가
# 나왔고, 35.9 초 정지 주행에서 그것이 +41.7 도의 가짜 회전으로 쌓여
# yaw_err 가 +32.2 도까지 갔다 (실제 회전은 -7.3 도뿐이었다). 사용자가 본
# "발이 조금씩 돌아간다" 가 이것이다.
#
# 토크를 끈 정지 상태에서 imu_check 로 재 보니 gyro_z 는 거의 0 이다.
# 그래서 0 으로 둔다. 필요하면 --gyro-bias 로 넣는다.
GYRO_BIAS_DEFAULT = 0.0


def _wrap(a: float) -> float:
    """(-pi, pi] 로 감는다."""
    return math.atan2(math.sin(a), math.cos(a))

# 2026-08-09: 오른다리(특히 hip_roll/yaw)가 실기에서 왼다리와 다르게 움직이는
# 문제를 사후분석하려고 매 스텝을 CSV로 남긴다. 매번 덮어쓴다 — 직전 실행만 본다.
LOG_PATH = os.path.expanduser("~/rl_walk_log.csv")

# ── IMU (BNO055, imu_check.py 와 같은 레지스터) ─────────────────────────────
IMU_BUS = 7
IMU_ADDR = 0x28
REG_ACC_DATA = 0x08
REG_GRV_DATA = 0x2E   # NDOF 융합이 분리해 낸 **중력만** 벡터
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
        """(gyro[rad/s], accel[m/s^2, 중력 포함], grav[m/s^2, 중력만]) — 몸통 프레임.

        accel 과 grav 를 **따로** 돌려주는 이유는 심에서 이 둘의 출처가 다르기
        때문이다 (joystick_env.py._get_observations):
          - 관측의 accelerometer 3칸 = `imu.data.lin_acc_b` — 접지 충격까지 들어간
            더러운 센서값이다. 실기도 생 ACC_DATA 를 그대로 써야 맞다.
          - 관측의 projected_gravity 3칸 = `robot.data.projected_gravity_b` —
            몸통 쿼터니언에서 뽑은 **참 중력방향**이고, 노이즈는 ±0.1 균등뿐이다.
            실기에서 생 가속도를 정규화해 쓰면 접지 충격이 그대로 섞여, 20 ms 에
            30° 씩 튀는 값이 학습 분포 밖에서 들어간다.
        BNO055 는 NDOF 융합으로 중력만 분리한 벡터를 GRV_DATA(0x2E)에 이미
        내놓고 있다 (2026-08-10 실측: 크기 9.806, std 0.0000 / 생 ACC 는 9.746,
        std 0.023). 주소 하나 차이다.
        """
        a = self._vec(REG_ACC_DATA, ACC_LSB)
        v = self._vec(REG_GRV_DATA, ACC_LSB)
        g = self._vec(REG_GYR_DATA, GYR_LSB)
        g = tuple(x * math.pi / 180.0 for x in g)
        m = self.axis_matrix
        rmp = lambda u: tuple(sum(m[i][j] * u[j] for j in range(3)) for i in range(3))
        return rmp(g), rmp(a), rmp(v)


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
    ap.add_argument("--current", type=int, default=700,
                     help="Goal Current 상한 (1 unit=2.69mA). 기본 700 = 1.88 A = 3.16 N·m 로 "
                          "심의 effort_limit 과 맞춘 값이다 — 낮추면 정책이 배운 토크를 "
                          "못 내고 무릎이 밀린다 (독스트링 참고). 발열/Overload 시에만 낮출 것")
    ap.add_argument("--pgain", type=int, default=LEG_P_MATCHED,
                     help=f"다리 10축 Position P Gain. 기본 {LEG_P_MATCHED} = 심의 "
                          f"stiffness 37.65 N·m/rad 와 같은 강성. 0 을 주면 펌웨어 "
                          f"기본값 800(=21.5 N·m/rad, 심의 57%%)을 그대로 둔다")
    ap.add_argument("--gyro-bias", type=float, default=GYRO_BIAS_DEFAULT,
                    help="자이로 z 바이어스(rad/s). 기본 0 — 실측상 무시할 수준이다.")
    ap.add_argument("--path-imu", action="store_true",
                    help="방향 오차를 자이로 z 적분으로 복원해 path_error 에 넣는다 "
                         "(기본은 [0,1,0] 상수). 회전·드리프트 보정용.")
    ap.add_argument("--dgain", type=int, default=0,
                     help="다리 10축 Position D Gain. 0 = 펌웨어 기본값 4700 유지. "
                          "머리 3축은 2026-08-09 계단응답으로 2000 이 최적이었다")
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
    ap.add_argument("--grav-src", choices=("fused", "accel"), default="fused",
                    help="projected_gravity 관측의 출처. fused=BNO055 GRV_DATA(0x2E, "
                         "융합이 분리한 중력만·기본), accel=생 가속도 정규화(예전 동작). "
                         "관측의 accelerometer 3칸은 어느 쪽이든 항상 생 가속도다.")
    ap.add_argument("--cmd-udp-port", type=int, default=None, metavar="PORT",
                    help="심(play_fixed_cmd.py --cmd-udp)이 보내는 속도 명령을 이 포트로 "
                         "받아서 --vx/--vy/--wz 대신 쓴다. 같은 조이스틱 입력으로 심과 "
                         "실기를 나란히 보려는 것. 패킷이 300 ms 끊기면 자동으로 정지 "
                         "명령이 되고, 패드 A 를 누르면 즉시 홀드로 빠진다.")
    ap.add_argument("--joy", nargs="?", const="/dev/input/js0", default=None,
                    metavar="DEV",
                    help="젯슨에 직접 꽂은 조이스틱으로 명령을 받는다 (기본 "
                         "/dev/input/js0). RT 를 당긴 동안만 명령이 나가고, 패드가 "
                         "끊기면 즉시 정지한다. --vx/--vy/--wz 나 --cmd-udp-port 와는 "
                         "같이 못 쓴다 — 명령원이 둘이면 어느 쪽이 이겼는지 알 수 없다. "
                         "붙이는 법은 bt_pad.py 참고.")
    ap.add_argument("--joy-map", default=None, metavar="k=n,…",
                    help="조이스틱 버튼 번호 덮어쓰기 (estop=0,start=11,…). "
                         "기본은 버튼 개수를 보고 자동으로 고른다")
    ap.add_argument("--joy-no-deadman", action="store_true",
                    help="RT 를 안 잡아도 명령이 나간다 (권하지 않는다)")
    ap.add_argument("--seconds", type=float, default=20.0, help="정책 루프 실행 시간(초)")
    ap.add_argument("--dry-run", action="store_true", help="모터/IMU 안 건드리고 로드만 확인")
    args = ap.parse_args()

    # 명령원은 하나여야 한다. 고정 명령과 조이스틱을 같이 주면 매 스텝 조이스틱이
    # command[:3] 을 덮어써서 --vx 가 조용히 무시된다 — 로그만 보고는 왜 안 갔는지
    # 알 수 없다. 기본값 0 과 "직접 0 을 준 것" 을 구분해야 하므로 argv 를 본다.
    if args.joy is not None:
        given = [f for f in ("--vx", "--vy", "--wz")
                 if any(a == f or a.startswith(f + "=") for a in sys.argv)]
        if given:
            ap.error(f"--joy 와 {'/'.join(given)} 는 같이 못 쓴다. "
                     f"조이스틱을 쓰려면 속도 인자를 빼고, 고정 명령으로 돌리려면 "
                     f"--joy 를 빼라.")
        if args.cmd_udp_port is not None:
            ap.error("--joy 와 --cmd-udp-port 는 같이 못 쓴다. 조이스틱을 젯슨에 "
                     "직접 꽂았으면 --joy, 심에서 중계받으려면 --cmd-udp-port 다.")

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
    # 명령이 고정이면 한 번만 계산하면 된다. --cmd-udp-port 로 명령이 매 스텝
    # 바뀌면 아래 _alpha_for() 로 루프 안에서 다시 구한다.
    a_move = float(meta.get("action_lowpass_alpha", 0.0))
    a_still = float(meta.get("action_lowpass_alpha_standstill", a_move))
    blend_lo, blend_hi = meta.get("action_lowpass_blend", [0.01, 0.05])
    cmd_norm = float(np.linalg.norm([args.vx, args.vy, args.wz]))
    _t = min(1.0, max(0.0, (cmd_norm - blend_lo) / max(blend_hi - blend_lo, 1e-6)))
    _t = _t * _t * (3.0 - 2.0 * _t)
    alpha_auto = a_still + (a_move - a_still) * _t

    def _alpha_for(cmd3):
        """이 명령 크기에서 쓸 저역통과 알파 (시뮬 _pre_physics_step 과 같은 식)."""
        n = float(np.linalg.norm(cmd3))
        t = min(1.0, max(0.0, (n - blend_lo) / max(blend_hi - blend_lo, 1e-6)))
        t = t * t * (3.0 - 2.0 * t)
        return a_still + (a_move - a_still) * t

    # CLI 로 알파를 직접 준 경우엔 명령이 바뀌어도 그 값을 유지한다.
    args.action_lpf_alpha_auto = args.action_lpf_alpha is None
    # READY 를 메타에서 덮어쓴다 (관측 기준점이라 어긋나면 전 관절이 밀린다).
    _r = _ready_from_meta(meta)
    if _r is not None:
        READY_ARR[:] = _r

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

    # ── 나머지 배포 계약도 메타에서 받는다 ─────────────────────────────────
    # READY 는 위에서 _ready_from_meta() 가 이미 덮었다. 스케일 3개는 지금까지
    # 전 계열이 같은 값이었지만 정책마다 바뀔 수 있는 값이라 같이 받는다.
    # globals() 로 쓰는 이유: READY_ARR 을 위에서 이미 참조해 global 선언을 못 건다.
    for key, name in (("action_scale", "ACTION_SCALE"),
                      ("dof_vel_scale", "DOF_VEL_SCALE"),
                      ("max_motor_velocity", "MAX_MOTOR_VEL")):
        if meta.get(key) is not None:
            cur = globals()[name]
            if abs(float(meta[key]) - cur) > 1e-9:
                print(f"[rl_walk] {name} {cur} -> {meta[key]} (meta)")
            globals()[name] = float(meta[key])
    if meta.get("lock_head_joints") is False:
        print("[rl_walk] !! 경고: 이 정책은 lock_head_joints=false 로 학습됐는데 "
              "이 스크립트는 머리를 항상 READY 로 고정한다 — 학습/배포 불일치다.")

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

    # 명령 수신기를 **모터 토크를 켜기 전에** 연다. 포트가 이미 물려 있거나
    # 하면 여기서 죽는데, 그때 로봇은 아직 아무것도 안 한 상태여야 한다.
    cmd_rx = None
    cmd_src = "심 중계" if args.cmd_udp_port is not None else "조이스틱"
    if args.cmd_udp_port is not None:
        from cmd_udp import CommandReceiver
        cmd_rx = CommandReceiver(port=args.cmd_udp_port)
        print(f"[rl_walk] 명령을 UDP {args.cmd_udp_port} 에서 받는다 — "
              f"--vx/--vy/--wz 는 무시된다. 패킷이 오기 전까지는 정지 명령이다.")
    elif args.joy is not None:
        # joy_local.JoystickCommand 는 CommandReceiver 와 같은 인터페이스
        # (get / stale / estopped / stats / close) 라 아래 루프는 그대로다.
        from joy_local import JoystickCommand, default_buttons
        if not os.path.exists(args.joy):
            raise SystemExit(f"{args.joy} 이 없다. 패드를 먼저 붙일 것:  "
                             f"python3 ~/bt_pad.py")
        btn = default_buttons(args.joy)
        if args.joy_map:
            for part in args.joy_map.split(","):
                k, _, v = part.partition("=")
                if k.strip() in btn and v.strip().isdigit():
                    btn[k.strip()] = int(v)
        cmd_rx = JoystickCommand(dev=args.joy, buttons=btn,
                                 deadman=not args.joy_no_deadman)
        print(f"[rl_walk] 명령을 조이스틱 {args.joy} 에서 받는다 — "
              f"RT 를 당긴 동안만 움직인다. A 를 누르면 즉시 홀드.")

    hwi = HWI(port=args.port, current_limit=args.current,
              leg_p=args.pgain or None, leg_d=args.dgain or None)
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

        # 지금 자세가 이 정책의 READY 와 얼마나 다른가. goto_ready 를 **다른**
        # 정책으로 돌려 놓고 이걸 켜면 램프인이 그 차이를 3 초에 메우면서
        # 로봇을 확 주저앉힌다. goto_ready 는 인자를 안 주면 '메타가 가장
        # 최근인' 정책을 고르므로, 정책을 새로 뽑을 때마다 그 선택이 바뀐다
        # — 조용히 어긋나기 딱 좋은 자리라 여기서 눈에 보이게 만든다.
        _dd = np.degrees(start_pos - READY_ARR)
        _big = [(n, v) for n, v in zip(NAMES, _dd) if abs(v) > 15.0]
        if _big:
            print(f"[rl_walk] 시작 자세가 이 정책의 READY 와 다르다 "
                  f"(최대 {max(_dd, key=abs):+.1f}°) — 램프인 {args.rampin:.1f}초 동안 메운다:")
            for n, v in sorted(_big, key=lambda x: -abs(x[1]))[:6]:
                print(f"           {n:16} {v:+7.1f}°")
            print(f"           로봇이 쓰러져 있으면 정상이다. 서 있는데 이게 뜨면 "
                  f"goto_ready 를 다른 정책으로 돌린 것이다 — "
                  f"goto_ready.py --policy <이 정책 폴더> 로 다시 할 것.")

        # 토크를 켜기 **전에** 전압부터 본다. 무부하에서도 낮으면 부하가 걸리는
        # 순간 규격 밖으로 떨어진다 — 2026-08-12 에 무부하 11.3 V 로 시작해
        # 2 초 만에 8.7 V 로 무너졌다. 무부하 11.3 은 3S 리포로 치면 셀당
        # 3.77 V, 이미 절반쯤 쓴 상태였다.
        _vb = hwi.io.sync_read_raw_data(LEG_IDS, 144, 2)
        v0 = [(b[0] | (b[1] << 8)) / 10.0 for b in _vb if len(b) == 2]
        if v0:
            print(f"[rl_walk] 무부하 전압 {min(v0):.1f}~{max(v0):.1f} V")
            if min(v0) < BROWNOUT_V + 1.5:
                print(f"[rl_walk] !! 경고: 무부하에서 이미 {min(v0):.1f} V 다. 보행 중 다리 "
                      f"10축 합계가 순간 5 A 를 넘으므로 곧 {BROWNOUT_V:.1f} V 아래로 "
                      f"떨어진다. 충전하거나 파워서플라이 전류한계를 올리고 다시 할 것.")

        hwi.arm()
        # 전류 상한을 **토크로 환산해서** 찍는다. 암페어만 보면 심의 effort_limit
        # 과 비교할 수가 없어서, 2026-08-12 까지 실기가 심의 42 % 토크로 도는 걸
        # 아무도 눈치채지 못했다. τ = 1.96·(I − 0.27), robot_cfg.py 와 같은 식.
        amp = args.current * 2.69 / 1000.0
        tau = 1.96 * max(amp - 0.27, 0.0)
        sim_tau = float(meta.get("effort_limit") or SIM_EFFORT_LIMIT)
        print(f"[rl_walk] 토크 켬 14축 · 전류 상한 {amp:.2f} A -> {tau:.2f} N·m "
              f"(심 effort_limit {sim_tau:.2f} N·m 의 {100*tau/sim_tau:.0f} %)")
        if tau < 0.9 * sim_tau:
            print(f"[rl_walk] !! 경고: 실기 토크가 심보다 {sim_tau/max(tau,1e-6):.1f}배 작다. "
                  f"정책은 {sim_tau:.2f} N·m 를 쓸 수 있다고 배웠다 — 체중을 받는 순간 "
                  f"무릎이 밀린다. --current {int((sim_tau/1.96 + 0.27)*1000/2.69)} 로 맞출 것.")

        # 게인은 **써 놓고 끝내지 않고 다시 읽어서** 확인한다. 모드 전환이
        # 게인을 덮으므로 (rustypot_hwi 주석 참고) 순서가 어긋나면 조용히
        # 무효가 되는데, 읽어 보지 않으면 그걸 알 방법이 없다.
        gains = hwi.read_leg_gains()
        ps = [p for p, _d in gains if p is not None]
        if ps:
            lo, hi = min(ps), max(ps)
            kp = joint_stiffness(lo)
            tag = f"{lo}" if lo == hi else f"{lo}~{hi} (축마다 다르다!)"
            print(f"[rl_walk] 다리 P gain {tag} -> 관절강성 {kp:.1f} N·m/rad "
                  f"(심 stiffness {SIM_STIFFNESS:.2f} 의 {100*kp/SIM_STIFFNESS:.0f} %)"
                  f" · D gain {gains[0][1]}")
            if args.pgain and lo != args.pgain:
                print(f"[rl_walk] !! 경고: --pgain {args.pgain} 을 걸었는데 실제로는 {lo} 이다 "
                      f"— 모드 전환이 덮었을 수 있다.")
            # 강성이 크게 어긋나면 정책이 명령한 위치에 실제 관절이 못 간다.
            if abs(kp - SIM_STIFFNESS) > 0.15 * SIM_STIFFNESS:
                print(f"[rl_walk] !! 경고: 실기 관절이 심보다 "
                      f"{SIM_STIFFNESS/max(kp,1e-6):.2f}배 무르다 — 같은 부하에서 그만큼 "
                      f"더 밀린다. --pgain {LEG_P_MATCHED} 로 맞출 것.")
        # 전류 상한이 만드는 **포화 각도** — 이 각도를 넘는 오차는 토크가 더
        # 안 늘어난다. 350 틱일 때 겨우 4.9° 였다.
        if ps:
            sat_deg = math.degrees(args.current / (lo / 128.0) * TICK_RAD)
            print(f"[rl_walk] 위치오차 {sat_deg:.1f}° 를 넘으면 전류 상한에 포화한다 "
                  f"(그 너머는 오차가 커져도 토크가 안 는다)")

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
        _cmd_seen = False
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
        # 액션->하드웨어 파이프라인의 **모든 단계**를 남긴다. 어느 한 단계라도
        # 빼면 나중에 되짚어 복원해야 하고, 그 복원에서 틀리기 쉽다
        # (2026-08-10: 클램프 전/후 목표를 혼동해 두 번 오진).
        #   action -> actf(저역필터) -> tgt(READY+scale, 머리고정)
        #          -> mtgt(속도제한 클립) -> goal(tick_of URDF 클램프, 실제 전송값)
        log_w.writerow(
            ["t", "step", "dt", "imitation_i",
             "ms_imu", "ms_read", "ms_infer", "ms_write", "ms_total",
             # --path-imu 상태. 켰는지조차 로그로 확인할 수 없었다 (2026-08-18).
             "path_yaw_err", "gyro_bias"]
            + [f"pos_{n}" for n in NAMES] + [f"vel_{n}" for n in NAMES]
            + [f"action_{n}" for n in NAMES] + [f"actf_{n}" for n in NAMES]
            + [f"tgt_{n}" for n in NAMES] + [f"mtgt_{n}" for n in NAMES]
            + [f"goal_{n}" for n in NAMES]
            + [f"cur_{n}" for n in LEG_NAMES] + [f"pwm_{n}" for n in LEG_NAMES]
            + [f"tqen_{n}" for n in LEG_NAMES] + [f"err_{n}" for n in LEG_NAMES]
            + [f"volt_{n}" for n in LEG_NAMES] + [f"temp_{n}" for n in LEG_NAMES]
            + ["contact_l", "contact_r", "gyro_x", "gyro_y", "gyro_z",
               "accel_x", "accel_y", "accel_z", "proj_grav_x", "proj_grav_y", "proj_grav_z",
               "phase_cos", "phase_sin",
               # 실제로 쓴 명령. 2026-08-10: 심은 걷는데 실기는 안 걷는 걸
               # 추적하려는데 명령이 안 남아 있어서 원인을 못 좁혔다.
               "cmd_vx", "cmd_vy", "cmd_wz", "cmd_stale"])
        max_delta = MAX_MOTOR_VEL * DT

        # 로그 옆에 실행 조건을 통째로 남긴다. CSV 만 나중에 봐도 어떤 정책·
        # 어떤 상한·어떤 상수로 돈 건지 되짚을 수 있어야 갭 비교가 가능하다.
        side = {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "onnx": os.path.abspath(args.onnx) if args.onnx else None,
            "policy_meta": meta,
            "command": {"vx": args.vx, "vy": args.vy, "wz": args.wz},
            "action_lpf_alpha": args.action_lpf_alpha,
            "grav_src": args.grav_src,
            "current_limit_ticks": args.current,
            "current_limit_A": args.current * 2.69 / 1000.0,
            # 실제로 서보에서 읽어 온 값이다 (명령값이 아니라). 모드 전환이
            # 게인을 덮을 수 있어서 명령값을 남기면 거짓말이 된다.
            "leg_p_gain": [p for p, _d in gains],
            "leg_d_gain": [d for _p, d in gains],
            "joint_stiffness_Nm_rad": joint_stiffness(gains[0][0]) if gains[0][0] else None,
            "sim_stiffness_Nm_rad": SIM_STIFFNESS,
            "torque_ceiling_Nm": 4.1 * (args.current * 2.69 / 1000.0) / 2.3,
            "constants": {
                "ACTION_SCALE": ACTION_SCALE, "DOF_VEL_SCALE": DOF_VEL_SCALE,
                "MAX_MOTOR_VEL": MAX_MOTOR_VEL, "GAIT_PERIOD_STEPS": GAIT_PERIOD_STEPS,
                "CONTROL_HZ": CONTROL_HZ, "STANDSTILL_HOLD_THRESH": STANDSTILL_HOLD_THRESH,
                "max_delta_rad": float(max_delta),
            },
            "ready_joint_pos": {n: float(v) for n, v in zip(NAMES, READY_ARR)},
            "contract_source": "policy.meta.json" if meta.get("ready_joint_pos") else "rl_walk.py 하드코딩",
            "joint_order": list(NAMES),
            "leg_order": list(LEG_NAMES),
            "urdf_limits_deg": {n: [math.degrees(BY_NAME[n][3]), math.degrees(BY_NAME[n][4])]
                                for n in NAMES},
            "shutdown_mask": list(hwi.shutdown_mask),
            "imu_axis_matrix": imu.axis_matrix,
            "imu_calib_stat": imu._rd(0x35, 1)[0],
            "notes": "각도는 전부 rad(URDF 관절각). 파이프라인 단계: "
                     "action -> actf(저역필터) -> tgt(READY+scale, 머리고정) -> "
                     "mtgt(속도제한) -> goal(URDF 클램프, 실제 전송). "
                     "err/volt/temp 는 느린 주기로 읽고 사이는 직전 값 유지. "
                     "머리 4축 pos/vel 은 안 읽는다(lock_head_joints) — READY/0 으로 채운 값.",
        }
        with open(os.path.splitext(LOG_PATH)[0] + ".meta.json", "w") as _sf:
            json.dump(side, _sf, ensure_ascii=False, indent=1)

        t_start = time.time()
        step = 0
        over_budget_count = 0
        over_budget_worst = 0.0
        t_window0 = time.time()
        nonfatal_seen = set()
        fatal_pending = None
        # --path-imu 상태. 로봇 방향은 자이로 z 적분, 경로 방향은 명령 적분.
        robot_yaw = 0.0
        path_yaw = 0.0
        gyro_bias = args.gyro_bias
        path_err_arr = PATH_ERROR_FIXED
        path_yaw_err = 0.0
        tqen = [1] * len(LEG_NAMES)
        leg_err = [0] * len(LEG_NAMES)
        # 전압/온도는 초 단위로만 변하니 느린 주기로 읽고 사이에는 직전 값을 쓴다.
        volts = [0.0] * len(LEG_NAMES)
        temps = [0] * len(LEG_NAMES)
        # 100(2초) 이었는데 25(0.5초) 로 당겼다 — 브라운아웃 가드가 여기 붙어
        # 있어서다. 2026-08-12 에 전원이 2 초 만에 무너졌는데 2 초 주기로는
        # 그걸 한 번 읽고 끝난다. SyncRead 3바이트×10축이 ~16 ms 라 25 스텝
        # 주기면 스텝당 0.6 ms, 20 ms 예산의 3 % 다.
        VOLT_PERIOD = 25
        brownout = 0
        goal_sent = READY_ARR.copy()
        ms = dict(imu=0.0, read=0.0, infer=0.0, write=0.0)
        t_prev = None
        # 이 버스에서 hw_error SyncRead 하나가 ~16ms 다. 2026-08-09 로그 실측:
        # step%10==0 인 루프만 84.9ms, 나머지는 20.2ms 였다(전체의 10%, 100초 중
        # 25초 손실). 그 루프에서 read 를 4번 했기 때문이다 —
        #   get_fatal_errors() -> get_hw_errors() 2회(에러비트가 떠 있으면 재확인)
        #   + 아래에서 get_hw_errors() 또 2회.
        # 이제 루프당 read 는 1회고(confirm=False), 주기도 늘렸다. 과열/과부하는
        # 초 단위로 쌓이는 현상이라 0.5초에 한 번이면 충분하다.
        HW_ERR_PERIOD = 25
        # 위치+속도는 매 루프 필수(관측벡터에 직접 들어감) — 한 번의 SyncRead로 묶는다.
        # hw_error 는 안전 확인용이라 몇 루프에 한 번이면 충분하다.
        while not _hold["stop"] and (time.time() - t_start) < args.seconds:
            t0 = time.time()

            if cmd_rx is not None:
                if not _cmd_seen and not cmd_rx.stale:
                    _cmd_seen = True
                    print(f"[rl_walk] ** {cmd_src} 첫 명령 도착 **", flush=True)
                if cmd_rx.estopped:
                    print("\n[rl_walk] !! 패드 A 비상정지 — 즉시 홀드한다", flush=True)
                    break
                # 워치독이 걸렸으면 get() 이 (0,0,0) 을 준다. command[3:] (머리
                # 4칸)는 학습에서 항상 0 이므로 건드리지 않는다.
                command[0], command[1], command[2] = cmd_rx.get()
                if args.action_lpf_alpha_auto:
                    args.action_lpf_alpha = _alpha_for(command[:3])

            _ta = time.time()
            gyro, accel, grav = imu.read()
            if args.path_imu:
                if np.linalg.norm(command[:3]) <= STANDSTILL_HOLD_THRESH:
                    # 정지는 정지다 (2026-08-18, 사용자 결정).
                    #
                    # 래치(멈춘 순간 방향을 목표로 붙잡기)도 해 봤지만 사용자가
                    # "정지는 잘되는데 오차 때문에 발이 조금씩 돌아간다" 고 했다.
                    # 로봇은 실제로 안 도는데(35.9 초에 -7.3 도) 적분 오차만
                    # 쌓여서 정책이 헛되이 되돌리려 한 것이다. 명령이 없으면
                    # 따라갈 경로도 없으니 오차는 0 이 맞다.
                    robot_yaw = path_yaw = 0.0
                    path_err_arr = PATH_ERROR_FIXED
                    path_yaw_err = 0.0
                else:
                    robot_yaw = _wrap(robot_yaw + (float(gyro[2]) - gyro_bias) * DT)
                    # 경로 방향은 **명령**을 적분한다 — 심의 path frame 과 같다.
                    path_yaw = _wrap(path_yaw + float(command[2]) * DT)
                    ye = _wrap(robot_yaw - path_yaw)
                    path_err_arr = np.array(
                        [0.0, math.cos(ye), math.sin(ye)], dtype=np.float32)
                    path_yaw_err = ye
            _tb = time.time()
            leg_pos, leg_vel, leg_cur, leg_pwm = hwi.get_leg_pos_vel()
            _tc = time.time()
            ms["imu"] = (_tb - _ta) * 1e3
            ms["read"] = (_tc - _tb) * 1e3
            # 머리 4축은 lock_head_joints=true 라 안 읽는다(SyncRead 14->10축,
            # 버스 시간 절약) — 항상 READY 로 고정 명령 나가니 rel=0/vel=0 으로 채운다.
            pos = READY_ARR.copy()
            vel = np.zeros(14, dtype=np.float32)
            pos[LEG_IDX] = leg_pos
            vel[LEG_IDX] = leg_vel
            contact = np.array(feet.get(), dtype=np.float32)

            if step % HW_ERR_PERIOD == 0 or fatal_pending is not None:
                # Shutdown 마스크에 걸린 에러(과열/전기충격/과부하)만 정지 사유다.
                # 마스크 밖 비트는 모터가 계속 도는 상태라 멈출 이유가 없다.
                #
                # 2026-08-09: 이 자리에서 Input Voltage(bit0) 를 보고 서보를
                # reboot 했었는데, 그 비트는 이 로봇 Shutdown 마스크에 없어서
                # 토크를 끊지도 않는 정보성 비트였다. 불필요한 복구가 보행 중
                # 1초에 한 번씩 400ms 씩 제어를 멈춰 로봇이 발을 구르며 주저앉는
                # 원인이 됐다. 이제 세지만 하고 넘어간다.
                # addr 64(torque_enable) ~ 70(hw_error) 은 7바이트 연속이라 한 번에
                # 읽는다. 토크가 런 중에 꺼지는지 보려는 것 — 2026-08-10 보행에서
                # left_knee 가 PWM/전류 0 으로 멈춰 서는 구간이 나왔다.
                blk = hwi.io.sync_read_raw_data(LEG_IDS, 64, 7)
                tqen = [b[0] for b in blk]
                leg_err = [b[6] for b in blk]
                # 치명 판정은 머리까지 포함해야 하므로 14축 이름 순서로 되돌린다.
                raw_err = [0] * 14
                for k, idx in enumerate(LEG_IDX):
                    raw_err[idx] = leg_err[k]
                fatal = hwi.mask_fatal(raw_err)
                if any(fatal):
                    if fatal_pending is None:
                        # 응답이 쪼개져 온 한 번의 오파싱으로 보행 중에 멈추면
                        # 그게 더 위험하다. 다음 루프에서 곧바로 다시 읽고,
                        # 같은 축이 두 번 연속일 때만 정지한다(확인을 한 루프
                        # 안에서 하지 않고 루프 사이로 나눈다 — 예산 때문에).
                        fatal_pending = fatal
                    else:
                        bad = [(n, e) for n, e, p in zip(NAMES, fatal, fatal_pending) if e and p]
                        fatal_pending = None
                        if bad:
                            print(f"[rl_walk] !! 치명 하드웨어 에러 {bad} — 정지한다")
                            break
                else:
                    fatal_pending = None
                if not nonfatal_seen:
                    nonfatal = [n for n, e in zip(NAMES, raw_err) if e]
                    if nonfatal:
                        nonfatal_seen = set(nonfatal)
                        print(f"[rl_walk] (참고) 비치명 에러 비트 — {sorted(nonfatal_seen)}. "
                              f"토크는 안 끊긴다. 보행 부하로 전압이 처진 흔적일 수 있다.")

            if step % VOLT_PERIOD == 0:
                # addr 144(present_input_voltage, 2B) + 146(present_temperature, 1B)
                # 이 연속이라 3바이트 한 번에. 브라운아웃 이력이 있어 전압은
                # 갭 분석에 필요하고, 온도는 전류 상한을 올린 뒤 안전 확인용이다.
                vb = hwi.io.sync_read_raw_data(LEG_IDS, 144, 3)
                volts = [(b[0] | (b[1] << 8)) / 10.0 for b in vb]
                temps = [b[2] for b in vb]

                # ── 브라운아웃 가드 ────────────────────────────────────────
                # 2026-08-12: v46 런에서 t=2.0 s 에 전원이 11.3 V -> 8.7 V 로
                # 무너지고 끝까지 회복하지 않았다. XM430 최소 동작전압은 9.5 V
                # (Min Voltage Limit 레지스터 95) 라 그 아래에서는 규격 밖이다.
                # 그 상태의 로그는 전부 못 믿는다 — 실제로 무릎이 PWM 100 %
                # 포화인데 전류가 0.003 A 였고, 전 축 InputVoltage 에러가 떠
                # 있었다. 그걸 모르고 게인·전류상한·기구를 며칠 뒤졌다.
                # 그러니 여기서 멈춘다. 낮은 전압으로 계속 돌려 봐야 얻을
                # 데이터가 없고, 배터리만 더 망가진다.
                vlo = min(volts) if volts else 99.0
                if vlo < BROWNOUT_V:
                    brownout += 1
                    if brownout == 1:
                        print(f"\n[rl_walk] !! 전압 {vlo:.1f} V — 최소 동작전압 "
                              f"{BROWNOUT_V:.1f} V 아래다. {BROWNOUT_HOLD}회 연속이면 멈춘다.",
                              flush=True)
                    if brownout >= BROWNOUT_HOLD:
                        print(f"\n[rl_walk] !! 브라운아웃 정지: 전압 {vlo:.1f} V "
                              f"(축별 {['%.1f' % v for v in volts]})\n"
                              f"           이 아래에서는 명령해도 토크가 안 난다. "
                              f"배터리 충전/교체 또는 파워서플라이 전류한계를 볼 것 "
                              f"(다리 10축 합계가 순간 5 A 를 넘는다).", flush=True)
                        _hold["stop"] = True
                else:
                    brownout = 0

            joint_pos_rel = pos - READY_ARR
            joint_vel_scaled = vel * DOF_VEL_SCALE
            phase = 2.0 * math.pi * imitation_i / GAIT_PERIOD_STEPS
            imitation_phase = np.array([math.cos(phase), math.sin(phase)], dtype=np.float32)

            accel_arr = np.array(accel, dtype=np.float32)   # 관측 accelerometer 칸 = 생 센서값 (심의 lin_acc_b)
            grav_src = np.array(grav if args.grav_src == "fused" else accel, dtype=np.float32)
            grav_norm = np.linalg.norm(grav_src) or 1.0
            projected_gravity = -grav_src / grav_norm

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
                path_err_arr,
                projected_gravity,
            ]).astype(np.float32)
            assert obs.shape[0] == 107, f"obs 차원 {obs.shape[0]} != 107"

            _td = time.time()
            if zero_action:
                action = np.zeros(14, dtype=np.float32)
            else:
                action = sess.run(None, {"obs": obs.reshape(1, 107)})[0].reshape(14)
            ms["infer"] = (time.time() - _td) * 1e3

            # 액션 저역필터 (docs/reports/lowpass_2026-08-09.md 실험 A). alpha=0이면
            # action_filt == action 그대로라 no-op. last_act(obs 이력)은 v35가 학습
            # 때 겪은 그대로 raw action 을 쓴다 — 필터는 target 계산에만 넣는다
            # (env 자체에 필터가 없는 v35 로는 이게 원 논문 실험 A 와 같은 구조).
            alpha = args.action_lpf_alpha
            action_filt = alpha * action_filt + (1.0 - alpha) * action

            target = READY_ARR + action_filt * ACTION_SCALE
            target[HEAD_IDX] = READY_ARR[HEAD_IDX]  # lock_head_joints=true

            motor_targets = np.clip(target, motor_targets - max_delta, motor_targets + max_delta)
            _te = time.time()
            # 반환값 = tick_of 클램프까지 거쳐 **실제로 서보에 나간** 각도.
            goal_sent = np.array(hwi.set_position_vec(motor_targets), dtype=np.float32)
            ms["write"] = (time.time() - _te) * 1e3

            _now = time.time()
            _dt = (_now - t_prev) if t_prev is not None else 0.0
            t_prev = _now
            log_w.writerow(
                [f"{_now-t_start:.4f}", step, f"{_dt:.4f}", imitation_i,
                 f"{ms['imu']:.2f}", f"{ms['read']:.2f}", f"{ms['infer']:.2f}",
                 f"{ms['write']:.2f}", f"{(_now-t0)*1e3:.2f}",
                 f"{path_yaw_err:.5f}", f"{gyro_bias:.5f}"]
                + list(pos) + list(vel)
                + list(action) + list(action_filt) + list(target) + list(motor_targets)
                + list(goal_sent)
                + list(leg_cur) + list(leg_pwm) + list(tqen) + list(leg_err)
                + list(volts) + list(temps)
                + [contact[0], contact[1], gyro[0], gyro[1], gyro[2],
                   accel_arr[0], accel_arr[1], accel_arr[2],
                   projected_gravity[0], projected_gravity[1], projected_gravity[2],
                   imitation_phase[0], imitation_phase[1],
                   command[0], command[1], command[2],
                   1 if (cmd_rx is not None and cmd_rx.stale) else 0])

            if step % 25 == 0:
                # 2026-08-09 브라운아웃 때 로그가 0바이트였다 — open(...,"w") 가
                # 즉시 truncate 하는데 버퍼가 안 비워진 채 전원이 나가서다.
                # 0.5초마다 비워 두면 다음에 갑자기 죽어도 직전까지는 남는다.
                log_f.flush()

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
                         if over_budget_count else "")
                      # UDP 명령을 쓰는 중이면 **받고 있는지**를 같이 보여준다.
                      # 이게 없으면 로봇이 안 움직일 때 패킷이 안 오는 건지
                      # 명령이 0 인 건지 화면만 봐서는 구별할 수 없다.
                      + (f"  | cmd=({command[0]:+.3f},{command[1]:+.3f},{command[2]:+.3f})"
                         f" {'STALE' if cmd_rx.stale else 'LIVE '} {cmd_rx.stats()}"
                         if cmd_rx is not None else ""),
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
        if cmd_rx is not None:
            cmd_rx.close()


if __name__ == "__main__":
    main()
