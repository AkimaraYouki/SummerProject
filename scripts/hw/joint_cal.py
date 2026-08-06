#!/usr/bin/env python3
"""XM430 14축 **부호(방향) 확인** — 한 축씩, 대화형.

Jetson 에서 **사용자 터미널로 직접** 돌린다. 원격 에이전트가 끼어들 여지가 없어야
한다 — 정지 명령이 네트워크 왕복을 타면 늦기 때문이다. Ctrl+C 는 이 프로세스 안에서
곧바로 전축 토크를 끊는다.

    ssh -t parksuho@192.168.137.7 'python3 ~/joint_cal.py'

--------------------------------------------------------------------------
무엇을 재는가
--------------------------------------------------------------------------
**영점은 이미 끝나 있다.** 사용자가 조립 시 2048 = 각 관절의 기구학적 0 으로 맞춰
두었다. 그래서 OFFSET_TICK 은 전부 2048 이고 이 스크립트로 잴 것이 없다.

남은 미지수는 **부호** 하나다: 모터가 tick 이 커지는 쪽으로 돌 때, URDF 관절각도
같이 커지는가(+1) 반대인가(-1).

판정 기준은 READY 자세다. READY 는 물리적으로 명확한 **웅크림** 자세이고, 각 관절의
READY 값 부호가 곧 "+ 가 어느 쪽인지" 를 말해준다. 그래서 조그를 READY 쪽으로
시키고 관절이 아래 '정상이면 이렇게' 대로 움직이면 +1, 반대면 -1 이다.

READY 값이 0 인 관절(hip_yaw, hip_roll, head_yaw, head_roll)은 이 방법으로 판정이
안 된다. 그 6 축은 UNDECIDABLE 로 표시되고 마지막에 몰아 두었다 — 시뮬과 직접
대조해야 한다 (docs/handoff/project_hardware_joint_calibration.md 의 rpy 축 표).

--------------------------------------------------------------------------
안전
--------------------------------------------------------------------------
  1. **전류 제한.** Current-based Position Control (mode 5) + Goal Current 상한.
     기본 250 unit = 약 0.67 A, 스톨 전류 2.3 A 의 29 %. 관절이 걸려도 모터가
     밀어붙이지 못하고 그냥 진다. `,` / `.` 로 실시간 조절.
  2. **한 번에 한 축만 토크가 켜진다.** 축을 옮기면 이전 축은 자동으로 풀린다.
  3. **절대 위치 점프 없음.** 전부 상대 조그, 기본 33 tick (0.05 rad = 2.9°).
     이동 범위는 그 관절의 URDF 한계로 잘린다.
  4. **온도 감시.** 55 °C 를 넘으면 즉시 전축 토크 해제 후 종료.
  5. **어떤 경로로 끝나든 토크 해제** (정상 종료 / Ctrl+C / 예외).

산출물 `joint_calibration.json`:

    {"left_knee": {"id": 10, "offset_tick": 2048, "sign": -1, "confirmed": true}, ...}
"""

import argparse
import json
import math
import os
import signal
import sys
import termios
import time
import tty

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

BAUD = 57600
TICK_RAD = 2.0 * math.pi / 4096.0
CENTER = 2048

ADDR_OPERATING_MODE = 11
ADDR_HOMING_OFFSET = 20
ADDR_TEMP_LIMIT = 31
ADDR_CURRENT_LIMIT = 38
ADDR_MAX_POS_LIMIT = 48
ADDR_MIN_POS_LIMIT = 52
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_TEMP = 146
DRIVE_MODE_ADDR = 10

MODE_CURRENT_POSITION = 5
CURRENT_UNIT_MA = 2.69
TEMP_CUTOFF = 55

# 관절 이름 -> DXL ID. 2026-08-06 사용자 지정값.
DXL_ID = {
    "left_hip_yaw": 3, "left_hip_roll": 8, "left_hip_pitch": 9,
    "left_knee": 10, "left_ankle": 11,
    "neck_pitch": 2, "head_pitch": 12, "head_yaw": 13, "head_roll": 14,
    "right_hip_yaw": 1, "right_hip_roll": 4, "right_hip_pitch": 5,
    "right_knee": 6, "right_ankle": 7,
}

JOINT_BY_ID = {i: n for n, i in DXL_ID.items()}

LIMIT_RAD = {
    "left_hip_yaw": (-0.5236, 0.5236), "left_hip_roll": (-0.4363, 0.4363),
    "left_hip_pitch": (-0.5236, 1.2217), "left_knee": (-2.0944, 2.0944),
    "left_ankle": (-1.5708, 1.5708),
    "neck_pitch": (-0.3491, 1.1345), "head_pitch": (-0.7854, 0.7854),
    "head_yaw": (-2.7925, 2.7925), "head_roll": (-0.5236, 0.5236),
    "right_hip_yaw": (-0.5236, 0.5236), "right_hip_roll": (-0.4363, 0.4363),
    "right_hip_pitch": (-0.5236, 1.2217), "right_knee": (-2.0944, 2.0944),
    "right_ankle": (-1.5708, 1.5708),
}

# READY_JOINT_POS_H175 (머리 4축은 0). 부호 판정의 정답지.
READY_RAD = {
    "left_hip_yaw": 0.0003, "left_hip_roll": 0.0213, "left_hip_pitch": 0.9910,
    "left_knee": -1.7852, "left_ankle": 0.8647,
    "neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": -0.0005, "right_hip_roll": -0.0092, "right_hip_pitch": 1.0114,
    "right_knee": 1.8163, "right_ankle": -0.8754,
}

# d 를 눌러 **URDF + 방향**으로 움직였을 때 눈에 보여야 하는 것.
#
# ⚠️ 2026-08-06 정정 — 아래 다섯 줄이 **반대로 적혀 있었다.**
#
# 예전 주석은 "머리를 +45° 세우면 CoM 이 -x 로 가고 로봇이 앞으로 넘어졌다
# => -x 가 앞" 이라는 닻에서 전부를 유도했다. CoM 이동량(-14.3 mm)은 재현되지만
# 그 전도는 정적 전복이 아니라 **정책이 무너진 동적 결과**라 방향 판정의 근거가
# 되지 못한다. 실제 몸통 프레임은 **+x 앞 / +y 좌 / +z 위** 다 (근거는 imu_map.py:
# 전진 시 디딤발이 -29.8 mm 흐르고, 발이 발목 기준 +x 로 63.8 / -x 로 40.1 mm 뻗는다).
#
# 아래 표는 URDF FK 로 직접 재서(관절 +0.4 rad 을 주고 대상 프레임의 로컬 축이
# 어디로 도는지) 다시 쓴 것이다.
#
#   관절            몸통프레임 축      + 방향의 물리적 의미      예전 주석
#   left_hip_roll   (+1, 0, 0)        왼다리 바깥(좌)           (같음)
#   right_hip_roll  (-1, 0, 0)        오른다리 바깥(우)         (같음)
#                                     <- 좌우 축이 물리적으로 반대라 MIRROR_SIGN=+1 과 일치
#   left/right_hip_yaw (0, 0, -1)     둘 다 발끝이 **오른쪽**    "왼쪽" 이라 적혀 있었음
#   neck_pitch      (0, -1, 0)        목이 **뒤로 젖혀짐**       "앞으로 숙여짐"
#   head_pitch      (0, +1, 0)        코끝이 **아래로**          "위로"
#   head_yaw        (0, 0, +1)        고개가 **왼쪽**            "오른쪽"
#   head_roll       (-1, 0, 0)        머리 꼭대기가 왼쪽         (같음)
#
# **부호(joint_calibration.json)는 이 정정의 영향을 받지 않는다.** 사용자가 실기를
# 눈으로 보면서 축마다 직접 맞춘 값이고, 그게 기준이다. 문구가 반대로 적혀 있던
# 동안에도 판정은 실물을 보고 한 것이라 14축 전부 confirmed 로 유효하다.
# 여기서 고친 것은 **문구뿐**이다 — 다음 사람이 이 표를 읽고 방향을 거꾸로
# 유도하는 일이 없게 하려는 것.
EXPECT = {
    "left_hip_pitch": "왼쪽 허벅지가 앞/위로 접힌다 (웅크리는 쪽).",
    "right_hip_pitch": "오른쪽 허벅지가 앞/위로 접힌다 (웅크리는 쪽).",
    "left_knee": "왼쪽 무릎이 굽는다 (종아리가 접힌다).",
    "right_knee": "오른쪽 무릎이 굽는다 (종아리가 접힌다).",
    "left_ankle": "왼쪽 발바닥이 정강이 쪽으로 당겨진다 (발등 굽힘).",
    "right_ankle": "오른쪽 발바닥이 정강이 쪽으로 당겨진다 (발등 굽힘).",
    "left_hip_roll": "왼다리가 **바깥(왼쪽)으로 벌어진다**.",
    "right_hip_roll": "오른다리가 **바깥(오른쪽)으로 벌어진다**.",
    "left_hip_yaw": "왼발 끝이 **오른쪽으로**(안쪽으로) 돌아간다.",
    "right_hip_yaw": "오른발 끝이 **오른쪽으로**(바깥으로) 돌아간다.",
    "neck_pitch": "목이 **뒤로 젖혀진다** (턱이 들린다).",
    "head_pitch": "코끝이 **아래로 내려간다** (고개를 숙인다).",
    "head_yaw": "고개가 **왼쪽으로** 돌아간다.",
    "head_roll": "머리 꼭대기가 **왼쪽으로** 기운다.",
}

# d 테스트에서 URDF + 로 얼마나 움직여 볼지 (rad). 8.6°.
TEST_RAD = 0.15

# 판정 가능한 6 축을 먼저, 그중에서도 가동범위가 작은 발목부터. 무릎은 ±120° 로
# 가장 크고 잘못 돌리면 다리가 몸통을 치므로 그 다음. 판정 불가 6 축은 맨 뒤.
ORDER = [
    "left_ankle", "right_ankle",
    "left_hip_pitch", "right_hip_pitch",
    "left_knee", "right_knee",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "left_hip_yaw", "right_hip_yaw", "left_hip_roll", "right_hip_roll",
]
assert set(ORDER) == set(DXL_ID) and len(ORDER) == 14

CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_calibration.json")

HELP = """
 ┌ 할 일 ─────────────────────────────────────────────────────────────┐
 │ 축마다:  d  를 누른다 (URDF + 방향으로 8.6° 왕복 4회, 약 16초)     │
 │   화면의 '부호 X 가 맞다면:' 대로 움직이면  ->  c   (확정, 자동저장)│
 │   반대로 움직이면                           ->  s  누르고 다시 d   │
 │   끝나면  h  로 0(2048) 복귀 후  ->  n  다음 축                    │
 └────────────────────────────────────────────────────────────────────┘
 d        URDF + 방향 왕복 테스트 (이게 주력이다)
 r / R    READY 쪽으로 한 스텝 / 다섯 스텝 (READY 가 0 이 아닌 축에서만 의미)
 j / k    - / + 한 스텝 (tick 기준 생조그)
 [ / ]    스텝 크기        , / .   전류 상한
 s        부호 반전        c       이 축 방향 확인됨으로 표시
 h        0 rad (2048) 로 복귀      t   토크 on/off
 n / p    다음 / 이전 축   w   저장   q   종료 (Ctrl+C 도 동일)
"""


def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class Cal:
    def __init__(self, port_name, goal_current):
        self.packet = PacketHandler(2.0)
        self.port = PortHandler(port_name)
        if not self.port.openPort() or not self.port.setBaudRate(BAUD):
            raise SystemExit(f"포트 열기 실패: {port_name}")
        self.goal_current = goal_current
        self.torqued = None
        self.step = 33
        self.idx = 0
        # 영점은 조립 때 이미 2048 로 맞춰져 있다. 부호만 미지수라 +1 에서 출발.
        self.cal = {n: {"id": DXL_ID[n], "offset_tick": CENTER, "sign": +1, "confirmed": False}
                    for n in DXL_ID}
        self._load()

    # ── 저수준 ──────────────────────────────────────────────────────────
    def r1(self, i, a):
        v, r, e = self.packet.read1ByteTxRx(self.port, i, a)
        return v if r == COMM_SUCCESS and not e else None

    def r2(self, i, a):
        v, r, e = self.packet.read2ByteTxRx(self.port, i, a)
        if r != COMM_SUCCESS or e:
            return None
        return v - 65536 if v > 32767 else v

    def r4(self, i, a):
        v, r, e = self.packet.read4ByteTxRx(self.port, i, a)
        if r != COMM_SUCCESS or e:
            return None
        return v - 4294967296 if v > 2147483647 else v

    def _load(self):
        if not os.path.exists(CAL_PATH):
            return
        with open(CAL_PATH) as f:
            d = json.load(f)
        for n, e in d.items():
            if n in self.cal:
                self.cal[n].update(e)
        done = sum(1 for c in self.cal.values() if c["confirmed"])
        print(f"기존 결과 읽음: {CAL_PATH} (확인된 축 {done}/14)")

    def save(self, quiet=False):
        # 임시파일 -> rename 으로 원자적으로 쓴다. 저장 도중 죽어도 기존 결과가
        # 반쯤 덮인 채 남지 않는다.
        tmp = CAL_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.cal, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, CAL_PATH)
        if quiet:
            return
        done = [n for n in ORDER if self.cal[n]["confirmed"]]
        print(f"\n저장됨 -> {CAL_PATH}  (확인 {len(done)}/14)")
        left = [n for n in ORDER if not self.cal[n]["confirmed"]]
        if left:
            print(f"남은 축: {', '.join(left)}")

    # ── 토크 ────────────────────────────────────────────────────────────
    def torque_off_all(self):
        for i in DXL_ID.values():
            self.packet.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, 0)
        self.torqued = None

    def torque_on(self, dxl_id):
        if self.torqued is not None and self.torqued != dxl_id:
            self.packet.write1ByteTxRx(self.port, self.torqued, ADDR_TORQUE_ENABLE, 0)
        self.packet.write1ByteTxRx(self.port, dxl_id, ADDR_TORQUE_ENABLE, 0)
        self.packet.write1ByteTxRx(self.port, dxl_id, ADDR_OPERATING_MODE, MODE_CURRENT_POSITION)
        pos = self.r4(dxl_id, ADDR_PRESENT_POSITION)
        self.packet.write4ByteTxRx(self.port, dxl_id, ADDR_GOAL_POSITION, pos & 0xFFFFFFFF)
        self.packet.write1ByteTxRx(self.port, dxl_id, ADDR_TORQUE_ENABLE, 1)
        self.packet.write2ByteTxRx(self.port, dxl_id, ADDR_GOAL_CURRENT, self.goal_current)
        self.torqued = dxl_id
        return pos

    # ── 관절 ────────────────────────────────────────────────────────────
    def joint(self):
        return ORDER[self.idx]

    def bounds(self, name):
        c = self.cal[name]
        lo_r, hi_r = LIMIT_RAD[name]
        a = c["offset_tick"] + c["sign"] * lo_r / TICK_RAD
        b = c["offset_tick"] + c["sign"] * hi_r / TICK_RAD
        return int(round(min(a, b))), int(round(max(a, b)))

    def ready_tick(self, name):
        c = self.cal[name]
        lo, hi = self.bounds(name)
        t = c["offset_tick"] + c["sign"] * READY_RAD[name] / TICK_RAD
        return max(lo, min(hi, int(round(t))))

    def angle(self, name, pos):
        c = self.cal[name]
        return c["sign"] * (pos - c["offset_tick"]) * TICK_RAD

    def status(self):
        name = self.joint()
        i = DXL_ID[name]
        pos = self.r4(i, ADDR_PRESENT_POSITION)
        if pos is None:
            return f"[{self.idx + 1}/14] {name} (ID {i})  ** 응답 없음 **"
        cur = self.r2(i, ADDR_PRESENT_CURRENT) or 0
        tmp = self.r1(i, ADDR_PRESENT_TEMP) or 0
        c = self.cal[name]
        lo, hi = self.bounds(name)
        rt = self.ready_tick(name)
        exp = EXPECT[name]
        tq = "ON " if self.torqued == i else "off"
        mark = "확인됨" if c["confirmed"] else "미확인"
        out = [
            "",
            "=" * 72,
            f"[{self.idx + 1}/14] {name}   ID {i}   부호 {c['sign']:+d}  ({mark})",
            f"  현재 {pos:5d} tick = {math.degrees(self.angle(name, pos)):+7.1f}°"
            f"    한계 [{lo}, {hi}]",
            f"  READY {rt:5d} tick = {math.degrees(READY_RAD[name]):+7.1f}°"
            f"    남은 거리 {rt - pos:+5d} tick",
            f"  토크 {tq}  전류 {abs(cur) * CURRENT_UNIT_MA:4.0f}mA"
            f" (상한 {self.goal_current * CURRENT_UNIT_MA:.0f})  온도 {tmp}°C"
            f"  스텝 {self.step}tick({math.degrees(self.step * TICK_RAD):.1f}°)",
        ]
        out.append(f"  d 를 누르면 (부호 {c['sign']:+d} 기준) : {exp}")
        return "\n".join(out)

    # ── 동작 ────────────────────────────────────────────────────────────
    def goto(self, target, settle=0.4):
        name = self.joint()
        i = DXL_ID[name]
        lo, hi = self.bounds(name)
        target = max(lo, min(hi, int(target)))
        if self.torqued != i:
            self.torque_on(i)
        self.packet.write4ByteTxRx(self.port, i, ADDR_GOAL_POSITION, target & 0xFFFFFFFF)
        time.sleep(settle)
        return target

    def jog(self, ticks):
        pos = self.r4(DXL_ID[self.joint()], ADDR_PRESENT_POSITION)
        if pos is None:
            return
        want = pos + ticks
        got = self.goto(want)
        if got != want:
            print(f"  한계에 걸려 {got} 까지만 (요청 {want})")

    def jog_toward_ready(self, n_steps):
        name = self.joint()
        pos = self.r4(DXL_ID[name], ADDR_PRESENT_POSITION)
        if pos is None:
            return
        rt = self.ready_tick(name)
        if rt == pos:
            print("  이미 READY 위치다.")
            return
        d = self.step * n_steps
        delta = min(d, abs(rt - pos)) * (1 if rt > pos else -1)
        self.goto(pos + delta)

    def direction_test(self, cycles=4):
        """URDF + 방향으로 TEST_RAD 만큼 갔다 오기를 반복한다.

        READY 값이 0 인 축(hip_yaw, hip_roll, head 4축)은 'READY 쪽으로' 가 성립하지
        않아서 r 을 못 쓴다. 대신 URDF + 가 물리적으로 무엇인지(EXPECT)를 보고 맞는지
        판정한다. 여러 번 왕복시키는 것은 한 번 스쳐 지나가면 방향을 놓치기 때문이다.

        목표에 못 미치고 멈추면 그대로 보고한다 — 전류를 제한해 두었으므로 그것은
        기구 스토퍼나 자가충돌에 닿았다는 뜻이지 모터가 약해서가 아니다.
        """
        name = self.joint()
        i = DXL_ID[name]
        c = self.cal[name]
        start = self.r4(i, ADDR_PRESENT_POSITION)
        if start is None:
            print("  응답 없음")
            return
        lo, hi = self.bounds(name)
        far = max(lo, min(hi, int(round(start + c["sign"] * TEST_RAD / TICK_RAD))))
        print(f"\n  URDF + 방향으로 {math.degrees(TEST_RAD):.1f}° "
              f"({start} -> {far} tick) 왕복 {cycles}회. 잘 보라:")
        print(f"  >>> 부호 {c['sign']:+d} 가 맞다면: {EXPECT[name]}")
        stalled = False
        for n in range(cycles):
            self.goto(far, settle=2.0)
            got = self.r4(i, ADDR_PRESENT_POSITION)
            self.goto(start, settle=2.0)
            back = self.r4(i, ADDR_PRESENT_POSITION)
            short = abs(got - far)
            if short > 25:
                stalled = True
            print(f"    {n + 1}회: {got} (목표 {far}, 부족 {short}) -> 복귀 {back}")
        if stalled:
            print("  ** 목표까지 못 갔다. 전류를 제한해 뒀으니 기구 스토퍼나 자가충돌에")
            print("     닿았다는 뜻이다. 그 방향으로 더 밀지 말 것. **")
        print("  맞으면 c, 반대로 움직였으면 s 누르고 다시 d.")

    def check_temp(self):
        i = DXL_ID[self.joint()]
        tmp = self.r1(i, ADDR_PRESENT_TEMP)
        if tmp is not None and tmp >= TEMP_CUTOFF:
            self.torque_off_all()
            print(f"\n!! {self.joint()} (ID {i}) 온도 {tmp}°C — 전축 토크 해제. 식힌 뒤 재개.")
            return False
        return True

    def check_power(self):
        """전원이 안 들어와 있으면 여기서 끝낸다.

        U2D2 는 USB 로 전원을 받으므로 배터리가 꺼져 있어도 포트는 멀쩡히 열리고
        /dev/ttyUSB0 도 그대로 있다. 그래서 '포트가 열렸다' 는 것은 모터가 살아
        있다는 근거가 못 된다 — 핑을 쳐 봐야 안다.
        """
        alive = []
        for i in DXL_ID.values():
            _m, r, e = self.packet.ping(self.port, i)
            if r == COMM_SUCCESS and not e:
                alive.append(i)
        if not alive:
            print("\n" + "=" * 72)
            print("14축 전부 응답이 없다. **모터 전원이 안 들어와 있다.**")
            print("U2D2 는 USB 로 켜지므로 포트가 열린 것만으로는 아무것도 보장 못 한다.")
            print("확인할 것: 배터리 스위치 / 전압(XM430 은 12V 정격, 저전압이면 무응답) /")
            print("           U2D2-첫서보 3핀 TTL 케이블 / 전원 분배 커넥터")
            print("=" * 72)
            return False
        missing = [i for i in DXL_ID.values() if i not in alive]
        if missing:
            names = ", ".join(f"{JOINT_BY_ID[i]}(ID {i})" for i in sorted(missing))
            print(f"\n!! 응답 없는 축 {len(missing)}개: {names}")
            print("   데이지체인 중간이 빠졌을 수 있다. 계속하려면 그 축은 건드리지 말 것.")
        return True

    def dump_registers(self):
        def f(v, w):
            return ("-" if v is None else str(v)).rjust(w)

        print(f"\n{'ID':>3} {'joint':16s} {'MinPos':>7s} {'MaxPos':>7s} {'Drive':>6s} "
              f"{'Homing':>7s} {'CurLim':>7s} {'TempLim':>7s} {'HWerr':>6s} {'Pos':>6s}")
        for name in ORDER:
            i = DXL_ID[name]
            print(f"{i:3d} {name:16s} {f(self.r4(i, ADDR_MIN_POS_LIMIT), 7)} "
                  f"{f(self.r4(i, ADDR_MAX_POS_LIMIT), 7)} {f(self.r1(i, DRIVE_MODE_ADDR), 6)} "
                  f"{f(self.r4(i, ADDR_HOMING_OFFSET), 7)} {f(self.r2(i, ADDR_CURRENT_LIMIT), 7)} "
                  f"{f(self.r1(i, ADDR_TEMP_LIMIT), 7)} {f(self.r1(i, ADDR_HARDWARE_ERROR), 6)} "
                  f"{f(self.r4(i, ADDR_PRESENT_POSITION), 6)}")
        print("Drive Mode bit0=1 이면 Reverse Mode — 모터 + 방향이 이미 뒤집혀 있다는 뜻.")

    # ── 메인 루프 ────────────────────────────────────────────────────────
    def enter_joint(self):
        if self.torqued is not None:
            self.packet.write1ByteTxRx(self.port, self.torqued, ADDR_TORQUE_ENABLE, 0)
            self.torqued = None

    def run(self):
        if not self.check_power():
            return
        self.dump_registers()
        print(HELP)
        while True:
            print(self.status())
            key = getkey()
            name = self.joint()
            i = DXL_ID[name]

            if key == "q":
                return
            elif key == "d":
                self.direction_test()
            elif key == "r":
                self.jog_toward_ready(1)
            elif key == "R":
                self.jog_toward_ready(5)
            elif key == "j":
                self.jog(-self.step)
            elif key == "k":
                self.jog(+self.step)
            elif key == "[":
                self.step = max(1, self.step // 2)
            elif key == "]":
                self.step = min(256, self.step * 2)
            elif key == ",":
                self.goal_current = max(20, self.goal_current - 25)
                if self.torqued == i:
                    self.packet.write2ByteTxRx(self.port, i, ADDR_GOAL_CURRENT, self.goal_current)
            elif key == ".":
                self.goal_current = min(1193, self.goal_current + 25)
                if self.torqued == i:
                    self.packet.write2ByteTxRx(self.port, i, ADDR_GOAL_CURRENT, self.goal_current)
            elif key == "s":
                self.cal[name]["sign"] *= -1
                self.cal[name]["confirmed"] = False
                print(f"  {name} 부호 -> {self.cal[name]['sign']:+d}. r 로 다시 확인할 것.")
                self.save(quiet=True)
            elif key == "c":
                self.cal[name]["confirmed"] = True
                print(f"  {name} 부호 {self.cal[name]['sign']:+d} 확인됨.")
                self.save()
            elif key == "h":
                self.goto(self.cal[name]["offset_tick"], settle=1.0)
            elif key == "t":
                if self.torqued == i:
                    self.packet.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, 0)
                    self.torqued = None
                else:
                    self.torque_on(i)
            elif key in ("n", "p"):
                self.idx = (self.idx + (1 if key == "n" else -1)) % 14
                self.enter_joint()
            elif key == "w":
                self.save()
            elif key == "?":
                print(HELP)

            if not self.check_temp():
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--current", type=int, default=250,
                    help="Goal Current 상한 (1 unit = 2.69 mA). 기본 250 = 0.67 A")
    ap.add_argument("--joint", default=None,
                    help="이 관절부터 시작한다 (예: --joint right_knee). n 을 여러 번 "
                         "누르지 않아도 되고, 그 과정에서 엉뚱한 축을 건드릴 일도 없다.")
    ap.add_argument("--pending", action="store_true",
                    help="아직 확인 안 된 첫 축부터 시작한다")
    args = ap.parse_args()

    if args.joint and args.joint not in ORDER:
        raise SystemExit(f"모르는 관절 이름: {args.joint}\n선택지: {', '.join(ORDER)}")

    if not sys.stdin.isatty():
        raise SystemExit(
            "TTY 가 아니다. 사용자 터미널에서 직접 실행할 것:\n"
            "  ssh -t parksuho@192.168.137.7 'python3 ~/joint_cal.py'"
        )

    cal = Cal(args.port, args.current)
    if args.joint:
        cal.idx = ORDER.index(args.joint)
    elif args.pending:
        left = [n for n in ORDER if not cal.cal[n]["confirmed"]]
        if not left:
            print("14축 모두 확인 완료. 할 일이 없다.")
            cal.port.closePort()
            return
        cal.idx = ORDER.index(left[0])
        print(f"미확인 {len(left)}축부터 시작: {', '.join(left)}")

    # ssh 세션이 끊기면 기본 동작이 즉시 종료라 finally 가 안 돈다 — 2026-08-06 에
    # 실제로 이걸로 결과 한 세션치를 통째로 날렸다. 예외로 바꿔서 finally 를 타게 한다.
    def _bye(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for _sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(_sig, _bye)

    try:
        cal.run()
    except KeyboardInterrupt:
        print("\nCtrl+C")
    finally:
        cal.torque_off_all()
        print("전축 토크 해제 완료.")
        cal.save()
        cal.port.closePort()


if __name__ == "__main__":
    main()
