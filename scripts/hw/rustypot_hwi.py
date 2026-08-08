#!/usr/bin/env python3
"""14축 다이나믹셀(XM430) ↔ 실기 하드웨어 인터페이스. rustypot(Rust) 백엔드.

원본 리포(Open_Duck_Mini_Runtime)의 rustypot_position_hwi.py 를 이 로봇의 실기
구성(ID/부호/제한각)에 맞춰 이식한 것. dxl_bridge.py/play_ref_gait.py 가 쓰던
dynamixel_sdk(순수 파이썬) 대신 rustypot(Rust + PyO3 바인딩)을 쓴다.

rustypot 은 XM430 전용 클래스가 없지만, XM430 은 XL430 과 같은 Protocol 2.0
X-시리즈 레지스터 맵(주소 완전 동일: torque_enable=64, goal_current=102,
goal_position=116, present_position=132, hw_error_status=70 등)을 쓰므로
`Xl430PyController` 를 그대로 쓸 수 있다. 2026-08-07 실기 14축 전부 ping/읽기
확인 완료.

JOINTS 표는 기본적으로 dxl_bridge.py 와 같다 — 2026-08-06 실기에서 부호까지
확인된 값이다. neck_pitch(ID2)/head_pitch(ID12)는 2026-08-08 실측(Z자 목
수평유지 테스트로 확인)으로 부호를 뒤집었고, dxl_bridge.py/goto_ready.py/
play_ref_gait.py 도 같은 날 전부 맞춰 고쳤다 — 이제 네 파일 모두 일치한다.
"""

import math
import struct
import time

import rustypot

BAUD = 1_000_000
TICK_RAD = 2.0 * math.pi / 4096.0
CENTER = 2048
CURRENT_UNIT_MA = 2.69
VEL_UNIT_RAD_S = 0.229 * 2.0 * math.pi / 60.0  # 0.229 rpm/unit -> rad/s

ADDR_OPERATING_MODE = 11
ADDR_HW_ERROR_STATUS = 70
MODE_CURRENT_POSITION = 5

# (관절, ID, 부호, URDF 하한, 상한) — dxl_bridge.py / goto_ready.py / play_ref_gait.py 와 같은 표.
JOINTS = [
    ("left_hip_yaw",     3, -1, -0.5236, 0.5236),
    ("left_hip_roll",    8, -1, -0.4363, 0.4363),
    ("left_hip_pitch",   9, -1, -0.5236, 1.2217),
    ("left_knee",       10, +1, -2.0944, 2.0944),
    ("left_ankle",      11, +1, -1.5708, 1.5708),
    ("neck_pitch",       2, +1, -0.3491, 1.1345),  # 2026-08-08: -1 이었으나 실측(손으로 앞으로 젖히면 reading 증가,
                                                     # +45 명령 시 실제로 앞으로 쏠림 확인)으로 부호 반전. 아래 참고.
    ("head_pitch",      12, -1, -0.7854, 0.7854),  # 2026-08-08: +1 이었으나 실측(+30 명령이 반대로 움직임)으로 부호 반전.
    ("head_yaw",        13, -1, -2.7925, 2.7925),
    ("head_roll",       14, -1, -0.5236, 0.5236),
    ("right_hip_yaw",    1, -1, -0.5236, 0.5236),
    ("right_hip_roll",   4, +1, -0.4363, 0.4363),
    ("right_hip_pitch",  5, +1, -0.5236, 1.2217),
    ("right_knee",       6, -1, -2.0944, 2.0944),
    ("right_ankle",      7, -1, -1.5708, 1.5708),
]
NAMES = [j[0] for j in JOINTS]
IDS = [j[1] for j in JOINTS]
BY_NAME = {j[0]: j for j in JOINTS}

# lock_head_joints=True 환경에서는 머리 4축을 정책이 안 건드리고 항상 READY로
# 고정 명령만 나간다 — 매 루프 SyncRead 에 넣을 이유가 없다. 뺀 만큼(14->10축)
# 버스 시간이 대략 비례해서 줄어든다(SyncRead 는 ID 당 응답 패킷 하나).
HEAD_NAMES = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
LEG_NAMES = [n for n in NAMES if n not in HEAD_NAMES]
LEG_IDS = [BY_NAME[n][1] for n in LEG_NAMES]

# 머리 3축(head_pitch 12 / head_yaw 13 / head_roll 14) 위치 PID 게인.
#
# 2026-08-08 에 head_roll 떨림을 잡겠다고 P=800->200, D=4700->0 으로 낮췄었는데
# **그게 틀렸다.** 그때는 평형 위치에서만 봐서 조용해 보였을 뿐, D=0 은 감쇠를
# 완전히 없앤 것이라 외란(런 중 몸통 진동)이 들어오면 크게 출렁인다.
#
# 2026-08-09 계단응답 실측(+100 tick, 정착 후 잔여 변동):
#     축          P=200/D=0    P=200/D=2000   P=800/D=4700(공장)
#     head_pitch    9 tick        1 tick          1 tick
#     head_yaw     66 tick        0 tick          1 tick
#     head_roll    95 tick        1 tick          3 tick
# P=200/D=2000 이 세 축 모두에서 잔여 진동이 가장 작고 오버슈트도 작다
# (head_roll +4 tick). 공장값도 안정적이지만 강성이 높아 전류를 더 쓴다.
# neck_pitch(ID2)는 요청 범위 밖이라 공장값 그대로 둔다.
HEAD_GAIN_IDS = (12, 13, 14)
HEAD_P = 200
HEAD_D = 2000


def tick_of(name, rad):
    _, _, sign, lo, hi = BY_NAME[name]
    a, b = CENTER + sign * lo / TICK_RAD, CENTER + sign * hi / TICK_RAD
    lo_t, hi_t = int(round(min(a, b))), int(round(max(a, b)))
    return max(lo_t, min(hi_t, int(round(CENTER + sign * rad / TICK_RAD))))


def rad_of(name, tick):
    return BY_NAME[name][2] * (tick - CENTER) * TICK_RAD


class HWI:
    """14축 일괄 read/write. 각도는 전부 rad (URDF 관절각), 내부에서만 tick 변환."""

    def __init__(self, port="/dev/ttyUSB0", current_limit=350):
        self.current_limit = current_limit
        self.io = rustypot.Xl430PyController(port, BAUD, 0.05)
        # Shutdown 레지스터(addr 63, EEPROM)에 켜진 비트만 실제로 토크를 끊는다.
        # 여기 없는 에러 비트는 상태 기록일 뿐 모터는 계속 돈다. 이 로봇은
        # 0b00110100 = 과열(2)/전기충격(4)/과부하(5) 이고 **Input Voltage(0)는
        # 빠져 있다** — 2026-08-09 확인. 이걸 모르고 전압 비트에 반응해 리부팅
        # 하다가 보행 중 1초마다 400ms 씩 제어를 멈추는 사고를 냈다.
        self.shutdown_mask = [b[0] for b in self.io.sync_read_raw_data(IDS, 63, 1)]

    def arm(self):
        """토크 켜기 전 준비: 토크 끄고 -> 전류제한위치제어 모드 -> 전류상한 -> 토크 켜기.

        이전 세션에서 겪은 문제(재연결 후 stale 레지스터로 예상 밖 동작)를
        피하려고 매번 명시적으로 전 축을 같은 상태로 강제한다.
        """
        self.io.sync_write_torque_enable(IDS, [0] * 14)
        self.io.sync_write_operating_mode(IDS, [MODE_CURRENT_POSITION] * 14)
        self.io.sync_write_current_limit(IDS, [self.current_limit] * 14)
        self.io.sync_write_torque_enable(IDS, [1] * 14)

        self._apply_head_gains()

    def _apply_head_gains(self, ids=None):
        """머리 3축(head_pitch/yaw/roll)의 위치 PID 게인을 HEAD_P/HEAD_D 로 건다.

        P/D 는 RAM 레지스터라 전원 재인가·reboot 시 공장값으로 돌아간다 —
        arm() 과 reboot 복구에서 매번 다시 걸어야 한다.
        """
        for i in (ids if ids is not None else HEAD_GAIN_IDS):
            if i in HEAD_GAIN_IDS:
                self.io.write_position_p_gain(i, HEAD_P)
                self.io.write_position_d_gain(i, HEAD_D)

    def disarm(self):
        self.io.sync_write_torque_enable(IDS, [0] * 14)

    def set_position_all(self, joints_rad: dict):
        """joints_rad: {관절이름: rad}. 없는 관절은 건드리지 않는다."""
        ids, ticks = [], []
        for name, rad in joints_rad.items():
            ids.append(BY_NAME[name][1])
            ticks.append(tick_of(name, rad))
        self.io.sync_write_goal_position(ids, ticks)

    def set_position_vec(self, rad_vec):
        """rad_vec: NAMES 순서의 14-길이 배열/리스트."""
        ticks = [tick_of(NAMES[k], rad_vec[k]) for k in range(14)]
        self.io.sync_write_goal_position(IDS, ticks)

    def get_present_positions(self):
        """NAMES 순서의 14-길이 리스트 (rad)."""
        ticks = self.io.sync_read_present_position(IDS)
        return [rad_of(NAMES[k], ticks[k]) for k in range(14)]

    def get_present_velocities(self):
        """NAMES 순서의 14-길이 리스트 (rad/s). 부호는 위치와 같은 관절 부호를 따른다."""
        raw = self.io.sync_read_present_velocity(IDS)
        return [BY_NAME[NAMES[k]][2] * raw[k] * VEL_UNIT_RAD_S for k in range(14)]

    def _sync_read_pos_vel_raw(self, ids, retries=6):
        """sync_read_raw_data(ids, 128, 8) 을 하되, 응답이 손상된(8바이트가 아닌)
        축이 있으면 몇 번 재시도한다.

        2026-08-08 실기에서 겪음: arm() 직후 램프인(연속 SyncWrite)이 끝나자마자
        첫 SyncRead 가 struct.unpack 에서 죽었다 — 축 하나의 응답이 8바이트가
        아니었다(버스 전환 시점의 일시적 충돌로 추정, 재현은 못 함). 한 축
        읽기가 가끔 깨지는 것 때문에 전체 루프가 죽으면 안 되니 재시도로
        흡수한다. 재시도로도 안 되면 진짜 문제이니 그대로 예외를 던진다 —
        조용히 stale 값을 쓰는 것보다 멈추는 게 안전하다.

        2026-08-09: FTDI latency_timer 를 16->1ms 로 낮추자 이 손상이 훨씬 잦아졌다
        (응답 패킷이 USB 전송 여러 개로 쪼개져 도착하는 빈도가 늘어난 것으로 추정).
        2ms 로 절충하고, 재시도 횟수도 2->6 으로 올려 여유를 더 뒀다.
        """
        for attempt in range(retries + 1):
            raw = self.io.sync_read_raw_data(ids, 128, 8)
            bad = [i for i, r in enumerate(raw) if len(r) != 8]
            if not bad:
                return raw
            if attempt == retries:
                raise RuntimeError(
                    f"SyncRead 응답 손상 (ID {[ids[i] for i in bad]}, "
                    f"바이트수 {[len(raw[i]) for i in bad]}) — 재시도 {retries}회 실패")
        return raw  # unreachable

    def get_present_pos_vel(self):
        """위치+속도를 한 번의 SyncRead 로. (positions_rad, velocities_rad_s) 튜플, 둘 다 NAMES 순서.

        Present Velocity(addr 128, 4바이트) 와 Present Position(addr 132, 4바이트)이
        레지스터 맵에서 붙어 있어(128~135) 8바이트 한 번에 읽을 수 있다 — read를
        둘로 나누면 각각 고정 오버헤드(~16ms/축14개)가 또 붙는다. 이 버스에서
        14축 SyncRead 하나가 ~16-28ms 걸리는 게 실측됐다(2026-08-08) — 50Hz(20ms)
        예산 안에서 read를 여러 번 쪼개면 못 맞춘다.
        """
        raw = self._sync_read_pos_vel_raw(IDS)
        pos, vel = [], []
        for k in range(14):
            v_raw, p_raw = struct.unpack("<ii", raw[k])
            sign = BY_NAME[NAMES[k]][2]
            pos.append(rad_of(NAMES[k], p_raw))
            vel.append(sign * v_raw * VEL_UNIT_RAD_S)
        return pos, vel

    def get_leg_pos_vel(self):
        """다리 10축만 pos+vel 을 한 번의 SyncRead 로. (positions_rad, velocities_rad_s)
        튜플, 둘 다 LEG_NAMES 순서. 머리 4축을 뺀 버전 — get_present_pos_vel() 참고
        (같은 8바이트 결합 read, ID 수만 14->10).

        2026-08-08 실측: 14축 combined read 가 ~28ms 로 50Hz(20ms) 예산을 그 자체로
        넘겼다. 머리 4축(lock_head_joints=True 라 정책이 안 씀)을 빼면 ID 수가
        10/14 로 줄어 버스 시간도 대략 그 비율로 준다.
        """
        raw = self._sync_read_pos_vel_raw(LEG_IDS)
        pos, vel = [], []
        for k, name in enumerate(LEG_NAMES):
            v_raw, p_raw = struct.unpack("<ii", raw[k])
            sign = BY_NAME[name][2]
            pos.append(rad_of(name, p_raw))
            vel.append(sign * v_raw * VEL_UNIT_RAD_S)
        return pos, vel

    def get_hw_errors(self):
        """NAMES 순서의 14-길이 리스트. 값은 원본 hw_error_status 비트마스크
        (0=정상, bit0=Input Voltage, bit2=Overheating, bit3=MotorEncoder,
        bit4=ElectricalShock, bit5=Overload).

        두 번 연속 읽어서 같은 축이 에러일 때만 믿는다 — sync_read_raw_data 처럼
        길이체크가 있는 게 아니라 rustypot 내장 파서라, 응답이 쪼개져 도착하면
        예외 없이 조용히 틀린 값을 만들 수 있어서다.

        2026-08-09 실기 확인: 램프인 직후 첫 체크에서 여러 축에 Input Voltage
        Error(bit0)가 반복적으로 뜬다(그때그때 다른 조합) — 전압 자체는 재보면
        14축 다 11.7~11.8V 로 정상이라, 급격해진 다축 동시 가감속의 역기전력
        스파이크가 래치되는 것으로 추정(latency_timer 를 낮춰 루프가 빨라진 뒤
        생겼다). reboot 로 풀리는 걸 확인했다 — recover_input_voltage_errors() 참고.
        """
        first = list(self.io.sync_read_hardware_error_status(IDS))
        if not any(first):
            return first
        second = list(self.io.sync_read_hardware_error_status(IDS))
        return [a if (a and b) else 0 for a, b in zip(first, second)]

    def get_fatal_errors(self):
        """get_hw_errors() 를 각 축의 Shutdown 마스크로 걸러, **실제로 토크가
        끊기는 에러만** 남긴 14-길이 리스트. 정지 판단은 이걸로 해야 한다.

        마스크 밖 비트(이 로봇에서는 Input Voltage)는 모터가 계속 도는 상태라
        멈출 이유가 없다 — 알고 싶으면 get_hw_errors() 원본을 따로 보면 된다.
        """
        return [e & m for e, m in zip(self.get_hw_errors(), self.shutdown_mask)]

    def recover_input_voltage_errors(self, raw_errors):
        """get_hw_errors() 결과 중 **Input Voltage Error(bit0)만 단독으로** 켜진
        축을 reboot 후 재설정해서 복구한다. 다른 비트가 섞였으면 안 건드리고
        그대로 돌려준다 — 과열/과부하/전기충격/인코더 에러는 진짜 문제일 수
        있으니 호출자가 정지 여부를 판단해야 한다.

        reboot 은 그 축의 RAM 설정(토크/모드/전류상한, head_roll 이면 P/D 게인)을
        전부 날리므로 arm() 과 같은 순서로 다시 건다.

        반환: (recovered_names, remaining_names).
        """
        recovered_names, remaining_names, recovered_ids = [], [], []
        for name, err in zip(NAMES, raw_errors):
            if not err:
                continue
            if err == 1:  # bit0 단독
                recovered_names.append(name)
                recovered_ids.append(BY_NAME[name][1])
            else:
                remaining_names.append(name)
        if recovered_ids:
            for i in recovered_ids:
                self.io.reboot(i)
            time.sleep(0.3)
            self.io.sync_write_torque_enable(recovered_ids, [0] * len(recovered_ids))
            self.io.sync_write_operating_mode(recovered_ids, [MODE_CURRENT_POSITION] * len(recovered_ids))
            self.io.sync_write_current_limit(recovered_ids, [self.current_limit] * len(recovered_ids))
            self.io.sync_write_torque_enable(recovered_ids, [1] * len(recovered_ids))
            self._apply_head_gains(recovered_ids)
        return recovered_names, remaining_names

    def hold_here(self):
        """현재 위치를 목표로 덮어써 그 자리에 고정한다 (토크는 켠 채)."""
        pos = self.get_present_positions()
        self.set_position_vec(pos)
        return pos
