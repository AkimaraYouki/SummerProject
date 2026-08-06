"""실기 다이나믹셀 14축 ↔ 시뮬 관절의 단일 진실원천.

`joint_order.py` 가 시뮬에서 하는 역할을 실기에서 하는 파일이다. 실기를 건드리는
모든 코드는 ID/오프셋/부호를 직접 적지 말고 여기서 임포트한다.

하드웨어: XM430-W350 (model 1020) ×14, 프로토콜 2.0 / 57600,
U2D2 는 **Jetson** `parksuho@192.168.137.7` 의 `/dev/ttyUSB0` 에 있다 (데스크탑 아님).
자세한 배경은 docs/handoff/project_hardware_joint_calibration.md.

--------------------------------------------------------------------------
확정된 것 / 아직 아닌 것
--------------------------------------------------------------------------
확정  DXL_ID      — 2026-08-06 사용자가 직접 지정해둔 값을 받아 기록. 조립된
                    실기의 실제 설정이다.
미확정 OFFSET_TICK — 각 관절의 URDF 영점이 몇 tick 인지. 지금은 전부 2048
                    (모터 중앙 = 관절 0) 이라고 **가정**하고 있을 뿐, 측정하지
                    않았다. 절차: 토크를 풀고 관절을 기구학적 영점에 손으로 맞춘
                    뒤 present position 을 읽는다.
미확정 SIGN        — 모터 +방향과 URDF +방향이 같은지. 지금은 전부 +1 **가정**.
                    절차: 한 관절씩 +33 tick (+0.05 rad) 만 주고 실제 움직인
                    방향을 시뮬과 대조. 반대면 -1.

측정 전까지 rad→tick 환산은 믿을 수 없다. `to_ticks()` 가 STRICT=True 일 때
일부러 예외를 던지는 이유다 — 가정값이 조용히 실기로 나가는 것을 막는다.
"""

import json
import math
import os

from .joint_order import ACTUATOR_JOINT_NAMES

# 1 tick = 360/4096 도 = 0.0879° = 0.001534 rad. 2048 = 모터 중앙.
TICK_RAD = 2.0 * math.pi / 4096.0
CENTER_TICK = 2048

BAUDRATE = 57600
PROTOCOL_VERSION = 2.0
PORT = "/dev/ttyUSB0"
HOST = "parksuho@192.168.137.7"  # U2D2 가 물려 있는 Jetson

# 관절 이름 -> 다이나믹셀 ID. 2026-08-06 사용자 제공.
# 로봇에 이미 설정돼 있는 값이므로 여기를 고치지 말고, 바꾸려면 모터 쪽부터 바꿀 것.
DXL_ID = {
    "left_hip_yaw": 3,
    "left_hip_roll": 8,
    "left_hip_pitch": 9,
    "left_knee": 10,
    "left_ankle": 11,
    "neck_pitch": 2,
    "head_pitch": 12,
    "head_yaw": 13,
    "head_roll": 14,
    "right_hip_yaw": 1,
    "right_hip_roll": 4,
    "right_hip_pitch": 5,
    "right_knee": 6,
    "right_ankle": 7,
}

# ID -> 관절 이름 (역방향 조회용).
JOINT_BY_ID = {i: n for n, i in DXL_ID.items()}

# 관절의 URDF 영점에 해당하는 tick, 그리고 모터 +방향이 URDF +방향과 같은지(+1/-1).
# 기본은 "중앙 = 0 rad, 부호 그대로" 라는 **가정**이다. scripts/hw/joint_cal.py 로
# 실기에서 재고 나온 joint_calibration.json 이 아래 경로에 있으면 그 값이 이긴다.
OFFSET_TICK = {n: CENTER_TICK for n in ACTUATOR_JOINT_NAMES}
SIGN = {n: +1 for n in ACTUATOR_JOINT_NAMES}

_CAL_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_calibration.json")

# 14축 전부가 실측되기 전에는 True 로 남는다. True 인 동안 to_ticks() 는 예외를
# 던져서, 가정값 그대로 실기에 각도를 내보내는 사고를 막는다.
CALIBRATION_PENDING = True

if os.path.exists(_CAL_JSON):
    with open(_CAL_JSON) as _f:
        _cal = json.load(_f)
    for _n, _e in _cal.items():
        if _n not in OFFSET_TICK:
            raise ValueError(f"joint_calibration.json 에 모르는 관절 이름: {_n}")
        if _e["id"] != DXL_ID[_n]:
            raise ValueError(
                f"{_n}: joint_calibration.json 의 ID {_e['id']} 가 DXL_ID 의 {DXL_ID[_n]} 와 다르다"
            )
        OFFSET_TICK[_n] = _e["offset_tick"]
        SIGN[_n] = _e["sign"]
    # 파일에 14축이 다 들어 있다고 끝난 게 아니다 — joint_cal.py 는 미확인 축도
    # 기본값(+1)으로 함께 쓴다. 사용자가 실기에서 눈으로 확인한 축만 confirmed 다.
    UNCONFIRMED = sorted(n for n in ACTUATOR_JOINT_NAMES if not _cal.get(n, {}).get("confirmed"))
    CALIBRATION_PENDING = bool(UNCONFIRMED)
else:
    UNCONFIRMED = sorted(ACTUATOR_JOINT_NAMES)

# robot/robot.urdf 의 revolute joint limit (rad). 조그·이동 전 클램프용.
JOINT_LIMIT_RAD = {
    "left_hip_yaw": (-0.5236, 0.5236),
    "left_hip_roll": (-0.4363, 0.4363),
    "left_hip_pitch": (-0.5236, 1.2217),
    "left_knee": (-2.0944, 2.0944),
    "left_ankle": (-1.5708, 1.5708),
    "neck_pitch": (-0.3491, 1.1345),
    "head_pitch": (-0.8727, 0.8727),  # ±50도. CAD 클램프 45도는 기구 한계가 아니다 (실제 설계 ~±60도, 2026-08-07 사용자 확인)
    "head_yaw": (-2.7925, 2.7925),
    "head_roll": (-0.5236, 0.5236),
    "right_hip_yaw": (-0.5236, 0.5236),
    "right_hip_roll": (-0.4363, 0.4363),
    "right_hip_pitch": (-0.5236, 1.2217),
    "right_knee": (-2.0944, 2.0944),
    "right_ankle": (-1.5708, 1.5708),
}

assert set(DXL_ID) == set(ACTUATOR_JOINT_NAMES), "hardware_map 의 관절 이름이 joint_order 와 다르다"
assert len(set(DXL_ID.values())) == 14, "ID 가 중복됐다"
assert set(JOINT_LIMIT_RAD) == set(ACTUATOR_JOINT_NAMES)


def limit_ticks(joint: str) -> tuple[int, int]:
    """관절의 URDF 한계를 tick 범위로. 오프셋/부호가 미확정이면 가정값 기준이다."""
    lo, hi = JOINT_LIMIT_RAD[joint]
    a = OFFSET_TICK[joint] + SIGN[joint] * lo / TICK_RAD
    b = OFFSET_TICK[joint] + SIGN[joint] * hi / TICK_RAD
    return (int(round(min(a, b))), int(round(max(a, b))))


def to_ticks(joint: str, rad: float, strict: bool = True) -> int:
    """URDF 관절각(rad) -> 다이나믹셀 goal position(tick), 한계 안으로 클램프.

    strict=True 이고 캘리브레이션이 안 끝났으면 예외를 던진다. 검증 목적으로
    가정값을 일부러 쓰려면 strict=False 를 명시할 것.
    """
    if strict and joint in UNCONFIRMED:
        raise RuntimeError(
            f"{joint}: 부호가 실기에서 확인되지 않았다 (joint_calibration.json 의 "
            f"confirmed=false). 미확인 {len(UNCONFIRMED)}축: {', '.join(UNCONFIRMED)}. "
            "가정값으로 움직이려면 to_ticks(..., strict=False) 로 명시할 것."
        )
    lo, hi = limit_ticks(joint)
    t = OFFSET_TICK[joint] + SIGN[joint] * rad / TICK_RAD
    return max(lo, min(hi, int(round(t))))


def to_rad(joint: str, tick: int) -> float:
    """다이나믹셀 present position(tick) -> URDF 관절각(rad)."""
    return SIGN[joint] * (tick - OFFSET_TICK[joint]) * TICK_RAD
