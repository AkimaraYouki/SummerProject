"""실기 IMU(BNO055) ↔ 시뮬 IMU 관측의 단일 진실원천.

`hardware_map.py` 가 관절에 대해 하는 역할을 IMU 에 대해 한다. 실기 IMU 를
읽는 코드는 축맵을 직접 적지 말고 여기서 임포트한다.

--------------------------------------------------------------------------
몸통 프레임: **+x 앞 / +y 좌 / +z 위** (표준 오른손 프레임)
--------------------------------------------------------------------------
독립적으로 두 번 확인했다 (2026-08-06):

  1) 전진 명령(vx=+0.15)에서 디딤발이 몸통 기준 x 로 **-29.8 mm** 흐른다.
     발이 지면에 고정돼 있으므로 몸통이 +x 로 전진한다는 뜻이다. 후진 명령에서는
     **+42.9 mm** 로 정확히 뒤집힌다. 크기도 0.15 m/s × 접지 0.378 s = 56.7 mm
     와 같은 자릿수다.
  2) 발목 프레임 기준으로 발이 +x 로 **63.8 mm**, -x 로 **40.1 mm** 뻗는다 --
     발가락이 앞, 뒤꿈치가 뒤인 정상 형상.

  좌우는 URDF 로 자명하다: left_foot y=+90.9 mm, right_foot y=-91.7 mm -> +y 가 좌.

⚠️ `docs/handoff/project_hardware_bringup_2026-08-06.md` 가 처음에 "-x 가 앞"
이라고 적었던 것은 **틀렸다**(2026-08-06 정정). 그 근거는 "머리를 +45° 세우면
CoM 이 x 로 -14.3 mm 가고 로봇이 앞으로 넘어졌다" 였는데, CoM 이동량 자체는
재현되지만(-14.26 mm) 그 전도는 정적 전복이 아니라 **정책이 무너진 동적 결과**라
방향 판정의 근거가 되지 못한다. 관절 SIGN 값들은 실기에서 눈으로 직접 확인한
것이라 이 정정과 무관하게 유효하다.

--------------------------------------------------------------------------
시뮬이 기대하는 값 (joystick_env.py `_get_observations`)
--------------------------------------------------------------------------
관측 앞 6차원이 IMU 다:

  gyro  = imu.data.ang_vel_b   [rad/s]  몸통 프레임
  accel = imu.data.lin_acc_b   [m/s^2]  몸통 프레임, **중력 포함**

`lin_acc_b` 에 중력이 포함되는 것이 중요하다 -- Isaac Lab 의 Imu 센서는
`gravity_bias=(0,0,+9.81)` 을 월드 가속도에 더한 뒤 몸통 프레임으로 돌린다
(isaaclab/sensors/imu/imu.py). 즉 **실제 가속도계가 재는 비력(specific force)과
같은 규약**이고, 직립 정지 시 (0, 0, +9.81) 이다. BNO055 의 ACC 출력을 그대로
쓰면 되고 부호를 뒤집을 필요가 없다.

`use_gravity_obs=True` 인 설정(v33 계열)은 여기에 `projected_gravity_b` 3 차원이
더 붙는다. 이건 중력 **방향**이라 직립 시 (0, 0, -1) 이다:

    projected_gravity_b = -normalize(accel)      (정지 상태에서)

가속 중에는 이 근사가 틀리므로, BNO055 를 NDOF 모드로 두고 GRAVITY 레지스터
(0x2E)의 융합 중력 벡터를 쓰는 편이 낫다 -- 그 경우에도 부호는 같다
(projected_gravity = -normalize(gravity_vector)).
"""

import json
import os

# BNO055 접속 정보 (Jetson).
I2C_BUS = 7
I2C_ADDR = 0x28

_CAL_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imu_calibration.json")

# 칩 축 -> 몸통 축 변환. a_몸통 = AXIS_MATRIX @ a_IMU.
# 기본은 항등이고, imu_calibration.json 이 있으면 그 값이 이긴다.
AXIS_MATRIX = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
AXIS_RESIDUAL = None
IMU_CALIBRATED = False

if os.path.exists(_CAL_JSON):
    with open(_CAL_JSON) as _f:
        _cal = json.load(_f)
    AXIS_MATRIX = [[float(v) for v in row] for row in _cal["matrix"]]
    AXIS_RESIDUAL = float(_cal.get("residual", 0.0))
    IMU_CALIBRATED = True

# 2026-08-06 실측 결과: **항등**. 칩이 몸통 축에 정렬돼 붙어 있다.
# 세 자세(정립 / 코 아래 90° / 오른쪽 눕힘 90°)의 가속도계 평균에서 구한
# 측정 행렬의 대각 성분이 0.998 / 0.999 / 0.997 이고, 이상축과의 각도가
# 각각 3.63° / 2.09° / 4.64° 였다 -- 사람이 손으로 자세를 잡을 때 생기는
# 오차 범위이므로 직각 축맵으로 스냅했다 (잔차 0.078).
# 칩의 AXIS_MAP_CONFIG(0x41)/AXIS_MAP_SIGN(0x42) 는 기본값 0x24/0x00 그대로다.
# 굳이 레지스터에 굽지 않는다 -- BNO055 는 전원이 끊기면 축맵이 날아가므로
# 소프트웨어에서 적용하는 편이 버전 관리도 되고 재현도 된다.

# IMU 가 몸통 원점에서 떨어진 위치 (m). robot/robot.urdf 의 `imu_frame` 고정관절
# 값이다 (부모 trunk_assembly 기준). 시뮬 ImuCfg.offset 이 이 값과 같아야 한다.
MOUNT_POS = (-0.0388, 0.0, 0.0914)


def to_trunk(v):
    """칩 프레임 3벡터 -> 몸통 프레임. 가속도·자이로 둘 다 같은 변환이다."""
    m = AXIS_MATRIX
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def projected_gravity(accel_trunk):
    """몸통 프레임 가속도 -> 시뮬의 projected_gravity_b 규약 (직립 시 (0,0,-1))."""
    n = sum(c * c for c in accel_trunk) ** 0.5 or 1.0
    return tuple(-c / n for c in accel_trunk)
