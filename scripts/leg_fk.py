"""URDF에서 뽑아온 다리 체인의 순기구학. 의존성 없음 (numpy만).

랩PC의 Isaac Sim 파이썬에서는 pinocchio 임포트가 실패한다
(libhpp-fcl.so가 Isaac이 번들한 Assimp와 심볼 충돌).
재생 오버레이에서 "레퍼런스가 발을 놓아야 할 위치"를 그리려면 순기구학이
필요하므로, URDF의 고정 변환을 여기 박아두고 직접 계산한다.
값은 robot/robot.urdf에서 추출했고 맥의 pinocchio 결과와 대조 검증했다.
"""

import numpy as np

# (관절명, 원점 xyz, 원점 rpy) — 모든 회전축은 Z
LEGS = {
    "left": [
        ("left_hip_yaw",   [-0.019,  0.035,  0.04744], [ 3.14159,  0.0,     0.0]),
        ("left_hip_roll",  [ 0.019, -0.00011, 0.04599], [-1.5708,  0.0,    -1.5708]),
        ("left_hip_pitch", [ 0.0753, -0.0,   -0.03569], [ 3.14159, -1.5708, 0.0]),
        ("left_knee",      [ 0.0,    0.07865, 0.0],     [ 0.0,     0.0,     0.0]),
        ("left_ankle",     [ 0.0,    0.07865, 0.0],     [ 0.0,     0.0,     0.0]),
    ],
    "right": [
        ("right_hip_yaw",   [-0.019, -0.035,  0.04744], [ 3.14159,  0.0,     0.0]),
        ("right_hip_roll",  [ 0.019, -0.00011, 0.04599], [ 1.5708,  0.0,    -1.5708]),
        ("right_hip_pitch", [-0.0753, 0.0,     0.03569], [ 0.0,     1.5708,  0.0]),
        ("right_knee",      [ 0.0,    0.07865, 0.038],   [-3.14159, 0.0,     0.0]),
        ("right_ankle",     [ 0.0,   -0.07865, -0.0],    [ 0.0,     0.0,     0.0]),
    ],
}
# 마지막 고정 변환 (ankle -> foot frame)
FOOT_FIXED = {
    "left":  ([0.0005,  0.03623, -0.01955], [1.5708, 0.0, 0.0]),
    "right": ([0.0005, -0.03623,  0.01955], [-1.5708, 0.0, 0.0]),
}


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _T(xyz, rpy):
    M = np.eye(4)
    M[:3, :3] = _rpy(*rpy)
    M[:3, 3] = xyz
    return M


def _Rz(a):
    M = np.eye(4)
    c, s = np.cos(a), np.sin(a)
    M[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return M


def foot_in_trunk(joint_angles: dict) -> dict:
    """{관절명: 각도} -> {"left"/"right": 몸통 기준 발 위치 (3,)}"""
    out = {}
    for side, chain in LEGS.items():
        M = np.eye(4)
        for nm, xyz, rpy in chain:
            M = M @ _T(xyz, rpy) @ _Rz(float(joint_angles[nm]))
        M = M @ _T(*FOOT_FIXED[side])
        out[side] = M[:3, 3].copy()
    return out
