#!/usr/bin/env python3
"""서보 게인/상한 레지스터를 **읽기만** 해서 공장값과 나란히 찍는다 (Jetson).

    ssh parksuho@192.168.137.7 'python3 ~/gain_report.py'

토크를 켜지 않고 아무것도 쓰지 않는다. 매달아 놓지 않아도 안전하다.

왜 필요한가 (2026-08-13).

P/D/I 와 Profile Velocity 는 **RAM** 이라 전원 재인가·reboot·모드 전환에서
공장값으로 돌아간다. Current Limit(38)/Velocity Limit(44)/PWM Limit(36)은
EEPROM 이라 남는다. 그래서 "지금 이 로봇에 뭐가 걸려 있나" 는 읽어 보기 전엔
알 수 없고, rl_walk.py 가 arm() 에서 거는 값과 실제로 얹힌 값이 다를 수 있다
(모드 전환이 게인을 덮는 순서 문제를 이미 한 번 겪었다).

각 레지스터가 물리적으로 무슨 뜻인지도 같이 환산해 찍는다 — 숫자만 보면
"1402 가 800 보다 크다" 밖에 안 나온다.
"""

import math
import sys

from rustypot_hwi import (BY_NAME, CURRENT_UNIT_MA, IDS, LEG_IDS, LEG_NAMES,
                          NAMES, TICK_RAD, TORQUE_PER_AMP, HEAD_P, HEAD_D,
                          SIM_STIFFNESS, LEG_P_MATCHED, joint_stiffness)
import rustypot

BAUD = 1_000_000
PWM_LIMIT_FULL = 885          # PWM Limit 기본값 = 100 % duty
STALL_TORQUE = 4.1            # N·m @ 12 V (XM430-W350 데이터시트)
STALL_CURRENT = 2.3           # A  @ 12 V
NO_LOAD_RADS = 4.82           # rad/s @ 12 V (46 rpm)
VEL_UNIT_RPM = 0.229

#: 모드 5(전류기반 위치제어) 진입 시 펌웨어가 다시 넣는 값. 2026-08-12 실측.
MODE5_DEFAULT = {"P": 800, "I": 0, "D": 4700}
#: EEPROM 공장 출하값 (XM430-W350).
#: velocity_limit 200 은 레지스터 범위 최대(1023)가 아니라 **출하 초기값**이다 —
#: 200 × 0.229 rpm = 45.8 rpm 으로 이 모델의 무부하 속도 46 rpm 과 같다. 우리
#: 코드는 이 레지스터를 쓴 적이 없다(2026-08-13 grep 확인). 즉 이건 "우리가 건
#: 제한" 이 아니라 모터가 원래 낼 수 있는 속도 그 자체다 — 1023 으로 올려도
#: 물리적으로 더 못 돈다.
FACTORY_EEPROM = {"current_limit": 1193, "velocity_limit": 200, "pwm_limit": 885}

REG_MAX = {"current_limit": 1193, "velocity_limit": 1023, "pwm_limit": 885,
           "P": 16383, "I": 16383, "D": 16383}


def sat_deg(p_gain, cap, full):
    """위치오차가 몇 도를 넘으면 위치 PID 출력이 cap 에 물리나.

    출력 = (P/128)·오차틱 이고 full 이 그 출력의 100 % 지점이다.
    """
    if not p_gain:
        return float("inf")
    return math.degrees(cap * 128.0 / p_gain * TICK_RAD)


def main():
    try:
        io = rustypot.Xl430PyController(serial_port="/dev/ttyUSB0", baudrate=BAUD,
                                        timeout=0.5)
    except Exception as e:
        sys.exit(f"버스를 못 열었다: {e}\n로봇 전원과 /dev/ttyUSB0 을 확인할 것.")

    def rd(fn, ids):
        try:
            return fn(ids)
        except Exception as e:
            sys.exit(f"읽기 실패({fn.__name__}): {e}\n전원이 꺼져 있거나 다른 프로세스가 포트를 쥐고 있다.")

    mode = rd(io.sync_read_operating_mode, IDS)
    tq = rd(io.sync_read_torque_enable, IDS)
    p = rd(io.sync_read_position_p_gain, IDS)
    i = rd(io.sync_read_position_i_gain, IDS)
    d = rd(io.sync_read_position_d_gain, IDS)
    clim = rd(io.sync_read_current_limit, IDS)
    gcur = rd(io.sync_read_goal_current, IDS)
    vlim = rd(io.sync_read_velocity_limit, IDS)
    plim = rd(io.sync_read_pwm_limit, IDS)
    pvel = rd(io.sync_read_profile_velocity, IDS)

    print(f"토크 상태: {'켜짐 ⚠' if any(tq) else '꺼짐'}   "
          f"동작모드: {sorted(set(mode))}\n")

    print("═" * 92)
    print("축별 현재값")
    print("═" * 92)
    print(f"{'관절':16s} {'ID':>3} {'P':>6} {'I':>4} {'D':>6} "
          f"{'전류상한':>8} {'GoalCur':>8} {'속도상한':>8} {'PWM상한':>7} {'ProfVel':>7}")
    for k, n in enumerate(NAMES):
        print(f"{n:16s} {IDS[k]:3d} {p[k]:6d} {i[k]:4d} {d[k]:6d} "
              f"{clim[k]:8d} {gcur[k]:8d} {vlim[k]:8d} {plim[k]:7d} {pvel[k]:7d}")

    li = [NAMES.index(n) for n in LEG_NAMES]
    one = lambda v: (v[li[0]] if len(set(v[x] for x in li)) == 1 else None)
    P, D, I = one(p), one(d), one(i)
    CL, VL, PL = one(clim), one(vlim), one(plim)

    print()
    print("═" * 92)
    print("다리 10축: 공장값 vs 지금")
    print("═" * 92)
    if P is None or D is None:
        print("!! 다리 축끼리 게인이 다르다 — 위 표를 직접 볼 것.")
    rows = [
        ("Position P Gain", MODE5_DEFAULT["P"], P, REG_MAX["P"], "RAM"),
        ("Position I Gain", MODE5_DEFAULT["I"], I, REG_MAX["I"], "RAM"),
        ("Position D Gain", MODE5_DEFAULT["D"], D, REG_MAX["D"], "RAM"),
        ("Current Limit", FACTORY_EEPROM["current_limit"], CL, REG_MAX["current_limit"], "EEPROM"),
        ("Velocity Limit", FACTORY_EEPROM["velocity_limit"], VL, REG_MAX["velocity_limit"], "EEPROM"),
        ("PWM Limit", FACTORY_EEPROM["pwm_limit"], PL, REG_MAX["pwm_limit"], "EEPROM"),
    ]
    print(f"{'레지스터':18s} {'공장값':>8} {'지금':>8} {'배율':>7} {'레지스터max':>11}  저장")
    for name, fac, now, mx, where in rows:
        if now is None:
            print(f"{name:18s} {fac:8d} {'축마다다름':>8}")
            continue
        ratio = f"{now/fac:.2f}x" if fac else "—"
        mark = "" if now == fac else "  <- 바꿈"
        print(f"{name:18s} {fac:8d} {now:8d} {ratio:>7} {mx:11d}  {where}{mark}")

    print()
    print("═" * 92)
    print("물리적으로 뭐가 달라지나 (다리 10축)")
    print("═" * 92)

    def block(tag, pg, cl):
        k = joint_stiffness(pg)
        amp = cl * CURRENT_UNIT_MA / 1000.0
        tau = TORQUE_PER_AMP * max(amp - 0.27, 0.0)
        tau = min(tau, STALL_TORQUE)
        print(f"  [{tag}]  P={pg}  전류상한={cl}")
        print(f"     관절강성          {k:6.2f} N·m/rad   (심 {SIM_STIFFNESS} 의 {100*k/SIM_STIFFNESS:.0f} %)")
        print(f"     전류 포화 오차    {sat_deg(pg, cl, cl):6.2f}°        이 각도를 넘으면 토크가 더 안 는다")
        print(f"     PWM 포화 오차     {sat_deg(pg, PWM_LIMIT_FULL, PWM_LIMIT_FULL):6.2f}°        "
              f"이 각도를 넘으면 duty 100 % 로 물린다")
        print(f"     토크 상한         {tau:6.2f} N·m       (모터 스톨 {STALL_TORQUE} N·m)")

    block("공장값", MODE5_DEFAULT["P"], FACTORY_EEPROM["current_limit"])
    print()
    if P:
        block("지금", P, CL if CL else FACTORY_EEPROM["current_limit"])
    print()
    block("rl_walk 기본", LEG_P_MATCHED, 700)
    print()
    block("전류 최대", LEG_P_MATCHED, REG_MAX["current_limit"])

    if VL:
        print(f"\n  속도상한 {VL} 틱 = {VL*VEL_UNIT_RPM:.1f} rpm = "
              f"{VL*VEL_UNIT_RPM*2*math.pi/60:.2f} rad/s   "
              f"(모터 무부하 {NO_LOAD_RADS} rad/s — 이걸 넘겨 놔도 물리적으로 못 넘는다)")
    if pvel and one(pvel) == 0:
        print("  ProfileVelocity 0 = 궤적생성 끔(계단입력). 이때 속도는 Velocity Limit 이 캡한다.")

    print(f"\n  참고: rl_walk.py 기본 --pgain 은 {LEG_P_MATCHED} (심 stiffness {SIM_STIFFNESS} 에 맞춘 값)")
    print(f"        머리 3축(12/13/14)은 의도적으로 P={HEAD_P} D={HEAD_D} — 2026-08-09 계단응답 최적")


if __name__ == "__main__":
    main()
