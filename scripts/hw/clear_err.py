#!/usr/bin/env python3
"""하드웨어 에러 래치를 reboot 으로 지운다 (Jetson).

    python3 ~/clear_err.py              # 에러 있는 축 전부
    python3 ~/clear_err.py left_knee    # 특정 축만

Overload(bit5) 같은 래치는 전원 재인가나 reboot 전에는 안 풀린다.
reboot 은 그 축의 RAM 설정(토크/모드/전류상한)을 공장값으로 되돌리므로
모드와 전류상한을 다시 건다. **토크는 켜지 않는다** — 사람이 로봇을 잡고
있을 때 갑자기 힘이 들어가면 위험하다. 토크는 rl_walk / goto_ready 가 켠다.
"""
import sys, time
sys.path.insert(0, "/home/parksuho")
from rustypot_hwi import HWI, NAMES, BY_NAME, MODE_CURRENT_POSITION

BITS = ["InputVoltage", "AngleLimit", "Overheating", "Range", "Checksum", "Overload", "InstError"]
want = sys.argv[1:] if len(sys.argv) > 1 else None

h = HWI()
ids = [BY_NAME[n][1] for n in NAMES]
err = h.io.sync_read_hardware_error_status(ids)
print(f"{'ID':>4} {'name':>18} {'err':>4}  bits")
targets = []
for n, i, e in zip(NAMES, ids, err):
    bits = ",".join(BITS[b] for b in range(7) if (e >> b) & 1) or "-"
    mark = ""
    # bit0(InputVoltage) 단독은 이 로봇의 Shutdown 마스크 밖이라 정보성이다.
    if e > 1 and (want is None or n in want):
        targets.append((n, i)); mark = "  <- 리부트"
    print(f"{i:4} {n:>18} {e:4}  {bits}{mark}")

if not targets:
    print("\n지울 래치가 없다 (err<=1 은 정보성 InputVoltage).")
    raise SystemExit

print(f"\n리부트: {[n for n, _ in targets]}")
for _, i in targets:
    h.io.reboot(i)
time.sleep(0.5)
tid = [i for _, i in targets]
h.io.sync_write_torque_enable(tid, [0] * len(tid))
h.io.sync_write_operating_mode(tid, [MODE_CURRENT_POSITION] * len(tid))
h.io.sync_write_current_limit(tid, [h.current_limit] * len(tid))
time.sleep(0.2)

err2 = h.io.sync_read_hardware_error_status(ids)
print(f"\n{'ID':>4} {'name':>18} {'전':>4} {'후':>4}")
for n, i, a, b in zip(NAMES, ids, err, err2):
    if a != b or (n, i) in targets:
        print(f"{i:4} {n:>18} {a:4} {b:4}" + ("   지워짐" if b <= 1 < a else ""))
print("\n토크는 꺼진 채로 둔다. 다음 실행이 켠다 (goto_ready / rl_walk / joint_step_test).")
