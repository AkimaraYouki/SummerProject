import sys, time
sys.path.insert(0, "/home/parksuho")
from rustypot_hwi import HWI, IDS, NAMES
h = HWI()
# XM430 Present Input Voltage = addr 144, 2바이트, 0.1 V 단위
raw = h.io.sync_read_raw_data(IDS, 144, 2)
v = [(b[0] | (b[1] << 8)) / 10.0 for b in raw]
err = h.io.sync_read_hardware_error_status(IDS)
print("%18s %7s %5s" % ("name", "volt", "err"))
for n, vv, e in zip(NAMES, v, err):
    flag = "  <- 낮음" if vv < 10.5 else ""
    print("%18s %6.1fV %5d%s" % (n, vv, e, flag))
print("\nXM430 정격 12V, 최소 동작 9.5V.  11V 아래면 InputVoltage 비트가 뜬다.")
print("평균 %.1fV  최소 %.1fV" % (sum(v)/len(v), min(v)))
