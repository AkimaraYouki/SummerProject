"""필터의 계산 시간. 실기 제어 주기(50 Hz = 20 ms) 안에 들어와야 한다."""
import sys, time, numpy as np, torch
sys.path.insert(0, "source")
from open_duck_mini_isaaclab.safety_filter import ClearanceFilter

for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
    f = ClearanceFilter(sys.argv[1], dev, margin_mm=5.0)
    for n in (1, 16):
        q = torch.zeros(n, 14, device=dev)
        d = np.load("/home/parksuho/odm_out/gait_v28.npz")
        names = [str(x) for x in d["leg_names"]]
        L = ["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle"]
        R = ["right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle"]
        li = [names.index(x) for x in L]; ri = [names.index(x) for x in R]
        src = torch.tensor(d["forward__q"][100:100+n, 0], dtype=torch.float32, device=dev)
        q[:, li] = src[:, li]; q[:, ri] = src[:, ri]
        for _ in range(20):
            f(q, li, ri)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            f(q, li, ri)
        if dev == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 200 * 1000
        print(f"@@{dev:5s} 환경 {n:2d}개 · {ms:6.3f} ms/스텝 · 20 ms 예산의 {ms/20*100:4.1f}%")
