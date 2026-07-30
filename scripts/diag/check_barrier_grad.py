"""수동 역전파가 autograd 와 같은 값을 주는지, 추론 모드에서도 도는지 확인."""
import sys, torch
sys.path.insert(0, "source")
from open_duck_mini_isaaclab.safety_filter import ClearanceFilter
f = ClearanceFilter(sys.argv[1], "cpu", margin_mm=5.0)
q = torch.rand(8, 5) * 0.4 - 0.2 + (f.lo + f.hi) / 2
h, g = f._h(q)
# 수치 미분과 대조
eps, num = 1e-4, torch.zeros_like(g)
for j in range(5):
    d = torch.zeros_like(q); d[:, j] = eps
    num[:, j] = (f._h(q + d)[0] - f._h(q - d)[0]) / (2 * eps)
print(f"@@해석 vs 수치 미분 최대 오차 {float((g - num).abs().max()):.6f}")
with torch.inference_mode():
    h2, g2 = f._h(q)
    _ = f._project(q)
print(f"@@inference_mode 안에서 정상 동작 · h 일치 {bool(torch.allclose(h, h2))}")
