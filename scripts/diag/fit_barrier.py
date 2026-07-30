"""CBF 장벽함수 h(q_leg) 를 5차원 신경망으로 피팅한다.

왜 신경망인가. 형상 근사를 두 번 시도해 두 번 실패했다 -- 단일 캡슐은 몸통
반지름이 112.8 mm 로 나왔고, 구 64+24 개는 보수성은 지켰지만(위반 0건) 실제보다
평균 30.8 mm 적게 보고해서 모든 자세가 관통으로 판정됐다. 실제 여유가 5~14 mm 인
문제에서는 둘 다 쓸 수 없다.

대신 함수를 직접 배운다. `check5d.py` 가 확인한 두 가지 덕분에 5차원이면 된다:
반대쪽 다리와 무관(오차 0.0000 mm), 좌우 미러 대칭(최대 0.22 mm).

**보수성 확보**: 학습만으로는 보장이 안 된다. 검증셋에서 **최대 과대추정치**를
재고 그만큼 빼서, 예측이 실제 간격을 넘는 일이 없게 만든다. 그래야 필터가
"안전하다"고 틀리지 않는다.
"""
import argparse, numpy as np, torch, torch.nn as nn


class Barrier(nn.Module):
    def __init__(self, w=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(),
            nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="barrier_h5d.pt")
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()

    d = np.load(a.data)
    S, D = d["S"].astype(np.float32), d["D"].astype(np.float32)
    lo, hi = d["lo"].astype(np.float32), d["hi"].astype(np.float32)
    X = (S - lo) / (hi - lo) * 2 - 1                     # [-1, 1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(X); k = int(n * 0.9)
    g = np.random.default_rng(0).permutation(n)
    tr, va = g[:k], g[k:]
    Xt = torch.tensor(X[tr], device=dev); Yt = torch.tensor(D[tr] * 1000, device=dev)
    Xv = torch.tensor(X[va], device=dev); Yv = torch.tensor(D[va] * 1000, device=dev)

    m = Barrier().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    for e in range(a.epochs):
        m.train()
        p = m(Xt)
        r = p - Yt
        # 과대추정(예측 > 실제)에 3배 벌점. 오프셋을 작게 만들어 필터가 덜 답답해진다.
        loss = (torch.where(r > 0, 3.0 * r, -r) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if (e + 1) % 100 == 0:
            m.eval()
            with torch.no_grad():
                rv = m(Xv) - Yv
            print(f"[{e+1:4d}] train {loss.item():7.3f} · val |오차| 평균 "
                  f"{rv.abs().mean():.3f} mm · 최대 과대추정 {rv.max():.3f} mm", flush=True)

    m.eval()
    with torch.no_grad():
        over = float((m(Xv) - Yv).max())
    print(f"\n[보수 오프셋] 검증셋 최대 과대추정 {over:.3f} mm")
    print(f"  h(q) = net(q) - {over:.3f} mm 로 쓰면 예측이 실제를 넘지 않는다")
    torch.save({"state": m.state_dict(), "lo": lo, "hi": hi, "offset_mm": over,
                "joints": [str(x) for x in d["joints"]]}, a.out)
    print(f"[ok] {a.out}")


if __name__ == "__main__":
    main()
