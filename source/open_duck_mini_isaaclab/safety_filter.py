"""다리-몸통 간격을 지키는 런타임 안전 필터 (CBF).

**왜 필터인가.** 실기에서 다리가 몸통에 닿으면 액추에이터가 깨진다. 리워드 항은
위반을 통계적으로 줄일 뿐 0 을 약속하지 못한다 -- v32 가 5 mm 위반을 38% 에서
1.0% 로 낮췄지만 1.0% 는 0 이 아니다. 보장은 런타임에서만 나온다.

**왜 QP 를 안 푸는가.** 표준 CBF-QP 는

    min |u - u_policy|^2   s.t.  grad_h . (q + u) >= -alpha * h

인데, 여기서는 제약이 다리당 **한 개**다 (h 는 스칼라). 제약이 하나인 등방
2차 문제의 해는 닫힌 형태로 나온다 -- grad_h 방향으로의 최소 투영이다.
QP 솔버도, 반복도 필요 없다. 다리당 신경망 1회 + 역전파 1회가 전부다.

**보수성.** h 는 학습된 근사이므로 그대로 믿으면 안 된다. `fit_barrier.py` 가
검증셋에서 잰 최대 과대추정(1.680 mm)을 빼고, 좌우를 한 함수로 처리하는 데서
오는 미러 잔차(0.22 mm, check_barrier_dims.py)도 뺀다. 그래야 필터가
"안전하다" 고 틀리는 경우가 없다.
"""

from __future__ import annotations

import torch

MIRROR_RESIDUAL_MM = 0.22   # check_barrier_dims.py: 좌우를 한 함수로 쓰는 대가
# 왼다리 자세를 오른다리에 대응시키는 부호 (symmetry.py 에서 유도·검증한 값)
MIRROR_SIGN = (-1.0, 1.0, 1.0, -1.0, -1.0)


class ClearanceFilter:
    """정책이 낸 관절 목표를 다리-몸통 간격이 안전한 쪽으로 최소한만 민다.

    Args:
        ckpt: `fit_barrier.py` 가 저장한 파일.
        margin_mm: 지켜야 할 간격. 사용자 지정 실기 기준 5 mm.
        alpha: 반복 투영에서 한 스텝의 크기(1.0 = 뉴턴 완전 스텝).
    """

    def __init__(self, ckpt: str, device, margin_mm: float = 5.0, alpha: float = 1.0):
        from open_duck_mini_isaaclab.tasks.velocity.barrier_net import Barrier

        d = torch.load(ckpt, map_location=device, weights_only=False)
        net = Barrier().to(device).eval()
        net.load_state_dict(d["state"])
        # 가중치만 꺼내 쓰고 nn.Module 은 버린다. 순전파와 역전파를 손으로
        # 계산하기 위해서다 -- autograd 를 쓰면 `torch.inference_mode()` 안에서
        # 터진다 (play_fixed_cmd.py 가 그렇게 돌린다: "element 0 of tensors does
        # not require grad"). 제어 루프에 들어갈 코드가 호출 문맥에 따라
        # 죽으면 안 되고, 4 층 MLP 라 손미분이 어렵지도 않다.
        self.W = [l.weight.detach() for l in net.net if isinstance(l, torch.nn.Linear)]
        self.b = [l.bias.detach() for l in net.net if isinstance(l, torch.nn.Linear)]
        self.lo = torch.as_tensor(d["lo"], device=device)
        self.hi = torch.as_tensor(d["hi"], device=device)
        # 학습 오차와 미러 잔차를 둘 다 빼서 절대 낙관하지 않게 한다
        self.bias = float(d["offset_mm"]) + MIRROR_RESIDUAL_MM
        self.margin = margin_mm
        self.alpha = alpha
        self.sign = torch.tensor(MIRROR_SIGN, device=device)

    def _h(self, q5: torch.Tensor):
        """h(q) 와 dh/dq. 단위는 mm. autograd 를 쓰지 않는다.

        SiLU: s(z) = z * sig(z),  s'(z) = sig(z) * (1 + z * (1 - sig(z)))
        """
        x = (q5 - self.lo) / (self.hi - self.lo) * 2.0 - 1.0
        a, ds = x, []
        for k in range(3):                       # 은닉 3 층
            z = a @ self.W[k].T + self.b[k]
            sig = torch.sigmoid(z)
            a = z * sig
            ds.append(sig * (1.0 + z * (1.0 - sig)))
        val = (a @ self.W[3].T + self.b[3]).squeeze(-1) - self.bias - self.margin

        g = self.W[3].expand(q5.shape[0], -1)    # [N, w]
        for k in (2, 1, 0):
            g = g * ds[k]
            g = g @ self.W[k]
        return val, g * 2.0 / (self.hi - self.lo)

    def _project(self, q5: torch.Tensor, iters: int = 12):
        """h(q) >= 0 이 될 때까지 grad 방향으로 최소 이동을 반복한다.

        한 번의 CBF 스텝은 위반량의 alpha 배만 줄인다. 목표를 **실제로** 안전
        집합 안에 넣으려면 반복해야 한다 -- 제약이 하나뿐이라 각 스텝이 닫힌
        형태(뉴턴 투영)라 12회를 돌아도 비용이 신경망 12 회에 불과하다.
        건드리는 것은 위반한 환경뿐이고, 여유 있는 자세는 그대로 통과한다.
        """
        q = q5
        for _ in range(iters):
            h, g = self._h(q)
            bad = h < 0
            if not bool(bad.any()):
                break
            gg = (g * g).sum(-1).clamp_min(1e-9)
            step = (-h / gg).clamp(min=0.0).unsqueeze(-1) * g
            q = q + bad.float().unsqueeze(-1) * step
        return q, self._h(q)[0]

    def __call__(self, target: torch.Tensor, left_idx, right_idx):
        """`target` [N, nj] 의 다리 관절만 안전 쪽으로 민다. 나머지는 그대로."""
        out = target.clone()
        ql = target[:, left_idx]
        qr = target[:, right_idx] * self.sign      # 오른다리를 왼다리 좌표계로
        nl, hl = self._project(ql)
        nr, hr = self._project(qr)
        out[:, left_idx] = nl
        out[:, right_idx] = nr * self.sign
        self.last_h = torch.stack([hl, hr], dim=-1)   # 로깅용 (mm, 0 이 임계)
        return out
