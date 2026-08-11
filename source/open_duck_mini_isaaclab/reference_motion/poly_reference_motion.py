"""Batched torch port of Open_Duck_Playground's
playground/common/poly_reference_motion.py::PolyReferenceMotion.

Pure torch, zero Isaac Sim / physics-engine dependency — this module is
fully unit-testable on a CPU-only machine (see
tests/test_poly_reference_motion_cpu.py), unlike almost everything else in
this package.

Reference-frame layout (36-dim). The original module's docstring described a
40-dim layout (16 joints incl. 2 antennas) — this rebuild's robot has no
antenna hardware at all (confirmed 2026-07-26, see joint_order.py), so the
reference-motion generator no longer records them and this is 14 joints:
  0:14   joint pos  (14 joints, order = joint_order.REF_JOINT_NAMES)
  14:28  joint vel  (same 14-joint order)
  28:30  foot contacts (left, right)
  30:33  base linear velocity (world frame)
  33:36  base angular velocity (world frame)
"""

from __future__ import annotations

import pickle

import torch

REF_FRAME_DIM = 36


class PolyReferenceMotion:
    def __init__(self, polynomial_coefficients_path: str, device: torch.device):
        self.device = device
        with open(polynomial_coefficients_path, "rb") as f:
            data = pickle.load(f)
        self._process(data)

    def _process(self, data: dict) -> None:
        dxs, dys, dthetas = [], [], []
        parsed: dict[float, dict[float, dict[float, list]]] = {}

        self.period = None
        self.fps = None
        self.frame_offsets = None
        self.startend_double_support_ratio = None
        self.nb_steps_in_period = None

        for name, entry in data.items():
            dx_str, dy_str, dtheta_str = name.split("_")
            dx, dy, dtheta = float(dx_str), float(dy_str), float(dtheta_str)

            if self.period is None:
                self.period = entry["period"]
                self.fps = entry["fps"]
                self.frame_offsets = entry["frame_offsets"]
                self.startend_double_support_ratio = entry["startend_double_support_ratio"]
                self.nb_steps_in_period = int(self.period * self.fps)

            if dx not in dxs:
                dxs.append(dx)
            if dy not in dys:
                dys.append(dy)
            if dtheta not in dthetas:
                dthetas.append(dtheta)

            # entry["coefficients"] is {"dim_0": [...], "dim_1": [...], ...}.
            # reference_motion_generator/scripts/fit_poly.py fits with
            # np.polyfit (which returns highest-degree-first) and stores
            # np.flip(coeffs) — i.e. the pkl holds LOWEST-degree-first
            # lists. The original JAX loader (poly_reference_motion.py)
            # does one more jp.flip to get back to highest-degree-first
            # before jp.polyval (which expects highest-degree-first, same
            # as np.polyval). We reproduce that single un-flip here so
            # `get_reference_motion`'s Horner loop below (which also
            # expects coeffs[0] == highest-degree term) gets the same
            # values the JAX version does.
            coeffs_by_dim = entry["coefficients"]
            coeffs = [list(reversed(coeffs_by_dim[f"dim_{d}"])) for d in range(len(coeffs_by_dim))]

            parsed.setdefault(dx, {}).setdefault(dy, {})[dtheta] = coeffs

        self.dxs = sorted(dxs)
        self.dys = sorted(dys)
        self.dthetas = sorted(dthetas)
        self.dx_range = (self.dxs[0], self.dxs[-1])
        self.dy_range = (self.dys[0], self.dys[-1])
        self.dtheta_range = (self.dthetas[0], self.dthetas[-1])

        nb_dx, nb_dy, nb_dtheta = len(self.dxs), len(self.dys), len(self.dthetas)
        # parsed[dx][dy][dtheta] is a list of D=36 per-dimension coefficient
        # lists, each of length K=degree+1 — grab the first one's length.
        _sample_coeffs = next(iter(next(iter(next(iter(parsed.values())).values())).values()))
        degree_plus_1 = len(_sample_coeffs[0])

        # [nb_dx, nb_dy, nb_dtheta, D, K] — K = polynomial degree + 1,
        # highest-degree coefficient first (Horner's method convention).
        # The recorded set can be a SPARSE subset of the full dx*dy*dtheta
        # grid (e.g. a speed filter drops out-of-band (dx,dy,dtheta) combos
        # individually — see auto_waddle.py's step-5 filter, fixed 2026-07-26).
        # get_reference_motion() below still does per-axis-independent
        # nearest-grid-point snapping, so every cell of this dense grid must
        # be populated. For any (dx,dy,dtheta) combo that was filtered out,
        # fall back to the coefficients of the nearest recorded combo (plain
        # Euclidean distance over (dx,dy,dtheta)) instead of crashing.
        all_points = [
            (dx, dy, dtheta)
            for dx, dy_map in parsed.items()
            for dy, dtheta_map in dy_map.items()
            for dtheta in dtheta_map
        ]

        def nearest_coeffs(dx: float, dy: float, dtheta: float) -> list:
            best = min(
                all_points,
                key=lambda p: (p[0] - dx) ** 2 + (p[1] - dy) ** 2 + (p[2] - dtheta) ** 2,
            )
            return parsed[best[0]][best[1]][best[2]]

        # **float64 여야 한다.** 15 차 다항식을 t∈[0,1) 에서 맞추면 계수가 1e8 대까지
        # 커지고 Horner 평가에서 대규모 자릿수 소거가 일어난다. float32(유효숫자 7)로는
        # 그 소거를 못 버텨서 결과가 무너진다 — 2026-08-11 실측으로 무릎 진폭 좌우차가
        # -3 % 여야 할 것이 +176 % 까지 튀었다. 표 자체는 작고(수십만 원소) 평가도
        # [N,36,16] 뿐이라 float64 비용은 무시할 만하다. 반환할 때 float32 로 되돌린다.
        grid = torch.zeros(nb_dx, nb_dy, nb_dtheta, REF_FRAME_DIM, degree_plus_1,
                           device=self.device, dtype=torch.float64)
        for xi, dx in enumerate(self.dxs):
            for yi, dy in enumerate(self.dys):
                for ti, dtheta in enumerate(self.dthetas):
                    if dtheta in parsed.get(dx, {}).get(dy, {}):
                        coeffs = parsed[dx][dy][dtheta]  # list of D lists, each length K
                    else:
                        coeffs = nearest_coeffs(dx, dy, dtheta)
                    grid[xi, yi, ti] = torch.tensor(coeffs, device=self.device, dtype=torch.float64)

        self.coeffs = grid
        self.dxs_t = torch.tensor(self.dxs, device=self.device)
        self.dys_t = torch.tensor(self.dys, device=self.device)
        self.dthetas_t = torch.tensor(self.dthetas, device=self.device)

    def get_reference_motion(self, dx: torch.Tensor, dy: torch.Tensor, dtheta: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Batched reference-frame lookup.

        Args:
            dx, dy, dtheta: [N] per-env commanded velocities.
            i: [N] per-env step counter (any int dtype); internally
               modulo'd by nb_steps_in_period, matching the JAX version.
        Returns:
            [N, 36] reference frame for each env's current command + phase.
        """
        dx = dx.clamp(self.dx_range[0], self.dx_range[1])
        dy = dy.clamp(self.dy_range[0], self.dy_range[1])
        dtheta = dtheta.clamp(self.dtheta_range[0], self.dtheta_range[1])

        ix = torch.argmin(torch.abs(self.dxs_t.view(1, -1) - dx.view(-1, 1)), dim=1)
        iy = torch.argmin(torch.abs(self.dys_t.view(1, -1) - dy.view(-1, 1)), dim=1)
        ith = torch.argmin(torch.abs(self.dthetas_t.view(1, -1) - dtheta.view(-1, 1)), dim=1)

        env_coeffs = self.coeffs[ix, iy, ith]  # [N, 36, K]

        t = (i.double() % self.nb_steps_in_period) / self.nb_steps_in_period
        t = t.clamp(0.0, 1.0)  # [N], float64 — 위 grid 주석 참고

        num_envs, dim, degree_plus_1 = env_coeffs.shape
        result = torch.zeros(num_envs, dim, device=self.device, dtype=torch.float64)
        for k in range(degree_plus_1):
            result = result * t.unsqueeze(-1) + env_coeffs[:, :, k]
        return result.float()
