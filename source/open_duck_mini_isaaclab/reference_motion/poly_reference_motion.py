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
        grid = torch.zeros(nb_dx, nb_dy, nb_dtheta, REF_FRAME_DIM, degree_plus_1, device=self.device)
        for xi, dx in enumerate(self.dxs):
            for yi, dy in enumerate(self.dys):
                for ti, dtheta in enumerate(self.dthetas):
                    coeffs = parsed[dx][dy][dtheta]  # list of D lists, each length K
                    grid[xi, yi, ti] = torch.tensor(coeffs, device=self.device)

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

        t = (i.float() % self.nb_steps_in_period) / self.nb_steps_in_period
        t = t.clamp(0.0, 1.0)  # [N]

        num_envs, dim, degree_plus_1 = env_coeffs.shape
        result = torch.zeros(num_envs, dim, device=self.device)
        for k in range(degree_plus_1):
            result = result * t.unsqueeze(-1) + env_coeffs[:, :, k]
        return result
