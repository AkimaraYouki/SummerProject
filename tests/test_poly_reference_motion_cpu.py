"""Pure-torch, CPU-only test for PolyReferenceMotion — runs on this Mac
without Isaac Sim/placo. Builds a small synthetic polynomial_coefficients.pkl
in the exact format reference_motion_generator/scripts/fit_poly.py produces
(np.polyfit output, then np.flip'd to lowest-degree-first before storage),
and checks the torch port's grid lookup + Horner evaluation against
numpy.polyval computed independently.
"""

import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from open_duck_mini_isaaclab.reference_motion.poly_reference_motion import (  # noqa: E402
    REF_FRAME_DIM,
    PolyReferenceMotion,
)

FPS = 50
PERIOD = 1.0  # seconds -> nb_steps_in_period = 50
STARTEND_DOUBLE_SUPPORT_RATIO = 0.1


def _make_pkl(tmp_path, grid_points, degree=3, seed=0):
    """grid_points: list of (dx, dy, dtheta) tuples. Each dim's true
    polynomial is randomly generated per grid cell so distinct cells are
    distinguishable in the test."""
    rng = np.random.default_rng(seed)
    data = {}
    true_polys = {}  # (dx,dy,dtheta) -> [D] list of highest-degree-first coeff arrays
    for dx, dy, dtheta in grid_points:
        key = f"{dx}_{dy}_{dtheta}"
        polys_highest_first = [rng.uniform(-1, 1, size=degree + 1) for _ in range(REF_FRAME_DIM)]
        true_polys[(dx, dy, dtheta)] = polys_highest_first
        # mimic fit_poly.py: np.polyfit gives highest-first, then np.flip -> lowest-first storage
        coefficients = {f"dim_{d}": list(np.flip(polys_highest_first[d])) for d in range(REF_FRAME_DIM)}
        data[key] = {
            "coefficients": coefficients,
            "period": PERIOD,
            "fps": FPS,
            "frame_offsets": {},
            "startend_double_support_ratio": STARTEND_DOUBLE_SUPPORT_RATIO,
        }
    pkl_path = os.path.join(tmp_path, "polynomial_coefficients.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    return pkl_path, true_polys


def test_grid_metadata(tmp_path):
    grid_points = [(-0.05, 0.0, 0.0), (0.0, 0.0, 0.0), (0.05, 0.0, 0.0)]
    pkl_path, _ = _make_pkl(tmp_path, grid_points)
    prm = PolyReferenceMotion(str(pkl_path), device=torch.device("cpu"))
    assert prm.nb_steps_in_period == PERIOD * FPS == 50
    assert prm.dxs == [-0.05, 0.0, 0.05]


def test_horner_matches_numpy_polyval(tmp_path):
    grid_points = [(0.0, 0.0, 0.0)]
    pkl_path, true_polys = _make_pkl(tmp_path, grid_points, degree=5)
    prm = PolyReferenceMotion(str(pkl_path), device=torch.device("cpu"))

    for i in [0, 10, 25, 49]:  # a few phase indices within one period
        t_expected = (i % 50) / 50
        expected = np.array([np.polyval(true_polys[(0.0, 0.0, 0.0)][d], t_expected) for d in range(REF_FRAME_DIM)])

        dx = torch.tensor([0.0])
        dy = torch.tensor([0.0])
        dtheta = torch.tensor([0.0])
        ii = torch.tensor([i])
        got = prm.get_reference_motion(dx, dy, dtheta, ii)[0].numpy()

        assert np.allclose(got, expected, atol=1e-5), f"i={i}: got {got[:3]}..., expected {expected[:3]}..."


def test_batched_matches_per_env_and_picks_nearest_cell(tmp_path):
    grid_points = [(-0.05, 0.0, 0.0), (0.0, 0.0, 0.0), (0.05, 0.0, 0.0)]
    pkl_path, true_polys = _make_pkl(tmp_path, grid_points, degree=4, seed=1)
    prm = PolyReferenceMotion(str(pkl_path), device=torch.device("cpu"))

    # 3 envs: commands closest to each of the 3 grid cells respectively.
    dx = torch.tensor([-0.049, 0.001, 0.048])
    dy = torch.tensor([0.0, 0.0, 0.0])
    dtheta = torch.tensor([0.0, 0.0, 0.0])
    ii = torch.tensor([5, 5, 5])

    batched = prm.get_reference_motion(dx, dy, dtheta, ii).numpy()

    expected_cells = [(-0.05, 0.0, 0.0), (0.0, 0.0, 0.0), (0.05, 0.0, 0.0)]
    t_expected = 5 / 50
    for env_idx, cell in enumerate(expected_cells):
        expected = np.array([np.polyval(true_polys[cell][d], t_expected) for d in range(REF_FRAME_DIM)])
        assert np.allclose(batched[env_idx], expected, atol=1e-5), f"env {env_idx} did not select grid cell {cell}"


def test_phase_wraps_at_period_boundary(tmp_path):
    grid_points = [(0.0, 0.0, 0.0)]
    pkl_path, true_polys = _make_pkl(tmp_path, grid_points, degree=3, seed=2)
    prm = PolyReferenceMotion(str(pkl_path), device=torch.device("cpu"))

    dx = torch.tensor([0.0])
    dy = torch.tensor([0.0])
    dtheta = torch.tensor([0.0])

    at_0 = prm.get_reference_motion(dx, dy, dtheta, torch.tensor([0])).numpy()
    at_period = prm.get_reference_motion(dx, dy, dtheta, torch.tensor([50])).numpy()
    assert np.allclose(at_0, at_period, atol=1e-6), "i=0 and i=nb_steps_in_period should give the same phase (t=0)"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_grid_metadata(d)
        test_horner_matches_numpy_polyval(d)
        test_batched_matches_per_env_and_picks_nearest_cell(d)
        test_phase_wraps_at_period_boundary(d)
    print("All poly_reference_motion tests passed.")
