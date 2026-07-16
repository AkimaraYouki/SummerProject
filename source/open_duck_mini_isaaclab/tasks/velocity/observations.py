"""Noise injection + delay ring buffers, ported from Playground's
joystick.py::_get_obs (uniform per-channel noise, NOT Isaac Lab's generic
Gaussian observation-noise wrapper — Playground's noise is uniform and
scaled per-channel by a global `level` multiplier, which a generic Gaussian
noise cfg would not reproduce).

See docs/decisions.md "Hard Piece 2" for the delay ring buffer design.
"""

from __future__ import annotations

import torch


def apply_uniform_noise(x: torch.Tensor, level: float, scale: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """x + U(-1,1) * level * scale, matching Playground's noise formula exactly."""
    return x + (2.0 * torch.rand_like(x) - 1.0) * level * scale


class DelayBuffer:
    """Batched, per-env random-delay ring buffer.

    Mirrors joystick.py's `action_history`/`imu_history` pattern: each step,
    the newest value is pushed to index 0 and older values shift back; a
    fresh random per-env delay index is drawn every step and used to read a
    (possibly stale) value back out. `min_delay`/`max_delay` are in units of
    env steps, with `max_delay` EXCLUSIVE (matching JAX's
    `jax.random.randint(minval, maxval)`, which torch.randint also treats as
    exclusive on `high`) — preserve this exactly, an off-by-one here changes
    the effective delay distribution.
    """

    def __init__(self, num_envs: int, dim: int, max_delay: int, device: torch.device):
        # max_delay==0 is a valid "delay disabled" config — keep a
        # length-1 buffer so indexing always works, but push()/sample() will
        # always return the current-step value in that case.
        self._capacity = max(max_delay, 1)
        self.buffer = torch.zeros(num_envs, self._capacity, dim, device=device)
        self.device = device

    def push(self, value: torch.Tensor) -> None:
        self.buffer = torch.roll(self.buffer, shifts=1, dims=1)
        self.buffer[:, 0, :] = value

    def sample(self, min_delay: int, max_delay: int) -> torch.Tensor:
        num_envs = self.buffer.shape[0]
        if max_delay > min_delay:
            delay_idx = torch.randint(min_delay, max_delay, (num_envs,), device=self.device)
        else:
            delay_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        return self.buffer[torch.arange(num_envs, device=self.device), delay_idx]

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        self.buffer[env_ids] = 0.0
