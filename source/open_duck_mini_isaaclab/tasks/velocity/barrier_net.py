"""장벽함수 신경망 구조. 학습(`scripts/diag/fit_barrier.py`)과 추론이 공유한다."""

import torch.nn as nn


class Barrier(nn.Module):
    """5 관절 -> 몸통-정강이 간격 [mm]."""

    def __init__(self, w: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(),
            nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)
