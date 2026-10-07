"""Common in-memory dataset used by both the synthetic and real-data paths."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dataset:
    coarse: np.ndarray            # (T, V, h, w) float32 coarse predictors
    coarse_vars: list[str]        # names of V predictors, must include "t" and "pr"
    coarse_lat: np.ndarray        # (h,) ascending
    coarse_lon: np.ndarray        # (w,) ascending
    fine_pr: np.ndarray           # (T, H, W) float32 target rainfall, mm/day
    fine_lat: np.ndarray          # (H,) ascending
    fine_lon: np.ndarray          # (W,) ascending
    years: np.ndarray             # (T,) int
    doy: np.ndarray               # (T,) int day of year
    topo: np.ndarray | None = None  # (H, W) km, optional static field
    source: str = "synthetic"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        T = self.fine_pr.shape[0]
        if self.coarse.shape[0] != T or self.years.shape[0] != T or self.doy.shape[0] != T:
            raise ValueError("coarse, fine_pr, years and doy must share the time axis")
        if self.coarse.shape[1] != len(self.coarse_vars):
            raise ValueError("coarse_vars does not match the predictor axis")
        for v in ("t", "pr"):
            if v not in self.coarse_vars:
                raise ValueError(f"coarse predictors must include '{v}'")
        if self.coarse.shape[2:] != (self.coarse_lat.size, self.coarse_lon.size):
            raise ValueError("coarse grid does not match coarse_lat/coarse_lon")
        if self.fine_pr.shape[1:] != (self.fine_lat.size, self.fine_lon.size):
            raise ValueError("fine grid does not match fine_lat/fine_lon")

    @property
    def n_days(self) -> int:
        return int(self.fine_pr.shape[0])

    def var(self, name: str) -> np.ndarray:
        return self.coarse[:, self.coarse_vars.index(name)]

    def subset(self, idx: np.ndarray) -> "Dataset":
        return Dataset(self.coarse[idx], list(self.coarse_vars), self.coarse_lat, self.coarse_lon,
                       self.fine_pr[idx], self.fine_lat, self.fine_lon, self.years[idx],
                       self.doy[idx], self.topo, self.source, dict(self.meta))
