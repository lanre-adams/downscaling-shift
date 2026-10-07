"""Experiment configuration.

Every run is fully described by one Config object. It is written into
results.json so a reader can reproduce any report exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class Config:
    # --- data source -------------------------------------------------------
    source: str = "synthetic"          # "synthetic" or "real"
    coarse_path: str = ""              # real data: predictors NetCDF (coarse grid)
    fine_path: str = ""                # real data: target rainfall NetCDF (fine grid)
    var_map: dict = field(default_factory=lambda: {
        "q": "q", "w": "w", "t": "t", "pr": "pr", "target": "precip"})

    # --- domain (West Africa, JJAS monsoon) --------------------------------
    lat_min: float = 4.0
    lat_max: float = 20.0
    lon_min: float = -10.0
    lon_max: float = 15.0
    year_start: int = 1981
    year_end: int = 2024
    day_step: int = 3                  # use every n-th JJAS day (speed)

    # --- synthetic world ---------------------------------------------------
    seed: int = 7
    fine_n: int = 64                   # fine grid is fine_n x fine_n
    factor: int = 8                    # coarse cell = factor x factor fine cells
    warming_rate: float = 0.03         # K per year (observed-like ~0.03)
    cc_scaling: float = 0.07           # moisture scaling with warming (Clausius-Clapeyron, /K)
    extreme_scaling: float = 0.03      # extra storm-intensity scaling (/K) -> "super-CC"
    monsoon_variability: float = 0.15  # interannual monsoon strength s.d.

    # --- experimental design -----------------------------------------------
    early_end: int = 2005              # early block: year_start..early_end
    late_start: int = 2010             # shifted block: late_start..year_end
    holdout_every: int = 5             # every 5th early year held out (in-distribution test)
    eval_days: int = 300               # cap on test days per set (speed); 0 = all
    run_drywet: bool = True            # second experiment: train on dry years, test on wet years

    # --- models ------------------------------------------------------------
    train_rows: int = 200_000          # pixel-day rows sampled for ML training
    max_iter: int = 150
    learning_rate: float = 0.08
    physics_beta: float = 0.07         # physics-aware model's prior scaling (/K)
    quantiles: tuple = (0.1, 0.5, 0.9)

    # --- outputs -----------------------------------------------------------
    out_dir: str = "results"
    spectrum_days: int = 150

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["quantiles"] = list(self.quantiles)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        names = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in names}
        if "quantiles" in clean:
            clean["quantiles"] = tuple(clean["quantiles"])
        return cls(**clean)


PRESETS: dict[str, dict[str, Any]] = {
    # Fast settings for tests and CI.
    "quick": dict(fine_n=32, factor=8, year_start=1981, year_end=2024, day_step=6,
                  train_rows=40_000, max_iter=60, spectrum_days=40),
    # Default dashboard run (~1-2 min on 2 CPUs).
    "default": dict(),
    # Stronger, future-like warming to stress the models.
    "future": dict(warming_rate=0.08),
    # Large shift (~6 K over the record): well outside the training range.
    "stress": dict(warming_rate=0.15),
}


def preset(name: str, **overrides: Any) -> Config:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Choose from {sorted(PRESETS)}")
    return Config(**{**PRESETS[name], **overrides})
