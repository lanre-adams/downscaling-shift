"""A synthetic 'West African monsoon' world with a known warming signal.

This is a teaching and testing world, not a climate model. It exists so the
whole shift-test harness can run anywhere, offline, in seconds, with a ground
truth whose response to warming is known by construction.

What it encodes (all deliberately simple):
  * a monsoon rain belt whose latitude migrates through June-September;
  * westward-travelling wave disturbances (loosely, African easterly waves);
  * fine-scale convective storm cells, more likely over high ground and where
    moisture and ascent coincide - the part a coarse model cannot resolve;
  * year-to-year monsoon strength (wet and dry years);
  * a warming trend: moisture scales with `cc_scaling` per K, and storm
    intensity with an *extra* `extreme_scaling` per K (so extremes grow faster
    than moisture - a crude stand-in for observed 'super-Clausius-Clapeyron'
    behaviour of convective extremes).

Coarse predictors are block-averages of fine fields, plus a noisy, smoothed
coarse rainfall, mimicking what a global model 'sees'.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import Config
from .data import Dataset

# (lat, lon, height km, width deg): rough stand-ins for real highlands
HIGHLANDS = [
    (9.9, 8.9, 1.2, 0.9),     # Jos Plateau
    (6.0, 10.5, 1.6, 1.0),    # Cameroon / Adamawa highlands
    (10.5, -10.0, 1.0, 1.2),  # Fouta Djallon / Guinea highlands
    (18.0, 8.5, 1.0, 1.4),    # Air Mountains
]

JJAS_START, JJAS_END = 152, 273  # 1 Jun .. 30 Sep (non-leap day-of-year)


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    f = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    s = f.std()
    return f / s if s > 0 else f


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Average the last two axes over non-overlapping f x f blocks."""
    *lead, H, W = a.shape
    return a.reshape(*lead, H // f, f, W // f, f).mean(axis=(-3, -1))


def generate(cfg: Config) -> Dataset:
    if cfg.fine_n % cfg.factor:
        raise ValueError("fine_n must be divisible by factor")
    rng = np.random.default_rng(cfg.seed)
    H = W = cfg.fine_n
    f = cfg.factor
    px_deg = (cfg.lon_max - cfg.lon_min) / W

    fine_lat = np.linspace(cfg.lat_min, cfg.lat_max, H)
    fine_lon = np.linspace(cfg.lon_min, cfg.lon_max, W)
    LAT, LON = np.meshgrid(fine_lat, fine_lon, indexing="ij")
    coarse_lat = fine_lat.reshape(-1, f).mean(1)
    coarse_lon = fine_lon.reshape(-1, f).mean(1)

    topo = np.zeros((H, W))
    for la, lo, hgt, wd in HIGHLANDS:
        topo += hgt * np.exp(-(((LAT - la) / wd) ** 2 + ((LON - lo) / wd) ** 2))

    years = np.arange(cfg.year_start, cfg.year_end + 1)
    doys = np.arange(JJAS_START, JJAS_END + 1, cfg.day_step)
    n_t = years.size * doys.size

    fine_pr = np.empty((n_t, H, W), np.float32)
    coarse = np.empty((n_t, 4, H // f, W // f), np.float32)
    yr_out = np.empty(n_t, np.int32)
    doy_out = np.empty(n_t, np.int32)

    monsoon = 1.0 + cfg.monsoon_variability * rng.standard_normal(years.size)
    dT_year = cfg.warming_rate * (years - years[0]) + 0.25 * rng.standard_normal(years.size)

    yy, xx = np.mgrid[0:H, 0:W]
    k = 0
    for iy, year in enumerate(years):
        m, dT = monsoon[iy], dT_year[iy]
        moist_scale = np.exp(cfg.cc_scaling * dT)
        storm_scale = np.exp(cfg.extreme_scaling * dT)
        phase = rng.uniform(0, 2 * np.pi)
        for doy in doys:
            t_season = (doy - JJAS_START) / (JJAS_END - JJAS_START)
            itcz = 7.5 + 5.0 * np.sin(np.pi * t_season)          # rain-belt latitude
            # moisture (g/kg): humid south, dry north, belt follows the ITCZ
            q = (3.0 + 15.0 * m * moist_scale / (1 + np.exp((LAT - (itcz + 2.5)) / 1.4))
                 + 0.8 * _smooth_noise(rng, (H, W), 6))
            q = np.clip(q, 0.5, None)
            # temperature (K): Sahel warmer, warming trend, weather noise
            t = 298.5 + 0.25 * (LAT - 4) + dT + 0.4 * _smooth_noise(rng, (H, W), 8)
            # ascent: westward-moving waves near the belt + large + small-scale noise
            wave = np.sin(2 * np.pi * (LON + 7.0 * doy) / 30.0 + phase)
            env = np.exp(-((LAT - itcz) / 4.0) ** 2)
            w_large = 0.7 * wave * env + 0.5 * _smooth_noise(rng, (H, W), 5)
            w_small = 0.7 * _smooth_noise(rng, (H, W), 1.2)
            w = w_large + w_small

            trigger = np.clip(w + 0.3, 0, None) * (q / 18.0) * (1.0 + 0.9 * topo)
            rain = 3.0 * trigger ** 1.5                                  # light, widespread rain

            n_cells = rng.poisson(18.0 * trigger.mean() * (H * W) / 4096)
            if n_cells:
                p = trigger.ravel() ** 2
                p = p / p.sum()
                pick = rng.choice(H * W, size=n_cells, p=p)
                cy, cx = np.divmod(pick, W)
                radius = rng.uniform(1.0, 3.2, n_cells) * (W / 64)  # fine pixels, scales with grid
                peak = (rng.gamma(2.0, 20.0, n_cells) * (q.ravel()[pick] / 15.0) * storm_scale)
                d2 = (yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2
                rain = rain + (peak[:, None, None] * np.exp(-d2 / (2 * radius[:, None, None] ** 2))).sum(0)

            rain = rain * rng.lognormal(0.0, 0.25, (H, W))
            rain[rain < 0.2] = 0.0

            pr_c = block_mean(rain, f)
            pr_c = gaussian_filter(pr_c, 0.8, mode="nearest") * rng.lognormal(0.0, 0.2, pr_c.shape)

            fine_pr[k] = rain
            coarse[k, 0] = block_mean(q, f)
            coarse[k, 1] = block_mean(w_large, f)  # coarse model resolves large-scale ascent only
            coarse[k, 2] = block_mean(t, f)
            coarse[k, 3] = pr_c
            yr_out[k], doy_out[k] = year, doy
            k += 1

    return Dataset(
        coarse=coarse, coarse_vars=["q", "w", "t", "pr"],
        coarse_lat=coarse_lat, coarse_lon=coarse_lon,
        fine_pr=fine_pr, fine_lat=fine_lat, fine_lon=fine_lon,
        years=yr_out, doy=doy_out, topo=topo.astype(np.float32), source="synthetic",
        meta={"px_deg": float(px_deg), "monsoon_index": monsoon.round(3).tolist(),
              "dT_year": dT_year.round(3).tolist(),
              "units": {"q": "g/kg", "w": "arbitrary", "t": "K", "pr": "mm/day"}},
    )
