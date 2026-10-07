"""Load real data from two NetCDF files into the common Dataset.

coarse file : daily predictors on a coarse lat/lon grid, variables named by
              cfg.var_map (defaults q, w, t, pr). 'w' is optional.
fine file   : daily rainfall (mm/day) on a finer lat/lon grid, variable
              cfg.var_map['target'] (default 'precip'). An optional 'topo'
              or 'elevation' variable (km or m) is used as a static feature.

Both files are cut to the configured box, years and June-September, aligned
on calendar date, and subsampled every cfg.day_step days. The scripts in
scripts/ produce files in exactly this layout from ERA5 and CHIRPS.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .data import Dataset

LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")
TIME_NAMES = ("time", "valid_time", "date")


def _name(ds, options, what):
    for n in options:
        if n in ds.dims or n in ds.coords:
            return n
    raise KeyError(f"Could not find a {what} coordinate (tried {options}) in {list(ds.coords)}")


def _standardise(ds, cfg: Config):
    import pandas as pd
    lat, lon, time = _name(ds, LAT_NAMES, "latitude"), _name(ds, LON_NAMES, "longitude"), _name(ds, TIME_NAMES, "time")
    ds = ds.rename({lat: "lat", lon: "lon", time: "time"})
    if float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    ds = ds.sortby("lat").sortby("lon")
    ds = ds.sel(lat=slice(cfg.lat_min, cfg.lat_max), lon=slice(cfg.lon_min, cfg.lon_max))
    t = pd.to_datetime(ds.time.values).normalize()
    ds = ds.assign_coords(time=t)
    keep = (t.month >= 6) & (t.month <= 9) & (t.year >= cfg.year_start) & (t.year <= cfg.year_end)
    return ds.isel(time=np.flatnonzero(keep))


def load_real(cfg: Config) -> Dataset:
    try:
        import xarray as xr
    except ImportError as e:  # pragma: no cover
        raise ImportError("Real-data mode needs xarray and netCDF4: pip install xarray netCDF4") from e
    if not cfg.coarse_path or not cfg.fine_path:
        raise ValueError("Real-data mode needs coarse_path and fine_path (see scripts/README.md)")
    vm = cfg.var_map
    c = _standardise(xr.open_dataset(cfg.coarse_path), cfg)
    f = _standardise(xr.open_dataset(cfg.fine_path), cfg)

    common = np.intersect1d(c.time.values, f.time.values)
    if common.size == 0:
        raise ValueError("The coarse and fine files share no dates in the selected years and JJAS season")
    common = common[:: max(1, cfg.day_step)]
    c, f = c.sel(time=common), f.sel(time=common)

    names, arrays = [], []
    for key in ("q", "w", "t", "pr"):
        src = vm.get(key, key)
        if src in c:
            names.append(key)
            arrays.append(c[src].transpose("time", "lat", "lon").values.astype(np.float32))
        elif key != "w":
            raise KeyError(f"Predictor '{src}' (for {key}) not found in {cfg.coarse_path}")
    coarse = np.stack(arrays, axis=1)
    target = f[vm.get("target", "precip")].transpose("time", "lat", "lon").values.astype(np.float32)

    # Missing values: drop days where the target is mostly missing, fill the rest.
    good = np.isfinite(target).mean(axis=(1, 2)) > 0.9
    coarse, target = coarse[good], target[good]
    common = common[good]
    target = np.nan_to_num(target, nan=0.0)
    for v in range(coarse.shape[1]):
        m = np.nanmean(coarse[:, v])
        coarse[:, v] = np.nan_to_num(coarse[:, v], nan=m)

    topo = None
    for tn in ("topo", "elevation", "orog"):
        if tn in f:
            topo = f[tn].values.astype(np.float32)
            if topo.ndim == 3:
                topo = topo[0]
            if np.nanmax(topo) > 50:  # metres -> km
                topo = topo / 1000.0
            topo = np.nan_to_num(topo)
            break

    import pandas as pd
    t = pd.to_datetime(common)
    return Dataset(coarse=coarse, coarse_vars=names, coarse_lat=c.lat.values.astype(float),
                   coarse_lon=c.lon.values.astype(float), fine_pr=target,
                   fine_lat=f.lat.values.astype(float), fine_lon=f.lon.values.astype(float),
                   years=t.year.values.astype(np.int32), doy=t.dayofyear.values.astype(np.int32),
                   topo=topo, source="real",
                   meta={"coarse_path": cfg.coarse_path, "fine_path": cfg.fine_path})
