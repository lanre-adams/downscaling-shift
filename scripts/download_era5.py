"""Download ERA5 predictors for West Africa, June-September, into data/coarse.nc.

Predictors (one snapshot per day at 12 UTC, a deliberate simplification):
    q  - specific humidity at 850 hPa (converted to g/kg)
    t  - temperature at 850 hPa (K)
    w  - vertical velocity at 700 hPa (Pa/s; negative = ascent, so we store -omega)
    pr - total precipitation, daily total (mm/day)

Needs a free Copernicus Climate Data Store account and the cdsapi package:
    pip install cdsapi
    # put your key in ~/.cdsapirc as described on the CDS website
    python scripts/download_era5.py --start 1981 --end 2024 --grid 1.0

Check before relying on it:
  * dataset and variable names on the CDS catalogue pages (the CDS API was
    redesigned in 2024; names below follow the current catalogue at the time
    of writing);
  * the precipitation source: here 'reanalysis-era5-single-levels' hourly
    total precipitation summed over the day. Requesting 24 hours per day is
    slower; the script requests one year at a time.

Not tested inside the build sandbox (no network access to the CDS).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1981)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--box", type=float, nargs=4, default=[4, 20, -10, 15], metavar=("S", "N", "W", "E"))
    ap.add_argument("--grid", type=float, default=1.0, help="output grid spacing in degrees (coarse)")
    ap.add_argument("--raw", default="data/raw/era5")
    ap.add_argument("--out", default="data/coarse.nc")
    a = ap.parse_args()

    import cdsapi
    import xarray as xr

    raw = Path(a.raw)
    raw.mkdir(parents=True, exist_ok=True)
    S, N, W, E = a.box
    area = [N, W, S, E]
    months = ["06", "07", "08", "09"]
    days = [f"{d:02d}" for d in range(1, 32)]
    c = cdsapi.Client()

    parts = []
    for year in range(a.start, a.end + 1):
        pl = raw / f"era5_pl_{year}.nc"
        sl = raw / f"era5_tp_{year}.nc"
        if not pl.exists():
            c.retrieve("reanalysis-era5-pressure-levels", {
                "product_type": ["reanalysis"],
                "variable": ["specific_humidity", "temperature", "vertical_velocity"],
                "pressure_level": ["700", "850"], "year": [str(year)], "month": months, "day": days,
                "time": ["12:00"], "area": area, "grid": [a.grid, a.grid],
                "data_format": "netcdf"}, str(pl))
        if not sl.exists():
            c.retrieve("reanalysis-era5-single-levels", {
                "product_type": ["reanalysis"], "variable": ["total_precipitation"],
                "year": [str(year)], "month": months, "day": days,
                "time": [f"{h:02d}:00" for h in range(24)], "area": area, "grid": [a.grid, a.grid],
                "data_format": "netcdf"}, str(sl))

        p = xr.open_dataset(pl)
        tname = "valid_time" if "valid_time" in p.dims else "time"
        lev = "pressure_level" if "pressure_level" in p.dims else "level"
        p = p.rename({tname: "time"})
        p["time"] = p.time.dt.floor("D")
        q = p["q"].sel({lev: 850}) * 1000.0           # kg/kg -> g/kg
        t = p["t"].sel({lev: 850})
        w = -p["w"].sel({lev: 700})                   # ascent positive
        s = xr.open_dataset(sl)
        sname = "valid_time" if "valid_time" in s.dims else "time"
        s = s.rename({sname: "time"})
        pr = (s["tp"] * 1000.0).resample(time="1D").sum()  # m -> mm per day
        pr = pr.sel(time=q.time)
        ds = xr.Dataset({"q": q.drop_vars(lev, errors="ignore"), "t": t.drop_vars(lev, errors="ignore"),
                         "w": w.drop_vars(lev, errors="ignore"), "pr": pr})
        parts.append(ds.load())
        print(f"{year}: {dict(ds.sizes)}")

    out = xr.concat(parts, "time")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(a.out)
    print(f"wrote {a.out}: {dict(out.sizes)}")


if __name__ == "__main__":
    main()
