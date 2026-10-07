"""Download CHIRPS daily rainfall for West Africa, June-September, into data/fine.nc.

CHIRPS (Funk et al. 2015) is a quasi-global, land-only gridded rainfall product
from 1981. This script fetches the 0.25 degree global daily files, one per year,
crops them to the study box and JJAS, and writes one NetCDF with variable
'precip' (mm/day).

Usage:
    python scripts/download_chirps.py --start 1981 --end 2024

Check before relying on it:
  * the base URL and file pattern below (CHIRPS 2.0 at the time of writing;
    a newer CHIRPS version may supersede it - update BASE/PATTERN if so);
  * the CHIRPS terms of use on the Climate Hazards Center website.
Each yearly global 0.25 degree file is a few hundred MB; downloads are cached
in data/raw/chirps/ and skipped if already present.

Not tested inside the build sandbox (no network access to the data server).
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25/"
PATTERN = "chirps-v2.0.{year}.days_p25.nc"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1981)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--box", type=float, nargs=4, default=[4, 20, -10, 15], metavar=("S", "N", "W", "E"))
    ap.add_argument("--raw", default="data/raw/chirps")
    ap.add_argument("--out", default="data/fine.nc")
    a = ap.parse_args()

    import xarray as xr

    raw = Path(a.raw)
    raw.mkdir(parents=True, exist_ok=True)
    parts = []
    for year in range(a.start, a.end + 1):
        fn = raw / PATTERN.format(year=year)
        if not fn.exists():
            url = BASE + fn.name
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, fn)
        ds = xr.open_dataset(fn)
        ds = ds.sortby("latitude").sel(latitude=slice(a.box[0], a.box[1]), longitude=slice(a.box[2], a.box[3]))
        ds = ds.sel(time=ds.time.dt.month.isin([6, 7, 8, 9]))
        parts.append(ds[["precip"]].load())
        print(f"{year}: {ds.sizes}")
    out = xr.concat(parts, "time")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(a.out)
    print(f"wrote {a.out}: {dict(out.sizes)}")


if __name__ == "__main__":
    main()
