# Real data: ERA5 predictors and CHIRPS rainfall

The pipeline runs on any pair of NetCDF files in this layout:

| File | Grid | Variables | Units |
|---|---|---|---|
| `data/coarse.nc` | coarse (e.g. 1°) | `q`, `w` (optional), `t`, `pr` on `(time, lat, lon)` | g/kg, ascent, K, mm/day |
| `data/fine.nc` | fine (e.g. 0.25°) | `precip` on `(time, lat, lon)`, optional `topo` | mm/day, km or m |

Coordinate names `latitude`/`longitude`/`valid_time` are also recognised, and
longitudes in 0–360 are converted. Only June–September days present in both
files are used.

## 1. Rainfall target (CHIRPS)

```bash
python scripts/download_chirps.py --start 1981 --end 2024
```

Downloads one global 0.25° file per year (cached in `data/raw/chirps/`), crops
to 4–20°N, 10°W–15°E and JJAS, and writes `data/fine.nc`.

## 2. Predictors (ERA5)

```bash
pip install cdsapi          # and set up ~/.cdsapirc with your CDS key
python scripts/download_era5.py --start 1981 --end 2024 --grid 1.0
```

Writes `data/coarse.nc` with 850 hPa humidity and temperature, 700 hPa ascent
(at 12 UTC) and daily total precipitation, all on a 1° grid.

## 3. Run

```bash
python -m shifttest run --coarse data/coarse.nc --fine data/fine.nc --out results/real
# or with Docker:
docker compose --profile real run --rm realdata
```

The real run appears in the dashboard as an extra scenario.

## Status of these scripts

The NetCDF loader is covered by the test suite. The two download scripts were
written against the public CHIRPS and Copernicus CDS interfaces but have **not
been run** in the environment where this repository was built (no network
access to those servers). Check dataset names, variable names and URLs on the
providers' websites before a long download, and start with two or three years.
