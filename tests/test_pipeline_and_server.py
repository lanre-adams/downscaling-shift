import json
import time
from pathlib import Path

import numpy as np
import pytest

from shifttest.config import preset
from shifttest.pipeline import run, update_index


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    cfg = preset("quick", day_step=8, run_drywet=True)
    cfg.out_dir = str(out / "quick")
    res = run(cfg)
    update_index(out, "quick", "Quick test", res)
    return out, res


def test_pipeline_outputs(quick_run):
    out, res = quick_run
    d = out / "quick"
    for f in ("results.json", "report.md", "report.html", "figures/example_day.png", "figures/change_map.png"):
        assert (d / f).exists(), f
    saved = json.loads((d / "results.json").read_text())
    assert set(saved["scores"]) == {"in_dist", "shifted"}
    for split in ("in_dist", "shifted"):
        for k in ("qm", "gbm", "gbm_phys"):
            v = saved["scores"][split][k]
            assert np.isfinite(v["rmse"]) and v["rmse"] >= 0
        assert 0 <= saved["scores"][split]["interval"]["coverage"] <= 1
    assert saved["drywet"]["scores"]
    assert len(saved["findings"]) >= 6
    assert json.loads((out / "index.json").read_text())["runs"][0]["id"] == "quick"


def test_report_mentions_status_and_disclosure(quick_run):
    out, _ = quick_run
    md = (out / "quick" / "report.md").read_text()
    assert "Data status: synthetic" in md
    assert "AI assistance" in md
    html = (out / "quick" / "report.html").read_text()
    assert "<table>" in html and 'src="figures/example_day.png"' in html


def test_pipeline_is_reproducible(tmp_path):
    a = preset("quick", day_step=12, run_drywet=False, out_dir=str(tmp_path / "a"))
    b = preset("quick", day_step=12, run_drywet=False, out_dir=str(tmp_path / "b"))
    ra, rb = run(a), run(b)
    assert ra["scores"]["shifted"]["gbm"]["rmse"] == rb["scores"]["shifted"]["gbm"]["rmse"]


def test_server(quick_run, monkeypatch):
    out, _ = quick_run
    monkeypatch.setenv("SHIFTTEST_RESULTS", str(out))
    monkeypatch.setenv("SHIFTTEST_NO_AUTORUN", "1")
    import importlib

    import shifttest.server as server
    importlib.reload(server)
    from fastapi.testclient import TestClient

    with TestClient(server.app) as c:
        assert c.get("/api/health").json()["ok"] is True
        assert "Does it survive" in c.get("/").text
        assert c.get("/results/index.json").json()["runs"][0]["id"] == "quick"
        assert c.get("/results/quick/report.html").status_code == 200
        assert c.post("/api/run", json={"preset": "nope"}).status_code == 422
        assert c.post("/api/run", json={"preset": "quick", "warming_rate": 5}).status_code == 422
        r = c.post("/api/run", json={"preset": "quick", "warming_rate": 0.1, "run_drywet": False})
        assert r.status_code == 202
        assert c.post("/api/run", json={"preset": "quick"}).status_code == 409
        for _ in range(240):
            st = c.get("/api/status").json()
            if st["state"] != "running":
                break
            time.sleep(0.5)
        assert st["state"] == "done", st
        ids = [r["id"] for r in c.get("/results/index.json").json()["runs"]]
        assert "custom" in ids


def test_export_static(quick_run, tmp_path):
    from shifttest.server import export_static
    out, _ = quick_run
    site = export_static(str(out), str(tmp_path / "site"))
    assert (site / "index.html").exists() and (site / "results" / "index.json").exists()
    assert (site / ".nojekyll").exists()


def test_real_data_loader(tmp_path):
    """Exercise the NetCDF path with tiny synthetic files in the expected layout."""
    xr = pytest.importorskip("xarray")
    import pandas as pd
    t = pd.date_range("2000-05-25", "2001-10-05", freq="D")
    clat, clon = np.arange(4, 21, 4.0), np.arange(-10, 16, 5.0)
    flat, flon = np.arange(4, 20.01, 1.0), np.arange(-10, 15.01, 1.0)
    rng = np.random.default_rng(0)
    c = xr.Dataset({v: (("time", "latitude", "longitude"), rng.random((t.size, clat.size, clon.size)))
                    for v in ("q", "w", "t", "pr")}, coords={"time": t, "latitude": clat[::-1], "longitude": clon})
    f = xr.Dataset({"precip": (("time", "lat", "lon"), rng.random((t.size, flat.size, flon.size)))},
                   coords={"time": t, "lat": flat, "lon": flon})
    c.to_netcdf(tmp_path / "coarse.nc"); f.to_netcdf(tmp_path / "fine.nc")
    from shifttest.realdata import load_real
    cfg = preset("default", source="real", coarse_path=str(tmp_path / "coarse.nc"),
                 fine_path=str(tmp_path / "fine.nc"), year_start=2000, year_end=2001, day_step=1)
    ds = load_real(cfg)
    assert ds.source == "real"
    assert ds.coarse_vars == ["q", "w", "t", "pr"]
    assert ds.n_days == 2 * 122
    assert set(np.unique(ds.years)) == {2000, 2001}
    assert ds.coarse_lat[0] < ds.coarse_lat[-1]
