"""Web dashboard: FastAPI serves the single-page frontend, the results, and a
small API to launch new runs. `export_static` builds the same site without the
API for GitHub Pages."""
from __future__ import annotations

import os
import shutil
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import PRESETS, preset

WEB_DIR = Path(__file__).parent / "web"
RESULTS_DIR = Path(os.environ.get("SHIFTTEST_RESULTS", "results")).resolve()

app = FastAPI(title="downscaling-shift-test", version=__version__)

_lock = threading.Lock()
_job = {"state": "idle", "message": "", "progress": 0.0, "run_id": None, "error": None}


class RunRequest(BaseModel):
    preset: str = Field("default", description="quick | default | future | stress")
    warming_rate: float | None = Field(None, ge=0.0, le=0.3, description="K per year")
    seed: int | None = Field(None, ge=0, le=10_000)
    run_drywet: bool = True


def _set(**kw):
    with _lock:
        _job.update(kw)


def _worker(req: RunRequest, run_id: str) -> None:
    from .pipeline import run, update_index
    try:
        over = {}
        if req.warming_rate is not None:
            over["warming_rate"] = req.warming_rate
        if req.seed is not None:
            over["seed"] = req.seed
        cfg = preset(req.preset, **over, run_drywet=req.run_drywet)
        cfg.out_dir = str(RESULTS_DIR / run_id)
        res = run(cfg, lambda m, f: _set(message=m, progress=round(f, 3)))
        update_index(RESULTS_DIR, run_id, f"Your run ({cfg.warming_rate} K/yr, seed {cfg.seed})", res)
        _set(state="done", message="Run finished", progress=1.0)
    except Exception as e:  # report the failure in the UI rather than crash the server
        traceback.print_exc()
        _set(state="error", message="Run failed", error=f"{type(e).__name__}: {e}")


def _sweep_worker() -> None:
    from .pipeline import sweep
    try:
        sweep(str(RESULTS_DIR), progress=lambda m, f: _set(message=m, progress=round(f, 3)))
        _set(state="done", message="First results ready", progress=1.0)
    except Exception as e:
        traceback.print_exc()
        _set(state="error", message="Preparing results failed", error=f"{type(e).__name__}: {e}")


@app.on_event("startup")
def _ensure_results() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not (RESULTS_DIR / "index.json").exists() and os.environ.get("SHIFTTEST_NO_AUTORUN") != "1":
        _set(state="running", message="Preparing the first results", progress=0.0, run_id="sweep")
        threading.Thread(target=_sweep_worker, daemon=True).start()


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__, "results": (RESULTS_DIR / "index.json").exists()}


@app.get("/api/status")
def status():
    with _lock:
        return dict(_job)


@app.get("/api/presets")
def presets():
    return {k: preset(k).to_dict() for k in PRESETS}


@app.post("/api/run", status_code=202)
def start_run(req: RunRequest):
    if req.preset not in PRESETS:
        raise HTTPException(422, f"Unknown preset '{req.preset}'. Choose one of {sorted(PRESETS)}.")
    with _lock:
        if _job["state"] == "running":
            raise HTTPException(409, "A run is already in progress. Wait for it to finish, then try again.")
        _job.update(state="running", message="Starting", progress=0.0, run_id="custom", error=None)
    threading.Thread(target=_worker, args=(req, "custom"), daemon=True).start()
    return {"accepted": True, "run_id": "custom"}


app.mount("/results", StaticFiles(directory=str(RESULTS_DIR), check_dir=False), name="results")
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def export_static(results: str = "results", out: str = "site") -> Path:
    """Copy the frontend and the results into one folder that any static host can serve."""
    src, dst = Path(results), Path(out)
    if not (src / "index.json").exists():
        raise FileNotFoundError(f"No {src}/index.json - run `python -m shifttest sweep` first")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(WEB_DIR, dst)
    shutil.copytree(src, dst / "results")
    (dst / ".nojekyll").write_text("")
    return dst
