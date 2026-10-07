"""End-to-end shift test: data -> splits -> models -> metrics -> figures -> report."""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__, metrics
from .config import Config
from .data import Dataset
from .models import (FeatureBuilder, GBMDownscaler, GBMQuantiles, PhysicsAwareGBM, QuantileMapping,
                     warming_anomaly)

Progress = Callable[[str, float], None]


def _noop(msg: str, frac: float) -> None:  # pragma: no cover
    pass


def load_data(cfg: Config) -> Dataset:
    if cfg.source == "synthetic":
        from .synthetic import generate
        return generate(cfg)
    if cfg.source == "real":
        from .realdata import load_real
        return load_real(cfg)
    raise ValueError(f"Unknown source '{cfg.source}' (use 'synthetic' or 'real')")


def make_splits(ds: Dataset, cfg: Config) -> dict[str, np.ndarray]:
    y = ds.years
    early = y <= cfg.early_end
    holdout = early & (((y - cfg.year_start) % cfg.holdout_every) == cfg.holdout_every - 1)
    splits = {
        "train": np.flatnonzero(early & ~holdout),
        "in_dist": np.flatnonzero(holdout),
        "shifted": np.flatnonzero(y >= cfg.late_start),
    }
    if cfg.eval_days:
        rng = np.random.default_rng(cfg.seed + 11)
        for k in ("in_dist", "shifted"):
            if splits[k].size > cfg.eval_days:
                splits[k] = np.sort(rng.choice(splits[k], cfg.eval_days, replace=False))
    for k, v in splits.items():
        if v.size == 0:
            raise ValueError(f"Split '{k}' is empty - check year_start/early_end/late_start")
    return splits


def drywet_splits(ds: Dataset, cap: int = 0, seed: int = 0) -> dict:
    years = np.unique(ds.years)
    totals = np.array([ds.fine_pr[ds.years == yr].mean() for yr in years])
    order = years[np.argsort(totals)]
    n = years.size
    dry, wet = order[: n // 2], order[-max(1, n // 4):]
    test = np.flatnonzero(np.isin(ds.years, wet))
    if cap and test.size > cap:
        test = np.sort(np.random.default_rng(seed).choice(test, cap, replace=False))
    return {"train": np.flatnonzero(np.isin(ds.years, dry)),
            "test": test,
            "dry_years": dry.tolist(), "wet_years": wet.tolist()}


def _years(ds: Dataset, idx: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(ds.years[idx]))


def run(cfg: Config, progress: Progress = _noop) -> dict:
    t0 = time.time()
    out = Path(cfg.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 1)

    progress("Loading data", 0.02)
    ds = load_data(cfg)
    splits = make_splits(ds, cfg)
    dT = warming_anomaly(ds, splits["train"])
    fb = FeatureBuilder(ds)
    factor = max(1, round(ds.fine_lat.size / ds.coarse_lat.size))

    models = [QuantileMapping(), GBMDownscaler(cfg), PhysicsAwareGBM(cfg)]
    qmodel = GBMQuantiles(cfg)
    preds: dict[str, dict[str, np.ndarray]] = {}
    for i, m in enumerate(models + [qmodel]):
        progress(f"Training: {m.label}", 0.08 + 0.12 * i)
        m.fit(ds, splits["train"], fb, rng=rng, dT=dT)
    for i, m in enumerate(models):
        progress(f"Predicting: {m.label}", 0.56 + 0.06 * i)
        preds[m.key] = {s: m.predict(ds, splits[s], fb, dT=dT) for s in ("in_dist", "shifted")}
    progress("Predicting: uncertainty interval", 0.74)
    qpred = {s: qmodel.predict(ds, splits[s], fb) for s in ("in_dist", "shifted")}

    progress("Scoring", 0.80)
    obs = {s: ds.fine_pr[splits[s]] for s in ("in_dist", "shifted")}
    scores: dict = {s: {} for s in obs}
    spectra: dict = {s: {} for s in obs}
    for s in obs:
        for m in models:
            sc = metrics.deterministic(preds[m.key][s], obs[s])
            ratio, spec = metrics.small_scale_ratio(preds[m.key][s], obs[s], factor, cfg.spectrum_days, cfg.seed)
            sc["small_scale_ratio"] = ratio
            scores[s][m.key] = sc
            spectra[s]["obs"] = spec["obs"]
            spectra[s][m.key] = spec["pred"]
            spectra[s]["k"] = spec["k"]
        scores[s]["interval"] = metrics.interval(qpred[s], obs[s])
    change = {m.key: metrics.change_signal(preds[m.key]["in_dist"], preds[m.key]["shifted"],
                                           obs["in_dist"], obs["shifted"]) for m in models}

    drywet = None
    if cfg.run_drywet:
        progress("Experiment 2: train on dry years, test on wet years", 0.84)
        dw = drywet_splits(ds, cfg.eval_days, cfg.seed + 12)
        dT2 = warming_anomaly(ds, dw["train"])
        drywet = {"dry_years": dw["dry_years"], "wet_years": dw["wet_years"], "scores": {}}
        for m in [QuantileMapping(), GBMDownscaler(cfg), PhysicsAwareGBM(cfg)]:
            m.fit(ds, dw["train"], fb, rng=rng, dT=dT2)
            p = m.predict(ds, dw["test"], fb, dT=dT2)
            drywet["scores"][m.key] = metrics.deterministic(p, ds.fine_pr[dw["test"]])

    progress("Drawing figures", 0.92)
    from .figures import draw_all
    figs = draw_all(ds, splits, preds, out / "figures")

    dTv = dT
    results = {
        "meta": {
            "tool": "downscaling-shift-test", "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runtime_s": round(time.time() - t0, 1),
            "source": ds.source, "python": platform.python_version(),
            "grid": {"fine": list(ds.fine_pr.shape[1:]), "coarse": list(ds.coarse.shape[2:]),
                     "factor": factor, "days": ds.n_days},
            "config": cfg.to_dict(),
        },
        "models": {m.key: m.label for m in models + [qmodel]},
        "splits": {
            "train": {"label": "Training (early, minus held-out years)", "years": _years(ds, splits["train"]),
                      "n_days": int(splits["train"].size), "mean_dT": float(dTv[splits["train"]].mean())},
            "in_dist": {"label": "In-distribution test (held-out early years)",
                        "years": _years(ds, splits["in_dist"]), "n_days": int(splits["in_dist"].size),
                        "mean_dT": float(dTv[splits["in_dist"]].mean())},
            "shifted": {"label": "Shifted test (later, warmer years)", "years": _years(ds, splits["shifted"]),
                        "n_days": int(splits["shifted"].size), "mean_dT": float(dTv[splits["shifted"]].mean())},
        },
        "scores": scores,
        "spectra": spectra,
        "change": change,
        "drywet": drywet,
        "qq_levels": metrics.QQ_LEVELS,
        "figures": figs,
    }
    q = ds.var("q")
    qmax = q[splits["train"]].max(axis=0)
    results["out_of_range"] = {s_: float((q[splits[s_]] > qmax).mean()) for s_ in ("in_dist", "shifted")}
    results["findings"] = findings(results)

    progress("Writing report", 0.97)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    from .report import write_report
    write_report(results, out)
    progress("Done", 1.0)
    return results


def findings(r: dict) -> list[str]:
    """Plain-language findings computed from the numbers (no hand-written claims)."""
    s, labels = r["scores"], r["models"]
    out = []
    dT = r["splits"]["shifted"]["mean_dT"] - r["splits"]["in_dist"]["mean_dT"]
    out.append(f"The shifted test period is {dT:+.2f} K warmer than the in-distribution test years "
               f"(domain-mean coarse temperature).")
    oor = r.get("out_of_range", {})
    if oor:
        out.append(f"{100 * oor['shifted']:.1f}% of shifted-period coarse moisture values lie above the "
                   f"training maximum for their grid cell ({100 * oor['in_dist']:.1f}% for held-out early years): "
                   "this is how much true extrapolation the shift demands.")
    for k in ("qm", "gbm", "gbm_phys"):
        a, b = s["in_dist"][k]["p99_bias_pct"], s["shifted"][k]["p99_bias_pct"]
        out.append(f"{labels[k]}: 99th-percentile rainfall bias moves from {a:+.1f}% in-distribution "
                   f"to {b:+.1f}% on the shifted period ({b - a:+.1f} points).")
    best = min(("qm", "gbm", "gbm_phys"), key=lambda k: abs(s["shifted"][k]["p99_bias_pct"]))
    out.append(f"Smallest extreme-rainfall bias on the shifted period: {labels[best]}.")
    for k in ("gbm", "gbm_phys"):
        c = r["change"][k]
        out.append(f"{labels[k]} reproduces {c['p99_captured_pct']:.0f}% of the observed change in "
                   f"99th-percentile rainfall ({c['pred_p99_change_pct']:+.1f}% vs {c['obs_p99_change_pct']:+.1f}%).")
    iv_a, iv_b = s["in_dist"]["interval"], s["shifted"]["interval"]
    out.append(f"The 10-90% interval covers {100 * iv_a['coverage_wet']:.0f}% of wet-day observations "
               f"in-distribution and {100 * iv_b['coverage_wet']:.0f}% on the shifted period "
               f"(nominal 80%).")
    g = s["shifted"]["gbm"]["small_scale_ratio"]
    out.append(f"The ML downscaler keeps {100 * g:.0f}% of the observed fine-scale variability "
               f"(power at scales finer than the coarse grid); below 100% means it is too smooth.")
    return out


SCENARIOS = [
    ("default", "Observed-like warming"),
    ("future", "Future-like warming"),
    ("stress", "Stress test"),
]


def update_index(root: Path, run_id: str, label: str, results: dict) -> dict:
    """Record a finished run in results/index.json so the dashboard can list it."""
    root = Path(root)
    idx_path = root / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {"runs": []}
    sp = results["splits"]
    entry = {"id": run_id, "label": label, "path": f"{run_id}/results.json",
             "warming_rate": results["meta"]["config"]["warming_rate"],
             "shift_K": round(sp["shifted"]["mean_dT"] - sp["in_dist"]["mean_dT"], 2),
             "source": results["meta"]["source"], "generated_at": results["meta"]["generated_at"]}
    index["runs"] = [r for r in index["runs"] if r["id"] != run_id] + [entry]
    order = {k: i for i, (k, _) in enumerate(SCENARIOS)}
    index["runs"].sort(key=lambda r: (order.get(r["id"], 99), r["id"]))
    index["updated"] = results["meta"]["generated_at"]
    idx_path.write_text(json.dumps(index, indent=1))
    return index


def sweep(root: str = "results", presets: list[str] | None = None, overrides: dict | None = None,
          progress: Progress = _noop) -> dict:
    """Run the standard scenarios (increasing warming) into results/<scenario>/."""
    from .config import preset as make
    names = presets or [k for k, _ in SCENARIOS]
    labels = dict(SCENARIOS)
    index = {}
    for i, name in enumerate(names):
        cfg = make(name, **(overrides or {}))
        cfg.out_dir = str(Path(root) / name)
        def p(msg, frac, i=i, name=name):
            progress(f"{labels.get(name, name)}: {msg}", (i + frac) / len(names))
        res = run(cfg, p)
        label = f"{labels.get(name, name)} ({cfg.warming_rate} K/yr)"
        index = update_index(Path(root), name, label, res)
    return index
