"""Evaluation metrics. All take arrays shaped (days, H, W) in mm/day."""
from __future__ import annotations

import numpy as np

QQ_LEVELS = [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
WET = 1.0  # mm/day wet-day threshold


def _pct(a: float, b: float) -> float:
    return float(100.0 * (a - b) / b) if b else float("nan")


def radial_spectrum(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean radially averaged power spectrum over days. Returns (k, power) with
    k in cycles per pixel (0..0.5]."""
    n, H, W = fields.shape
    win = np.outer(np.hanning(H), np.hanning(W))
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)
    bins = np.linspace(0, 0.5, min(H, W) // 2 + 1)
    which = np.digitize(kr.ravel(), bins) - 1
    power = np.zeros(bins.size - 1)
    for f in fields:
        a = (f - f.mean()) * win
        p = np.abs(np.fft.fft2(a)) ** 2
        power += np.bincount(which, weights=p.ravel(), minlength=bins.size)[: bins.size - 1]
    counts = np.bincount(which, minlength=bins.size)[: bins.size - 1]
    power = power / np.maximum(counts, 1) / n
    k = 0.5 * (bins[1:] + bins[:-1])
    return k[1:], power[1:]  # drop the k~0 bin


def small_scale_ratio(pred: np.ndarray, obs: np.ndarray, factor: int, n_days: int = 150,
                      seed: int = 0) -> tuple[float, dict]:
    """Ratio of predicted to observed power at scales finer than the coarse grid
    (wavelength < 2 coarse cells). 1 = right amount of fine-scale structure,
    < 1 = too smooth, > 1 = too noisy."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(obs.shape[0], size=min(n_days, obs.shape[0]), replace=False)
    k, po = radial_spectrum(obs[pick])
    _, pp = radial_spectrum(pred[pick])
    fine = k > 1.0 / (2 * factor)
    ratio = float(np.exp(np.mean(np.log((pp[fine] + 1e-12) / (po[fine] + 1e-12)))))
    return ratio, {"k": k.tolist(), "obs": po.tolist(), "pred": pp.tolist()}


def deterministic(pred: np.ndarray, obs: np.ndarray) -> dict:
    o, p = obs.ravel(), pred.ravel()
    q_o = np.quantile(o, QQ_LEVELS)
    q_p = np.quantile(p, QQ_LEVELS)
    i99 = QQ_LEVELS.index(0.99)
    i999 = QQ_LEVELS.index(0.999)
    return {
        "mean_obs": float(o.mean()),
        "mean_pred": float(p.mean()),
        "mean_bias_pct": _pct(p.mean(), o.mean()),
        "rmse": float(np.sqrt(np.mean((p - o) ** 2))),
        "corr": float(np.corrcoef(p, o)[0, 1]) if p.std() > 0 else 0.0,
        "p99_obs": float(q_o[i99]),
        "p99_pred": float(q_p[i99]),
        "p99_bias_pct": _pct(q_p[i99], q_o[i99]),
        "p999_bias_pct": _pct(q_p[i999], q_o[i999]),
        "wet_freq_obs": float((o > WET).mean()),
        "wet_freq_pred": float((p > WET).mean()),
        "qq_obs": q_o.tolist(),
        "qq_pred": q_p.tolist(),
    }


def interval(qs: dict, obs: np.ndarray) -> dict:
    lo, hi = min(qs), max(qs)
    inside = (obs >= qs[lo]) & (obs <= qs[hi])
    nominal = hi - lo
    wet = obs > WET
    loss = 0.0
    for q, p in qs.items():
        d = obs - p
        loss += float(np.mean(np.maximum(q * d, (q - 1) * d)))
    return {
        "nominal": float(nominal),
        "coverage": float(inside.mean()),
        "coverage_wet": float(inside[wet].mean()) if wet.any() else float("nan"),
        "above_upper_wet": float((obs[wet] > qs[hi][wet]).mean()) if wet.any() else float("nan"),
        "pinball": loss / len(qs),
    }


def change_signal(pred_early: np.ndarray, pred_late: np.ndarray,
                  obs_early: np.ndarray, obs_late: np.ndarray) -> dict:
    """How much of the observed early->late change does the model reproduce?"""
    def ch(a, b, fn):
        return _pct(fn(b), fn(a))
    p99 = lambda x: float(np.quantile(x, 0.99))
    mean = lambda x: float(np.mean(x))
    out = {
        "obs_mean_change_pct": ch(obs_early, obs_late, mean),
        "pred_mean_change_pct": ch(pred_early, pred_late, mean),
        "obs_p99_change_pct": ch(obs_early, obs_late, p99),
        "pred_p99_change_pct": ch(pred_early, pred_late, p99),
    }
    for k in ("mean", "p99"):
        o, p = out[f"obs_{k}_change_pct"], out[f"pred_{k}_change_pct"]
        out[f"{k}_captured_pct"] = float(100 * p / o) if abs(o) > 0.5 else float("nan")
    return out
