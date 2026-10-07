"""Static map figures (PNG). Charts are drawn in the browser from results.json."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import Dataset  # noqa: E402

CITIES = {"Lagos": (6.52, 3.38), "Uyo": (5.04, 7.91), "Kano": (12.00, 8.52)}
INK, MUTED = "#1C2422", "#5B6663"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#B9C2BE",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 10, "axes.titlecolor": INK})


def _extent(ds: Dataset):
    return [ds.fine_lon[0], ds.fine_lon[-1], ds.fine_lat[0], ds.fine_lat[-1]]


def _cities(ax, ds: Dataset):
    for name, (la, lo) in CITIES.items():
        if ds.fine_lat[0] <= la <= ds.fine_lat[-1] and ds.fine_lon[0] <= lo <= ds.fine_lon[-1]:
            ax.plot(lo, la, "o", ms=3.5, mfc="white", mec=INK, mew=0.9)
            ax.text(lo + 0.35, la + 0.25, name, fontsize=7.5, color=INK)


def example_day(ds: Dataset, splits, preds, path: Path) -> dict:
    idx = splits["shifted"]
    obs = ds.fine_pr[idx]
    j = int(np.argmax(np.quantile(obs.reshape(obs.shape[0], -1), 0.995, axis=1)))
    day = int(idx[j])
    coarse_pr = ds.var("pr")[day]
    f = ds.fine_pr.shape[1] // coarse_pr.shape[0]
    coarse_up = np.kron(coarse_pr, np.ones((f, f)))[: ds.fine_pr.shape[1], : ds.fine_pr.shape[2]]
    panels = [("Coarse input", coarse_up), ("Observed (target)", obs[j]),
              ("Quantile mapping", preds["qm"]["shifted"][j]), ("ML downscaler", preds["gbm"]["shifted"][j]),
              ("ML + warming prior", preds["gbm_phys"]["shifted"][j])]
    vmax = float(np.quantile(obs[j], 0.995)) or 1.0
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.2), constrained_layout=True)
    for ax, (title, a) in zip(axes, panels):
        im = ax.imshow(a, origin="lower", extent=_extent(ds), cmap="YlGnBu", vmin=0, vmax=vmax,
                       interpolation="nearest", aspect="auto")
        ax.set_title(title)
        _cities(ax, ds)
        ax.set_xticks([-10, 0, 10]); ax.set_yticks([5, 10, 15, 20])
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
    cb.set_label("mm/day")
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return {"file": f"figures/{path.name}", "title": "One heavy-rain day from the shifted period",
            "caption": (f"Day {int(ds.doy[day])} of {int(ds.years[day])}: the shifted-period day with the "
                        "heaviest local rainfall. Each panel shares one colour scale. The coarse input is "
                        "what a global model provides; it looks faint because coarse cells spread rain thinly. "
                        "The other panels are attempts to recover the target from it.")}


def change_map(ds: Dataset, splits, preds, path: Path) -> dict:
    o_e, o_l = ds.fine_pr[splits["in_dist"]].mean(0), ds.fine_pr[splits["shifted"]].mean(0)
    def pct(a, b):
        return 100 * (b - a) / np.maximum(a, 0.5)
    panels = [("Observed change", pct(o_e, o_l))]
    for k, t in (("gbm", "ML downscaler"), ("gbm_phys", "ML + warming prior")):
        panels.append((t, pct(preds[k]["in_dist"].mean(0), preds[k]["shifted"].mean(0))))
    lim = float(np.nanquantile(np.abs(panels[0][1]), 0.98)) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), constrained_layout=True)
    for ax, (title, a) in zip(axes, panels):
        im = ax.imshow(a, origin="lower", extent=_extent(ds), cmap="BrBG", vmin=-lim, vmax=lim,
                       interpolation="nearest", aspect="auto")
        ax.set_title(title)
        _cities(ax, ds)
        ax.set_xticks([-10, 0, 10]); ax.set_yticks([5, 10, 15, 20])
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
    cb.set_label("% change in mean rainfall")
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return {"file": f"figures/{path.name}", "title": "Mean-rainfall change, early to late period",
            "caption": ("Change in mean JJAS rainfall from the held-out early years to the shifted period. "
                        "Brown is drier, green is wetter. A downscaler suitable for climate projection "
                        "must reproduce this pattern, not just today's climate.")}


def draw_all(ds: Dataset, splits, preds, out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    return [example_day(ds, splits, preds, out / "example_day.png"),
            change_map(ds, splits, preds, out / "change_map.png")]
