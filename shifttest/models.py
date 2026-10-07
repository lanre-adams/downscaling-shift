"""Downscaling models compared in the shift test.

* QuantileMapping   - bilinear interpolation of coarse rainfall, then per-pixel
                      empirical quantile mapping to observed rainfall (a standard
                      statistical baseline).
* GBMDownscaler     - gradient-boosted trees (Poisson loss) mapping interpolated
                      coarse predictors + static fields to fine-scale rainfall.
                      Stands in for 'an ML downscaler'; trees cannot extrapolate
                      beyond the training range, which is the point of the test.
* GBMQuantiles      - the same features with quantile loss (10/50/90%) to give
                      an uncertainty interval; evaluated by interval coverage.
* PhysicsAwareGBM   - the GBM trained in a 'warming-normalised' space: moisture,
                      temperature and rainfall are rescaled to a reference
                      temperature with a prior scaling (default 7 %/K, Clausius-
                      Clapeyron), so the trees see a stationary problem; output is
                      scaled back with the same prior.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import Dataset

FEATURES = ["q", "w", "t", "pr", "topo", "lat", "lon", "season"]


# --------------------------------------------------------------------------
# Feature construction (coarse -> fine bilinear interpolation, vectorised)
# --------------------------------------------------------------------------
class Interpolator:
    """Precomputed bilinear weights from the coarse grid to the fine grid."""

    def __init__(self, ds: Dataset):
        self.iy0, self.iy1, self.wy = self._axis(ds.coarse_lat, ds.fine_lat)
        self.ix0, self.ix1, self.wx = self._axis(ds.coarse_lon, ds.fine_lon)

    @staticmethod
    def _axis(c: np.ndarray, f: np.ndarray):
        fc = np.clip(f, c[0], c[-1])
        i1 = np.clip(np.searchsorted(c, fc), 1, c.size - 1)
        i0 = i1 - 1
        w = (fc - c[i0]) / (c[i1] - c[i0])
        return i0, i1, w

    def __call__(self, a: np.ndarray) -> np.ndarray:
        """a: (..., h, w) -> (..., H, W)."""
        top = a[..., self.iy0, :] * (1 - self.wy)[:, None] + a[..., self.iy1, :] * self.wy[:, None]
        return top[..., self.ix0] * (1 - self.wx) + top[..., self.ix1] * self.wx


def warming_anomaly(ds: Dataset, train_idx: np.ndarray) -> np.ndarray:
    """Domain-mean coarse temperature anomaly per day vs the training climatology
    for the same day of year (K). Positive = warmer than training."""
    tmean = ds.var("t").mean(axis=(1, 2))
    clim = {}
    for d in np.unique(ds.doy[train_idx]):
        clim[d] = tmean[train_idx][ds.doy[train_idx] == d].mean()
    fallback = tmean[train_idx].mean()
    return np.array([tmean[i] - clim.get(ds.doy[i], fallback) for i in range(ds.n_days)], np.float32)


class FeatureBuilder:
    def __init__(self, ds: Dataset):
        self.ds = ds
        self.interp = Interpolator(ds)
        H, W = ds.fine_pr.shape[1:]
        LAT, LON = np.meshgrid(ds.fine_lat, ds.fine_lon, indexing="ij")
        topo = ds.topo if ds.topo is not None else np.zeros((H, W), np.float32)
        self.static = np.stack([topo, LAT, LON]).astype(np.float32)  # (3, H, W)

    def days(self, idx: np.ndarray, dT: np.ndarray | None = None, beta: float = 0.0) -> np.ndarray:
        """Feature cube for the given days: (n, H, W, F). If dT is given, moisture,
        rainfall and temperature are normalised to the reference climate."""
        ds = self.ds
        c = ds.coarse[idx].astype(np.float32).copy()          # (n, V, h, w)
        if dT is not None:
            s = np.exp(-beta * dT[idx])[:, None, None]
            for v in ("q", "pr"):
                if v in ds.coarse_vars:
                    c[:, ds.coarse_vars.index(v)] *= s
            c[:, ds.coarse_vars.index("t")] -= dT[idx][:, None, None]
        fine = self.interp(c)                                  # (n, V, H, W)
        n, _, H, W = fine.shape
        order = [ds.coarse_vars.index(v) if v in ds.coarse_vars else None for v in ("q", "w", "t", "pr")]
        cols = [fine[:, i] if i is not None else np.zeros((n, H, W), np.float32) for i in order]
        cols += [np.broadcast_to(self.static[j], (n, H, W)) for j in range(3)]
        season = np.sin(np.pi * (ds.doy[idx] - 152) / 121.0).astype(np.float32)
        cols.append(np.broadcast_to(season[:, None, None], (n, H, W)))
        return np.stack(cols, axis=-1)

    def sample(self, idx: np.ndarray, n_rows: int, rng: np.random.Generator,
               dT=None, beta=0.0, target_scale: np.ndarray | None = None):
        """Random (day, pixel) rows for training."""
        H, W = self.ds.fine_pr.shape[1:]
        per_day = max(1, int(np.ceil(n_rows / idx.size)))
        Xs, ys = [], []
        for chunk in np.array_split(idx, max(1, idx.size // 64)):
            X = self.days(chunk, dT, beta).reshape(chunk.size, H * W, -1)
            y = self.ds.fine_pr[chunk].reshape(chunk.size, H * W)
            if target_scale is not None:
                y = y * target_scale[chunk][:, None]
            pick = rng.integers(0, H * W, size=(chunk.size, min(per_day, H * W)))
            Xs.append(np.take_along_axis(X, pick[..., None], 1).reshape(-1, X.shape[-1]))
            ys.append(np.take_along_axis(y, pick, 1).ravel())
        X, y = np.concatenate(Xs), np.concatenate(ys)
        if X.shape[0] > n_rows:
            keep = rng.choice(X.shape[0], n_rows, replace=False)
            X, y = X[keep], y[keep]
        return X, y


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class QuantileMapping:
    key = "qm"
    label = "Quantile mapping (baseline)"
    levels = np.linspace(0, 1, 201)

    def fit(self, ds: Dataset, train_idx: np.ndarray, fb: FeatureBuilder, **_):
        x = fb.interp(ds.var("pr")[train_idx])               # (n, H, W)
        y = ds.fine_pr[train_idx]
        self.qx = np.quantile(x, self.levels, axis=0)          # (L, H, W)
        self.qy = np.quantile(y, self.levels, axis=0)
        return self

    def predict(self, ds: Dataset, idx: np.ndarray, fb: FeatureBuilder, **_) -> np.ndarray:
        x = fb.interp(ds.var("pr")[idx])
        out = np.empty_like(x, dtype=np.float32)
        H, W = x.shape[1:]
        for i in range(H):
            for j in range(W):
                qx, qy = self.qx[:, i, j], self.qy[:, i, j]
                qx_u, k = np.unique(qx, return_index=True)
                col = np.interp(x[:, i, j], qx_u, qy[k])
                above = x[:, i, j] > qx_u[-1]
                col[above] = qy[-1] + (x[above, i, j] - qx_u[-1])   # additive extrapolation
                out[:, i, j] = col
        return np.clip(out, 0, None)


class GBMDownscaler:
    key = "gbm"
    label = "ML downscaler (gradient boosting)"

    def __init__(self, cfg):
        self.cfg = cfg

    def _model(self, **kw):
        return HistGradientBoostingRegressor(max_iter=self.cfg.max_iter, learning_rate=self.cfg.learning_rate,
                                             max_leaf_nodes=31, min_samples_leaf=40,
                                             random_state=self.cfg.seed, **kw)

    def fit(self, ds, train_idx, fb, rng, **_):
        X, y = fb.sample(train_idx, self.cfg.train_rows, rng)
        self.m = self._model(loss="poisson").fit(X, y)
        return self

    def _predict_chunks(self, fn, ds, idx, fb, dT=None, beta=0.0):
        H, W = ds.fine_pr.shape[1:]
        out = np.empty((idx.size, H, W), np.float32)
        pos = 0
        for chunk in np.array_split(idx, max(1, idx.size // 48)):
            X = fb.days(chunk, dT, beta).reshape(-1, len(FEATURES))
            out[pos:pos + chunk.size] = fn(X).reshape(chunk.size, H, W)
            pos += chunk.size
        return out

    def predict(self, ds, idx, fb, **_):
        return np.clip(self._predict_chunks(self.m.predict, ds, idx, fb), 0, None)


class GBMQuantiles(GBMDownscaler):
    key = "gbm_q"
    label = "ML downscaler, 10-90% interval"

    def fit(self, ds, train_idx, fb, rng, **_):
        X, y = fb.sample(train_idx, self.cfg.train_rows, rng)
        self.ms = {q: self._model(loss="quantile", quantile=q).fit(X, y) for q in self.cfg.quantiles}
        return self

    def predict(self, ds, idx, fb, **_):
        return {q: np.clip(self._predict_chunks(m.predict, ds, idx, fb), 0, None) for q, m in self.ms.items()}


class PhysicsAwareGBM(GBMDownscaler):
    key = "gbm_phys"
    label = "ML downscaler + warming prior"

    def fit(self, ds, train_idx, fb, rng, dT=None, **_):
        self.beta = self.cfg.physics_beta
        scale = np.exp(-self.beta * dT)
        X, y = fb.sample(train_idx, self.cfg.train_rows, rng, dT=dT, beta=self.beta, target_scale=scale)
        self.m = self._model(loss="poisson").fit(X, y)
        return self

    def predict(self, ds, idx, fb, dT=None, **_):
        p = self._predict_chunks(self.m.predict, ds, idx, fb, dT=dT, beta=self.beta)
        return np.clip(p * np.exp(self.beta * dT[idx])[:, None, None], 0, None)
