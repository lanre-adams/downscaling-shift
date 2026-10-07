import json

import numpy as np
import pytest

from shifttest import metrics
from shifttest.config import Config, preset
from shifttest.data import Dataset
from shifttest.models import FeatureBuilder, Interpolator, QuantileMapping, warming_anomaly
from shifttest.pipeline import drywet_splits, make_splits
from shifttest.synthetic import block_mean, generate


@pytest.fixture(scope="module")
def tiny():
    return generate(preset("quick", day_step=12, year_start=1981, year_end=2024))


def test_synthetic_shapes_and_determinism(tiny):
    T, H, W = tiny.fine_pr.shape
    assert (H, W) == (32, 32)
    assert tiny.coarse.shape == (T, 4, 4, 4)
    assert np.isfinite(tiny.fine_pr).all() and (tiny.fine_pr >= 0).all()
    again = generate(preset("quick", day_step=12))
    np.testing.assert_array_equal(tiny.fine_pr, again.fine_pr)


def test_warming_raises_heavy_rain():
    cold = generate(preset("quick", warming_rate=0.0, day_step=6, seed=3))
    hot = generate(preset("quick", warming_rate=0.15, day_step=6, seed=3))
    late = lambda d: d.fine_pr[d.years >= 2010]
    assert np.quantile(late(hot), 0.99) > 1.2 * np.quantile(late(cold), 0.99)


def test_block_mean():
    a = np.arange(16, dtype=float).reshape(4, 4)
    np.testing.assert_allclose(block_mean(a, 2), [[2.5, 4.5], [10.5, 12.5]])


def test_interpolator_reproduces_linear_field(tiny):
    interp = Interpolator(tiny)
    LAT, LON = np.meshgrid(tiny.coarse_lat, tiny.coarse_lon, indexing="ij")
    field = 2 * LAT + 0.5 * LON
    fine = interp(field)
    FL, FO = np.meshgrid(tiny.fine_lat, tiny.fine_lon, indexing="ij")
    inside = ((FL >= tiny.coarse_lat[0]) & (FL <= tiny.coarse_lat[-1])
              & (FO >= tiny.coarse_lon[0]) & (FO <= tiny.coarse_lon[-1]))
    np.testing.assert_allclose(fine[inside], (2 * FL + 0.5 * FO)[inside], rtol=1e-6)


def test_splits_are_disjoint_and_ordered(tiny):
    cfg = Config(eval_days=0)
    s = make_splits(tiny, cfg)
    assert not set(s["train"]) & set(s["in_dist"])
    assert tiny.years[s["shifted"]].min() >= cfg.late_start
    assert tiny.years[s["in_dist"]].max() <= cfg.early_end
    dw = drywet_splits(tiny)
    assert not set(dw["dry_years"]) & set(dw["wet_years"])


def test_quantile_mapping_identity_on_training(tiny):
    s = make_splits(tiny, Config(eval_days=0))
    fb = FeatureBuilder(tiny)
    qm = QuantileMapping().fit(tiny, s["train"], fb)
    pred = qm.predict(tiny, s["train"], fb)
    obs = tiny.fine_pr[s["train"]]
    # distribution (not day-by-day values) should match on training data
    assert abs(np.quantile(pred, 0.99) - np.quantile(obs, 0.99)) / np.quantile(obs, 0.99) < 0.1


def test_warming_anomaly_zero_mean_on_training(tiny):
    s = make_splits(tiny, Config(eval_days=0))
    dT = warming_anomaly(tiny, s["train"])
    assert abs(dT[s["train"]].mean()) < 1e-3


def test_metrics_perfect_prediction():
    rng = np.random.default_rng(0)
    obs = rng.gamma(0.5, 8.0, (20, 32, 32)).astype(np.float32)
    d = metrics.deterministic(obs, obs)
    assert d["mean_bias_pct"] == 0 and d["p99_bias_pct"] == 0 and d["rmse"] == 0
    ratio, _ = metrics.small_scale_ratio(obs, obs, factor=8, n_days=20)
    assert ratio == pytest.approx(1.0)
    smooth = np.repeat(np.repeat(obs[:, ::8, ::8], 8, 1), 8, 2)
    r2, _ = metrics.small_scale_ratio(smooth, obs, factor=8, n_days=20)
    assert r2 < 1.0


def test_interval_coverage():
    obs = np.linspace(0, 10, 1001)
    qs = {0.1: np.full_like(obs, 1.0), 0.9: np.full_like(obs, 9.0)}
    iv = metrics.interval(qs, obs)
    assert iv["coverage"] == pytest.approx(0.8, abs=0.01)


def test_change_signal():
    a = np.ones((10, 4, 4)); b = 1.2 * a
    c = metrics.change_signal(a, b, a, b)
    assert c["mean_captured_pct"] == pytest.approx(100.0)


def test_dataset_validation():
    with pytest.raises(ValueError):
        Dataset(coarse=np.zeros((2, 1, 2, 2)), coarse_vars=["q"], coarse_lat=np.arange(2.), coarse_lon=np.arange(2.),
                fine_pr=np.zeros((2, 4, 4)), fine_lat=np.arange(4.), fine_lon=np.arange(4.),
                years=np.array([1, 2]), doy=np.array([1, 2]))


def test_config_roundtrip():
    c = preset("future", seed=11)
    c2 = Config.from_dict(json.loads(json.dumps(c.to_dict())))
    assert c2 == c
