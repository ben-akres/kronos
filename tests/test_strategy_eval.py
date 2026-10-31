"""Plumbing tests for strategy_eval (no network, random-init tiny model)."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from strategy_eval import backtest as bt
from strategy_eval.data import align, synthetic_universe
from strategy_eval.forecast import forecast_features, rolling_forecasts
from strategy_eval.run import tiny_random_kronos

H, L = 5, 64


@pytest.fixture(scope="module")
def wide():
    return align(synthetic_universe(n_assets=4, n_bars=200))


def test_simulate_buy_hold_matches_price_ratio(wide):
    pos = list(range(100, 180, H))
    sym = wide["close"].columns[0]
    sim = bt.simulate(bt.buy_hold(wide, pos, [sym]), wide, pos, H, cost_bps=0.0)
    o = wide["open"][sym]
    expected = o.iloc[pos[-1] + 1 + H] / o.iloc[pos[0] + 1]
    assert sim["equity"].iloc[-1] == pytest.approx(expected, rel=1e-10)


def test_costs_reduce_returns(wide):
    pos = list(range(100, 180, H))
    W = bt.xs_reversal(wide, pos, 2, H)
    free = bt.simulate(W, wide, pos, H, 0.0)["equity"].iloc[-1]
    paid = bt.simulate(W, wide, pos, H, 50.0)["equity"].iloc[-1]
    assert paid < free


def test_heuristics_do_not_look_ahead(wide):
    pos = list(range(100, 180, H))
    k = 3
    poisoned = {f: w.copy() for f, w in wide.items()}
    for f in ("open", "high", "low", "close"):
        poisoned[f].iloc[pos[k] + 1:] *= 3.0  # change everything after decision k
    for fn in (lambda w: bt.ts_momentum(w, pos), lambda w: bt.sma_trend(w, pos),
               lambda w: bt.xs_momentum(w, pos, 2), lambda w: bt.xs_reversal(w, pos, 2, H)):
        pd.testing.assert_frame_equal(fn(wide).iloc[: k + 1], fn(poisoned).iloc[: k + 1])


def test_kronos_forecasts_do_not_look_ahead(wide):
    tok, mdl, ctx = tiny_random_kronos()
    pos = [100, 105, 110]
    a = rolling_forecasts(wide, pos, tok, mdl, ctx, L, H, n_samples=3, seed=1)
    poisoned = {f: w.copy() for f, w in wide.items()}
    for f in poisoned:
        poisoned[f].iloc[pos[1] + 1:] *= 2.0
    b = rolling_forecasts(poisoned, pos, tok, mdl, ctx, L, H, n_samples=3, seed=1)
    np.testing.assert_allclose(a["close_paths"][:2], b["close_paths"][:2], rtol=1e-5)
    assert a["close_paths"].shape == (3, 4, 3, H)
    feats = forecast_features(a, wide)
    assert set(feats) == {"pred_ret", "prob_up", "pred_disp", "pred_vol"}
    assert feats["prob_up"].values.min() >= 0 and feats["prob_up"].values.max() <= 1
