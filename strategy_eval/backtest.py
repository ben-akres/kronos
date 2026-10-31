"""Strategies, a simple long/short portfolio simulator, and evaluation metrics.

Timing convention (identical for every strategy)
-----------------------------------------------
* A decision is made after the close of bar ``t`` using only bars ``<= t``.
* Trades fill at the **open of bar t+1** (no same-bar execution).
* Positions are held, drifting with prices, until the next decision's fill
  ``horizon`` bars later (open of ``t+1+horizon``).
* Costs: ``cost_bps`` per unit of one-way turnover, charged at each fill.
  Shorts pay no borrow fee and cash earns nothing (both favour short-heavy
  and cash-heavy strategies slightly).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def simulate(weights: pd.DataFrame, wide: dict[str, pd.DataFrame], decision_pos: list[int],
             horizon: int, cost_bps: float) -> dict:
    """Run target weights through open-to-open returns.

    ``weights`` has one row per decision date and one column per asset. A row of
    all-NaN means "do not trade, let positions drift" (used by buy & hold).
    """
    opn = wide["open"][weights.columns].values
    cal = wide["open"].index
    c = cost_bps / 1e4
    first_exec = decision_pos[0] + 1
    last_exec_end = decision_pos[-1] + 1 + horizon
    assert last_exec_end < len(cal), "need open prices through the end of the last holding period"

    V, cash = 1.0, 1.0
    h = np.zeros(opn.shape[1])
    values, dates, turnovers, costs, gross = [], [], [], [], []
    W = weights.values
    for k, t in enumerate(decision_pos):
        e = t + 1
        end = decision_pos[k + 1] + 1 if k + 1 < len(decision_pos) else last_exec_end
        w = W[k]
        if not np.isnan(w).all():
            w = np.nan_to_num(w)
            cur = h / V
            to = np.abs(w - cur).sum()
            cost = V * to * c
            V -= cost
            h = w * V
            cash = V - h.sum()
            turnovers.append(to); costs.append(cost)
        if k == 0:
            values.append(V); dates.append(cal[e])
        for s in range(e, end):
            h = h * (opn[s + 1] / opn[s])
            V = cash + h.sum()
            values.append(V); dates.append(cal[s + 1]); gross.append(np.abs(h).sum() / V)
    eq = pd.Series(values, index=pd.DatetimeIndex(dates))
    return {"equity": eq, "turnover": float(np.sum(turnovers)), "cost": float(np.sum(costs)),
            "exposure": float(np.mean(gross))}


def perf(sim: dict, periods_per_year: int) -> dict:
    eq = sim["equity"]
    r = eq.pct_change().dropna()
    years = len(r) / periods_per_year
    total = eq.iloc[-1] / eq.iloc[0] - 1
    vol = r.std() * np.sqrt(periods_per_year)
    sharpe = r.mean() / r.std() * np.sqrt(periods_per_year) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1 if years > 0 else np.nan,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sharpe_tstat": sharpe * np.sqrt(years) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": dd,
        "turnover_per_year": sim["turnover"] / years if years > 0 else np.nan,
        "cost_drag": sim["cost"],
        "avg_gross_exposure": sim["exposure"],
    }


# ---------------------------------------------------------------------------
# Strategies. Each returns a weights DataFrame (decision dates x assets) and
# only ever reads data at positions <= t.
# ---------------------------------------------------------------------------
def _frame(wide, decision_pos):
    cal = wide["close"].index
    return pd.DataFrame(np.nan, index=cal[decision_pos], columns=wide["close"].columns)


def _topk(score: pd.Series, k: int) -> pd.Series:
    w = pd.Series(0.0, index=score.index)
    w[score.nlargest(k).index] = 1.0 / k
    return w


def buy_hold(wide, decision_pos, symbols=None):
    W = _frame(wide, decision_pos)
    cols = symbols or list(W.columns)
    W.iloc[0] = 0.0
    W.iloc[0,[W.columns.get_loc(c) for c in cols]] = 1.0 / len(cols)
    return W


def equal_weight_rebalanced(wide, decision_pos):
    W = _frame(wide, decision_pos)
    W[:] = 1.0 / W.shape[1]
    return W


def random_topk(wide, decision_pos, k, rng):
    W = _frame(wide, decision_pos)
    n = W.shape[1]
    for i in range(len(W)):
        w = np.zeros(n)
        w[rng.choice(n, k, replace=False)] = 1.0 / k
        W.iloc[i] = w
    return W


def random_timing(wide, decision_pos, rng):
    """Each asset independently in (1/N) or out with probability 1/2."""
    W = _frame(wide, decision_pos)
    n = W.shape[1]
    W[:] = (rng.random((len(W), n)) < 0.5) / n
    return W


def _trailing_return(wide, t, n):
    c = wide["close"]
    return c.iloc[t] / c.iloc[t - n] - 1


def ts_momentum(wide, decision_pos, lookback=60):
    W = _frame(wide, decision_pos)
    n = W.shape[1]
    for i, t in enumerate(decision_pos):
        W.iloc[i] = (_trailing_return(wide, t, lookback) > 0).astype(float) / n
    return W


def sma_trend(wide, decision_pos, window=50):
    W = _frame(wide, decision_pos)
    n = W.shape[1]
    c = wide["close"]
    for i, t in enumerate(decision_pos):
        sma = c.iloc[t - window + 1: t + 1].mean()
        W.iloc[i] = (c.iloc[t] > sma).astype(float) / n
    return W


def xs_momentum(wide, decision_pos, k, lookback=60):
    W = _frame(wide, decision_pos)
    for i, t in enumerate(decision_pos):
        W.iloc[i] = _topk(_trailing_return(wide, t, lookback), k)
    return W


def xs_reversal(wide, decision_pos, k, lookback=5):
    W = _frame(wide, decision_pos)
    for i, t in enumerate(decision_pos):
        W.iloc[i] = _topk(-_trailing_return(wide, t, lookback), k)
    return W


def ewma_vol(wide, t, lam=0.94, window=120):
    r = np.log(wide["close"].iloc[t - window: t + 1]).diff().dropna()
    wts = lam ** np.arange(len(r))[::-1]
    return np.sqrt((wts[:, None] * r.values ** 2).sum(axis=0) / wts.sum())


def inverse_vol(wide, decision_pos, vol_by_date):
    """Long-only, fully invested, weight proportional to 1/forecast vol."""
    W = _frame(wide, decision_pos)
    for i in range(len(W)):
        iv = 1.0 / np.maximum(np.asarray(vol_by_date[i], dtype=float), 1e-6)
        W.iloc[i] = iv / iv.sum()
    return W


def from_signal_ts(feat: pd.DataFrame, decision_pos, wide, threshold=0.0):
    W = _frame(wide, decision_pos)
    n = W.shape[1]
    W[:] = (feat.values > threshold).astype(float) / n
    return W


def from_signal_topk(feat: pd.DataFrame, decision_pos, wide, k):
    W = _frame(wide, decision_pos)
    for i in range(len(W)):
        W.iloc[i] = _topk(feat.iloc[i], k)
    return W


def from_signal_long_short(feat: pd.DataFrame, decision_pos, wide, k):
    """Dollar-neutral: +0.5/k on top-k, -0.5/k on bottom-k (gross exposure 1)."""
    W = _frame(wide, decision_pos)
    for i in range(len(W)):
        s = feat.iloc[i]
        w = pd.Series(0.0, index=s.index)
        w[s.nlargest(k).index] = 0.5 / k
        w[s.nsmallest(k).index] = -0.5 / k
        W.iloc[i] = w
    return W


# ---------------------------------------------------------------------------
# Forecast-quality diagnostics (independent of portfolio construction)
# ---------------------------------------------------------------------------
def realized(wide, decision_pos, horizon):
    c, o = wide["close"], wide["open"]
    cal = c.index
    rows_cc, rows_oo, rows_vol = [], [], []
    for t in decision_pos:
        rows_cc.append(c.iloc[t + horizon] / c.iloc[t] - 1)
        rows_oo.append(o.iloc[t + 1 + horizon] / o.iloc[t + 1] - 1)
        rows_vol.append(np.log(c.iloc[t: t + horizon + 1]).diff().dropna().std())
    idx = cal[decision_pos]
    return (pd.DataFrame(rows_cc, index=idx), pd.DataFrame(rows_oo, index=idx), pd.DataFrame(rows_vol, index=idx))


def rank_ic(signal: pd.DataFrame, target: pd.DataFrame) -> dict:
    ics = []
    for d in signal.index:
        a, b = signal.loc[d], target.loc[d]
        if a.nunique() > 1 and b.nunique() > 1:
            ics.append(a.rank().corr(b.rank()))
    ics = np.array(ics)
    m, s = ics.mean(), ics.std(ddof=1)
    return {"rank_ic_mean": m, "rank_ic_std": s, "ic_tstat": m / s * np.sqrt(len(ics)) if s > 0 else np.nan,
            "ic_positive_frac": (ics > 0).mean(), "n_dates": len(ics)}


def direction_stats(signal: pd.DataFrame, target: pd.DataFrame) -> dict:
    s, y = np.sign(signal.values.ravel()), np.sign(target.values.ravel())
    m = (s != 0) & (y != 0)
    up_rate = (y[m] > 0).mean()
    return {"hit_rate": (s[m] == y[m]).mean(), "always_up_hit_rate": up_rate,
            "frac_pred_up": (s[m] > 0).mean(), "n": int(m.sum())}


def vol_stats(forecast: pd.DataFrame, realized_vol: pd.DataFrame) -> dict:
    f = forecast.values.ravel().astype(float)
    r = realized_vol.values.ravel().astype(float)
    m = (f > 0) & (r > 0) & np.isfinite(f) & np.isfinite(r)
    f, r = f[m], r[m]
    ratio = r ** 2 / f ** 2
    return {"spearman": pd.Series(f).rank().corr(pd.Series(r).rank()),
            "qlike": float(np.mean(ratio - np.log(ratio) - 1)),
            "mean_forecast_over_realized": float(np.mean(f) / np.mean(r))}
