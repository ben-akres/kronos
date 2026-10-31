"""Out-of-sample Kronos strategy backtest vs random, buy & hold and heuristics.

Example (from the repo root):

    python -m strategy_eval.run --universe us_etf            # last 12 months, Kronos-small
    python -m strategy_eval.run --universe crypto --horizon 7 --cost-bps 10
    python -m strategy_eval.run --synthetic                  # plumbing smoke test, no network

Writes ``report.md``, ``metrics.csv``, ``equity.png`` and the cached forecasts
under ``--out``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from strategy_eval import backtest as bt
from strategy_eval.data import UNIVERSES, align, load_universe, synthetic_universe
from strategy_eval.forecast import forecast_features, load_kronos, rolling_forecasts


def tiny_random_kronos():
    """Randomly initialised tiny Kronos, used only by --synthetic smoke runs."""
    from model.kronos import Kronos, KronosTokenizer
    tok = KronosTokenizer(d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
                          ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, s1_bits=4, s2_bits=4,
                          beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4).eval()
    mdl = Kronos(s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=2, ff_dim=64, ffn_dropout_p=0.0,
                 attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True).eval()
    return tok, mdl, 512


def decision_positions(cal: pd.DatetimeIndex, test_start, test_end, horizon, min_hist):
    first = max(int(cal.searchsorted(pd.Timestamp(test_start))), min_hist)
    pos, t = [], first
    while t + 1 + horizon < len(cal) and cal[t] <= pd.Timestamp(test_end):
        pos.append(t)
        t += horizon
    if len(pos) < 4:
        raise SystemExit(f"Only {len(pos)} decisions fit in the window; extend the data or shorten --horizon.")
    return pos


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", default="us_etf", choices=sorted(UNIVERSES))
    p.add_argument("--synthetic", action="store_true", help="GBM data + random-init tiny model (smoke test)")
    p.add_argument("--test-start", default=None, help="default: 365 days before --test-end")
    p.add_argument("--test-end", default=None, help="default: last available bar")
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--lookback", type=int, default=256)
    p.add_argument("--horizon", type=int, default=5, help="bars per holding period = forecast length")
    p.add_argument("--samples", type=int, default=10, help="sampled paths per forecast")
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k-assets", type=int, default=None, help="assets held by cross-sectional strategies")
    p.add_argument("--cost-bps", type=float, default=None, help="one-way cost; default 5 (equities) / 10 (crypto)")
    p.add_argument("--n-random", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-limit", type=int, default=512)
    p.add_argument("--out", default="strategy_eval/outputs")
    p.add_argument("--refresh-data", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out) / ("synthetic" if a.synthetic else a.universe)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # ---------------- data ----------------
    if a.synthetic:
        raw, bench = synthetic_universe(), "SYN0"
        crypto = False
    else:
        end = pd.Timestamp(a.test_end) if a.test_end else pd.Timestamp.today().normalize()
        start = pd.Timestamp(a.test_start) if a.test_start else end - pd.Timedelta(days=365)
        hist_days = int(max(a.lookback, 130) * (1.0 if a.universe == "crypto" else 1.5)) + 30
        data_start = (start - pd.Timedelta(days=hist_days)).strftime("%Y-%m-%d")
        data_end = (end + pd.Timedelta(days=3 * a.horizon + 10)).strftime("%Y-%m-%d")
        raw, bench = load_universe(a.universe, data_start, data_end, Path(a.out) / "data", a.refresh_data)
        crypto = UNIVERSES[a.universe]["source"] == "binance"
    wide = align(raw)
    cal = wide["close"].index
    symbols = list(wide["close"].columns)
    ppy = 365 if (cal.dayofweek >= 5).any() else 252
    cost_bps = a.cost_bps if a.cost_bps is not None else (10.0 if crypto else 5.0)
    k = a.top_k_assets or max(1, round(len(symbols) / 4))
    test_end = a.test_end or cal[-1]
    test_start = a.test_start or (pd.Timestamp(test_end) - pd.Timedelta(days=365))
    if a.synthetic:
        test_start, test_end = cal[-260], cal[-1]
    pos = decision_positions(cal, test_start, test_end, a.horizon, min_hist=max(a.lookback, 130))
    print(f"{len(symbols)} assets, {len(pos)} decisions from {cal[pos[0]].date()} to {cal[pos[-1]].date()}, "
          f"holding {a.horizon} bars, cost {cost_bps} bps, top-k {k}")

    # ---------------- Kronos forecasts ----------------
    if a.synthetic:
        tok, mdl, max_ctx = tiny_random_kronos()
        model_name = "random-init tiny Kronos (smoke test only)"
    else:
        tok, mdl, max_ctx = load_kronos(a.model, a.device)
        model_name = a.model
    tag = f"{model_name.split('/')[-1]}_L{a.lookback}_H{a.horizon}_S{a.samples}_T{a.T}_p{a.top_p}_seed{a.seed}"
    fc = rolling_forecasts(wide, pos, tok, mdl, max_ctx, a.lookback, a.horizon, a.samples, T=a.T,
                           top_p=a.top_p, seed=a.seed, device=a.device, batch_limit=a.batch_limit,
                           cache_path=None if a.synthetic else out / f"forecasts_{tag}.npz")
    F = forecast_features(fc, wide)
    risk_adj = F["pred_ret"] / F["pred_disp"].replace(0, np.nan)
    risk_adj = risk_adj.fillna(0.0)

    # ---------------- strategies ----------------
    ewma = [bt.ewma_vol(wide, t) for t in pos]  # columns in `symbols` order
    strategies = {
        f"buy_hold_{bench}": bt.buy_hold(wide, pos, [bench]),
        "buy_hold_equal_weight": bt.buy_hold(wide, pos),
        "equal_weight_rebalanced": bt.equal_weight_rebalanced(wide, pos),
        "heur_ts_momentum_60": bt.ts_momentum(wide, pos, 60),
        "heur_sma50_trend": bt.sma_trend(wide, pos, 50),
        f"heur_xs_momentum_60_top{k}": bt.xs_momentum(wide, pos, k, 60),
        f"heur_xs_reversal_{a.horizon}_top{k}": bt.xs_reversal(wide, pos, k, a.horizon),
        "heur_inverse_vol_ewma": bt.inverse_vol(wide, pos, ewma),
        "kronos_ts_long_if_up": bt.from_signal_ts(F["pred_ret"], pos, wide),
        f"kronos_xs_top{k}": bt.from_signal_topk(F["pred_ret"], pos, wide, k),
        f"kronos_xs_riskadj_top{k}": bt.from_signal_topk(risk_adj, pos, wide, k),
        f"kronos_long_short_{k}x{k}": bt.from_signal_long_short(F["pred_ret"], pos, wide, k),
        "kronos_inverse_vol": bt.inverse_vol(wide, pos, F["pred_vol"].values),
    }
    rows, curves = {}, {}
    for name, W in strategies.items():
        sim = bt.simulate(W, wide, pos, a.horizon, cost_bps)
        rows[name] = bt.perf(sim, ppy)
        curves[name] = sim["equity"]

    rng = np.random.default_rng(a.seed)
    rand = {"random_topk": [], "random_timing": []}
    for _ in range(a.n_random):
        rand["random_topk"].append(bt.perf(bt.simulate(bt.random_topk(wide, pos, k, rng), wide, pos, a.horizon, cost_bps), ppy))
        rand["random_timing"].append(bt.perf(bt.simulate(bt.random_timing(wide, pos, rng), wide, pos, a.horizon, cost_bps), ppy))
    rand = {n: pd.DataFrame(v) for n, v in rand.items()}

    res = pd.DataFrame(rows).T
    for n, df in rand.items():
        res[f"pctile_vs_{n}"] = [100 * (df["total_return"] < r).mean() for r in res["total_return"]]
    res = res.sort_values("sharpe", ascending=False)

    # ---------------- forecast diagnostics ----------------
    cc, oo, rv = bt.realized(wide, pos, a.horizon)
    mom = pd.DataFrame([bt._trailing_return(wide, t, 60) for t in pos], index=cal[pos])
    rev = pd.DataFrame([-bt._trailing_return(wide, t, a.horizon) for t in pos], index=cal[pos])
    ic = pd.DataFrame({
        "kronos_pred_ret vs close-to-close": bt.rank_ic(F["pred_ret"], cc),
        "kronos_pred_ret vs traded open-to-open": bt.rank_ic(F["pred_ret"], oo),
        "kronos_prob_up vs traded open-to-open": bt.rank_ic(F["prob_up"], oo),
        "momentum_60 vs traded open-to-open": bt.rank_ic(mom, oo),
        f"reversal_{a.horizon} vs traded open-to-open": bt.rank_ic(rev, oo),
    }).T
    dirs = pd.DataFrame({"kronos_pred_ret vs close-to-close": bt.direction_stats(F["pred_ret"], cc),
                         "momentum_60 vs close-to-close": bt.direction_stats(mom, cc)}).T
    ewma_df = pd.DataFrame(ewma, index=cal[pos], columns=symbols)
    trail20 = pd.DataFrame([np.log(wide["close"].iloc[t - 20: t + 1]).diff().dropna().std() for t in pos], index=cal[pos])
    vols = pd.DataFrame({"kronos_pred_vol": bt.vol_stats(F["pred_vol"], rv),
                         "ewma_0.94": bt.vol_stats(ewma_df, rv),
                         "trailing_20d": bt.vol_stats(trail20, rv)}).T

    # ---------------- outputs ----------------
    res.to_csv(out / "metrics.csv")
    for n, df in rand.items():
        df.to_csv(out / f"{n}_distribution.csv", index=False)
    ic.to_csv(out / "forecast_rank_ic.csv"); vols.to_csv(out / "vol_forecast.csv"); dirs.to_csv(out / "direction.csv")
    pd.DataFrame(curves).to_csv(out / "equity_curves.csv")
    _plot(curves, rand, out / "equity.png")
    _report(out / "report.md", a, model_name, symbols, cal, pos, cost_bps, k, ppy, res, rand, ic, dirs, vols)
    print((out / "report.md").read_text())
    return res


def _plot(curves, rand, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    for n, eq in curves.items():
        style = "-" if n.startswith("kronos") else ("--" if n.startswith("heur") else ":")
        ax.plot(eq.index, eq.values, style, lw=1.6 if n.startswith("kronos") else 1.1, label=n)
    ax.set_title("Equity curves (net of costs); band = 5-95% of random top-k portfolios (final value)")
    lo, hi = np.percentile(rand["random_topk"]["total_return"], [5, 95]) + 1
    ax.axhspan(lo, hi, color="grey", alpha=0.12)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fmt(df):
    pct = {"total_return", "cagr", "ann_vol", "max_drawdown", "cost_drag"}
    out = df.copy().astype(object)
    for c in df.columns:
        if c in pct:
            out[c] = [f"{v:.1%}" for v in df[c]]
        elif df[c].dtype.kind == "f":
            out[c] = [f"{v:.2f}" for v in df[c]]
    return out.to_markdown()


def _report(path, a, model_name, symbols, cal, pos, cost_bps, k, ppy, res, rand, ic, dirs, vols):
    rsum = pd.DataFrame({n: {"median_total_return": df["total_return"].median(),
                             "p5_total_return": df["total_return"].quantile(0.05),
                             "p95_total_return": df["total_return"].quantile(0.95),
                             "median_sharpe": df["sharpe"].median(),
                             "p95_sharpe": df["sharpe"].quantile(0.95)} for n, df in rand.items()}).T
    lines = [
        "# Kronos out-of-sample strategy backtest", "",
        f"- Universe: **{'synthetic GBM (smoke test)' if a.synthetic else a.universe}**, {len(symbols)} assets: {', '.join(symbols)}",
        f"- Test decisions: {len(pos)} from {cal[pos[0]].date()} to {cal[pos[-1]].date()}, rebalanced every {a.horizon} bars "
        f"(signal at close t, fill at open t+1), last position closed {cal[pos[-1] + 1 + a.horizon].date()}",
        f"- Model: {model_name}; lookback {a.lookback}; {a.samples} sampled paths per forecast; T={a.T}, top_p={a.top_p}, seed {a.seed}",
        f"- Costs: {cost_bps} bps per unit one-way turnover; no borrow fee; cash earns 0. Cross-sectional strategies hold top {k}.",
        f"- Annualisation: {ppy} bars/year. Random baselines: {a.n_random} seeds each.",
        "- Caveats: universe chosen today (survivorship bias); one ~12-month window is one regime; "
        "Sharpe t-stat ≈ Sharpe × √years, so |t| < 2 is not distinguishable from luck.", "",
        "## Strategy performance (net of costs, sorted by Sharpe)", "",
        "`pctile_vs_random_*` = % of random portfolios with a lower total return.", "",
        _fmt(res), "",
        "## Random baselines", "", _fmt(rsum), "",
        "## Forecast quality: cross-sectional rank IC per decision date", "", _fmt(ic), "",
        "## Direction (pooled over assets and dates)", "", _fmt(dirs), "",
        f"## Volatility forecasts vs realised {a.horizon}-bar daily vol (QLIKE: lower is better)", "", _fmt(vols), "",
        "![equity](equity.png)", "",
    ]
    Path(path).write_text("\n".join(lines))
    (Path(path).parent / "run_config.json").write_text(json.dumps(vars(a), indent=2, default=str))


if __name__ == "__main__":
    main()
