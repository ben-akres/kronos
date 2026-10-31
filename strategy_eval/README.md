# strategy_eval: out-of-sample Kronos strategy backtest

This module tests whether zero-shot Kronos forecasts make money **on data the
model can't have seen**. Kronos was released in August 2025, so the default test
window is the most recent 12 months. Each Kronos strategy is compared with:

- **Buy and hold**: the benchmark (SPY or BTC), and an equal-weight basket that is never rebalanced.
- **Equal weight, rebalanced** every holding period.
- **Random**: 500 seeds each of random top-k selection and random in/out timing. This gives a distribution of luck to rank every strategy against.
- **Simple heuristics**:
  - time-series momentum (60-bar);
  - SMA-50 trend filter;
  - cross-sectional 60-bar momentum;
  - short-term reversal;
  - inverse-volatility weighting with EWMA volatility.
- **Kronos variants, fixed before seeing results**:
  - long an asset when its mean forecast return is above 0;
  - top-k by forecast return;
  - top-k by forecast return divided by the spread of the sampled paths;
  - dollar-neutral long/short;
  - inverse-volatility weighting using the Kronos path volatility.

It also scores the forecasts directly, without any portfolio:
- cross-sectional rank IC;
- hit rate against the "always up" baseline;
- Kronos volatility forecasts against EWMA and trailing-20d volatility, scored with QLIKE.

## Run

```bash
pip install -r requirements.txt yfinance requests tabulate pytest
python -m strategy_eval.run --universe us_etf                         # 24 liquid ETFs, Kronos-small
python -m strategy_eval.run --universe crypto --horizon 7 --cost-bps 10
python -m strategy_eval.run --universe us_mega --samples 20
python -m strategy_eval.run --synthetic --samples 4 --n-random 50       # offline plumbing check
pytest tests/test_strategy_eval.py
```

Results go to `strategy_eval/outputs/<universe>/`: `report.md`, `metrics.csv`,
`equity.png`, the random-baseline distributions, and the cached forecast paths
(`forecasts_*.npz`). Re-runs reuse the cache.

**Network hosts needed:** `huggingface.co` (plus its file CDNs `cdn-lfs*.hf.co` / `cas-bridge.xethub.hf.co`) for the
weights; `query1.finance.yahoo.com`, `query2.finance.yahoo.com` and `fc.yahoo.com` for the US universes;
`data-api.binance.vision` for crypto.

**Runtime:** the model has no KV cache, so each generated bar re-runs the full context.
Kronos-small on 4 CPU cores takes roughly 1–2 minutes per decision date with 24 assets × 10 paths, so the
default weekly schedule (~50 decisions) takes about 1–2 hours. A GPU takes minutes. Use `--samples` and
`--lookback` to trade accuracy for speed.

## Methodology

- **Timing:** the signal is computed at the close of bar *t*, the fill happens at the open of *t+1*, and the position is held for `--horizon` bars with weights drifting. Every strategy uses this same timing.
- **No look-ahead:** each forecast normalises its input using only the lookback window, and uses a per-decision random seed. The tests confirm that changing future prices doesn't change earlier forecasts or weights.
- **Paths are kept:** `KronosPredictor.predict` only returns the mean of the sampled paths. Here every path is kept, so we get the probability of an up move, the dispersion, and a volatility forecast.
- **Costs:** `--cost-bps` is charged per unit of one-way turnover. There is no borrow fee and cash earns nothing.

## Limitations

- The universes were chosen today, so survivorship bias is built in.
- Twelve months is one market regime. The Sharpe t-stat is roughly Sharpe × √years, so one year can't separate a Sharpe-1 strategy from luck.
- The Kronos strategy variants, the sampling settings (T, top_p, samples) and the lookback are all trials. Changing them after looking at results invalidates the comparison against random.
- Daily bars only, with no intraday execution modelling, no market impact, and no taxes.
