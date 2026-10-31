"""Rolling, look-ahead-free Kronos forecasts that keep every sampled path.

``KronosPredictor.predict`` averages the sampled paths and returns only the
mean (``model/kronos.py`` ``auto_regressive_inference``). For trading we
also want the spread of the paths (probability of an up move, a volatility
forecast), so this module repeats each context ``n_samples`` times and calls
``auto_regressive_inference`` with ``sample_count=1``. That yields one
independent path per copy.

Each decision at calendar position ``t`` sees only bars ``[t-lookback+1, t]``.
The future timestamps passed to the model are the next ``horizon`` calendar
dates. The model receives only their calendar features (weekday, day, month);
the trading calendar is known in advance, so this does not leak information.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference, calc_time_stamps

FEATS = ["open", "high", "low", "close", "volume", "amount"]

MODEL_ZOO = {
    "NeoQuasar/Kronos-mini": ("NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "NeoQuasar/Kronos-small": ("NeoQuasar/Kronos-Tokenizer-base", 512),
    "NeoQuasar/Kronos-base": ("NeoQuasar/Kronos-Tokenizer-base", 512),
}


def load_kronos(model_id: str, device: str = "cpu"):
    tok_id, max_ctx = MODEL_ZOO[model_id]
    tokenizer = KronosTokenizer.from_pretrained(tok_id).to(device).eval()
    model = Kronos.from_pretrained(model_id).to(device).eval()
    return tokenizer, model, max_ctx


def _future_stamps(cal: pd.DatetimeIndex, t: int, horizon: int) -> pd.Series:
    fut = list(cal[t + 1: t + 1 + horizon])
    # 7-day calendar (crypto) vs business-day calendar (equities).
    step = pd.Timedelta(days=1) if (cal.dayofweek >= 5).any() else pd.tseries.offsets.BDay()
    last = fut[-1] if fut else cal[t]
    while len(fut) < horizon:  # beyond the end of the data: extrapolate the calendar
        last = last + step
        fut.append(last)
    return pd.Series(pd.DatetimeIndex(fut))


def rolling_forecasts(wide: dict[str, pd.DataFrame], decision_pos: list[int], tokenizer, model,
                      max_context: int, lookback: int, horizon: int, n_samples: int,
                      T: float = 1.0, top_p: float = 0.9, clip: float = 5.0, seed: int = 0,
                      device: str = "cpu", batch_limit: int = 512, cache_path: Path | None = None):
    """Return dict with arrays of shape (n_decisions, n_assets, n_samples, horizon)."""
    symbols = list(wide["close"].columns)
    cal = wide["close"].index
    if cache_path is not None and Path(cache_path).exists():
        z = np.load(cache_path, allow_pickle=True)
        if list(z["symbols"]) == symbols and list(z["decision_pos"]) == list(decision_pos):
            print(f"Loaded cached forecasts from {cache_path}")
            return {k: z[k] for k in z.files}
        print("Cache does not match this run; recomputing.")

    n_d, n_a = len(decision_pos), len(symbols)
    close_paths = np.zeros((n_d, n_a, n_samples, horizon), dtype=np.float32)
    high_paths = np.zeros_like(close_paths)
    low_paths = np.zeros_like(close_paths)
    t0 = time.time()
    for k, t in enumerate(decision_pos):
        assert t - lookback + 1 >= 0, "not enough history before first decision"
        xs, x_st, y_st, mus, sds = [], [], [], [], []
        x_ts = pd.Series(cal[t - lookback + 1: t + 1])
        y_ts = _future_stamps(cal, t, horizon)
        x_stamp = calc_time_stamps(x_ts).values.astype(np.float32)
        y_stamp = calc_time_stamps(y_ts).values.astype(np.float32)
        for s in symbols:
            x = np.stack([wide[f][s].values[t - lookback + 1: t + 1] for f in FEATS], axis=1).astype(np.float32)
            mu, sd = x.mean(axis=0), x.std(axis=0)
            xn = np.clip((x - mu) / (sd + 1e-5), -clip, clip)
            xs.append(xn); mus.append(mu); sds.append(sd)
            x_st.append(x_stamp); y_st.append(y_stamp)
        X = np.repeat(np.stack(xs), n_samples, axis=0)
        XS = np.repeat(np.stack(x_st), n_samples, axis=0)
        YS = np.repeat(np.stack(y_st), n_samples, axis=0)

        torch.manual_seed(seed + k)  # per-decision seed: earlier forecasts never depend on later data
        outs = []
        for b in range(0, len(X), batch_limit):
            preds = auto_regressive_inference(
                tokenizer, model,
                torch.from_numpy(X[b:b + batch_limit]).to(device),
                torch.from_numpy(XS[b:b + batch_limit]).to(device),
                torch.from_numpy(YS[b:b + batch_limit]).to(device),
                max_context=max_context, pred_len=horizon, clip=clip, T=T, top_k=0, top_p=top_p,
                sample_count=1)
            outs.append(preds[:, -horizon:, :])
        P = np.concatenate(outs).reshape(n_a, n_samples, horizon, len(FEATS))
        mu = np.stack(mus)[:, None, None, :]
        sd = np.stack(sds)[:, None, None, :]
        P = P * (sd + 1e-5) + mu
        close_paths[k] = P[..., 3]
        high_paths[k] = P[..., 1]
        low_paths[k] = P[..., 2]
        el = time.time() - t0
        print(f"  forecast {k + 1}/{n_d} {cal[t].date()}  elapsed {el:,.0f}s  eta {el / (k + 1) * (n_d - k - 1):,.0f}s",
              flush=True)

    out = dict(close_paths=close_paths, high_paths=high_paths, low_paths=low_paths,
               symbols=np.array(symbols), decision_pos=np.array(decision_pos))
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **out)
    return out


def forecast_features(fc: dict, wide: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Summaries of the path distribution per (decision date, asset)."""
    cal = wide["close"].index
    pos = list(fc["decision_pos"])
    idx = cal[pos]
    cols = list(fc["symbols"])
    c_t = wide["close"].iloc[pos][cols].values[:, :, None]  # (D, A, 1)
    paths = fc["close_paths"].astype(np.float64)              # (D, A, S, H)
    end_ret = paths[..., -1] / c_t - 1                        # (D, A, S)
    logp = np.log(np.concatenate([np.repeat(c_t[..., None], paths.shape[2], axis=2), paths], axis=-1))
    step_vol = np.diff(logp, axis=-1).std(axis=-1).mean(axis=-1)  # mean per-path daily vol
    mk = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    return {
        "pred_ret": mk(end_ret.mean(axis=-1)),
        "prob_up": mk((end_ret > 0).mean(axis=-1)),
        "pred_disp": mk(end_ret.std(axis=-1)),
        "pred_vol": mk(step_vol),
    }
