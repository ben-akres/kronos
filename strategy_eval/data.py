"""Download and cache daily OHLCV bars from public sources.

Sources
-------
- ``yahoo``  : US equities / ETFs via the ``yfinance`` package (split- and
               dividend-adjusted OHLC, ``auto_adjust=True``). Needs network
               access to ``query1.finance.yahoo.com``, ``query2.finance.yahoo.com``
               and ``fc.yahoo.com``.
- ``binance``: Spot crypto daily klines from Binance's public market-data
               mirror ``data-api.binance.vision`` (no API key required).

Every loader returns ``dict[symbol -> DataFrame]`` indexed by a tz-naive
``DatetimeIndex`` with columns ``open, high, low, close, volume, amount``.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["open", "high", "low", "close", "volume", "amount"]

# ---------------------------------------------------------------------------
# Universes. Every list below was picked *today*, which builds in survivorship
# bias (these names all survived to now). The multi-asset ETF list suffers
# least from this and is the default.
# ---------------------------------------------------------------------------
UNIVERSES = {
    "us_etf": {
        "source": "yahoo",
        "benchmark": "SPY",
        "symbols": [
            "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
            "GLD", "SLV", "USO", "VNQ", "XLE", "XLF", "XLK", "XLV", "XLY", "XLP",
            "XLI", "XLU", "XLB", "XLRE",
        ],
    },
    "us_mega": {
        "source": "yahoo",
        "benchmark": "SPY",
        "symbols": [
            "SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "V",
            "UNH", "XOM", "JNJ", "PG", "MA", "HD", "COST", "ABBV", "MRK", "PEP",
            "KO", "AVGO", "ADBE", "CVX", "WMT", "BAC", "CRM", "NFLX", "AMD", "TSLA",
        ],
    },
    "crypto": {
        "source": "binance",
        "benchmark": "BTCUSDT",
        "symbols": [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
            "TRXUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT",
            "UNIUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT", "NEARUSDT", "APTUSDT",
        ],
    },
}


def _fetch_yahoo(symbols, start, end) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    out = {}
    for sym in symbols:
        df = yf.download(sym, start=start, end=end, interval="1d", auto_adjust=True,
                         progress=False, threads=False)
        if df is None or df.empty:
            print(f"  [warn] no data for {sym}")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        out[sym] = df[COLS].astype(float)
        print(f"  {sym}: {len(df)} bars {df.index.min().date()} -> {df.index.max().date()}")
    return out


def _fetch_binance(symbols, start, end) -> dict[str, pd.DataFrame]:
    import requests

    base = "https://data-api.binance.vision/api/v3/klines"
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    out = {}
    for sym in symbols:
        rows, cursor = [], start_ms
        while cursor < end_ms:
            r = requests.get(base, params={"symbol": sym, "interval": "1d", "startTime": cursor,
                                           "endTime": end_ms, "limit": 1000}, timeout=30)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            cursor = batch[-1][0] + 86_400_000
            time.sleep(0.2)
        if not rows:
            print(f"  [warn] no data for {sym}")
            continue
        df = pd.DataFrame(rows).iloc[:, :8]
        df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "amount"]
        df.index = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
        df = df[COLS].astype(float)
        df = df[~df.index.duplicated(keep="last")]
        # Drop today's still-forming candle.
        df = df[df.index < pd.Timestamp.utcnow().tz_localize(None).normalize()]
        out[sym] = df
        print(f"  {sym}: {len(df)} bars {df.index.min().date()} -> {df.index.max().date()}")
    return out


def load_universe(name: str, start: str, end: str, cache_dir: Path, refresh: bool = False):
    """Return (data_dict, benchmark_symbol), caching raw bars as CSV per symbol."""
    spec = UNIVERSES[name]
    cache = Path(cache_dir) / name
    cache.mkdir(parents=True, exist_ok=True)
    missing = [s for s in spec["symbols"] if refresh or not (cache / f"{s}.csv").exists()]
    if missing:
        print(f"Downloading {len(missing)} symbols from {spec['source']} ...")
        fetch = _fetch_yahoo if spec["source"] == "yahoo" else _fetch_binance
        for sym, df in fetch(missing, start, end).items():
            df.to_csv(cache / f"{sym}.csv", index_label="date")
    data = {}
    for sym in spec["symbols"]:
        p = cache / f"{sym}.csv"
        if p.exists():
            df = pd.read_csv(p, index_col="date", parse_dates=True)
            data[sym] = df.loc[(df.index >= start) & (df.index <= end), COLS]
    return data, spec["benchmark"]


def align(data: dict[str, pd.DataFrame], min_coverage: float = 0.98):
    """Put all symbols on one calendar.

    Symbols that miss more than ``1 - min_coverage`` of the calendar are dropped
    (e.g. a coin listed half-way through). Remaining small gaps are forward
    filled for prices (volume set to 0) so that every asset has a bar on every
    date. Returns a dict of wide DataFrames: field -> (date x symbol).
    """
    cal = sorted(set().union(*[df.index for df in data.values()]))
    cal = pd.DatetimeIndex(cal)
    keep = {s: df for s, df in data.items() if df.index.isin(cal).sum() >= min_coverage * len(cal)}
    dropped = set(data) - set(keep)
    if dropped:
        print(f"Dropping symbols with insufficient history: {sorted(dropped)}")
    wide = {}
    for f in COLS:
        w = pd.DataFrame({s: df[f] for s, df in keep.items()}).reindex(cal)
        if f in ("volume", "amount"):
            w = w.fillna(0.0)
        else:
            w = w.ffill()
        wide[f] = w
    # Rows before every symbol has started are not usable.
    first_valid = wide["close"].dropna().index.min()
    wide = {f: w.loc[first_valid:] for f, w in wide.items()}
    assert not np.isnan(wide["close"].values).any()
    return wide


def synthetic_universe(n_assets=6, n_bars=700, seed=0, start="2023-01-02"):
    """Geometric-Brownian-motion OHLCV, used only by the smoke test."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_bars)
    data = {}
    for i in range(n_assets):
        r = rng.normal(0.0003, 0.015, n_bars)
        close = 100 * np.exp(np.cumsum(r))
        open_ = close * np.exp(rng.normal(0, 0.003, n_bars))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_bars)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_bars)))
        vol = rng.lognormal(10, 0.3, n_bars)
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
        data[f"SYN{i}"] = df[COLS]
    return data
