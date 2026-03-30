"""
monitor.py
----------
Continuous live monitoring loop.  Models are trained ONCE at startup on
full daily history.  Every poll interval the latest intraday bar is fetched,
features are recomputed on the most recent window, and an updated BUY / HOLD
signal is printed for every watched ticker.

How intraday data is used
-------------------------
  yfinance returns 1-minute bars for the current trading day.
  We take the most recent minute bar and treat it as the "current bar":
    Open  = bar open
    High  = bar high  (best so far this minute)
    Low   = bar low
    Close = bar close (latest price)
    Volume= bar volume

  This bar is appended to the last N days of daily history so that all
  rolling-window features (MA50 needs 50 bars, etc.) can be computed
  without NaN.

Usage
-----
  python -m stock_prediction.monitor                        # default tickers, 5-min interval
  python -m stock_prediction.monitor --ticker AAPL MSFT     # specific tickers
  python -m stock_prediction.monitor --interval 1           # poll every 1 minute
  python -m stock_prediction.monitor --no-market-check      # run outside market hours (testing)

  OR via main.py:
  python stock_prediction/main.py --monitor
  python stock_prediction/main.py --monitor --interval 1 --ticker AAPL
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_prediction.data_collection import download_stock_data, TICKERS
from stock_prediction.features import build_features, get_feature_columns
from stock_prediction.models import (
    MajorityClassBaseline,
    build_logistic_regression,
    build_random_forest,
    MajorityVoteEnsemble,
)
from stock_prediction.rl_agent import train_dqn_agent


# ---------------------------------------------------------------------------
# Market hours (NYSE / NASDAQ)
# ---------------------------------------------------------------------------

MARKET_TZ    = pytz.timezone("America/New_York")
MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)


def is_market_open() -> bool:
    """Return True if the US equity market is currently open."""
    now_et = datetime.now(MARKET_TZ)
    if now_et.weekday() >= 5:          # Saturday or Sunday
        return False
    t = now_et.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def seconds_until_open() -> float:
    """Return seconds until the next market open (9:30 ET)."""
    now_et = datetime.now(MARKET_TZ)
    today_open = MARKET_TZ.localize(
        datetime(now_et.year, now_et.month, now_et.day, 9, 30)
    )
    if now_et < today_open and now_et.weekday() < 5:
        return (today_open - now_et).total_seconds()
    # next weekday open
    days_ahead = 1
    while True:
        candidate = today_open + pd.Timedelta(days=days_ahead)
        if candidate.weekday() < 5:
            return (candidate - now_et).total_seconds()
        days_ahead += 1


# ---------------------------------------------------------------------------
# Latest bar fetch
# ---------------------------------------------------------------------------

def fetch_latest_bar(ticker: str) -> pd.Series | None:
    """
    Download 1-minute bars for today and return the most recent complete
    minute bar as a Series with columns Open/High/Low/Close/Volume.
    Returns None if no intraday data is available yet.
    """
    try:
        raw = yf.download(
            ticker,
            period="1d",
            interval="1m",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return None

        # Flatten MultiIndex if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        bar = raw[["Open", "High", "Low", "Close", "Volume"]].iloc[-1]
        return bar
    except Exception as e:
        print(f"  [WARN] Could not fetch intraday bar for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Feature computation on live data
# ---------------------------------------------------------------------------

def build_live_feature_row(
    history_df: pd.DataFrame,
    live_bar: pd.Series,
) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """
    Append the live bar to historical daily data, recompute features on the
    extended dataset, and return the feature vector for the latest row.

    Returns (feature_vector, feature_cols) or (None, None) on failure.
    """
    # Build a one-row DataFrame for the live bar with a timestamp index
    now_ts = pd.Timestamp.now().normalize()           # today at midnight
    live_row = pd.DataFrame(
        {
            "Open":   [float(live_bar["Open"])],
            "High":   [float(live_bar["High"])],
            "Low":    [float(live_bar["Low"])],
            "Close":  [float(live_bar["Close"])],
            "Volume": [float(live_bar["Volume"])],
        },
        index=[now_ts],
    )

    # Remove today from history if it exists, then append the live bar
    combined = history_df.copy()
    if now_ts in combined.index:
        combined = combined.drop(index=now_ts)
    combined = pd.concat([combined, live_row])
    combined.sort_index(inplace=True)

    try:
        # build_features drops NaN rows + the last row (no next-day label)
        # We want the features *including* the live bar, so we temporarily
        # suppress the last-row drop by adding a dummy future row.
        dummy_ts  = now_ts + pd.Timedelta(days=1)
        dummy_row = pd.DataFrame(
            {col: [combined[col].iloc[-1]] for col in combined.columns},
            index=[dummy_ts],
        )
        extended = pd.concat([combined, dummy_row])

        feat_df      = build_features(extended)
        feature_cols = get_feature_columns(feat_df)

        # The last real row is the live bar's feature vector
        live_feat = feat_df.iloc[-2][feature_cols].values.reshape(1, -1)
        return live_feat, feature_cols
    except Exception as e:
        print(f"  [WARN] Feature computation failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# One-time model training
# ---------------------------------------------------------------------------

def train_models_on_history(ticker: str, history_df: pd.DataFrame) -> dict:
    """
    Train all models on the full available daily history.
    Called once at startup — this is the slow step.
    """
    print(f"  [{ticker}] Building features for training...")
    feat_df      = build_features(history_df)
    feature_cols = get_feature_columns(feat_df)

    X = feat_df[feature_cols].values
    y = feat_df["Target"].values

    print(f"  [{ticker}] Training on {len(X)} samples, {len(feature_cols)} features...")

    baseline = MajorityClassBaseline()
    baseline.fit(X, y)

    lr = build_logistic_regression()
    lr.fit(X, y)

    rf = build_random_forest()
    rf.fit(X, y)

    dqn = train_dqn_agent(
        X_train=X,
        daily_returns_train=feat_df["daily_return"].values,
        n_episodes=15,
    )

    ensemble = MajorityVoteEnsemble([lr, rf, dqn])

    print(f"  [{ticker}] Training complete.\n")

    return {
        "Baseline":  baseline,
        "LR":        lr,
        "RF":        rf,
        "DQN":       dqn,
        "Ensemble":  ensemble,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# Single poll
# ---------------------------------------------------------------------------

def poll_once(ticker: str, history_df: pd.DataFrame, trained: dict) -> None:
    """
    Fetch the latest bar, compute features, run all models, print signals.
    """
    live_bar = fetch_latest_bar(ticker)
    if live_bar is None:
        print(f"  [{ticker}] No intraday data — market may not have opened yet.")
        return

    feature_cols = trained["feature_cols"]
    X_live, _ = build_live_feature_row(history_df, live_bar)

    if X_live is None:
        print(f"  [{ticker}] Could not compute features for latest bar.")
        return

    current_price = float(live_bar["Close"])
    ts            = datetime.now().strftime("%H:%M:%S")

    models = {
        "Baseline":          trained["Baseline"],
        "Logistic Reg":      trained["LR"],
        "Random Forest":     trained["RF"],
        "DQN Agent":         trained["DQN"],
        "Ensemble":          trained["Ensemble"],
    }

    preds = {name: int(m.predict(X_live)[0]) for name, m in models.items()}

    # Consensus (exclude baseline)
    non_base = {k: v for k, v in preds.items() if k != "Baseline"}
    buy_votes = sum(non_base.values())
    total     = len(non_base)
    consensus = "BUY / HOLD" if buy_votes >= total / 2 else "HOLD CASH"
    arrow     = "▲" if consensus == "BUY / HOLD" else "▼"

    print(f"\n  [{ts}]  {ticker}  ${current_price:.2f}")
    print(f"  {'─'*50}")
    for name, pred in preds.items():
        label = "BUY/HOLD " if pred == 1 else "HOLD CASH"
        bar   = "█" * (pred * 8)
        print(f"  {name:<18} {label}  {bar}")
    print(f"  {'─'*50}")
    print(f"  CONSENSUS  {arrow}  {consensus}  ({buy_votes}/{total} models say BUY)")


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------

def run_monitor(
    tickers: list[str],
    interval_minutes: int = 5,
    skip_market_check: bool = False,
) -> None:
    """
    Continuously poll all tickers every `interval_minutes` minutes.
    Trains models once at startup, then loops until Ctrl-C.
    """
    print("\n" + "="*60)
    print("  LIVE SIGNAL MONITOR")
    print(f"  Tickers  : {', '.join(tickers)}")
    print(f"  Interval : every {interval_minutes} minute(s)")
    print(f"  Market   : {'always run' if skip_market_check else 'NYSE/NASDAQ hours only'}")
    print("  Press Ctrl-C to stop.")
    print("="*60 + "\n")

    # ------------------------------------------------------------------
    # Step 1 — download full history and train all models (done once)
    # ------------------------------------------------------------------
    from datetime import date
    today_str = date.today().isoformat()

    history   = {}   # ticker -> raw OHLCV DataFrame
    trained   = {}   # ticker -> dict of trained models

    for ticker in tickers:
        print(f"[STARTUP] Downloading history for {ticker}...")
        try:
            history[ticker] = download_stock_data(ticker, start="2015-01-01", end=today_str)
            print(f"[STARTUP] Training models for {ticker}...")
            trained[ticker] = train_models_on_history(ticker, history[ticker])
        except Exception as e:
            print(f"[ERROR] Could not initialise {ticker}: {e}")

    if not trained:
        print("[ERROR] No tickers could be initialised. Exiting.")
        return

    print("\n[STARTUP COMPLETE] Entering monitoring loop...\n")

    # ------------------------------------------------------------------
    # Step 2 — polling loop
    # ------------------------------------------------------------------
    try:
        while True:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not skip_market_check and not is_market_open():
                secs  = seconds_until_open()
                hrs   = int(secs // 3600)
                mins  = int((secs % 3600) // 60)
                print(f"[{now_str}] Market is closed. Next open in {hrs}h {mins}m. Sleeping...")
                # Sleep in 60-second increments so Ctrl-C is responsive
                for _ in range(min(int(secs), 300)):
                    time.sleep(1)
                continue

            print(f"\n{'='*60}")
            print(f"  POLL  {now_str}")
            print(f"{'='*60}")

            for ticker in tickers:
                if ticker not in trained:
                    continue
                try:
                    poll_once(ticker, history[ticker], trained[ticker])
                except Exception as e:
                    print(f"  [{ticker}] Poll error: {e}")

            print(f"\n  Next update in {interval_minutes} minute(s)...")
            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        print("\n\n[MONITOR] Stopped by user. Goodbye.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous live BUY/HOLD signal monitor."
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        default=TICKERS,
        metavar="TICKER",
        help=f"Ticker(s) to watch (default: {TICKERS})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        metavar="MINUTES",
        help="How often to refresh signals in minutes (default: 5)",
    )
    parser.add_argument(
        "--no-market-check",
        action="store_true",
        help="Disable market-hours gating (useful for testing outside trading hours)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_monitor(
        tickers=[t.upper() for t in args.ticker],
        interval_minutes=args.interval,
        skip_market_check=args.no_market_check,
    )


if __name__ == "__main__":
    main()
