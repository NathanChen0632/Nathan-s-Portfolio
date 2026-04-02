"""
live_signal.py
--------------
Generates a one-shot live BUY / HOLD CASH signal for today using a
DQN agent trained on the full available daily history (2015 → today).

Usage
-----
  python stock_prediction/main.py --signal
  python stock_prediction/main.py --signal --ticker AAPL MSFT
  python -m stock_prediction.live_signal --ticker SPY
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_prediction.data_collection import download_stock_data, TICKERS
from stock_prediction.features import build_features, get_feature_columns
from stock_prediction.rl_agent import train_dqn_agent


# ---------------------------------------------------------------------------
# Core signal generation
# ---------------------------------------------------------------------------

def generate_live_signal(ticker: str) -> None:
    today_str = date.today().isoformat()

    print(f"\n{'='*60}")
    print(f"  LIVE SIGNAL — {ticker}  ({today_str})")
    print(f"{'='*60}")

    # 1. Download data up to today
    df = download_stock_data(ticker, start="2015-01-01", end=today_str)

    # 2. Feature engineering
    feat_df      = build_features(df)
    feature_cols = get_feature_columns(feat_df)

    # 3. Train DQN on ALL available history with full trading rules
    X_train       = feat_df[feature_cols].values
    returns_train = feat_df["daily_return"].values
    prices_train  = df.loc[feat_df.index, "Close"].values.flatten()

    print(f"\n  Training DQN on {len(X_train)} days of history...")
    dqn = train_dqn_agent(
        X_train=X_train,
        daily_returns_train=returns_train,
        prices_train=prices_train,
        feature_cols=feature_cols,
        n_episodes=50,
    )

    # 4. Build feature row for the most recent available trading day
    X_today       = feat_df[feature_cols].iloc[[-1]].values
    latest_close  = float(df["Close"].iloc[-1])
    latest_date   = df.index[-1].date()

    print(f"\n  Latest data point : {latest_date}")
    print(f"  Last close price  : ${latest_close:.2f}")

    # 5. Predict using single-step method with cash context (no open position)
    pred  = dqn.predict_step(
        features=X_today[0],
        position=0,
        days_held=0,
        entry_price=0.0,
        stop_price=0.0,
        target_price=0.0,
        current_price=latest_close,
    )
    action    = "BUY / HOLD" if pred == 1 else "HOLD CASH"
    arrow     = "▲" if pred == 1 else "▼"

    print(f"\n  {'─'*50}")
    print(f"  DQN Signal  :  {arrow}  {action}")
    print(f"  {'─'*50}")

    if pred == 1:
        print(f"\n  The DQN agent predicts {ticker} will close HIGHER tomorrow.")
        print(f"  Suggested action: hold or buy {ticker}.\n")
    else:
        print(f"\n  The DQN agent predicts {ticker} will close LOWER or flat tomorrow.")
        print(f"  Suggested action: move to cash / avoid holding {ticker} overnight.\n")

    print("  NOTE: This is a research tool, not financial advice.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="One-shot live DQN signal.")
    parser.add_argument("--ticker", nargs="+", default=TICKERS, metavar="TICKER")
    return parser.parse_args()


def main():
    args = parse_args()
    for ticker in args.ticker:
        try:
            generate_live_signal(ticker.upper())
        except Exception as e:
            print(f"\n[ERROR] {ticker}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
