"""
live_signal.py
--------------
Generates a live BUY / HOLD CASH signal for today based on the most
recent available market data.

How it works
------------
  1. Download the full historical dataset up to today (yfinance)
  2. Engineer all features on that dataset
  3. Train all models (Baseline, LR, RF, DQN, Ensemble) on ALL available
     history — no train/test split, because we are not evaluating accuracy
     here, we are making a forward-looking prediction
  4. Extract the most recent complete feature row (yesterday's close is
     the last confirmed data point)
  5. Run every model's .predict() on that single row
  6. Print a clear recommendation for each model and an overall consensus

Usage
-----
  python -m stock_prediction.live_signal                  # default tickers
  python -m stock_prediction.live_signal --ticker AAPL
  python -m stock_prediction.live_signal --ticker AAPL MSFT NVDA

Why train on all data?
----------------------
  In backtest mode we hold out a test set to measure accuracy honestly.
  For a live signal there is no future data to hold out — we want the
  model to have learned from as much history as possible before making
  today's call.
"""

from __future__ import annotations

import argparse
import sys
import os
from datetime import date

import numpy as np
import pandas as pd

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
# Core signal generation
# ---------------------------------------------------------------------------

def generate_live_signal(ticker: str) -> None:
    """
    Download fresh data, train all models on full history, and print
    today's BUY / HOLD CASH recommendation for the given ticker.
    """
    print(f"\n{'='*60}")
    print(f"  LIVE SIGNAL — {ticker}  ({date.today()})")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Download data up to today
    # ------------------------------------------------------------------
    today_str = date.today().isoformat()
    df = download_stock_data(ticker, start="2015-01-01", end=today_str)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    feat_df = build_features(df)
    feature_cols = get_feature_columns(feat_df)

    # build_features drops the last row (no next-day label available).
    # For live prediction we need that last row, so we re-add it without
    # the Target column.
    last_row_features = pd.DataFrame(index=[df.index[-1]])
    for col in feature_cols:
        if col in feat_df.columns:
            last_row_features[col] = feat_df[col].iloc[-1]

    # Drop the final row from training data (it has no valid Target label)
    train_df      = feat_df.dropna(subset=["Target"])
    X_train       = train_df[feature_cols].values
    y_train       = train_df["Target"].values

    # The single feature vector we want to predict on
    X_today = last_row_features[feature_cols].values   # shape (1, n_features)

    latest_close  = df["Close"].iloc[-1]
    latest_date   = df.index[-1].date()

    print(f"\n  Latest data point : {latest_date}")
    print(f"  Last close price  : ${latest_close:.2f}")
    print(f"  Training samples  : {len(X_train)}")

    # ------------------------------------------------------------------
    # 3. Train all models on full history
    # ------------------------------------------------------------------
    print("\n  Training models on full history...")

    baseline = MajorityClassBaseline()
    baseline.fit(X_train, y_train)

    lr = build_logistic_regression()
    lr.fit(X_train, y_train)

    rf = build_random_forest()
    rf.fit(X_train, y_train)

    daily_returns_train = train_df["daily_return"].values
    dqn = train_dqn_agent(
        X_train=X_train,
        daily_returns_train=daily_returns_train,
        n_episodes=15,
    )

    ensemble = MajorityVoteEnsemble([lr, rf, dqn])

    # ------------------------------------------------------------------
    # 4. Predict on today's feature vector
    # ------------------------------------------------------------------
    models = {
        "Baseline (Majority Class)": baseline,
        "Logistic Regression":       lr,
        "Random Forest":             rf,
        "DQN (RL Agent)":            dqn,
        "Ensemble (LR + RF + DQN)":  ensemble,
    }

    print(f"\n  {'Model':<35} {'Signal':<12} {'Action'}")
    print(f"  {'-'*60}")

    signals = {}
    for name, model in models.items():
        pred = int(model.predict(X_today)[0])
        signals[name] = pred
        action = "BUY / HOLD" if pred == 1 else "HOLD CASH"
        marker = "  <--" if pred == 1 else ""
        print(f"  {name:<35} {pred:<12} {action}{marker}")

    # ------------------------------------------------------------------
    # 5. Consensus recommendation
    # ------------------------------------------------------------------
    # Exclude baseline from consensus — it never changes and adds no info
    non_baseline = {k: v for k, v in signals.items() if "Baseline" not in k}
    votes_for_buy = sum(non_baseline.values())
    total_votes   = len(non_baseline)
    consensus     = "BUY / HOLD" if votes_for_buy >= total_votes / 2 else "HOLD CASH"

    print(f"\n  {'-'*60}")
    print(f"  Consensus ({votes_for_buy}/{total_votes} models say BUY): "
          f">>> {consensus} <<<")
    print(f"  {'='*60}\n")

    # Plain-English explanation
    if consensus == "BUY / HOLD":
        print(f"  The majority of models predict {ticker} will close HIGHER")
        print(f"  tomorrow than today's close of ${latest_close:.2f}.")
        print(f"  Suggested action: hold or buy {ticker}.\n")
    else:
        print(f"  The majority of models predict {ticker} will close LOWER")
        print(f"  or flat tomorrow relative to today's close of ${latest_close:.2f}.")
        print(f"  Suggested action: move to cash / avoid holding {ticker} overnight.\n")

    print("  NOTE: This is a research tool, not financial advice.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a live BUY/HOLD signal for today."
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        default=TICKERS,
        metavar="TICKER",
        help=f"Ticker(s) to analyse (default: {TICKERS})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for ticker in args.ticker:
        try:
            generate_live_signal(ticker.upper())
        except Exception as e:
            print(f"\n[ERROR] Failed for {ticker}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
