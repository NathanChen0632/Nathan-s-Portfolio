"""
main.py
-------
Entry point for the CS5100 Stock Direction Prediction project.

Uses a single Deep Q-Network (DQN) reinforcement learning agent trained
on 10 years of daily data (2015-2025). The RL agent directly optimises
portfolio returns rather than prediction accuracy, making it better suited
to the trading objective than supervised classifiers.

Usage
-----
  python stock_prediction/main.py                        # backtest SPY, QQQ, XLK
  python stock_prediction/main.py --ticker AAPL          # single ticker
  python stock_prediction/main.py --signal               # live one-shot signal
  python stock_prediction/main.py --monitor              # continuous live monitor
  python stock_prediction/main.py --monitor --email      # monitor + email alerts

Workflow
--------
  1. Download 10 years of OHLCV data (2015-2025)
  2. Engineer technical features (RSI, MACD, MAs, volatility, volume)
  3. Chronological 70/15/15 train-val-test split
  4. Train DQN agent on train+val (full pre-test history)
  5. Evaluate on held-out test set
  6. Plot confusion matrix and equity curve vs buy-and-hold
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_prediction.data_collection import download_stock_data, TICKERS
from stock_prediction.features import build_features, get_feature_columns
from stock_prediction.models import chronological_split
from stock_prediction.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    print_summary_table,
)
from stock_prediction.backtesting import (
    run_backtest,
    compute_backtest_metrics,
    print_backtest_metrics,
    plot_equity_curve,
)
from stock_prediction.rl_agent import train_dqn_agent
from stock_prediction.live_signal import generate_live_signal
from stock_prediction.monitor import run_monitor


# ---------------------------------------------------------------------------
# Per-ticker pipeline
# ---------------------------------------------------------------------------

def run_pipeline(ticker: str):
    print(f"\n{'#'*60}")
    print(f"  TICKER: {ticker}")
    print(f"{'#'*60}")

    # 1. Download 10 years of data
    print("\n[1/4] Downloading 10 years of data (2015-2025)...")
    df = download_stock_data(ticker)

    # 2. Feature engineering
    print("\n[2/4] Engineering features...")
    feat_df      = build_features(df)
    feature_cols = get_feature_columns(feat_df)
    print(f"  Features: {len(feature_cols)}  |  Samples: {len(feat_df)}")

    # 3. Chronological split (70% train / 15% val / 15% test)
    print("\n[3/4] Splitting data...")
    splits = chronological_split(feat_df, feature_cols)
    print(f"  Train: {len(splits['X_train'])}  Val: {len(splits['X_val'])}  Test: {len(splits['X_test'])}")

    # Merge train+val — agent learns from all pre-test history
    X_train_full  = np.concatenate([splits["X_train"], splits["X_val"]])
    returns_train = feat_df["daily_return"].values[: len(X_train_full)]

    # 4. Train DQN agent on full pre-test history
    print("\n[4/4] Training DQN agent on 10 years of data (50 episodes)...")
    dqn = train_dqn_agent(
        X_train=X_train_full,
        daily_returns_train=returns_train,
        n_episodes=50,
    )

    # Evaluate on held-out test set
    print("\n  Evaluating on held-out test set...")
    results = {"DQN (RL Agent)": evaluate_model(
        "DQN (RL Agent)", dqn, splits["X_test"], splits["y_test"]
    )}
    plot_confusion_matrix(f"{ticker}_DQN", dqn, splits["X_test"], splits["y_test"])
    print_summary_table(results)

    # Backtest: DQN strategy vs buy-and-hold
    print(f"\n  Running backtest...")
    bt_df   = run_backtest(dqn, splits["test_df"], feature_cols)
    metrics = compute_backtest_metrics(bt_df)
    print_backtest_metrics(metrics, f"{ticker} — DQN (RL Agent)")
    plot_equity_curve(bt_df, f"{ticker}_DQN")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CS5100 — DQN RL Stock Trading Strategy"
    )
    parser.add_argument("--ticker", nargs="+", default=TICKERS, metavar="TICKER")
    parser.add_argument(
        "--signal", action="store_true",
        help="One-shot live signal for today.",
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Continuous live monitor (polls every --interval minutes).",
    )
    parser.add_argument("--interval", type=int, default=5, metavar="MINUTES")
    parser.add_argument("--no-market-check", action="store_true")
    parser.add_argument(
        "--email", action="store_true",
        help="Send email alerts on BUY/SELL signals.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.monitor:
        run_monitor(
            tickers=[t.upper() for t in args.ticker],
            interval_minutes=args.interval,
            skip_market_check=args.no_market_check,
            email_alerts=args.email,
        )
        return

    if args.signal:
        for ticker in args.ticker:
            try:
                generate_live_signal(ticker.upper())
            except Exception as e:
                print(f"\n[ERROR] {ticker}: {e}")
        return

    # Default: full backtest pipeline
    for ticker in args.ticker:
        try:
            run_pipeline(ticker.upper())
        except Exception as e:
            print(f"\n[ERROR] {ticker}: {e}")
            import traceback; traceback.print_exc()

    print("\n\nAll results saved to the 'results/' directory.")


if __name__ == "__main__":
    main()
