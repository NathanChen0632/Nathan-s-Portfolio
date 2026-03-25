"""
main.py
-------
Entry point for the CS5100 Stock Direction Prediction project.

Usage
-----
  python main.py                    # runs all tickers with default settings
  python main.py --ticker AAPL      # single ticker
  python main.py --ticker AAPL MSFT # multiple tickers

Workflow
--------
  1. Download historical OHLCV data (yfinance)
  2. Engineer technical features
  3. Chronological train / val / test split (70 / 15 / 15)
  4. Train Baseline, Logistic Regression, and Random Forest
  5. Evaluate on validation set (hyperparameter awareness) and test set
  6. Plot confusion matrices, feature importance, LR coefficients
  7. Run backtesting simulation vs buy-and-hold
"""

import argparse
import os
import sys

# Ensure the package root is on the path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_prediction.data_collection import download_stock_data, TICKERS
from stock_prediction.features import build_features, get_feature_columns
from stock_prediction.models import chronological_split, train_all_models
from stock_prediction.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_lr_coefficients,
    print_summary_table,
)
from stock_prediction.backtesting import (
    run_backtest,
    compute_backtest_metrics,
    print_backtest_metrics,
    plot_equity_curve,
)


# ---------------------------------------------------------------------------
# Per-ticker pipeline
# ---------------------------------------------------------------------------

def run_pipeline(ticker: str):
    print(f"\n{'#'*60}")
    print(f"  TICKER: {ticker}")
    print(f"{'#'*60}")

    # 1. Data collection
    print("\n[1/5] Downloading data...")
    df = download_stock_data(ticker)

    # 2. Feature engineering
    print("\n[2/5] Engineering features...")
    feat_df = build_features(df)
    feature_cols = get_feature_columns(feat_df)
    print(f"  Features: {len(feature_cols)}  |  Samples: {len(feat_df)}")

    # 3. Chronological split
    print("\n[3/5] Splitting data (70% train / 15% val / 15% test)...")
    splits = chronological_split(feat_df, feature_cols)
    print(f"  Train: {len(splits['X_train'])}  Val: {len(splits['X_val'])}  Test: {len(splits['X_test'])}")

    # 4. Train models
    print("\n[4/5] Training models...")
    models = train_all_models(splits["X_train"], splits["y_train"])

    # 5. Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    test_results = {}
    for name, model in models.items():
        test_results[name] = evaluate_model(name, model, splits["X_test"], splits["y_test"])
        plot_confusion_matrix(f"{ticker}_{name}", model, splits["X_test"], splits["y_test"])

    print_summary_table(test_results)

    # Plot model insights
    if "Random Forest" in models:
        plot_feature_importance(models["Random Forest"], feature_cols)
    if "Logistic Regression" in models:
        plot_lr_coefficients(models["Logistic Regression"], feature_cols)

    # 6. Backtesting
    print(f"\n[Backtesting] Running trading simulations on test set...")
    for name, model in models.items():
        if name == "Baseline (Majority Class)":
            continue  # baseline equity curve not informative
        bt_df   = run_backtest(model, splits["test_df"], feature_cols)
        metrics = compute_backtest_metrics(bt_df)
        print_backtest_metrics(metrics, f"{ticker} — {name}")
        plot_equity_curve(bt_df, f"{ticker}_{name}")

    return test_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CS5100 Stock Direction Prediction — ML Pipeline"
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        default=TICKERS,
        metavar="TICKER",
        help=f"Stock ticker(s) to analyse (default: {TICKERS})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_results = {}

    for ticker in args.ticker:
        try:
            all_results[ticker] = run_pipeline(ticker.upper())
        except Exception as e:
            print(f"\n[ERROR] Failed for {ticker}: {e}")
            import traceback; traceback.print_exc()

    print("\n\nAll results saved to the 'results/' directory.")


if __name__ == "__main__":
    main()
