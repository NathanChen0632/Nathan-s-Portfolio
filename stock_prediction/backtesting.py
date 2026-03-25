"""
backtesting.py
--------------
Simple daily trading simulation on the held-out test set.

Strategy (model-based)
----------------------
  - Start with $10,000 in cash.
  - Each day: if the model predicts UP (1), go long (hold the stock).
              if the model predicts DOWN (0), hold cash.
  - No transaction costs, no fractional shares, no short selling.

Benchmark: Buy-and-Hold
-----------------------
  - Invest $10,000 on day 1 and hold until end of test period.

Metrics reported
----------------
  - Final portfolio value
  - Total return (%)
  - Annualised return (%)
  - Sharpe ratio (annualised, risk-free rate = 0)
  - Max drawdown (%)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "results"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_backtest(
    model,
    test_df: pd.DataFrame,
    feature_cols: list,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """
    Run the model-based strategy on the test set.

    Parameters
    ----------
    model        : trained sklearn estimator
    test_df      : the test slice of the feature DataFrame (must contain 'Target'
                   and the raw close price can be reconstructed from daily_return)
    feature_cols : list of feature column names
    initial_capital : starting cash ($)

    Returns
    -------
    A DataFrame indexed by date with columns:
      close_return, signal, strategy_return, bah_return,
      strategy_equity, bah_equity
    """
    X_test = test_df[feature_cols].values
    signals = model.predict(X_test)   # 1 = long, 0 = cash

    # daily_return is already in test_df (engineered feature)
    daily_ret = test_df["daily_return"].values

    # Strategy: take the daily return only on days we're long
    strategy_ret = signals * daily_ret

    # Equity curves (compounded)
    strategy_equity = initial_capital * np.cumprod(1 + strategy_ret)
    bah_equity      = initial_capital * np.cumprod(1 + daily_ret)

    result = pd.DataFrame({
        "close_return":      daily_ret,
        "signal":            signals,
        "strategy_return":   strategy_ret,
        "bah_return":        daily_ret,
        "strategy_equity":   strategy_equity,
        "bah_equity":        bah_equity,
    }, index=test_df.index)

    return result


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _annualised_return(equity_series: pd.Series, trading_days: int = 252) -> float:
    total = equity_series.iloc[-1] / equity_series.iloc[0]
    n_years = len(equity_series) / trading_days
    return total ** (1 / n_years) - 1


def _sharpe_ratio(daily_returns: pd.Series, trading_days: int = 252) -> float:
    if daily_returns.std() == 0:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days)


def _max_drawdown(equity_series: pd.Series) -> float:
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    return drawdown.min()


def compute_backtest_metrics(bt_df: pd.DataFrame, initial_capital: float = 10_000.0) -> dict:
    metrics = {}
    for label, eq_col, ret_col in [
        ("Strategy",    "strategy_equity", "strategy_return"),
        ("Buy & Hold",  "bah_equity",      "bah_return"),
    ]:
        eq  = bt_df[eq_col]
        ret = bt_df[ret_col]
        metrics[label] = {
            "final_value":     eq.iloc[-1],
            "total_return":    (eq.iloc[-1] / initial_capital - 1) * 100,
            "annual_return":   _annualised_return(eq) * 100,
            "sharpe_ratio":    _sharpe_ratio(ret),
            "max_drawdown":    _max_drawdown(eq) * 100,
        }
    return metrics


def print_backtest_metrics(metrics: dict, model_name: str):
    print(f"\n{'='*60}")
    print(f"  Backtest Results — {model_name}")
    print(f"{'='*60}")
    print(f"  {'Metric':<22} {'Strategy':>12} {'Buy & Hold':>12}")
    print(f"  {'-'*46}")
    keys = [
        ("final_value",   "Final Value ($)",    ",.2f"),
        ("total_return",  "Total Return (%)",   ".2f"),
        ("annual_return", "Annual Return (%)",  ".2f"),
        ("sharpe_ratio",  "Sharpe Ratio",        ".3f"),
        ("max_drawdown",  "Max Drawdown (%)",    ".2f"),
    ]
    for key, label, fmt in keys:
        strat = metrics["Strategy"][key]
        bah   = metrics["Buy & Hold"][key]
        print(f"  {label:<22} {strat:>12{fmt}} {bah:>12{fmt}}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Equity curve plot
# ---------------------------------------------------------------------------

def plot_equity_curve(bt_df: pd.DataFrame, model_name: str):
    _ensure_output_dir()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bt_df.index, bt_df["strategy_equity"], label="Model Strategy",  color="#2c7bb6", linewidth=1.5)
    ax.plot(bt_df.index, bt_df["bah_equity"],      label="Buy & Hold",      color="#d7191c", linewidth=1.5, linestyle="--")
    ax.set_title(f"Equity Curve — {model_name}")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    path = os.path.join(OUTPUT_DIR, f"equity_{safe_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved equity curve → {path}")
