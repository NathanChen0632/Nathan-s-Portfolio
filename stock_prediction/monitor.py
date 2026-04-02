"""
monitor.py
----------
Continuous live trading strategy monitor.

Strategy state machine (per ticker)
-------------------------------------
  FLAT  →  consensus says UP   →  emit BUY  →  LONG
  LONG  →  consensus says UP   →  emit HOLD →  LONG
  LONG  →  consensus says DOWN →  emit SELL →  FLAT
  FLAT  →  consensus says DOWN →  emit WAIT →  FLAT

There is ONE strategy signal per ticker per poll.  The individual model
votes (LR, RF, DQN) are shown as supporting evidence but the user only
needs to act on the strategy signal at the top.

Usage
-----
  python stock_prediction/main.py --monitor
  python stock_prediction/main.py --monitor --ticker AAPL MSFT --interval 1
  python stock_prediction/main.py --monitor --no-market-check   # test outside hours

  python -m stock_prediction.monitor --ticker AAPL
"""

from __future__ import annotations

import argparse
import os
import smtplib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

# Auto-load .env file so credentials work without manually running `source .env`
try:
    from dotenv import load_dotenv
    # Walk up from this file's directory to find .env
    _here = os.path.dirname(os.path.abspath(__file__))
    for _candidate in [_here, os.path.dirname(_here)]:
        _env_path = os.path.join(_candidate, ".env")
        if os.path.exists(_env_path):
            load_dotenv(_env_path)
            break
except ImportError:
    pass  # dotenv not installed — fall back to manual env vars

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_prediction.data_collection import download_stock_data, TICKERS
from stock_prediction.features import build_features, get_feature_columns
from stock_prediction.rl_agent import train_dqn_agent


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

MARKET_TZ    = pytz.timezone("America/New_York")
MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)


def is_market_open() -> bool:
    now_et = datetime.now(MARKET_TZ)
    if now_et.weekday() >= 5:
        return False
    t = now_et.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def seconds_until_open() -> float:
    now_et     = datetime.now(MARKET_TZ)
    today_open = MARKET_TZ.localize(
        datetime(now_et.year, now_et.month, now_et.day, 9, 30)
    )
    if now_et < today_open and now_et.weekday() < 5:
        return (today_open - now_et).total_seconds()
    days_ahead = 1
    while True:
        candidate = today_open + pd.Timedelta(days=days_ahead)
        if candidate.weekday() < 5:
            return (candidate - now_et).total_seconds()
        days_ahead += 1


# ---------------------------------------------------------------------------
# Strategy state
# ---------------------------------------------------------------------------

class Position(Enum):
    FLAT = "FLAT"   # not holding — looking to buy
    LONG = "LONG"   # holding     — looking to sell


@dataclass
class StrategyState:
    """Tracks the live trading state for one ticker."""
    ticker:       str
    position:     Position = Position.FLAT
    entry_price:  float    = 0.0
    stop_price:   float    = 0.0
    target_price: float    = 0.0
    entry_time:   str      = ""
    days_held:    int      = 0
    trade_log:    list     = field(default_factory=list)


# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
BLUE   = "\033[94m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Email alerts
# ---------------------------------------------------------------------------

@dataclass
class EmailConfig:
    """
    Email credentials loaded from environment variables.

    Required env vars:
      SMTP_USER      — your Gmail address  (e.g. you@gmail.com)
      SMTP_PASSWORD  — Gmail App Password  (16-char, no spaces)
                       Generate at: Google Account → Security → App passwords
      ALERT_TO       — recipient address   (can be same as SMTP_USER)

    Optional:
      SMTP_HOST      — default: smtp.gmail.com
      SMTP_PORT      — default: 587
    """
    sender:   str
    password: str
    to:       str
    host:     str = "smtp.gmail.com"
    port:     int = 587

    @classmethod
    def from_env(cls) -> "EmailConfig | None":
        """
        Build an EmailConfig from environment variables.
        Returns None (with a warning) if any required variable is missing.
        """
        sender   = os.environ.get("SMTP_USER")
        password = os.environ.get("SMTP_PASSWORD")
        to       = os.environ.get("ALERT_TO")

        if not all([sender, password, to]):
            missing = [v for v, val in [
                ("SMTP_USER", sender), ("SMTP_PASSWORD", password), ("ALERT_TO", to)
            ] if not val]
            print(f"{YELLOW}[EMAIL] Missing env vars: {', '.join(missing)} — alerts disabled.{RESET}")
            print(f"{YELLOW}[EMAIL] Set them in a .env file or export before running.{RESET}")
            return None

        return cls(
            sender=sender,
            password=password,
            to=to,
            host=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
            port=int(os.environ.get("SMTP_PORT", 587)),
        )


def _build_email_body(
    action:      str,
    ticker:      str,
    price:       float,
    state:       "StrategyState",
    votes:       dict[str, int],
) -> tuple[str, str]:
    """Return (subject, html_body) for a BUY or SELL alert."""
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emoji    = "🟢" if action == "BUY" else "🔴"
    subject  = f"{emoji} {action} {ticker} @ ${price:.2f} — Trading Alert"

    if action == "BUY":
        action_line = f"<b style='color:green'>BUY {ticker} NOW @ ${price:.2f}</b>"
        detail_line = f"New position opened. Watching for SELL signal."
    else:
        pnl      = (price - state.entry_price) / state.entry_price * 100
        sign     = "+" if pnl >= 0 else ""
        colour   = "green" if pnl >= 0 else "red"
        action_line = f"<b style='color:red'>SELL {ticker} NOW @ ${price:.2f}</b>"
        detail_line = (
            f"Entry: ${state.entry_price:.2f} &nbsp;|&nbsp; "
            f"P&amp;L: <b style='color:{colour}'>{sign}{pnl:.2f}%</b>"
        )

    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:480px">
      <h2>{emoji} DQN Trading Signal — {ticker}</h2>
      <p style="font-size:1.3em">{action_line}</p>
      <p>{detail_line}</p>
      <p style="color:#888">Time: {ts}</p>
      <hr/>
      <p style="color:#aaa;font-size:0.8em">
        Generated by a Deep Q-Network trained on 10 years of market data.
        This is a research tool, not financial advice.
      </p>
    </body></html>
    """
    return subject, html


def send_email_alert(
    cfg:    "EmailConfig",
    action: str,
    ticker: str,
    price:  float,
    state:  "StrategyState",
    votes:  dict[str, int],
) -> None:
    """Send a BUY or SELL alert email. Silently logs on failure."""
    try:
        subject, html = _build_email_body(action, ticker, price, state, votes)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = cfg.sender
        msg["To"]      = cfg.to
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(cfg.host, cfg.port) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg.sender, cfg.password)
            server.sendmail(cfg.sender, cfg.to, msg.as_string())

        print(f"  {GREEN}[EMAIL] Alert sent to {cfg.to}{RESET}")
    except Exception as e:
        print(f"  {YELLOW}[EMAIL] Failed to send alert: {e}{RESET}")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_latest_bar(ticker: str) -> pd.Series | None:
    """Fetch the most recent 1-minute bar for the current trading session."""
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
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw[["Open", "High", "Low", "Close", "Volume"]].iloc[-1]
    except Exception as e:
        print(f"  [WARN] Could not fetch intraday bar for {ticker}: {e}")
        return None


def build_live_feature_row(
    history_df: pd.DataFrame,
    live_bar: pd.Series,
) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """
    Append the live bar to daily history and return the feature vector
    for that bar so all rolling-window indicators are correctly computed.
    """
    now_ts   = pd.Timestamp.now().normalize()
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

    combined = history_df.copy()
    if now_ts in combined.index:
        combined = combined.drop(index=now_ts)
    combined = pd.concat([combined, live_row]).sort_index()

    try:
        # Add a dummy next-day row so build_features doesn't drop the live bar
        dummy_ts  = now_ts + pd.Timedelta(days=1)
        dummy_row = pd.DataFrame(
            {col: [combined[col].iloc[-1]] for col in combined.columns},
            index=[dummy_ts],
        )
        feat_df      = build_features(pd.concat([combined, dummy_row]))
        feature_cols = get_feature_columns(feat_df)
        return feat_df.iloc[-2][feature_cols].values.reshape(1, -1), feature_cols
    except Exception as e:
        print(f"  [WARN] Feature computation failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Model training (once at startup)
# ---------------------------------------------------------------------------

def train_models_on_history(ticker: str, history_df: pd.DataFrame) -> dict:
    print(f"  [{ticker}] Building features...")
    feat_df      = build_features(history_df)
    feature_cols = get_feature_columns(feat_df)
    X            = feat_df[feature_cols].values
    prices       = history_df.loc[feat_df.index, "Close"].values.flatten()

    print(f"  [{ticker}] Training DQN with disciplined strategy on {len(X)} samples...")
    dqn = train_dqn_agent(
        X_train=X,
        daily_returns_train=feat_df["daily_return"].values,
        prices_train=prices,
        feature_cols=feature_cols,
        n_episodes=50,
    )

    print(f"  [{ticker}] Done.\n")
    return {
        "DQN":          dqn,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# Strategy signal resolution
# ---------------------------------------------------------------------------

def get_consensus(trained: dict, X_live: np.ndarray) -> tuple[int, dict[str, int]]:
    """
    Run the DQN agent and return its signal.
    Returns (signal: 0|1, votes dict for display).
    """
    pred  = int(trained["DQN"].predict(X_live)[0])
    votes = {"DQN Agent": pred}
    return pred, votes


def resolve_strategy_action(
    state:        StrategyState,
    signal:       int,
    price:        float,
    features:     np.ndarray,
    feature_cols: list,
) -> str:
    """
    Apply DQN signal to current position state.
    On BUY: computes ATR-based stop and 2:1 target and stores them in state.
    Updates state in place. Returns action string.
    """
    from stock_prediction.rl_agent import TradingConfig
    cfg = TradingConfig()

    if state.position == Position.FLAT:
        if signal == 1:
            # Compute stop and target at entry using ATR
            atr_idx    = feature_cols.index("atr14_pct") if "atr14_pct" in feature_cols else None
            atr_pct    = float(features[atr_idx]) if atr_idx is not None else 0.015
            stop_dist  = cfg.stop_atr_mult * atr_pct * price
            stop_price = price - stop_dist
            tgt_price  = price + cfg.min_rr_ratio * stop_dist

            state.position     = Position.LONG
            state.entry_price  = price
            state.stop_price   = stop_price
            state.target_price = tgt_price
            state.entry_time   = datetime.now().strftime("%H:%M:%S")
            state.days_held    = 0
            return "BUY"
        return "WAIT"

    else:  # Position.LONG
        state.days_held += 1

        # Check if stop or target was hit regardless of DQN signal
        if price <= state.stop_price:
            signal = 0   # force exit — stop hit
        elif price >= state.target_price:
            signal = 0   # force exit — target hit
        elif state.days_held >= cfg.max_holding_days:
            signal = 0   # force exit — time stop

        if signal == 0:
            pnl = (price - state.entry_price) / state.entry_price * 100
            reason = "stop hit" if price <= state.stop_price else (
                     "target hit" if price >= state.target_price else
                     "time stop" if state.days_held >= cfg.max_holding_days else "signal")
            state.trade_log.append({
                "entry":  state.entry_price,
                "exit":   price,
                "pnl%":   pnl,
                "time":   datetime.now().strftime("%H:%M:%S"),
                "reason": reason,
            })
            state.position     = Position.FLAT
            state.entry_price  = 0.0
            state.stop_price   = 0.0
            state.target_price = 0.0
            state.days_held    = 0
            state.entry_time   = ""
            return "SELL"
        return "HOLD"


# ---------------------------------------------------------------------------
# Poll display
# ---------------------------------------------------------------------------

def print_strategy_signal(
    action:     str,
    ticker:     str,
    price:      float,
    state:      StrategyState,
    votes:      dict[str, int],
    consensus:  int,
) -> None:
    ts = datetime.now().strftime("%H:%M:%S")

    # --- colour and banner by action ---
    if action == "BUY":
        colour  = GREEN
        banner  = f"  {BOLD}{GREEN}>>> BUY {ticker} NOW  @  ${price:.2f} <<<{RESET}"
        summary = (
            f"  {GREEN}Entry: ${price:.2f}  |  "
            f"Stop: ${state.stop_price:.2f}  |  "
            f"Target: ${state.target_price:.2f}  |  "
            f"R/R: 2:1{RESET}"
        )
    elif action == "SELL":
        colour  = RED
        pnl     = (price - state.entry_price) / state.entry_price * 100
        sign    = "+" if pnl >= 0 else ""
        reason  = state.trade_log[-1]["reason"] if state.trade_log else "signal"
        banner  = f"  {BOLD}{RED}>>> SELL {ticker} NOW  @  ${price:.2f} <<<{RESET}"
        summary = (
            f"  {RED}Reason: {reason}  |  Entry: ${state.entry_price:.2f}  "
            f"|  P&L: {sign}{pnl:.2f}%{RESET}"
        )
    elif action == "HOLD":
        colour  = BLUE
        pnl     = (price - state.entry_price) / state.entry_price * 100
        sign    = "+" if pnl >= 0 else ""
        banner  = f"  {BOLD}{BLUE}HOLD {ticker}  @  ${price:.2f}{RESET}"
        summary = (
            f"  {BLUE}Day {state.days_held}  |  "
            f"Entry: ${state.entry_price:.2f}  Stop: ${state.stop_price:.2f}  "
            f"Target: ${state.target_price:.2f}  |  "
            f"Unrealised P&L: {sign}{pnl:.2f}%{RESET}"
        )
    else:  # WAIT
        colour  = GREY
        banner  = f"  {GREY}WAIT — no position in {ticker}  (${price:.2f}){RESET}"
        summary = f"  {GREY}Watching for a valid BUY setup (R/R ≥ 2:1, volume confirmed)...{RESET}"

    dqn_signal = votes.get("DQN Agent", consensus)
    dot        = f"{GREEN}●{RESET}" if dqn_signal == 1 else f"{RED}●{RESET}"

    print(f"\n  {'═'*58}")
    print(f"  [{ts}]  {ticker}  ${price:.2f}")
    print(f"  {'═'*58}")
    print(banner)
    print(summary)
    print(f"  {'─'*58}")
    print(f"  DQN Signal:  {dot}  {'UP — expects price to rise' if dqn_signal == 1 else 'DOWN — expects price to fall'}")

    # Session trade log
    if state.trade_log:
        print(f"  {'─'*58}")
        print(f"  Session trades:")
        for t in state.trade_log[-3:]:
            sign = "+" if t["pnl%"] >= 0 else ""
            col  = GREEN if t["pnl%"] >= 0 else RED
            print(f"    {t['time']}  entry ${t['entry']:.2f}  "
                  f"exit ${t['exit']:.2f}  {col}{sign}{t['pnl%']:.2f}%{RESET}")

    print(f"  {'═'*58}")


# ---------------------------------------------------------------------------
# Single poll
# ---------------------------------------------------------------------------

def poll_once(
    ticker:     str,
    history_df: pd.DataFrame,
    trained:    dict,
    state:      StrategyState,
    email_cfg:  EmailConfig | None = None,
) -> None:
    live_bar = fetch_latest_bar(ticker)
    if live_bar is None:
        print(f"  [{ticker}] No intraday data — market may not have opened yet.")
        return

    X_live, _ = build_live_feature_row(history_df, live_bar)
    if X_live is None:
        print(f"  [{ticker}] Feature computation failed.")
        return

    price = float(live_bar["Close"])

    # Use predict_step so the agent has full trade context (stop, target, days held)
    pred = trained["DQN"].predict_step(
        features=X_live[0],
        position=1 if state.position == Position.LONG else 0,
        days_held=state.days_held,
        entry_price=state.entry_price,
        stop_price=state.stop_price,
        target_price=state.target_price,
        current_price=price,
    )

    votes  = {"DQN Agent": pred}
    action = resolve_strategy_action(state, pred, price, X_live[0], trained["feature_cols"])

    print_strategy_signal(action, ticker, price, state, votes, pred)

    # Send email only on actionable signals (not HOLD or WAIT)
    if email_cfg and action in ("BUY", "SELL"):
        send_email_alert(email_cfg, action, ticker, price, state, votes)


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------

def run_monitor(
    tickers:           list[str],
    interval_minutes:  int  = 5,
    skip_market_check: bool = False,
    email_alerts:      bool = False,
) -> None:

    # Load email config from env vars if alerts are requested
    email_cfg = EmailConfig.from_env() if email_alerts else None

    print("\n" + "="*60)
    print(f"  {BOLD}TRADING STRATEGY MONITOR{RESET}")
    print(f"  Tickers  : {', '.join(tickers)}")
    print(f"  Interval : every {interval_minutes} minute(s)")
    print(f"  Strategy : FLAT → BUY → HOLD → SELL → FLAT")
    print(f"  Market   : {'always run' if skip_market_check else 'NYSE/NASDAQ hours only'}")
    email_status = f"{GREEN}enabled → {email_cfg.to}{RESET}" if email_cfg else f"{GREY}disabled{RESET}"
    print(f"  Email    : {email_status}")
    print("  Press Ctrl-C to stop.")
    print("="*60 + "\n")

    from datetime import date
    today_str = date.today().isoformat()

    history = {}
    trained = {}
    states  = {}   # ticker -> StrategyState

    for ticker in tickers:
        print(f"[STARTUP] Downloading history for {ticker}...")
        try:
            history[ticker] = download_stock_data(ticker, start="2015-01-01", end=today_str)
            print(f"[STARTUP] Training models for {ticker}...")
            trained[ticker] = train_models_on_history(ticker, history[ticker])
            states[ticker]  = StrategyState(ticker=ticker)
        except Exception as e:
            print(f"[ERROR] Could not initialise {ticker}: {e}")

    if not trained:
        print("[ERROR] No tickers could be initialised. Exiting.")
        return

    print(f"\n{GREEN}[READY] Strategy is live. Watching: {', '.join(trained.keys())}{RESET}\n")

    try:
        while True:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not skip_market_check and not is_market_open():
                secs = seconds_until_open()
                hrs  = int(secs // 3600)
                mins = int((secs % 3600) // 60)
                print(f"[{now_str}] Market closed — next open in {hrs}h {mins}m. Sleeping...")
                for _ in range(min(int(secs), 300)):
                    time.sleep(1)
                continue

            for ticker in trained:
                try:
                    poll_once(ticker, history[ticker], trained[ticker], states[ticker], email_cfg)
                except Exception as e:
                    print(f"  [{ticker}] Poll error: {e}")

            print(f"\n  Next check in {interval_minutes} minute(s)...  [{now_str}]")
            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}[MONITOR] Stopped by user.{RESET}")
        # Print final session summary
        print(f"\n{'='*60}")
        print("  SESSION SUMMARY")
        print(f"{'='*60}")
        for ticker, state in states.items():
            print(f"\n  {ticker}")
            if state.trade_log:
                total_pnl = sum(t["pnl%"] for t in state.trade_log)
                wins      = sum(1 for t in state.trade_log if t["pnl%"] > 0)
                print(f"  Completed trades : {len(state.trade_log)}")
                print(f"  Win rate         : {wins}/{len(state.trade_log)}")
                print(f"  Total P&L        : {'+' if total_pnl >= 0 else ''}{total_pnl:.2f}%")
            else:
                status = "LONG (open position)" if state.position == Position.LONG else "FLAT"
                print(f"  No completed trades this session  |  Status: {status}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Live trading strategy monitor.")
    parser.add_argument("--ticker", nargs="+", default=TICKERS, metavar="TICKER")
    parser.add_argument("--interval", type=int, default=5, metavar="MINUTES")
    parser.add_argument("--no-market-check", action="store_true")
    parser.add_argument(
        "--email",
        action="store_true",
        help="Send email alerts on BUY/SELL signals. "
             "Requires SMTP_USER, SMTP_PASSWORD, and ALERT_TO env vars.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_monitor(
        tickers=[t.upper() for t in args.ticker],
        interval_minutes=args.interval,
        skip_market_check=args.no_market_check,
        email_alerts=args.email,
    )


if __name__ == "__main__":
    main()
