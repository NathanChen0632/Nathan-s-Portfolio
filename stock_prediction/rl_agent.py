"""
rl_agent.py
-----------
Deep Q-Network (DQN) agent with a disciplined trading strategy built
directly into the environment's reward function and state space.

Trading parameters encoded in the environment
---------------------------------------------
  Risk per trade    1% of account per trade (position sizing)
  Min R/R ratio     2:1 — the agent is penalised for entering setups
                    where potential gain < 2x potential loss
  Stop loss         ATR-based: entry - (stop_atr_mult * ATR14)
  Target            entry + (min_rr_ratio * stop_distance)
  Volume gate       Penalises entries on below-average volume
  MA proximity      Rewards pullback entries near the 20-day MA
  Holding period    Time-decay penalty after day 5; forced exit at day 10
  Transaction cost  0.05% charged on every position change

State space (n_features + 5 extra context dimensions)
------------------------------------------------------
  [original features] + [
    position           (0=cash, 1=long),
    days_held_norm     (days held / max_holding_days),
    unrealized_pnl     (% gain/loss since entry),
    stop_distance_norm (how far price is from stop as fraction of risk),
    target_distance_norm (how far price is from target),
  ]

The agent therefore knows not just market conditions but also
where it stands inside a live trade — making sell decisions much
more informed than a stateless policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Trading configuration
# ---------------------------------------------------------------------------

@dataclass
class TradingConfig:
    """
    All disciplined-trading parameters in one place.
    Change here to instantly affect every part of the pipeline.
    """
    account_size:       float = 10_000.0  # starting capital ($)
    risk_per_trade:     float = 0.01      # max 1% of account at risk per trade
    min_rr_ratio:       float = 2.0       # minimum reward:risk ratio to enter
    max_holding_days:   int   = 10        # forced exit after N days
    stop_atr_mult:      float = 2.0       # stop = entry - mult * ATR14
    min_volume_ratio:   float = 0.8       # minimum vol/avg_vol to confirm entry
    ma_proximity_bonus: float = 0.001     # extra reward for pullback near MA20
    time_decay_start:   int   = 5         # days after which time penalty starts
    time_decay_rate:    float = 0.0001    # reward penalty per extra day held
    transaction_cost:   float = 0.0005   # 0.05% per trade


# ---------------------------------------------------------------------------
# Trading Environment
# ---------------------------------------------------------------------------

class TradingEnv:
    """
    Market simulator that enforces all trading strategy parameters.

    Key feature indices are looked up from feature_cols at init time
    so the environment can extract ATR, volume ratio, and MA20 ratio
    directly from the feature vector without extra arrays.
    """

    def __init__(
        self,
        feature_array:  np.ndarray,
        daily_returns:  np.ndarray,
        prices:         np.ndarray,
        feature_cols:   list,
        config:         TradingConfig | None = None,
    ):
        self.features      = feature_array
        self.daily_returns = daily_returns
        self.prices        = prices
        self.feature_cols  = feature_cols
        self.cfg           = config or TradingConfig()

        # Look up feature indices needed by the reward function
        self._atr_idx  = feature_cols.index("atr14_pct")   if "atr14_pct"    in feature_cols else None
        self._vol_idx  = feature_cols.index("volume_ratio") if "volume_ratio" in feature_cols else None
        self._ma20_idx = feature_cols.index("ma20_ratio")   if "ma20_ratio"   in feature_cols else None

        self.n_steps    = len(feature_array)
        self.n_features = feature_array.shape[1]
        # 5 extra dimensions: position, days_norm, unrealized_pnl, stop_dist, target_dist
        self.state_size = self.n_features + 5
        self.reset()

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.t            = 0
        self.position     = 0
        self.entry_price  = 0.0
        self.stop_price   = 0.0
        self.target_price = 0.0
        self.days_held    = 0
        return self._build_state()

    def _build_state(self) -> np.ndarray:
        """Concatenate market features with current trade context."""
        days_norm = self.days_held / self.cfg.max_holding_days

        if self.position == 1 and self.entry_price > 0:
            curr          = self.prices[self.t]
            unrealized    = (curr - self.entry_price) / self.entry_price
            risk_dist     = self.entry_price - self.stop_price
            stop_dist     = (curr - self.stop_price)   / risk_dist  if risk_dist > 0 else 1.0
            target_range  = self.target_price - self.entry_price
            target_dist   = (self.target_price - curr) / target_range if target_range > 0 else 0.0
        else:
            unrealized = 0.0
            stop_dist  = 1.0
            target_dist = 1.0

        context = np.array([
            float(self.position),
            days_norm,
            unrealized,
            np.clip(stop_dist,   -2.0, 3.0),
            np.clip(target_dist, -1.0, 2.0),
        ])
        return np.concatenate([self.features[self.t], context])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        cfg         = self.cfg
        curr_price  = self.prices[self.t]
        daily_ret   = self.daily_returns[self.t]
        next_price  = curr_price * (1.0 + daily_ret)
        reward      = 0.0

        # ---------------------------------------------------------------
        # FLAT — agent wants to BUY
        # ---------------------------------------------------------------
        if self.position == 0 and action == 1:

            # ATR-based stop loss
            atr_pct      = self.features[self.t, self._atr_idx] if self._atr_idx is not None else 0.015
            stop_dist_$  = cfg.stop_atr_mult * atr_pct * curr_price
            stop_price   = curr_price - stop_dist_$
            target_price = curr_price + cfg.min_rr_ratio * stop_dist_$
            rr           = (target_price - curr_price) / stop_dist_$ if stop_dist_$ > 0 else 0.0

            if rr >= cfg.min_rr_ratio:
                # --- Valid R/R — enter the trade ---
                self.position     = 1
                self.entry_price  = curr_price
                self.stop_price   = stop_price
                self.target_price = target_price
                self.days_held    = 0
                reward           -= cfg.transaction_cost    # entry cost

                # Volume confirmation
                if self._vol_idx is not None:
                    vol = self.features[self.t, self._vol_idx]
                    if vol < cfg.min_volume_ratio:
                        reward -= 0.001   # below-average volume — weak signal
                    elif vol > 1.5:
                        reward += 0.0005  # strong institutional volume — bonus

                # MA20 pullback proximity bonus
                if self._ma20_idx is not None:
                    ma20_ratio = self.features[self.t, self._ma20_idx]
                    if 0.98 <= ma20_ratio <= 1.03:
                        reward += cfg.ma_proximity_bonus   # ideal pullback entry
            else:
                # --- Bad R/R — penalise wanting to enter a substandard setup ---
                reward -= 0.0015

        # ---------------------------------------------------------------
        # LONG — agent wants to SELL (voluntary exit)
        # ---------------------------------------------------------------
        elif self.position == 1 and action == 0:
            pnl    = (curr_price - self.entry_price) / self.entry_price
            risk_$ = self.entry_price - self.stop_price
            size   = cfg.risk_per_trade / (risk_$ / self.entry_price) if risk_$ > 0 else 1.0
            reward = pnl * min(size, 5.0) - cfg.transaction_cost
            self._reset_position()

        # ---------------------------------------------------------------
        # LONG — agent wants to HOLD
        # ---------------------------------------------------------------
        elif self.position == 1 and action == 1:
            self.days_held += 1
            risk_$ = self.entry_price - self.stop_price
            size   = cfg.risk_per_trade / (risk_$ / self.entry_price) if risk_$ > 0 else 1.0

            if next_price <= self.stop_price:
                # Stop loss hit — fixed loss of risk_per_trade
                pnl    = (self.stop_price - self.entry_price) / self.entry_price
                reward = pnl * min(size, 5.0) - cfg.transaction_cost - 0.002  # extra stop penalty
                self._reset_position()

            elif next_price >= self.target_price:
                # Target hit — reward scaled by R/R ratio
                pnl    = (self.target_price - self.entry_price) / self.entry_price
                reward = pnl * min(size, 5.0) - cfg.transaction_cost + 0.002  # target bonus
                self._reset_position()

            elif self.days_held >= cfg.max_holding_days:
                # Time stop — forced exit
                pnl    = (curr_price - self.entry_price) / self.entry_price
                reward = pnl * min(size, 5.0) - cfg.transaction_cost - 0.001
                self._reset_position()

            else:
                # Normal hold day
                reward = daily_ret * min(size, 5.0)
                # Time decay to discourage holding too long
                if self.days_held > cfg.time_decay_start:
                    reward -= cfg.time_decay_rate * (self.days_held - cfg.time_decay_start)

        # ---------------------------------------------------------------
        # FLAT — agent stays in cash (no reward)
        # ---------------------------------------------------------------
        # else: reward stays 0.0

        self.t += 1
        done       = self.t >= self.n_steps - 1
        next_state = self._build_state() if not done else np.zeros(self.state_size)
        return next_state, reward, done

    def _reset_position(self):
        self.position     = 0
        self.entry_price  = 0.0
        self.stop_price   = 0.0
        self.target_price = 0.0
        self.days_held    = 0


# ---------------------------------------------------------------------------
# Q-network (pure NumPy)
# ---------------------------------------------------------------------------

class _QNetwork:
    """Two-layer fully-connected net with ReLU. Xavier init."""

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int, lr: float):
        self.lr = lr
        s1 = np.sqrt(2.0 / input_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * s1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_actions) * s2
        self.b2 = np.zeros(n_actions)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._h = np.maximum(0.0, x @ self.W1 + self.b1)
        return self._h @ self.W2 + self.b2

    def backward(self, grad_out: np.ndarray) -> None:
        dW2 = self._h.T @ grad_out
        db2 = grad_out.sum(axis=0)
        dh  = (grad_out @ self.W2.T) * (self._h > 0)
        dW1 = self._x.T @ dh
        db1 = dh.sum(axis=0)
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2

    def get_weights(self):
        return (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def set_weights(self, w):
        self.W1, self.b1, self.W2, self.b2 = [x.copy() for x in w]


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """DQN with experience replay and target network."""

    N_ACTIONS = 2

    def __init__(
        self,
        state_size:         int,
        hidden_dim:         int   = 128,
        lr:                 float = 1e-3,
        gamma:              float = 0.99,
        epsilon_start:      float = 1.0,
        epsilon_end:        float = 0.05,
        epsilon_decay:      float = 0.995,
        batch_size:         int   = 64,
        replay_capacity:    int   = 10_000,
        target_update_freq: int   = 50,
        random_state:       int   = 42,
    ):
        np.random.seed(random_state)
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self._step              = 0
        self.q_net      = _QNetwork(state_size, hidden_dim, self.N_ACTIONS, lr)
        self.target_net = _QNetwork(state_size, hidden_dim, self.N_ACTIONS, lr)
        self.target_net.set_weights(self.q_net.get_weights())
        self.memory: deque = deque(maxlen=replay_capacity)

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        return int(np.argmax(self.q_net.forward(state.reshape(1, -1))))

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def replay(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        idx         = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch       = [self.memory[i] for i in idx]
        states      = np.array([b[0] for b in batch])
        actions     = np.array([b[1] for b in batch], dtype=int)
        rewards     = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones       = np.array([b[4] for b in batch], dtype=float)

        q_current = self.q_net.forward(states)
        q_next    = self.target_net.forward(next_states)
        q_target  = q_current.copy()
        for i in range(self.batch_size):
            td = rewards[i] if dones[i] else rewards[i] + self.gamma * np.max(q_next[i])
            q_target[i, actions[i]] = td

        grad = 2.0 * (q_current - q_target) / self.batch_size
        self.q_net.backward(grad)
        loss = float(np.mean((q_current - q_target) ** 2))

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._step  += 1
        if self._step % self.target_update_freq == 0:
            self.target_net.set_weights(self.q_net.get_weights())
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        X:                np.ndarray,
        prices:           np.ndarray | None = None,
        feature_cols:     list | None       = None,
        config:           TradingConfig | None = None,
        initial_position: int = 0,
    ) -> np.ndarray:
        """
        Run greedy policy over a feature array.

        If prices and feature_cols are supplied the full trading environment
        is simulated (stops, targets, time rules all apply).
        Otherwise falls back to simple Q-argmax with position tracking.
        """
        if prices is not None and feature_cols is not None:
            return self._predict_with_env(X, prices, feature_cols, config, initial_position)
        return self._predict_simple(X, initial_position)

    def _predict_with_env(self, X, prices, feature_cols, config, initial_position):
        """Full environment simulation at inference time."""
        cfg = config or TradingConfig()
        # Dummy returns array (not used for reward during inference)
        dummy_returns = np.zeros(len(X))
        env = TradingEnv(X, dummy_returns, prices, feature_cols, cfg)
        env.position = initial_position

        signals  = np.zeros(len(X), dtype=int)
        state    = env._build_state()
        for t in range(len(X) - 1):
            action           = int(np.argmax(self.q_net.forward(state.reshape(1, -1))))
            next_state, _, _ = env.step(action)
            signals[t]       = env.position   # record resulting position (after trading rules)
            state            = next_state
        signals[-1] = env.position
        return signals

    def _predict_simple(self, X, initial_position):
        """Fallback: greedy argmax, tracks position without trading rules."""
        signals  = np.zeros(len(X), dtype=int)
        position = initial_position
        for t, features in enumerate(X):
            state      = np.append(features, [float(position), 0.0, 0.0, 1.0, 1.0])
            action     = int(np.argmax(self.q_net.forward(state.reshape(1, -1))))
            signals[t] = action
            position   = action
        return signals

    def predict_step(
        self,
        features:      np.ndarray,
        position:      int,
        days_held:     int,
        entry_price:   float,
        stop_price:    float,
        target_price:  float,
        current_price: float,
        config:        TradingConfig | None = None,
    ) -> int:
        """
        Single-step prediction for live trading.
        The caller supplies full trade context so the agent can make
        a properly informed hold/sell decision.
        """
        cfg       = config or TradingConfig()
        days_norm = days_held / cfg.max_holding_days

        if position == 1 and entry_price > 0:
            unrealized  = (current_price - entry_price) / entry_price
            risk_dist   = entry_price - stop_price
            stop_dist   = (current_price - stop_price) / risk_dist if risk_dist > 0 else 1.0
            target_rng  = target_price - entry_price
            target_dist = (target_price - current_price) / target_rng if target_rng > 0 else 0.0
        else:
            unrealized = stop_dist = 0.0
            target_dist = 1.0

        state = np.append(features, [
            float(position), days_norm,
            unrealized,
            np.clip(stop_dist,   -2.0, 3.0),
            np.clip(target_dist, -1.0, 2.0),
        ])
        return int(np.argmax(self.q_net.forward(state.reshape(1, -1))))


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_dqn_agent(
    X_train:             np.ndarray,
    daily_returns_train: np.ndarray,
    prices_train:        np.ndarray | None  = None,
    feature_cols:        list | None        = None,
    config:              TradingConfig | None = None,
    n_episodes:          int   = 50,
    hidden_dim:          int   = 128,
    lr:                  float = 1e-3,
    transaction_cost:    float = 0.0005,
    random_state:        int   = 42,
) -> DQNAgent:
    """
    Train a DQN agent using the full disciplined trading environment.

    If prices_train and feature_cols are provided the agent learns with
    ATR stops, R/R gating, volume confirmation, MA proximity, and
    holding-period rules active.  Otherwise it falls back to the basic
    daily-return environment.
    """
    cfg = config or TradingConfig()

    if prices_train is not None and feature_cols is not None:
        env = TradingEnv(X_train, daily_returns_train, prices_train, feature_cols, cfg)
        mode = "disciplined (R/R + stops + volume + MA)"
    else:
        # Fallback: basic environment without trading rules
        from collections import namedtuple
        env = _BasicTradingEnv(X_train, daily_returns_train, transaction_cost)
        mode = "basic (daily return only)"

    agent = DQNAgent(
        state_size=env.state_size,
        hidden_dim=hidden_dim,
        lr=lr,
        random_state=random_state,
    )

    print(f"  Training DQN — {n_episodes} episodes | mode: {mode}")
    print(f"  State size: {env.state_size} | Hidden: {hidden_dim}")

    for ep in range(n_episodes):
        state        = env.reset()
        total_reward = 0.0
        losses       = []

        while True:
            action                   = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss is not None:
                losses.append(loss)
            total_reward += reward
            state         = next_state
            if done:
                break

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"    ep {ep+1:>2}/{n_episodes}  "
            f"reward={total_reward:+.4f}  "
            f"ε={agent.epsilon:.3f}  "
            f"loss={avg_loss:.6f}"
        )

    return agent


# ---------------------------------------------------------------------------
# Basic fallback environment (no trading rules)
# ---------------------------------------------------------------------------

class _BasicTradingEnv:
    """Original simple environment — used as fallback if prices not provided."""

    def __init__(self, feature_array, daily_returns, transaction_cost=0.0005):
        self.features         = feature_array
        self.daily_returns    = daily_returns
        self.transaction_cost = transaction_cost
        self.n_steps          = len(feature_array)
        self.n_features       = feature_array.shape[1]
        self.state_size       = self.n_features + 5   # match new state size
        self.reset()

    def reset(self):
        self.t        = 0
        self.position = 0
        return self._state()

    def _state(self):
        # Pad with zeros for the 5 extra context dims
        return np.append(self.features[self.t], [float(self.position), 0.0, 0.0, 1.0, 1.0])

    def step(self, action):
        daily_ret = self.daily_returns[self.t]
        tx        = self.transaction_cost * abs(action - self.position)
        reward    = action * daily_ret - tx
        self.position = action
        self.t       += 1
        done          = self.t >= self.n_steps - 1
        next_state    = self._state() if not done else np.zeros(self.state_size)
        return next_state, reward, done
