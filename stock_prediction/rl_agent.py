"""
rl_agent.py
-----------
Reinforcement Learning agent for stock trading using a Deep Q-Network (DQN).

Instead of learning "did the stock go up today?" (a supervised classification
problem), the RL agent learns a *policy*: given the current market state and
my current portfolio position, what action maximises my long-run returns?

Core concepts
-------------
  Environment   The market simulator.  At each step the agent sees a state
                (feature vector + current position) and picks an action.

  Actions       0 = move to cash (sell if currently holding)
                1 = go long      (buy if currently in cash, else hold)

  Reward        The daily portfolio return that results from the action:
                  reward = action * daily_return - tx_cost * |action - prev_action|
                Transaction costs are deducted on the days the agent switches
                positions (0→1 or 1→0), making it learn to avoid thrashing.

  Q-function    A small 2-layer neural network (pure NumPy — no external DL
                framework required) that estimates the expected future return
                for every (state, action) pair.

  Training      Epsilon-greedy exploration + experience replay + a separate
                target network for stable Bellman targets.

Why RL instead of (or in addition to) supervised learning?
----------------------------------------------------------
  Supervised models optimise accuracy — they penalise every wrong prediction
  equally.  The RL agent optimises *returns* — it learns that missing a small
  up-day is cheap, but holding through a large crash is very expensive.  This
  aligns the training objective directly with the trading goal.

Usage (from main.py)
---------------------
  from stock_prediction.rl_agent import train_dqn_agent

  agent = train_dqn_agent(X_train, daily_returns_train)
  signals = agent.predict(X_test)          # drop-in for model.predict()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple


# ---------------------------------------------------------------------------
# Trading Environment
# ---------------------------------------------------------------------------

class TradingEnv:
    """
    A minimal Markov Decision Process (MDP) wrapping the time-series data.

    State
    -----
    Concatenation of the current day's feature vector and the agent's
    current portfolio position (0 = cash, 1 = long).  Adding the position
    to the state lets the agent reason about transaction costs — it knows
    whether switching will incur a cost.

    Transition
    ----------
    Deterministic: step t → step t+1 in calendar order.

    Episode
    -------
    One full pass through the training time series (start → end).
    """

    def __init__(
        self,
        feature_array: np.ndarray,
        daily_returns: np.ndarray,
        transaction_cost: float = 0.0005,
    ):
        self.features = feature_array            # (T, n_features)
        self.daily_returns = daily_returns        # (T,)
        self.transaction_cost = transaction_cost
        self.n_steps = len(feature_array)
        self.n_features = feature_array.shape[1]
        self.state_size = self.n_features + 1    # +1 for position flag
        self.reset()

    def reset(self) -> np.ndarray:
        """Return to day 0, cash position, ready to start a new episode."""
        self.t = 0
        self.position = 0
        return self._state()

    def _state(self) -> np.ndarray:
        """Build the state vector for the current timestep."""
        return np.append(self.features[self.t], float(self.position))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one trading day.

        Parameters
        ----------
        action : 0 (cash) or 1 (long)

        Returns
        -------
        next_state : np.ndarray
        reward     : float  — daily portfolio return minus any transaction cost
        done       : bool   — True when the episode is over
        """
        daily_ret = self.daily_returns[self.t]
        tx = self.transaction_cost * abs(action - self.position)
        reward = action * daily_ret - tx

        self.position = action
        self.t += 1
        done = self.t >= self.n_steps - 1

        next_state = self._state() if not done else np.zeros(self.state_size)
        return next_state, reward, done


# ---------------------------------------------------------------------------
# Neural Network — Q-function approximator (pure NumPy)
# ---------------------------------------------------------------------------

class _QNetwork:
    """
    Two-layer fully-connected network with ReLU hidden activation.

    Input  → Hidden (ReLU) → Output (2 Q-values, one per action)

    Xavier initialisation keeps activations in a healthy range at the start
    of training.  Gradient descent with a fixed learning rate is simple but
    works well for the scale of this problem.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int, lr: float):
        self.lr = lr
        s1 = np.sqrt(2.0 / input_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * s1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_actions) * s2
        self.b2 = np.zeros(n_actions)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x shape: (batch, input_dim)."""
        self._x = x
        self._h = np.maximum(0.0, x @ self.W1 + self.b1)   # ReLU
        return self._h @ self.W2 + self.b2                   # Q-values

    def backward(self, grad_out: np.ndarray) -> None:
        """
        Backprop the MSE loss gradient and update weights in-place.

        grad_out shape: (batch, n_actions)
        """
        dW2 = self._h.T @ grad_out
        db2 = grad_out.sum(axis=0)
        dh  = grad_out @ self.W2.T
        dh_relu = dh * (self._h > 0)          # ReLU derivative
        dW1 = self._x.T @ dh_relu
        db1 = dh_relu.sum(axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def get_weights(self) -> tuple:
        return (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def set_weights(self, weights: tuple) -> None:
        self.W1, self.b1, self.W2, self.b2 = [w.copy() for w in weights]


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network agent with experience replay and a target network.

    Epsilon-greedy exploration
    --------------------------
    At the start of training epsilon=1.0, so the agent acts randomly and
    fills its replay buffer with diverse experiences.  Epsilon decays each
    step toward epsilon_end so the agent gradually exploits what it has
    learned rather than continuing to explore.

    Experience replay
    -----------------
    Instead of learning from (state, action, reward, next_state) tuples in
    the order they were collected — which would be highly correlated — we
    store them in a ring buffer and sample random mini-batches.  This breaks
    the temporal correlation and dramatically stabilises training.

    Target network
    --------------
    The Bellman update uses a *separate* copy of the Q-network that is only
    refreshed every `target_update_freq` steps.  Without this, the target
    values change every step, creating a moving-target problem that causes
    training to diverge.
    """

    N_ACTIONS = 2   # 0 = cash, 1 = long

    def __init__(
        self,
        state_size: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        replay_capacity: int = 10_000,
        target_update_freq: int = 50,
        random_state: int = 42,
    ):
        np.random.seed(random_state)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._step = 0

        self.q_net      = _QNetwork(state_size, hidden_dim, self.N_ACTIONS, lr)
        self.target_net = _QNetwork(state_size, hidden_dim, self.N_ACTIONS, lr)
        self.target_net.set_weights(self.q_net.get_weights())

        self.memory: deque = deque(maxlen=replay_capacity)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy: explore randomly with probability epsilon,
        otherwise take the greedy action (highest Q-value).
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        q = self.q_net.forward(state.reshape(1, -1))
        return int(np.argmax(q))

    # ------------------------------------------------------------------
    # Memory and learning
    # ------------------------------------------------------------------

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> float | None:
        """
        Sample a random mini-batch, compute Bellman targets, and update
        the Q-network.  Returns the mean batch loss, or None if the buffer
        is not yet large enough.
        """
        if len(self.memory) < self.batch_size:
            return None

        idx   = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in idx]

        states      = np.array([b[0] for b in batch])
        actions     = np.array([b[1] for b in batch], dtype=int)
        rewards     = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones       = np.array([b[4] for b in batch], dtype=float)

        # Current Q-values
        q_current = self.q_net.forward(states)

        # Bellman targets using the frozen target network
        q_next    = self.target_net.forward(next_states)
        q_target  = q_current.copy()
        for i in range(self.batch_size):
            td_target = rewards[i] if dones[i] else rewards[i] + self.gamma * np.max(q_next[i])
            q_target[i, actions[i]] = td_target

        # MSE loss gradient  dL/dQ = 2*(Q - target) / batch_size
        loss_grad = 2.0 * (q_current - q_target) / self.batch_size
        self.q_net.backward(loss_grad)
        loss = float(np.mean((q_current - q_target) ** 2))

        # Decay epsilon — less random as the agent learns more
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Sync target network periodically
        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.target_net.set_weights(self.q_net.get_weights())

        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray, initial_position: int = 0) -> np.ndarray:
        """
        Run the greedy (epsilon=0) policy over a feature array.

        The agent tracks its own position between steps so it can correctly
        account for transaction costs in its Q-value estimates.

        Compatible with the sklearn .predict() interface so it can be passed
        directly to run_backtest() and evaluate_model().

        Parameters
        ----------
        X                 : feature array (n_samples, n_features)
        initial_position  : starting position (0 = cash, 1 = long)

        Returns
        -------
        signals : np.ndarray of shape (n_samples,), values in {0, 1}
        """
        signals  = np.zeros(len(X), dtype=int)
        position = initial_position
        for t, features in enumerate(X):
            state   = np.append(features, float(position))
            q       = self.q_net.forward(state.reshape(1, -1))
            action  = int(np.argmax(q))
            signals[t] = action
            position   = action
        return signals


# ---------------------------------------------------------------------------
# Convenience training function (called from main.py)
# ---------------------------------------------------------------------------

def train_dqn_agent(
    X_train: np.ndarray,
    daily_returns_train: np.ndarray,
    n_episodes: int = 15,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    transaction_cost: float = 0.0005,
    random_state: int = 42,
) -> DQNAgent:
    """
    Train a DQN agent on the training split and return the fitted agent.

    Each *episode* is one full sequential pass through the training data.
    Running multiple episodes lets the agent revisit early experiences and
    refine its policy using the knowledge gained later in the series.

    Parameters
    ----------
    X_train              : feature array, shape (n_samples, n_features)
    daily_returns_train  : daily close-to-close returns aligned to X_train
    n_episodes           : number of full passes through the training data
    hidden_dim           : Q-network hidden layer width
    lr                   : gradient descent learning rate
    transaction_cost     : fraction of trade value charged on position changes
    random_state         : seed for reproducibility

    Returns
    -------
    Trained DQNAgent ready for .predict() inference.
    """
    env = TradingEnv(X_train, daily_returns_train, transaction_cost)
    agent = DQNAgent(
        state_size=env.state_size,
        hidden_dim=hidden_dim,
        lr=lr,
        random_state=random_state,
    )

    print(f"  Training DQN agent — {n_episodes} episodes, "
          f"hidden_dim={hidden_dim}, lr={lr}")

    for ep in range(n_episodes):
        state        = env.reset()
        total_reward = 0.0
        losses       = []

        while True:
            action                       = agent.act(state)
            next_state, reward, done     = env.step(action)
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
            f"    ep {ep + 1:>2}/{n_episodes}  "
            f"reward={total_reward:+.4f}  "
            f"ε={agent.epsilon:.3f}  "
            f"loss={avg_loss:.6f}"
        )

    return agent
