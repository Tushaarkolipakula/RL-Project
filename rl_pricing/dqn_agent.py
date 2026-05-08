from __future__ import annotations

import pickle
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

import numpy as np

from rl_pricing.config import PricingConfig
from rl_pricing.mdp import Observation, apply_action, expected_reward
from rl_pricing.policies import RateTrackingPolicy
from rl_pricing.state import encode_features


# ─────────────────────────────────────────────────────────────────────────────
# N-step replay buffer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Transition:
    state:      np.ndarray   # s_t,   shape (5,)
    action:     int          # a_t
    ret:        float        # n-step return G_t
    next_state: np.ndarray   # s_{t+n}, shape (5,)
    done:       bool         # whether episode ended within n steps


class _NStepBuffer:
    """Collects transitions within an episode and emits n-step returns.

    Calling push() returns a completed _Transition once the deque is full,
    or None while it is still filling up. flush() drains remaining transitions
    at episode end.
    """

    def __init__(self, n: int, gamma: float):
        self.n     = n
        self.gamma = gamma
        self._buf: deque = deque()   # (state, action, reward, next_state, done)

    def push(
        self,
        s: np.ndarray, a: int, r: float,
        s_next: np.ndarray, done: bool,
    ) -> list[_Transition]:
        self._buf.append((s, a, r, s_next, done))
        return self._drain(force=False)

    def flush(self) -> list[_Transition]:
        return self._drain(force=True)

    def reset(self) -> None:
        self._buf.clear()

    def _drain(self, force: bool) -> list[_Transition]:
        out: list[_Transition] = []
        while len(self._buf) >= self.n or (force and self._buf):
            # Compute n-step return from the oldest entry
            n_avail = min(self.n, len(self._buf))
            G = 0.0
            last_next = self._buf[-1][3]
            last_done = False
            for k, (_, _, rk, snk, dk) in enumerate(list(self._buf)[:n_avail]):
                G += (self.gamma ** k) * rk
                last_next = snk
                last_done = dk
                if dk:
                    break
            s0, a0, _, _, _ = self._buf[0]
            out.append(_Transition(
                state      = s0,
                action     = a0,
                ret        = G,
                next_state = last_next,
                done       = last_done,
            ))
            self._buf.popleft()
        return out


class _ReplayBuffer:
    """Fixed-capacity circular replay buffer storing completed transitions."""

    def __init__(self, capacity: int):
        self._buf: Deque[_Transition] = deque(maxlen=capacity)

    def push(self, t: _Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> list[_Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-NumPy MLP
# ─────────────────────────────────────────────────────────────────────────────

class _MLP:
    """Two-hidden-layer MLP, ReLU activations, Adam optimiser, float64."""

    def __init__(
        self,
        input_dim: int,
        h1: int,
        h2: int,
        output_dim: int,
        rng: np.random.Generator,
        bias_init: np.ndarray | None = None,   # shape (output_dim,) — warm start
    ):
        def _he(fan_in: int, fan_out: int) -> np.ndarray:
            lim = np.sqrt(6.0 / fan_in)
            return rng.uniform(-lim, lim, (fan_in, fan_out))

        self.W1 = _he(input_dim, h1)
        self.b1 = np.zeros(h1)
        self.W2 = _he(h1, h2)
        self.b2 = np.zeros(h2)
        self.W3 = _he(h2, output_dim)
        # Warm-start output biases to expected reward per action
        self.b3 = bias_init.copy() if bias_init is not None else np.zeros(output_dim)

        self._t: int = 0
        self._m = {n: np.zeros_like(p) for n, p in self._named_params()}
        self._v = {n: np.zeros_like(p) for n, p in self._named_params()}

    def _named_params(self):
        return [
            ("W1", self.W1), ("b1", self.b1),
            ("W2", self.W2), ("b2", self.b2),
            ("W3", self.W3), ("b3", self.b3),
        ]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        Z1 = X  @ self.W1 + self.b1
        A1 = np.maximum(0.0, Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = np.maximum(0.0, Z2)
        Q  = A2 @ self.W3 + self.b3
        return Q, {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    def predict(self, X: np.ndarray) -> np.ndarray:
        Q, _ = self.forward(X)
        return Q

    def update(
        self,
        cache: dict,
        dQ: np.ndarray,
        lr: float,
        max_grad_norm: float = 10.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        B = cache["X"].shape[0]
        X, Z1, A1, Z2, A2 = (cache[k] for k in ("X", "Z1", "A1", "Z2", "A2"))

        dW3 = (A2.T @ dQ) / B;  db3 = dQ.mean(0);  dA2 = dQ @ self.W3.T
        dZ2 = dA2 * (Z2 > 0);   dW2 = (A1.T @ dZ2) / B; db2 = dZ2.mean(0); dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0);   dW1 = (X.T  @ dZ1) / B; db1 = dZ1.mean(0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

        # Global gradient norm clipping
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if total_norm > max_grad_norm:
            scale = max_grad_norm / (total_norm + 1e-12)
            grads = {k: v * scale for k, v in grads.items()}

        self._t += 1
        for name, param in self._named_params():
            g = grads[name]
            self._m[name] = 0.9 * self._m[name] + 0.1 * g
            self._v[name] = beta2 * self._v[name] + (1 - beta2) * g * g
            m_hat = self._m[name] / (1 - 0.9  ** self._t)
            v_hat = self._v[name] / (1 - beta2 ** self._t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_weights_from(self, src: "_MLP") -> None:
        for name, param in self._named_params():
            param[:] = getattr(src, name)

    def polyak_update_from(self, src: "_MLP", tau: float) -> None:
        """θ_self ← τ·θ_src + (1-τ)·θ_self  (soft target update)."""
        for name, param in self._named_params():
            param[:] = tau * getattr(src, name) + (1.0 - tau) * param

    def state_dict(self) -> dict:
        return (
            {n: p.copy() for n, p in self._named_params()}
            | {"_t": self._t, "_m": dict(self._m), "_v": dict(self._v)}
        )

    def load_state_dict(self, d: dict) -> None:
        for name, _ in self._named_params():
            getattr(self, name)[:] = d[name]
        self._t = d["_t"]
        self._m = d["_m"]
        self._v = d["_v"]


# ─────────────────────────────────────────────────────────────────────────────
# Q-value warm-start initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_bias_init(cfg: PricingConfig) -> np.ndarray:
    """Compute output-layer bias init = E[expected_reward(rate_after_action)].

    Averages expected reward over the credit/demand distribution for each of
    the five actions, starting from the optimal rate (same prior as Q-table).
    This mirrors what agents.py does for tabular Q initialisation.
    """
    # Approximate starting rate: midpoint of r_min and r_max
    start_rate = (cfg.r_min + cfg.r_max) / 2.0
    biases = np.zeros(cfg.n_actions)
    credits = cfg.credit_categories
    demands = cfg.demand_levels
    c_probs = cfg.credit_probs
    d_probs = cfg.demand_probs

    for a in range(cfg.n_actions):
        rate = float(np.clip(start_rate + cfg.actions[a], cfg.r_min, cfg.r_max))
        ev = 0.0
        for c, cp in zip(credits, c_probs):
            for d, dp in zip(demands, d_probs):
                er = expected_reward(rate, c, d, cfg).expected_reward
                ev += cp * dp * er
        biases[a] = ev
    return biases


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent(RateTrackingPolicy):

    name = "dqn"

    def __init__(
        self,
        config: PricingConfig | None = None,
        seed: int = 0,
        lr: float | None = None,
    ):
        super().__init__(config)
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)

        cfg = self.config
        self.lr            = cfg.dqn_lr if lr is None else lr
        self.gamma         = cfg.discount
        self.epsilon       = cfg.dqn_epsilon_start
        self.epsilon_min   = cfg.dqn_epsilon_min
        self.epsilon_decay = cfg.dqn_epsilon_decay
        self.tau           = cfg.dqn_tau           # soft update coefficient
        self.n_step        = cfg.dqn_n_step        # n-step return horizon

        state_dim = 5   # encode_features() output dimension
        h1, h2    = cfg.dqn_hidden1, cfg.dqn_hidden2
        n_act     = cfg.n_actions   # 5

        bias_init = _compute_bias_init(cfg)
        self.online_net = _MLP(state_dim, h1, h2, n_act, self.rng, bias_init=bias_init)
        self.target_net = _MLP(state_dim, h1, h2, n_act, self.rng, bias_init=bias_init)

        self.replay      = _ReplayBuffer(cfg.dqn_replay_capacity)
        self.n_step_buf  = _NStepBuffer(self.n_step, self.gamma)
        self.batch_size  = cfg.dqn_batch_size
        self.grad_steps  = cfg.dqn_grad_steps
        self.warmup_steps = cfg.dqn_warmup_steps

        self._step_count:    int = 0
        self._episode_count: int = 0
        self._last_feat:     np.ndarray | None = None
        self._last_action:   int | None = None

    # ── RateTrackingPolicy interface ──────────────────────────────────────────

    def start_episode(self, obs: Observation) -> None:
        super().start_episode(obs)
        self.n_step_buf.reset()
        self._last_feat   = None
        self._last_action = None

    def act(self, obs: Observation, greedy: bool = False) -> int:
        rate_before = self.current_rate
        s = encode_features(obs, rate_before, self.config)

        if not greedy and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(self.config.n_actions))
        else:
            Q = self.online_net.predict(s[np.newaxis, :])
            action = int(np.argmax(Q[0]))

        if not greedy:
            self._last_feat   = s
            self._last_action = action

        self.current_rate = apply_action(rate_before, action, self.config)
        return action

    def observe(
        self,
        obs: Observation,
        action: int,
        reward: float,
        next_obs: Observation,
        done: bool,
    ) -> None:
        if self._last_feat is None:
            return

        next_feat = encode_features(next_obs, self.current_rate, self.config)

        # Collect n-step transitions
        completed = self.n_step_buf.push(
            self._last_feat, self._last_action, reward, next_feat, done
        )
        if done:
            completed += self.n_step_buf.flush()

        for t in completed:
            self.replay.push(t)

        self._last_feat   = None
        self._last_action = None
        self._step_count += 1

        if self._step_count >= self.warmup_steps and len(self.replay) >= self.batch_size:
            for _ in range(self.grad_steps):
                self._learn()
            # Soft target update every step
            self.target_net.polyak_update_from(self.online_net, self.tau)

        if done:
            self._episode_count += 1

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Learning step ─────────────────────────────────────────────────────────

    def _learn(self) -> None:
        batch = self.replay.sample(self.batch_size)
        B = len(batch)

        S      = np.stack([t.state      for t in batch])
        A      = np.array([t.action     for t in batch], dtype=np.int64)
        R      = np.array([t.ret        for t in batch])
        S_next = np.stack([t.next_state for t in batch])
        Done   = np.array([t.done       for t in batch], dtype=np.float64)

        # ── Reward normalisation (per mini-batch) ─────────────────────────────
        r_std = R.std()
        if r_std > 1e-8:
            R = (R - R.mean()) / (r_std + 1e-8)

        # ── Double-DQN target ─────────────────────────────────────────────────
        Q_online, cache  = self.online_net.forward(S)
        Q_next_online    = self.online_net.predict(S_next)
        next_actions     = np.argmax(Q_next_online, axis=1)
        Q_next_target    = self.target_net.predict(S_next)
        Q_next_val       = Q_next_target[np.arange(B), next_actions]

        # γ^n already baked into the n-step return; bootstrapped term uses γ^n
        gamma_n = self.gamma ** self.n_step
        target  = R + gamma_n * Q_next_val * (1.0 - Done)

        # ── Huber loss gradient ───────────────────────────────────────────────
        Q_taken = Q_online[np.arange(B), A]
        td_err  = Q_taken - target
        delta   = 1.0
        huber_g = np.where(np.abs(td_err) <= delta, td_err, delta * np.sign(td_err))

        dQ = np.zeros_like(Q_online)
        dQ[np.arange(B), A] = huber_g / B

        self.online_net.update(cache, dQ, lr=self.lr)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        payload = {
            "config":      self.config,
            "lr":          self.lr,
            "epsilon":     self.epsilon,
            "online_net":  self.online_net.state_dict(),
            "_step_count": self._step_count,
            "_ep_count":   self._episode_count,
        }
        with Path(path).open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "DQNAgent":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        agent = cls(config=payload["config"], seed=seed, lr=payload["lr"])
        agent.online_net.load_state_dict(payload["online_net"])
        agent.target_net.copy_weights_from(agent.online_net)
        agent.epsilon        = payload["epsilon"]
        agent._step_count    = payload["_step_count"]
        agent._episode_count = payload["_ep_count"]
        return agent
