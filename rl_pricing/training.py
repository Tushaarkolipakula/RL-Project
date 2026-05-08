"""Training orchestration for learning agents.

Trains Q-Learning, Policy Gradient, and DQN agents, then evaluates all
five methods:
  1. Fixed Pricing
  2. Rule-Based
  3. Q-Learning
  4. Policy Gradient
  5. DQN  (Deep Q-Network — Double DQN variant)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import time

from rl_pricing.agents import PolicyGradientAgent, ReplayQLearningAgent
from rl_pricing.config import DEFAULT_OUTPUT_DIR, PricingConfig
from rl_pricing.dataset import build_and_save as generate_dataset
from rl_pricing.dqn_agent import DQNAgent
from rl_pricing.environment import LoanPricingEnv
from rl_pricing.evaluation import MetricSummary, evaluate_agent
from rl_pricing.policies import FixedRatePolicy, RuleBasedPolicy


@dataclass
class TrainingResult:
    seed: int
    method: str
    train_rewards: list[float]
    final_epsilon: float
    model_path: str


# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def train_q_learning_agent(
    seed: int,
    config: PricingConfig,
    train_episodes: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[ReplayQLearningAgent, TrainingResult]:
    n_train = config.train_episodes if train_episodes is None else train_episodes
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    env   = LoanPricingEnv(config=config, seed=seed)
    agent = ReplayQLearningAgent(config=config, seed=seed)
    train_rewards: list[float] = []

    for _episode in range(n_train):
        obs = env.reset()
        agent.start_episode(obs)
        done = False
        rewards: list[float] = []
        while not done:
            action = agent.act(obs, greedy=False)
            next_obs, reward, done = env.step(action)
            agent.observe(obs, action, reward, next_obs, done)
            rewards.append(float(reward))
            obs = next_obs
        agent.decay_epsilon()
        train_rewards.append(float(sum(rewards) / len(rewards)))

    model_path = out_dir / f"q_learning_seed{seed}.pkl"
    agent.save(model_path)
    return agent, TrainingResult(
        seed=seed, method="q_learning",
        train_rewards=train_rewards,
        final_epsilon=agent.epsilon,
        model_path=str(model_path),
    )


# Keep old alias
train_replay_q_agent = train_q_learning_agent


# ─────────────────────────────────────────────────────────────────────────────
# Policy Gradient  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def train_policy_gradient_agent(
    seed: int,
    config: PricingConfig,
    train_episodes: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[PolicyGradientAgent, TrainingResult]:
    n_train = config.train_episodes if train_episodes is None else train_episodes
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    env   = LoanPricingEnv(config=config, seed=seed)
    agent = PolicyGradientAgent(config=config, seed=seed)
    train_rewards: list[float] = []

    for _episode in range(n_train):
        obs = env.reset()
        agent.start_episode(obs)
        done = False
        rewards: list[float] = []
        while not done:
            action = agent.act(obs, greedy=False)
            next_obs, reward, done = env.step(action)
            agent.observe(obs, action, reward, next_obs, done)
            rewards.append(float(reward))
            obs = next_obs
        train_rewards.append(float(sum(rewards) / len(rewards)))

    model_path = out_dir / f"policy_gradient_seed{seed}.pkl"
    agent.save(model_path)
    return agent, TrainingResult(
        seed=seed, method="policy_gradient",
        train_rewards=train_rewards,
        final_epsilon=0.0,
        model_path=str(model_path),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DQN  (new)
# ─────────────────────────────────────────────────────────────────────────────

def train_dqn_agent(
    seed: int,
    config: PricingConfig,
    train_episodes: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[DQNAgent, TrainingResult]:
    """Train one Double-DQN agent.

    The training loop is identical in structure to the Q-Learning loop so the
    experiment is a fair comparison:
      1. env.reset()  → agent.start_episode()
      2. agent.act()  → env.step()  → agent.observe()   (per step)
      3. agent.decay_epsilon()                           (per episode)

    The agent's observe() triggers internal gradient steps; the loop itself
    does not need to know about mini-batches or target-network updates.
    """
    n_train = config.train_episodes if train_episodes is None else train_episodes
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    env   = LoanPricingEnv(config=config, seed=seed)
    agent = DQNAgent(config=config, seed=seed)
    train_rewards: list[float] = []

    for _episode in range(n_train):
        obs = env.reset()
        agent.start_episode(obs)
        done = False
        rewards: list[float] = []
        while not done:
            action = agent.act(obs, greedy=False)
            next_obs, reward, done = env.step(action)
            agent.observe(obs, action, reward, next_obs, done)
            rewards.append(float(reward))
            obs = next_obs
        agent.decay_epsilon()
        train_rewards.append(float(sum(rewards) / len(rewards)))

    model_path = out_dir / f"dqn_seed{seed}.pkl"
    agent.save(model_path)
    return agent, TrainingResult(
        seed=seed, method="dqn",
        train_rewards=train_rewards,
        final_epsilon=agent.epsilon,
        model_path=str(model_path),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_full_experiment(
    config: PricingConfig,
    seeds: list[int] | None = None,
    train_episodes: int | None = None,
    eval_episodes: int | None = None,
    output_dir: str | Path | None = None,
    include_dqn: bool = True,
) -> dict:
    """Generate dataset, train all RL agents, evaluate all methods.

    Parameters
    ----------
    include_dqn : bool
        Set False to reproduce the original two-agent experiment exactly.
        Defaults to True so the full five-method comparison runs by default.
    """
    selected_seeds = list(config.seeds if seeds is None else seeds)
    n_train = config.train_episodes if train_episodes is None else train_episodes
    n_eval  = config.eval_episodes  if eval_episodes  is None else eval_episodes
    config  = replace(
        config,
        train_episodes=n_train,
        eval_episodes=n_eval,
        seeds=tuple(selected_seeds),
    )
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()

    # ── Step 1: Synthetic dataset ─────────────────────────────────────────────
    dataset_csv = out_dir / "data" / "synthetic_dataset.csv"
    if not dataset_csv.exists():
        print("Generating synthetic dataset …")
        generate_dataset(output_dir=out_dir, n_episodes=n_eval, config=config)
    else:
        print(f"Dataset already exists at {dataset_csv} — skipping generation.")

    # ── Step 2: Train Q-Learning ──────────────────────────────────────────────
    print("\nTraining Q-Learning agents …")
    q_agents: dict[int, ReplayQLearningAgent] = {}
    q_runs:   list[TrainingResult] = []
    for seed in selected_seeds:
        print(f"  seed {seed} …")
        agent, result = train_q_learning_agent(seed=seed, config=config, output_dir=out_dir)
        q_agents[seed] = agent
        q_runs.append(result)

    # ── Step 3: Train Policy Gradient ─────────────────────────────────────────
    print("\nTraining Policy Gradient agents …")
    pg_agents: dict[int, PolicyGradientAgent] = {}
    pg_runs:   list[TrainingResult] = []
    for seed in selected_seeds:
        print(f"  seed {seed} …")
        agent, result = train_policy_gradient_agent(seed=seed, config=config, output_dir=out_dir)
        pg_agents[seed] = agent
        pg_runs.append(result)

    # ── Step 4: Train DQN ─────────────────────────────────────────────────────
    dqn_agents: dict[int, DQNAgent] = {}
    dqn_runs:   list[TrainingResult] = []
    if include_dqn:
        print("\nTraining DQN agents …")
        for seed in selected_seeds:
            print(f"  seed {seed} …")
            agent, result = train_dqn_agent(seed=seed, config=config, output_dir=out_dir)
            dqn_agents[seed] = agent
            dqn_runs.append(result)

    # ── Step 5: Evaluate all methods ──────────────────────────────────────────
    print("\nEvaluating …")
    method_specs: list[tuple[str, object]] = [
        ("Fixed Pricing",   lambda seed: FixedRatePolicy(config=config)),
        ("Rule-Based",      lambda seed: RuleBasedPolicy(config=config)),
        ("Q-Learning",      lambda seed: q_agents[seed]),
        ("Policy Gradient", lambda seed: pg_agents[seed]),
    ]
    if include_dqn:
        method_specs.append(("DQN", lambda seed: dqn_agents[seed]))

    summaries: list[MetricSummary] = []
    for method, factory in method_specs:
        summary, _ = evaluate_agent(
            method=method,
            factory=factory,
            config=config,
            seeds=selected_seeds,
            episodes=None,
            greedy=True,
        )
        summaries.append(summary)
        print(f"  {method:<18} profit/step={summary.profit_mean:+.5f} ± {summary.profit_std:.4f}"
              f"  accept={summary.accept_mean:.3f}  rate={summary.rate_mean:.2f}%")

    # ── Step 6: Serialise ─────────────────────────────────────────────────────
    all_runs = q_runs + pg_runs + dqn_runs
    result = {
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in config.__dict__.items()},
        "runtime_seconds": time.time() - started,
        "training": [
            {
                "seed":          run.seed,
                "method":        run.method,
                "final_epsilon": run.final_epsilon,
                "model_path":    run.model_path,
                "train_rewards": run.train_rewards,
            }
            for run in all_runs
        ],
        "summaries": [s.as_dict() for s in summaries],
    }

    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["results_path"] = str(results_path)
    return result