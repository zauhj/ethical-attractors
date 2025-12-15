"""Minimal RL experiment for Ethical Attractors (fresh implementation).

We pit a tabular Q-learning agent against simple opponents in
iterated Prisoner's Dilemma payoff variants.

Payoff variants (R,T,S,P):
    prosocial        (3, 2, 0, 1)
    symmetric        (3, 5, 0, 1)
    competitive      (3, 6, 0, 1)
    reward_reversed  (3, 2, 0, 5)

Opponents:
    • tft     – Tit-for-tat (cooperate first, then mirror)
    • random  – Bernoulli(p=0.5)

Outputs JSON summary of mean cooperation rates for each pairing.
Designed for laptop-scale runtime (<30 s).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

Action = int  # 0=Cooperate, 1=Defect


PAYOFF_VARIANTS = {
    "prosocial": (3, 2, 0, 1),
    "symmetric": (3, 5, 0, 1),
    "competitive": (3, 6, 0, 1),
    "reward_reversed": (3, 2, 0, 5),
}


class QAgent:
    """Tabular Q-learning with epsilon-greedy decay."""
    """Tabular Q-learning with 1-step memory state."""

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.95,
        eps_start: float = 0.2,
        eps_end: float = 0.01,
        decay: float = 1e-5,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay = decay
        self.steps = 0
        # state: last joint action encoded 0..3 (CC, CD, DC, DD).
        self.q = np.zeros((4, 2))
        self.last_state = 0  # start with CC (assume coop)

    def _eps(self) -> float:
        return max(self.eps_end, self.eps_start - self.decay * self.steps)

    def act(self) -> Action:
        self.steps += 1
        if random.random() < self._eps():
            return random.randint(0, 1)
        return int(np.argmax(self.q[self.last_state]))

    def update(self, state: int, action: Action, reward: float, next_state: int):
        best_next = np.max(self.q[next_state])
        td = reward + self.gamma * best_next - self.q[state, action]
        self.q[state, action] += self.alpha * td
        self.last_state = next_state


class Opponent:
    @staticmethod
    def tft(history_self: List[Action], history_opp: List[Action]) -> Action:
        return 0 if not history_opp else history_opp[-1]

    @staticmethod
    def random(_: List[Action], __: List[Action]) -> Action:
        return random.randint(0, 1)


def ipd_step(a: Action, b: Action, variant: Tuple[int, int, int, int]) -> Tuple[float, float]:
    R, T, S, P = variant
    if a == 0 and b == 0:
        return R, R
    if a == 0 and b == 1:
        return S, T
    if a == 1 and b == 0:
        return T, S
    return P, P  # DD


def encode_state(a_last: Action, b_last: Action) -> int:
    return (a_last << 1) | b_last  # 0..3


def simulate(
    episodes: int,
    horizon: int,
    agent: QAgent,
    opp_policy,
    variant: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    coop_agent, coop_opp = 0, 0
    history_a: List[Action] = []
    history_b: List[Action] = []
    a_last, b_last = 0, 0  # start with CC
    for _ in range(episodes):
        agent.last_state = encode_state(a_last, b_last)
        for _ in range(horizon):
            a = agent.act()
            b = opp_policy(history_b, history_a)
            r_a, r_b = ipd_step(a, b, variant)
            next_state = encode_state(a, b)
            agent.update(agent.last_state, a, r_a, next_state)
            a_last, b_last = a, b
            history_a.append(a)
            history_b.append(b)
            coop_agent += int(a == 0)
            coop_opp += int(b == 0)
    denom = episodes * horizon
    return coop_agent / denom, coop_opp / denom


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal RL experiment for ethical attractors")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--output", type=str, default="analysis/rl")
    args = p.parse_args()

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        ROOT = Path(__file__).resolve().parent.parent
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"rl_{ts}.json"

    variants = list(PAYOFF_VARIANTS.keys())
    coop_agent_tft, coop_opp_tft = [], []
    coop_agent_rand, coop_opp_rand = [], []

    agent_hparams = {
        "alpha": float(QAgent().alpha),
        "gamma": float(QAgent().gamma),
        "eps_start": float(QAgent().eps_start),
        "eps_end": float(QAgent().eps_end),
        "eps_decay": float(QAgent().decay),
        "eps_schedule": "linear_per_step_clamped",
        "eps_formula": "eps=max(eps_end, eps_start - eps_decay*steps)",
    }

    random.seed(42)
    np.random.seed(42)

    for name in variants:
        v = PAYOFF_VARIANTS[name]
        ca_t, co_t, ca_r, co_r = [], [], [], []
        for _ in range(args.seeds):
            agent = QAgent()
            a_t, o_t = simulate(args.episodes, args.horizon, agent, Opponent.tft, v)
            agent = QAgent()
            a_r, o_r = simulate(args.episodes, args.horizon, agent, Opponent.random, v)
            ca_t.append(a_t)
            co_t.append(o_t)
            ca_r.append(a_r)
            co_r.append(o_r)
        coop_agent_tft.append(float(np.mean(ca_t)))
        coop_opp_tft.append(float(np.mean(co_t)))
        coop_agent_rand.append(float(np.mean(ca_r)))
        coop_opp_rand.append(float(np.mean(co_r)))

    with json_path.open("w") as fp:
        json.dump(
            {
                "variant": variants,
                "payoff_variants": PAYOFF_VARIANTS,
                "episodes": args.episodes,
                "horizon": args.horizon,
                "seeds": args.seeds,
                "rng_seed": 42,
                "agent_hparams": agent_hparams,
                "mean_coop_agent_tft": coop_agent_tft,
                "mean_coop_opp_tft": coop_opp_tft,
                "mean_coop_agent_rand": coop_agent_rand,
                "mean_coop_opp_rand": coop_opp_rand,
            },
            fp,
            indent=2,
        )
    print(f"[RL] wrote {json_path}")


if __name__ == "__main__":
    main()
