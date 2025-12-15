from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from abm import neighbour_sum


def step_soft_theta(state: np.ndarray, k: float, noise: float, theta: float) -> np.ndarray:
    """Soft-majority imitation with configurable neighbour threshold.

    Reduces to the original rule from abm.step_soft when theta=2.0
    on a 4-neighbour lattice.
    """

    neigh = neighbour_sum(state)
    prob_coop = 1.0 / (1.0 + np.exp(-k * (neigh - theta)))
    next_state = (np.random.rand(*state.shape) < prob_coop).astype(np.int8)
    if noise > 0.0:
        flip = np.random.rand(*state.shape) < noise
        next_state = np.where(flip, 1 - next_state, next_state)
    return next_state


def run_shock_observer(
    grid: tuple[int, int],
    steps: int,
    init_coop: float,
    noise: float,
    soft_k: float,
    soft_k_intervene: float,
    noise_intervene: float,
    t_shock: int,
    p_flip: float,
    mu_min: float,
    leak_len: int,
    intervene_len: int,
    theta_base: float,
    theta_intervene: float,
    seeds: int,
) -> Dict[str, Dict[str, List[float]]]:
    H, W = grid
    conditions = ["no_observer", "observer"]

    results = {}
    for cond in conditions:
        mu_runs = []
        time_below_runs = []
        for s in range(seeds):
            np.random.seed(s)
            state = (np.random.rand(H, W) < init_coop).astype(np.int8)
            mu = np.empty(steps + 1, dtype=float)
            mu[0] = state.mean()

            shocked = False
            below_count = 0
            intervene_remaining = 0
            time_below = 0

            for t in range(1, steps + 1):
                if t == t_shock:
                    shocked = True
                    coop_idx = np.argwhere(state == 1)
                    if coop_idx.size > 0 and p_flip > 0.0:
                        n_coop = coop_idx.shape[0]
                        n_flip = int(round(p_flip * n_coop))
                        n_flip = max(0, min(n_flip, n_coop))
                        if n_flip > 0:
                            choice_idx = np.random.choice(n_coop, size=n_flip, replace=False)
                            for (i, j) in coop_idx[choice_idx]:
                                state[i, j] = 0

                if cond == "observer" and shocked and (t - 1) >= t_shock:
                    if mu[t - 1] < mu_min:
                        below_count += 1
                    else:
                        below_count = 0
                    if below_count >= leak_len and intervene_remaining == 0:
                        intervene_remaining = intervene_len

                if cond == "observer" and intervene_remaining > 0:
                    k_use = soft_k_intervene
                    noise_use = noise_intervene
                    theta_use = theta_intervene
                    intervene_remaining -= 1
                else:
                    k_use = soft_k
                    noise_use = noise
                    theta_use = theta_base

                state = step_soft_theta(state, k_use, noise_use, theta_use)
                mu[t] = state.mean()

                if shocked and t > t_shock and mu[t] < mu_min:
                    time_below += 1

            mu_runs.append(mu.tolist())
            time_below_runs.append(float(time_below))

        mu_arr = np.array(mu_runs, dtype=float)
        mu_mean = mu_arr.mean(axis=0)
        mu_sem = mu_arr.std(axis=0, ddof=1) / np.sqrt(seeds)

        results[cond] = {
            "mu_mean": mu_mean.tolist(),
            "mu_sem": mu_sem.tolist(),
            "time_below_band": time_below_runs,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Shock + observer intervention experiment for lattice ABM")
    parser.add_argument("--grid", type=int, nargs=2, default=(64, 64))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--init-coop", type=float, default=0.5)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--soft-k", type=float, default=4.0)
    parser.add_argument("--soft-k-intervene", type=float, default=5.0)
    parser.add_argument("--noise-intervene", type=float, default=0.0)
    parser.add_argument("--t-shock", type=int, default=200)
    parser.add_argument("--p-flip", type=float, default=0.3)
    parser.add_argument("--mu-min", type=float, default=0.8)
    parser.add_argument("--leak-len", type=int, default=10)
    parser.add_argument("--intervene-len", type=int, default=50)
    parser.add_argument("--theta-base", type=float, default=2.0)
    parser.add_argument("--theta-intervene", type=float, default=2.0)
    parser.add_argument("--seeds", type=int, default=32)
    args = parser.parse_args()

    grid = tuple(args.grid)

    ROOT = Path(__file__).resolve().parent.parent
    ANALYSIS_DIR = ROOT / "analysis" / "abm"
    FIG_DIR = ROOT / "analysis" / "figures"
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = run_shock_observer(
        grid,
        args.steps,
        args.init_coop,
        args.noise,
        args.soft_k,
        args.soft_k_intervene,
        args.noise_intervene,
        args.t_shock,
        args.p_flip,
        args.mu_min,
        args.leak_len,
        args.intervene_len,
        args.theta_base,
        args.theta_intervene,
        args.seeds,
    )

    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = ANALYSIS_DIR / f"shock_observer_{ts}.json"
    with json_path.open("w") as fp:
        json.dump(
            {
                "grid": grid,
                "steps": args.steps,
                "init_coop": args.init_coop,
                "noise": args.noise,
                "soft_k": args.soft_k,
                "soft_k_intervene": args.soft_k_intervene,
                "noise_intervene": args.noise_intervene,
                "t_shock": args.t_shock,
                "p_flip": args.p_flip,
                "mu_min": args.mu_min,
                "leak_len": args.leak_len,
                "intervene_len": args.intervene_len,
                "theta_base": args.theta_base,
                "theta_intervene": args.theta_intervene,
                "seeds": args.seeds,
                "results": summary,
            },
            fp,
            indent=2,
        )

    fig_path = FIG_DIR / f"shock_observer_{ts}.png"

    t = np.arange(args.steps + 1)
    mu_no = np.array(summary["no_observer"]["mu_mean"])
    sem_no = np.array(summary["no_observer"]["mu_sem"])
    mu_obs = np.array(summary["observer"]["mu_mean"])
    sem_obs = np.array(summary["observer"]["mu_sem"])

    plt.figure(figsize=(6, 4))
    plt.axvline(args.t_shock, color="#999999", linestyle=":", label="shock time")
    plt.plot(t, mu_no, label="shock only", color="#4C72B0")
    plt.fill_between(t, mu_no - sem_no, mu_no + sem_no, color="#4C72B0", alpha=0.2)
    plt.plot(t, mu_obs, label="shock + observer", color="#DD8452", linestyle="--")
    plt.fill_between(t, mu_obs - sem_obs, mu_obs + sem_obs, color="#DD8452", alpha=0.2)
    plt.xlabel("step")
    plt.ylabel("mean cooperation")
    plt.title("Shock recovery with/without observer intervention")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    deterministic = FIG_DIR / "shock_observer.png"
    deterministic.write_bytes(fig_path.read_bytes())
    print(f"[abm_shock_observer] saved {fig_path}")


if __name__ == "__main__":
    main()
