"""Lattice-based ABM for Ethical Attractors (final, reproducible version).

Grid of binary agents (0 = defect, 1 = cooperate) updated synchronously or
asynchronously. Two update rules:
1. strict: flip to majority of 4-neighbour Moore neighbourhood (ties keep state).
2. soft: probability of cooperation = σ(k * (n_coop - 2)), logistic slope k.

Adds optional site-flip exploration noise at rate `noise`.
Outputs JSON with parameters, logistic-fit R², and optionally a PNG figure.
Designed for quick (<1 min) laptop runs and full reproducibility.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def logistic(t: np.ndarray, k: float, t50: float, mu_inf: float) -> np.ndarray:  # noqa: D401
    """Three-parameter logistic curve µ(t) = µ_inf / (1 + exp(-k (t - t50)))."""
    return mu_inf / (1.0 + np.exp(-k * (t - t50)))


def neighbour_sum(state: np.ndarray) -> np.ndarray:  # noqa: D401
    """4-neighbour periodic sum via np.roll."""
    s = state
    return (
        np.roll(s, 1, 0)
        + np.roll(s, -1, 0)
        + np.roll(s, 1, 1)
        + np.roll(s, -1, 1)
    )


# -----------------------------------------------------------------------------
# Update rules
# -----------------------------------------------------------------------------

def step_strict(state: np.ndarray, noise: float) -> np.ndarray:  # noqa: D401
    """Deterministic majority rule with optional bit-flip noise."""
    neigh = neighbour_sum(state)
    next_state = np.where(neigh > 2, 1, np.where(neigh < 2, 0, state))
    if noise > 0.0:
        flip = np.random.rand(*state.shape) < noise
        next_state = np.where(flip, 1 - next_state, next_state)
    return next_state


def step_soft(state: np.ndarray, k: float, noise: float) -> np.ndarray:  # noqa: D401
    """Stochastic soft-majority imitation with logistic slope *k*."""
    neigh = neighbour_sum(state)
    prob_coop = 1.0 / (1.0 + np.exp(-k * (neigh - 2.0)))
    next_state = (np.random.rand(*state.shape) < prob_coop).astype(np.int8)
    if noise > 0.0:
        flip = np.random.rand(*state.shape) < noise
        next_state = np.where(flip, 1 - next_state, next_state)
    return next_state


def step_voter(state: np.ndarray, noise: float) -> np.ndarray:  # noqa: D401
    """Classical voter-model synchronous update on a 4-neighbour lattice.

    Each site randomly samples one of its 4 neighbours and adopts its state.
    Optional bit-flip noise applied after copying.
    """
    # Stack the four neighbour grids along a new axis (H, W, 4)
    n_up = np.roll(state, 1, 0)
    n_down = np.roll(state, -1, 0)
    n_left = np.roll(state, 1, 1)
    n_right = np.roll(state, -1, 1)
    neigh_stack = np.stack([n_up, n_down, n_left, n_right], axis=2)
    # Choose one neighbour per site uniformly at random
    choice = np.random.randint(0, 4, size=state.shape)
    h_idx = np.arange(state.shape[0])[:, None]
    w_idx = np.arange(state.shape[1])[None, :]
    next_state = neigh_stack[h_idx, w_idx, choice]
    if noise > 0.0:
        flip = np.random.rand(*state.shape) < noise
        next_state = np.where(flip, 1 - next_state, next_state)
    return next_state


# -----------------------------------------------------------------------------
# Simulation wrapper
# -----------------------------------------------------------------------------

def simulate(
    grid: Tuple[int, int],
    steps: int,
    init_coop: float,
    noise: float,
    rule: str,
    soft_k: float,
    async_update: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:  # noqa: D401
    """Run the ABM and return (µ trajectory, final state)."""

    state = (np.random.rand(*grid) < init_coop).astype(np.int8)
    mu = np.empty(steps + 1)
    mu[0] = state.mean()
    h, w = grid
    n_sites = int(h * w)
    for t in range(1, steps + 1):
        if async_update:
            for _ in range(n_sites):
                i = np.random.randint(0, h)
                j = np.random.randint(0, w)
                sub = state[i, j]
                neigh = (
                    state[(i + 1) % h, j]
                    + state[(i - 1) % h, j]
                    + state[i, (j + 1) % w]
                    + state[i, (j - 1) % w]
                )
                if rule == "strict":
                    new_val = 1 if neigh > 2 else 0 if neigh < 2 else int(sub)
                elif rule == "soft":
                    p_c = 1.0 / (1.0 + np.exp(-soft_k * (neigh - 2.0)))
                    new_val = int(np.random.rand() < p_c)
                else:  # voter
                    r = np.random.randint(0, 4)
                    if r == 0:
                        new_val = int(state[(i + 1) % h, j])
                    elif r == 1:
                        new_val = int(state[(i - 1) % h, j])
                    elif r == 2:
                        new_val = int(state[i, (j + 1) % w])
                    else:
                        new_val = int(state[i, (j - 1) % w])
                if noise > 0.0 and np.random.rand() < noise:
                    new_val = 1 - new_val
                state[i, j] = new_val
        else:
            if rule == "strict":
                state = step_strict(state, noise)
            elif rule == "soft":
                state = step_soft(state, soft_k, noise)
            else:
                state = step_voter(state, noise)
        mu[t] = state.mean()
    return mu, state


# -----------------------------------------------------------------------------
# Logistic-fit helper
# -----------------------------------------------------------------------------

def fit_logistic(mu: np.ndarray) -> Tuple[float, Tuple[float, float, float]]:  # noqa: D401
    t = np.arange(mu.size)
    p0 = [0.05, mu.size / 2.0, mu[-1]]
    try:
        popt, _ = curve_fit(logistic, t, mu, p0=p0, maxfev=10000)
        k_hat, t50_hat, mu_inf_hat = popt
        mu_pred = logistic(t, *popt)
        ss_res = np.sum((mu - mu_pred) ** 2)
        ss_tot = np.sum((mu - mu.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
    except Exception:
        k_hat, t50_hat, mu_inf_hat, r2 = np.nan, np.nan, np.nan, -np.inf
    return r2, (k_hat, t50_hat, mu_inf_hat)


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Lattice ABM for Ethical Attractors")
    parser.add_argument("--grid", type=int, nargs=2, default=(64, 64))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--init-coop", type=float, default=0.5)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--rule", choices=["strict", "soft", "voter"], default="soft")
    parser.add_argument("--soft-k", type=float, default=4.0)
    parser.add_argument("--async", dest="async_update", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--output", type=str, default="analysis/abm")
    parser.add_argument("--diag", action="store_true")
    parser.add_argument("--no-fig", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    mu, _ = simulate(
        tuple(args.grid),
        args.steps,
        args.init_coop,
        args.noise,
        args.rule,
        args.soft_k,
        async_update=args.async_update,
    )
    r2, (k_hat, t50_hat, mu_inf_hat) = fit_logistic(mu)

    # ensure dirs
    out_base = Path(args.output)
    if not out_base.is_absolute():
        ROOT = Path(__file__).resolve().parent.parent
        out_base = ROOT / out_base
    out_base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_base / f"abm_{ts}.json"
    with json_path.open("w") as fp:
        json.dump(
            {
                "grid": args.grid,
                "steps": args.steps,
                "init_coop": args.init_coop,
                "noise": args.noise,
                "rule": args.rule,
                "soft_k": args.soft_k,
                "async": args.async_update,
                "seed": args.seed,
                "r2": r2,
                "k_hat": k_hat,
                "t50_hat": t50_hat,
                "mu_inf_hat": mu_inf_hat,
                "mu": mu.tolist(),
            },
            fp,
            indent=2,
        )
    print(f"[ABM] wrote {json_path} (R²={r2:.3f})")

    if args.no_fig:
        return

    fig_path = out_base.parent / "figures" / f"abm_{ts}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(mu.size)
    plt.plot(t, mu, label="simulation")
    if np.isfinite(r2):
        plt.plot(t, logistic(t, k_hat, t50_hat, mu_inf_hat), "--", label="fit")
    plt.xlabel("step")
    plt.ylabel("mean cooperation")
    plt.title("ABM convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"[ABM] figure saved to {fig_path}")

    deterministic = fig_path.parent / "abm_logistic.png"
    deterministic.write_bytes(fig_path.read_bytes())

    if args.diag and np.isfinite(r2):
        resid_path = out_base.parent / "figures" / f"abm_residual_{ts}.png"
        resid = mu - logistic(t, k_hat, t50_hat, mu_inf_hat)
        plt.figure(figsize=(6, 3))
        plt.plot(t, resid)
        plt.axhline(0.0, color="black", lw=1)
        plt.xlabel("step")
        plt.ylabel("residual")
        plt.title("Residual diagnostics")
        plt.tight_layout()
        plt.savefig(resid_path)
        plt.close()
        print(f"[ABM] residual plot saved to {resid_path}")

        resid_det = resid_path.parent / "abm_residual.png"
        resid_det.write_bytes(resid_path.read_bytes())


if __name__ == "__main__":
    main()
