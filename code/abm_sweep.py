"""Parameter sweep for ABM on multiple topologies (final).

See paper for details. Outputs JSON summary and plots.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np

import json
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent))
from abm import simulate, fit_logistic, logistic

PARAM_GRID = {
    "soft_k": [1.0, 2.0, 4.0],
    "steps": [400, 600, 800],
    "noise": [0.0, 0.005],
}

GRID_SIZE = (64, 64)
INIT_COOP = 0.5


def sweep() -> Dict[str, float]:
    combos = list(product(PARAM_GRID["soft_k"], PARAM_GRID["steps"], PARAM_GRID["noise"]))
    results: List[Tuple[float, Dict[str, float], np.ndarray]] = []
    for k, steps, noise in combos:
        mu, _ = simulate(GRID_SIZE, steps, INIT_COOP, noise, "soft", k)
        r2, _ = fit_logistic(mu)
        results.append((r2, {"k": k, "steps": steps, "noise": noise}, mu))
        print(f"k={k} steps={steps} noise={noise} -> R²={r2:.3f}")
    best_r2, best_params, best_mu = max(results, key=lambda x: x[0])
    print(f"[SWEEP] best R²={best_r2:.3f} with {best_params}")

    # Save outputs
    out_dir = Path("../analysis/abm")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"abm_sweep_{ts}.json"
    with json_path.open("w") as fp:
        json.dump(
            {
                "grid_size": GRID_SIZE,
                "init_coop": INIT_COOP,
                "param_grid": PARAM_GRID,
                "best_r2": best_r2,
                "best_params": best_params,
            },
            fp,
            indent=2,
        )
    print(f"[SWEEP] summary saved to {json_path}")

    # Figure for best run
    fig_path = out_dir.parent / "figures" / f"abm_best_{ts}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(best_mu.size)
    r2_plot, (k_hat, t50_hat, mu_inf_hat) = fit_logistic(best_mu)
    plt.plot(t, best_mu, label="simulation")
    if np.isfinite(r2_plot):
        plt.plot(t, logistic(t, k_hat, t50_hat, mu_inf_hat), "--", label="fit")
    plt.title("Best ABM run")
    plt.xlabel("step")
    plt.ylabel("mean cooperation")
    plt.legend()
    plt.savefig(fig_path)
    plt.close()
    print(f"[SWEEP] figure saved to {fig_path}")
    return best_params



if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Parameter sweep for ABM logistic fit")
    p.add_argument("--output", type=str, default="../analysis/abm", help="Output directory for JSON/figures")
    args = p.parse_args()

    # Override default output dir if provided
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        # Reconstruct relative path from script location
        out_dir = (Path(__file__).parent / args.output).resolve()
    # Monkey-patch path inside sweep() via closure variable shadowing
    PARAM_OUTPUT_DIR = out_dir  # noqa: N806 keep upper-case for clarity

    # Re-run sweep with customised output directory
    # We replicate sweep logic but with modified out_dir variable.
    combos = list(product(PARAM_GRID["soft_k"], PARAM_GRID["steps"], PARAM_GRID["noise"]))
    results: List[Tuple[float, Dict[str, float], np.ndarray]] = []
    for k, steps, noise in combos:
        mu, _ = simulate(GRID_SIZE, steps, INIT_COOP, noise, "soft", k)
        r2, _ = fit_logistic(mu)
        results.append((r2, {"k": k, "steps": steps, "noise": noise}, mu))
        print(f"k={k} steps={steps} noise={noise} -> R²={r2:.3f}")
    best_r2, best_params, best_mu = max(results, key=lambda x: x[0])
    print(f"[SWEEP] best R²={best_r2:.3f} with {best_params}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"abm_sweep_{ts}.json"
    with json_path.open("w") as fp:
        json.dump(
            {
                "grid_size": GRID_SIZE,
                "init_coop": INIT_COOP,
                "param_grid": PARAM_GRID,
                "best_r2": best_r2,
                "best_params": best_params,
            },
            fp,
            indent=2,
        )
    print(f"[SWEEP] summary saved to {json_path}")

    # Figure
    fig_path = out_dir.parent / "figures" / f"abm_best_{ts}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(best_mu.size)
    r2_plot, (k_hat, t50_hat, mu_inf_hat) = fit_logistic(best_mu)
    plt.plot(t, best_mu, label="simulation")
    if np.isfinite(r2_plot):
        plt.plot(t, logistic(t, k_hat, t50_hat, mu_inf_hat), "--", label="fit")
    plt.title("Best ABM run")
    plt.xlabel("step")
    plt.ylabel("mean cooperation")
    plt.legend()
    plt.savefig(fig_path)
    plt.close()
    print(f"[SWEEP] figure saved to {fig_path}")

