from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from abm import simulate


def run_async_vs_sync(
    grid: tuple[int, int],
    steps: int,
    init_coop: float,
    noise: float,
    rule: str,
    soft_k: float,
    seeds: int,
) -> dict:
    mu_sync = np.zeros((seeds, steps + 1), dtype=float)
    mu_async = np.zeros((seeds, steps + 1), dtype=float)

    for s in range(seeds):
        np.random.seed(s)
        mu_s, _ = simulate(grid, steps, init_coop, noise, rule, soft_k, async_update=False)
        mu_sync[s] = mu_s

        np.random.seed(s)
        mu_a, _ = simulate(grid, steps, init_coop, noise, rule, soft_k, async_update=True)
        mu_async[s] = mu_a

    t = np.arange(steps + 1)
    mu_sync_mean = mu_sync.mean(axis=0)
    mu_sync_sem = mu_sync.std(axis=0, ddof=1) / np.sqrt(seeds)
    mu_async_mean = mu_async.mean(axis=0)
    mu_async_sem = mu_async.std(axis=0, ddof=1) / np.sqrt(seeds)

    return {
        "t": t.tolist(),
        "mu_sync_mean": mu_sync_mean.tolist(),
        "mu_sync_sem": mu_sync_sem.tolist(),
        "mu_async_mean": mu_async_mean.tolist(),
        "mu_async_sem": mu_async_sem.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Async vs sync comparison for lattice ABM")
    parser.add_argument("--grid", type=int, nargs=2, default=(64, 64))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--init-coop", type=float, default=0.5)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--rule", choices=["strict", "soft", "voter"], default="soft")
    parser.add_argument("--soft-k", type=float, default=4.0)
    parser.add_argument("--seeds", type=int, default=32)
    args = parser.parse_args()

    grid = tuple(args.grid)

    ROOT = Path(__file__).resolve().parent.parent
    ANALYSIS_DIR = ROOT / "analysis" / "abm"
    FIG_DIR = ROOT / "analysis" / "figures"
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = run_async_vs_sync(
        grid,
        args.steps,
        args.init_coop,
        args.noise,
        args.rule,
        args.soft_k,
        args.seeds,
    )

    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = ANALYSIS_DIR / f"async_compare_{ts}.json"

    import json

    with json_path.open("w") as fp:
        json.dump(
            {
                "grid": grid,
                "steps": args.steps,
                "init_coop": args.init_coop,
                "noise": args.noise,
                "rule": args.rule,
                "soft_k": args.soft_k,
                "seeds": args.seeds,
                "summary": summary,
            },
            fp,
            indent=2,
        )

    fig_path = FIG_DIR / f"async_compare_{ts}.png"

    t = np.array(summary["t"])
    mu_sync_mean = np.array(summary["mu_sync_mean"])
    mu_sync_sem = np.array(summary["mu_sync_sem"])
    mu_async_mean = np.array(summary["mu_async_mean"])
    mu_async_sem = np.array(summary["mu_async_sem"])

    plt.figure(figsize=(6, 4))
    plt.plot(t, mu_sync_mean, label="sync", color="#4C72B0")
    plt.fill_between(
        t,
        mu_sync_mean - mu_sync_sem,
        mu_sync_mean + mu_sync_sem,
        color="#4C72B0",
        alpha=0.2,
    )
    plt.plot(t, mu_async_mean, label="async", color="#DD8452", linestyle="--")
    plt.fill_between(
        t,
        mu_async_mean - mu_async_sem,
        mu_async_mean + mu_async_sem,
        color="#DD8452",
        alpha=0.2,
    )
    plt.xlabel("step")
    plt.ylabel("mean cooperation")
    plt.title("Async vs sync soft-majority dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    deterministic = FIG_DIR / "async_compare.png"
    deterministic.write_bytes(fig_path.read_bytes())
    print(f"[plot_async_vs_sync] saved {fig_path}")


if __name__ == "__main__":
    main()
