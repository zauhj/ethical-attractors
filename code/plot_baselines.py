#!/usr/bin/env python3
"""Baseline comparison plots for Ethical Attractors.

Generates a topology comparison figure overlaying cooperation trajectories for
three rules (strict, soft, voter) on:
  - 64x64 lattice (periodic 4-neighbour)
  - Karate Club graph (real-world network)

Outputs: ../analysis/figures/abm_topology_compare.png
"""
from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

# Local imports
from abm import simulate as simulate_lattice
from abm_graph import simulate_graph, load_graph

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "analysis" / "figures"


def main() -> None:
    steps = 800
    noise = 0.005
    init = 0.5
    soft_k = 4.0

    # Lattice runs
    grid = (64, 64)
    rules = ["strict", "soft", "voter"]
    lattice = {}
    for r in rules:
        mu, _ = simulate_lattice(grid, steps, init, noise, r, soft_k, async_update=False)
        lattice[r] = mu

    # Karate graph runs
    G, gname = load_graph("karate", None)
    graph = {}
    for r in rules:
        mu, _ = simulate_graph(G, steps, init, noise, r, soft_k, seed=42)
        graph[r] = mu

    # Plot
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fig_path = FIG_DIR / f"abm_topology_compare_{ts}.png"

    colors = {"strict": "#4C72B0", "soft": "#DD8452", "voter": "#55A868"}
    styles = {"strict": "-", "soft": "--", "voter": ":"}

    t = np.arange(steps + 1)
    plt.figure(figsize=(10, 4))

    # Left: lattice
    ax1 = plt.subplot(1, 2, 1)
    for r in rules:
        ax1.plot(t, lattice[r], styles[r], color=colors[r], label=r)
    ax1.set_title("Lattice (64x64)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("mean cooperation")
    ax1.set_ylim(0, 1)
    ax1.legend(frameon=False)

    # Right: karate
    ax2 = plt.subplot(1, 2, 2)
    for r in rules:
        ax2.plot(t, graph[r], styles[r], color=colors[r], label=r)
    ax2.set_title("Karate Club graph")
    ax2.set_xlabel("step")
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)

    # Deterministic copy for LaTeX include
    (FIG_DIR / "abm_topology_compare.png").write_bytes(fig_path.read_bytes())
    print(f"[plot_baselines] saved {fig_path}")


if __name__ == "__main__":
    main()
