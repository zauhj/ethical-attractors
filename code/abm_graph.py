from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_graph(
    G: nx.Graph,
    steps: int,
    init_coop: float,
    noise: float,
    rule: str,
    soft_k: float,
    seed: int | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if seed is not None:
        np.random.seed(seed)

    nodes = list(G.nodes())
    n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    deg = np.array([G.degree[u] for u in nodes], dtype=np.int32)
    neigh = [np.fromiter((idx[v] for v in G.neighbors(u)), dtype=np.int32) for u in nodes]

    state = (np.random.rand(n) < init_coop).astype(np.int8)
    mu = np.empty(steps + 1)
    mu[0] = state.mean()

    for t in range(1, steps + 1):
        if rule == "strict":
            coop_counts = np.array([state[nbrs].sum() for nbrs in neigh], dtype=np.int32)
            next_state = np.where(
                coop_counts > (deg // 2), 1, np.where(coop_counts < (deg // 2), 0, state)
            )
        elif rule == "soft":
            coop_counts = np.array([state[nbrs].sum() for nbrs in neigh], dtype=np.float32)
            prob_coop = sigmoid(soft_k * (coop_counts - (deg.astype(np.float32) / 2.0)))
            next_state = (np.random.rand(n) < prob_coop).astype(np.int8)
        else:  # voter
            next_state = state.copy()
            for i, nbrs in enumerate(neigh):
                if nbrs.size > 0:
                    j = np.random.randint(0, nbrs.size)
                    next_state[i] = state[nbrs[j]]

        if noise > 0.0:
            flip = np.random.rand(n) < noise
            next_state = np.where(flip, 1 - next_state, next_state)

        state = next_state
        mu[t] = state.mean()

    meta = {"n": n, "m": int(G.number_of_edges()), "avg_deg": float(np.mean(deg))}
    return mu, meta


def load_graph(name: str, edgelist: str | None) -> Tuple[nx.Graph, str]:
    if edgelist:
        G = nx.read_edgelist(edgelist, nodetype=int)
        return G, Path(edgelist).stem
    if name == "karate":
        return nx.karate_club_graph(), "karate"
    raise ValueError(f"unknown graph: {name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Graph ABM for Ethical Attractors")
    p.add_argument("--graph", choices=["karate"], default="karate")
    p.add_argument("--edgelist", type=str, default=None)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--init-coop", type=float, default=0.5)
    p.add_argument("--noise", type=float, default=0.005)
    p.add_argument("--rule", choices=["strict", "soft", "voter"], default="soft")
    p.add_argument("--soft-k", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="analysis/graph")
    args = p.parse_args()

    G, gname = load_graph(args.graph, args.edgelist)
    mu, meta = simulate_graph(G, args.steps, args.init_coop, args.noise, args.rule, args.soft_k, args.seed)

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        ROOT = Path(__file__).resolve().parent.parent
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"graph_abm_{gname}_{args.rule}_{ts}.json"

    with json_path.open("w") as fp:
        json.dump(
            {
                "graph": gname,
                "rule": args.rule,
                "steps": args.steps,
                "init_coop": args.init_coop,
                "noise": args.noise,
                "soft_k": args.soft_k,
                "seed": args.seed,
                "meta": meta,
                "mu": mu.tolist(),
            },
            fp,
            indent=2,
        )

    print(f"[graph_abm] wrote {json_path}")


if __name__ == "__main__":
    main()
