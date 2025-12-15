#!/usr/bin/env python3
"""Generate cross-model convergence overlay (ABM vs RL).

Loads the most recent ABM JSON and RL JSON outputs that are saved by
`abm.py` and `rl.py` in the local `analysis/` sub-directories, then plots

    • ABM mean cooperation trajectory (µ_t) – solid line
    • Horizontal line at the mean cooperation of the prosocial RL variant

Saves to ../analysis/figures/cross_validation_<timestamp>.png and also
creates a deterministic copy cross_validation.png for LaTeX inclusion.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ABM_DIR = ROOT / "analysis" / "abm"
RL_DIR = ROOT / "analysis" / "rl"
FIG_DIR = ROOT / "analysis" / "figures"


def load_latest_json(directory: Path, pattern: str) -> Path:
    files = list(directory.glob(pattern))
    if not files:
        sys.exit(f"[plot_cross_validation] no files matching {pattern} in {directory}")
    return max(files, key=lambda p: p.stat().st_mtime)


def main() -> None:
    abm_json = load_latest_json(ABM_DIR, "abm_[0-9]*.json")
    rl_json = load_latest_json(RL_DIR, "rl_*.json")

    # ABM data
    with abm_json.open() as fp:
        abm_data = json.load(fp)
    mu = np.array(abm_data["mu"], dtype=float)
    t = np.arange(mu.size)

    # RL data – take prosocial variant (index 0 by convention)
    with rl_json.open() as fp:
        rl_data = json.load(fp)
    if "summary" in rl_data:
        try:
            rl_mean = float(rl_data["summary"]["prosocial"]["mean_agent_tft"])
        except KeyError:
            sys.exit("[plot_cross_validation] prosocial variant not found in RL json")
    else:
        try:
            variant_names = rl_data["variant"]
            mean_tft = rl_data["mean_coop_agent_tft"]
            idx = variant_names.index("prosocial")
            rl_mean = float(mean_tft[idx])
        except (KeyError, ValueError):
            sys.exit("[plot_cross_validation] prosocial variant not found in RL json")

    # Plot
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fig_path = FIG_DIR / f"cross_validation_{ts}.png"
    plt.figure(figsize=(6, 4))
    plt.plot(t, mu, label="ABM trajectory", color="#4C72B0")
    plt.axhline(rl_mean, ls="--", color="#DD8452", label="RL (prosocial) mean")
    plt.xlabel("step / episode")
    plt.ylabel("mean cooperation")
    plt.title("Cross-model convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # deterministic copy
    (FIG_DIR / "cross_validation.png").write_bytes(fig_path.read_bytes())
    print(f"[plot_cross_validation] saved {fig_path}")


if __name__ == "__main__":
    main()
