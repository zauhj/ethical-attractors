"""Generate RL cooperation bar chart for the manuscript.

Reads the most recent JSON output produced by rl.py in ../analysis/rl/ and
creates rl_coop_bar.png in ../analysis/figures/ . The figure matches the layout
assumed in paper text (Payoff variants on x-axis, agent-TFT cooperation rate).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_latest_json() -> Path:
    rl_dir = Path(__file__).resolve().parent.parent / "analysis" / "rl"
    jsons = list(rl_dir.glob("rl_*.json"))
    if not jsons:
        sys.exit("[plot_rl] no RL json files found — run rl.py first")
    latest = max(jsons, key=lambda p: p.stat().st_mtime)
    return latest


def main() -> None:
    latest = load_latest_json()
    with latest.open() as fp:
        data = json.load(fp)

    if "summary" in data:
        summary = data["summary"]
        variants = list(summary.keys())
        mean_agent_tft = np.array([summary[v]["mean_agent_tft"] for v in variants], dtype=float)
        se_agent_tft = np.array([summary[v]["std_agent_tft"] for v in variants], dtype=float)
    else:
        variants = data["variant"]
        mean_agent_tft = np.array(data["mean_coop_agent_tft"], dtype=float)
        se_agent_tft = np.zeros_like(mean_agent_tft)

    fig_dir = latest.parent.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "rl_coop_bar.png"

    plt.figure(figsize=(6, 3))
    x = np.arange(len(variants))
    plt.bar(x, mean_agent_tft, yerr=se_agent_tft, capsize=4, color="#4C72B0")
    plt.xticks(x, variants, rotation=15)
    plt.ylabel("Mean cooperation rate (agent vs TFT)")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[plot_rl] saved {fig_path}")


if __name__ == "__main__":
    main()
