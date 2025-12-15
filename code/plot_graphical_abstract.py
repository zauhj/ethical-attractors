from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ABM_DIR = ROOT / "analysis" / "abm"
RL_DIR = ROOT / "analysis" / "rl"
FIG_DIR = ROOT / "analysis" / "figures"


def latest_json(directory: Path, pattern: str) -> Path:
    files = list(directory.glob(pattern))
    if not files:
        raise SystemExit(f"[plot_graphical_abstract] no files matching {pattern} in {directory}")
    return max(files, key=lambda p: p.stat().st_mtime)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    abm_json = latest_json(ABM_DIR, "abm_[0-9]*.json")
    async_json = latest_json(ABM_DIR, "async_compare_*.json")
    shock_json = latest_json(ABM_DIR, "shock_observer_*.json")
    rl_json = latest_json(RL_DIR, "rl_*.json")

    with abm_json.open() as fp:
        abm_data = json.load(fp)
    mu = np.array(abm_data["mu"], dtype=float)
    t_mu = np.arange(mu.size, dtype=float)

    with async_json.open() as fp:
        async_data = json.load(fp)
    async_summary = async_data["summary"]
    t_async = np.array(async_summary["t"], dtype=float)
    mu_sync_mean = np.array(async_summary["mu_sync_mean"], dtype=float)
    mu_sync_sem = np.array(async_summary["mu_sync_sem"], dtype=float)
    mu_async_mean = np.array(async_summary["mu_async_mean"], dtype=float)
    mu_async_sem = np.array(async_summary["mu_async_sem"], dtype=float)

    with shock_json.open() as fp:
        shock_data = json.load(fp)
    shock_results = shock_data["results"]
    t_shock = int(shock_data["t_shock"])
    steps_shock = int(shock_data["steps"])
    t = np.arange(steps_shock + 1, dtype=float)
    mu_no = np.array(shock_results["no_observer"]["mu_mean"], dtype=float)
    sem_no = np.array(shock_results["no_observer"]["mu_sem"], dtype=float)
    mu_obs = np.array(shock_results["observer"]["mu_mean"], dtype=float)
    sem_obs = np.array(shock_results["observer"]["mu_sem"], dtype=float)

    with rl_json.open() as fp:
        rl_data = json.load(fp)

    if "summary" in rl_data:
        summary = rl_data["summary"]
        variants = list(summary.keys())
        variants_sorted = sorted([v for v in variants if v != "prosocial"])
        if "prosocial" in variants:
            variants_sorted = ["prosocial"] + variants_sorted
        variants = variants_sorted
        means = np.array([summary[v]["mean_agent_tft"] for v in variants], dtype=float)
        stds = np.array([summary[v]["std_agent_tft"] for v in variants], dtype=float)
    else:
        variants = list(rl_data.get("variant", []))
        means = np.array(rl_data.get("mean_coop_agent_tft", []), dtype=float)
        stds = np.zeros_like(means)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    ax = axs[0, 0]
    ax.plot(t_mu, mu, color="#4C72B0")
    ax.set_title("ABM trajectory")
    ax.set_xlabel("step")
    ax.set_ylabel("mean cooperation")
    ax.set_ylim(0.0, 1.0)

    ax = axs[0, 1]
    ax.plot(t_async, mu_sync_mean, label="sync", color="#4C72B0")
    ax.fill_between(t_async, mu_sync_mean - mu_sync_sem, mu_sync_mean + mu_sync_sem, color="#4C72B0", alpha=0.2)
    ax.plot(t_async, mu_async_mean, label="async", color="#DD8452", linestyle="--")
    ax.fill_between(
        t_async,
        mu_async_mean - mu_async_sem,
        mu_async_mean + mu_async_sem,
        color="#DD8452",
        alpha=0.2,
    )
    ax.set_title("Async vs sync")
    ax.set_xlabel("step")
    ax.set_ylabel("mean cooperation")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=9)

    ax = axs[1, 0]
    ax.axvline(t_shock, color="#999999", linestyle=":")
    ax.plot(t, mu_no, label="shock only", color="#4C72B0")
    ax.fill_between(t, mu_no - sem_no, mu_no + sem_no, color="#4C72B0", alpha=0.2)
    ax.plot(t, mu_obs, label="shock + observer", color="#DD8452", linestyle="--")
    ax.fill_between(t, mu_obs - sem_obs, mu_obs + sem_obs, color="#DD8452", alpha=0.2)
    ax.set_title("Observer intervention")
    ax.set_xlabel("step")
    ax.set_ylabel("mean cooperation")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=9)

    ax = axs[1, 1]
    x = np.arange(len(variants), dtype=float)
    ax.bar(x, means, yerr=stds, color="#55A868", alpha=0.9, capsize=3)
    ax.set_title("RL cooperation")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.set_ylabel("mean cooperation")
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    ts = time.strftime("%Y%m%d-%H%M%S")
    fig_path = FIG_DIR / f"graphical_abstract_{ts}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    (FIG_DIR / "graphical_abstract.png").write_bytes(fig_path.read_bytes())
    print(f"[plot_graphical_abstract] saved {fig_path}")


if __name__ == "__main__":
    main()
