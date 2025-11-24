#!/bin/bash
# Reproduce all results and figures for Ethical Attractors
set -e
cd "$(dirname "$0")"

# ABM (lattice) main run for cross-validation figure
python3 code/abm.py --grid 64 64 --steps 800 --noise 0.005 --rule soft --soft-k 4 --output analysis/abm

# ABM parameter sweep (soft rule)
python3 code/abm_sweep.py --output analysis/abm

# Async vs sync comparison (lattice)
python3 code/plot_async_vs_sync.py

# RL run (produces analysis/rl/rl_*.json)
python3 code/rl.py --output analysis/rl

# Graph ABM on Karate Club (baseline on real-world network)
python3 code/abm_graph.py --graph karate --steps 800 --noise 0.005 --rule soft --soft-k 4 --output analysis/graph

# Baseline plots
python3 code/plot_rl.py
python3 code/plot_cross_validation.py
python3 code/plot_baselines.py

# Shock + observer intervention experiment (lattice)
python3 code/abm_shock_observer.py

# Copy final figures for manuscript
cp analysis/figures/abm_topology_compare.png analysis/figures/rl_coop_bar.png analysis/figures/cross_validation.png .

