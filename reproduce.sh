#!/bin/bash
# Reproduce all results and figures for Ethical Attractors
set -e
cd "$(dirname "$0")"

# ABM main run
python3 code/abm.py --mode sync --seeds 30 --steps 800 --noise 0 --soft_k 4 --output analysis/abm/abm_sync.json

# ABM sweep
python3 code/abm_sweep.py --output analysis/abm/abm_sweep.json

# RL run
python3 code/rl.py --output analysis/rl/rl_results.json

# RL cross-validation
python3 code/plot_cross_validation.py --output analysis/figures/cross_validation.png

# Plot RL overlay
python3 code/plot_rl.py --output analysis/figures/rl_coop_bar.png

# Copy best figures for paper
cp analysis/figures/abm_logistic.png analysis/figures/abm_residual.png analysis/figures/abm_topology_compare.png analysis/figures/rl_coop_bar.png analysis/figures/cross_validation.png .

