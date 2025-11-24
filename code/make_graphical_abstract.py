#!/usr/bin/env python3
"""Create a simple two-panel graphical abstract from existing figures.

Inputs (default):
  - analysis/figures/abm_topology_compare.png
  - analysis/figures/cross_validation.png

Output:
  - analysis/figures/graphical_abstract.png (300 dpi)

Usage:
  python3 code/make_graphical_abstract.py [--left IMG] [--right IMG] [--output OUT]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "analysis" / "figures"


def build(left: Path, right: Path, out_path: Path) -> None:
    if not left.exists():
        raise FileNotFoundError(f"left image not found: {left}")
    if not right.exists():
        raise FileNotFoundError(f"right image not found: {right}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_l = mpimg.imread(left)
    img_r = mpimg.imread(right)

    # Create side-by-side panel with no axes or spines
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, img in zip(axes, (img_l, img_r)):
        ax.imshow(img)
        ax.axis("off")

    # Optional small panel labels
    axes[0].text(0.02, 0.96, "A", transform=axes[0].transAxes, fontsize=14, fontweight="bold", va="top")
    axes[1].text(0.02, 0.96, "B", transform=axes[1].transAxes, fontsize=14, fontweight="bold", va="top")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[make_graphical_abstract] wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build graphical abstract from two figures")
    p.add_argument("--left", type=str, default=str(FIG_DIR / "abm_topology_compare.png"))
    p.add_argument("--right", type=str, default=str(FIG_DIR / "cross_validation.png"))
    p.add_argument("--output", type=str, default=str(FIG_DIR / "graphical_abstract.png"))
    args = p.parse_args()

    build(Path(args.left), Path(args.right), Path(args.output))


if __name__ == "__main__":
    main()
