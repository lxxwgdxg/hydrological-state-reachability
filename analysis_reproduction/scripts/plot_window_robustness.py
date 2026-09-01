"""Rebuild the initialization/history robustness figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "window_seed_condition_summary.csv"
OUT = ROOT / "outputs"
CONDITIONS = ("zero_L365", "zero_L730", "zero_L1095", "smax_L365")
LABELS = ("365 d\nzero", "730 d\nzero", "1,095 d\nzero", "365 d\n$S_{max}$")
SEEDS = (11, 29, 47)
COLORS = {11: "#0072B2", 29: "#D55E00", 47: "#009E73"}


def main() -> None:
    table = pd.read_csv(INPUT)
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if set(zip(table.condition, table.seed)) != expected:
        raise RuntimeError("Unexpected condition-seed inventory")
    table = table.set_index(["condition", "seed"])
    metrics = (
        ("material_negative_storage_basin_fraction", "Basins with material\nnegative storage (%)", lambda x: x * 100.0, (0, 105)),
        ("median_maximum_storage_deficit_mm", "Median maximum\nstorage deficit (mm)", lambda x: x, (0, 52)),
        ("median_paired_nse_decline", "Median paired $\\Delta$NSE\n(original − reachable)", lambda x: x, (0, 0.56)),
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.45), constrained_layout=True)
    x = np.arange(len(CONDITIONS), dtype=float)
    for panel_index, (ax, metric) in enumerate(zip(axes, metrics)):
        column, ylabel, transform, ylim = metric
        seed_values = []
        for seed in SEEDS:
            values = transform(np.array([table.loc[(condition, seed), column] for condition in CONDITIONS]))
            seed_values.append(values)
            ax.plot(x, values, color=COLORS[seed], marker="o", markersize=4.8, linewidth=1.15, alpha=0.88, label=f"seed {seed}", zorder=2)
        ax.scatter(x, np.median(np.vstack(seed_values), axis=0), marker="D", s=30, facecolor="white", edgecolor="black", linewidth=1.05, zorder=3, label="cross-seed median" if panel_index == 0 else None)
        ax.axhline(0, color="#777777", linewidth=0.7, zorder=0)
        ax.set_xticks(x, LABELS)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.13, 1.04, chr(ord("a") + panel_index), transform=ax.transAxes, fontweight="bold")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8, handlelength=1.5)
    fig.suptitle("Unreachable storage persists under longer histories and generous initialization", fontsize=12, fontweight="semibold")
    fig.text(0.5, -0.015, "Fixed checkpoints; 24 basins per seed; held-out test period. Materiality threshold: storage < −0.1 mm.", ha="center", va="top", fontsize=8.3, color="#444444")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "WINDOW_INITIALIZATION_ROBUSTNESS_REPRODUCED.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "WINDOW_INITIALIZATION_ROBUSTNESS_REPRODUCED.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
