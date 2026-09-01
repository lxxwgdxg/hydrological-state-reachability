"""Rebuild the four-panel core figure from released derived tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
SEEDS = (11, 29, 47)
COLORS = {11: "#2878B5", 29: "#D95319", 47: "#2E8B57"}


def main() -> None:
    external = pd.read_csv(DATA / "core_figure_panel_a.csv")
    high_skill = pd.read_csv(DATA / "core_figure_panel_b.csv")
    recovery = pd.read_csv(DATA / "core_figure_panel_c.csv")
    external_retrain = pd.read_csv(DATA / "core_figure_panel_d.csv")
    if set(external.seed.unique()) != set(SEEDS) or any(len(external[external.seed == seed]) != 506 for seed in SEEDS):
        raise RuntimeError("Unexpected panel-a inventory")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(8.1, 6.5), gridspec_kw={"wspace": 0.30, "hspace": 0.42}
    )
    axes = axes.ravel()

    ax = axes[0]
    for seed in SEEDS:
        values = np.sort(
            external.loc[external.seed == seed, "negative_storage_test_day_fraction"].to_numpy()
        )
        survival = (len(values) - np.arange(len(values))) / len(values)
        ax.step(values, survival, where="post", color=COLORS[seed], lw=1.8, label=f"seed {seed}")
    ax.axvline(0.01, color="0.25", lw=1.0, ls="--")
    ax.text(0.018, 0.50, "1% materiality gate", color="0.25", fontsize=7.5, rotation=90, va="center")
    ax.set(xlim=(0, 1.0), ylim=(0, 1.02), xlabel="Fraction of test days with\nunreachable storage", ylabel="Fraction of 506 unseen basins\nat or above frequency")
    ax.set_title("a  Unreachable storage transfers", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[3]
    for seed in SEEDS:
        subset = external_retrain[
            (external_retrain.seed == seed)
            & (external_retrain.high_skill_unconstrained_original == True)
        ]
        ax.scatter(
            subset.nse_unconstrained_original,
            subset.nse_feasible_retrained,
            s=11,
            alpha=0.42,
            color=COLORS[seed],
            edgecolors="none",
        )
        median_gap = float(subset.transfer_gap_nse.median())
        within = float(subset.transfer_gap_within_0_10.mean())
        ax.text(
            0.02,
            0.05 + 0.075 * (2 - list(SEEDS).index(seed)),
            f"{seed}: median gap={median_gap:.3f}; {within:.0%} within 0.10",
            transform=ax.transAxes,
            color=COLORS[seed],
            fontsize=7.4,
        )
    limits = (-0.10, 0.92)
    ax.plot(limits, limits, color="0.25", lw=1.0, ls="--")
    ax.set(
        xlim=(0.49, 0.91),
        ylim=limits,
        xlabel="Original-model NSE\n(predeclared high-skill stratum)",
        ylabel="Reachable-trained NSE\n(omitted basins)",
    )
    ax.set_title("d  Reachable repair transfers", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for seed in SEEDS:
        subset = high_skill[high_skill.seed == seed]
        ax.scatter(subset.original_nse, subset.ordered_feasible_nse, s=11, alpha=0.48, color=COLORS[seed], edgecolors="none")
        med = float(subset.paired_nse_decline.median())
        frac = float(subset.nse_declines.mean())
        ax.text(0.02, 0.05 + 0.075 * (2 - list(SEEDS).index(seed)), f"{seed}: median ΔNSE={med:.3f}; {frac:.0%} decline", transform=ax.transAxes, color=COLORS[seed], fontsize=7.5)
    limits = (-4.1, 1.0)
    ax.plot(limits, limits, color="0.25", lw=1.0, ls="--")
    ax.set(xlim=(0.49, 0.91), ylim=limits, xlabel="Original-path NSE\n(predeclared high-skill stratum)", ylabel="Same weights, reachable-path NSE")
    ax.set_title("b  Skill uses unreachable states", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    x = np.array([0.0, 1.0])
    for seed in SEEDS:
        row = recovery[recovery.seed == seed].iloc[0]
        values = [row.same_checkpoint_median_nse_cost, row.after_feasible_retraining_median_nse_gap]
        ax.plot(x, values, marker="o", ms=5.5, lw=1.8, color=COLORS[seed])
    ax.axhline(0, color="0.35", lw=0.9)
    ax.axhline(0.05, color="0.45", lw=0.9, ls="--")
    ax.text(0.50, 0.058, "material cost threshold", color="0.35", ha="center", fontsize=7.4)
    ax.set(xlim=(-0.12, 1.18), ylim=(-0.04, 0.48), ylabel="Median NSE cost relative to\nunconstrained model (24 basins)")
    ax.set_xticks(x, ["Freeze weights;\nenforce reachability", "Retrain with\nreachable updates"])
    ax.set_title("c  The shortcut is avoidable", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.995, "Unreachable water states support skill, but reachable training preserves transfer", ha="center", va="top", fontsize=11.2, fontweight="bold")
    fig.subplots_adjust(top=0.91, bottom=0.09, left=0.09, right=0.98)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "CORE_FIGURE_REPRODUCED.png", dpi=400, facecolor="white")
    fig.savefig(OUT / "CORE_FIGURE_REPRODUCED.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
