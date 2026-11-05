#!/usr/bin/env python3
"""Generate the benzamidine--trypsin validation figures.

The repository stores processed, replica-level data rather than trajectories.
Exact event summaries were calculated from 0.1 ns data.  The retained traces
are plotted at 0.2 ns resolution.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

METHODS = ("Unbiased MD", "Protein-GaMD", "LiGaMD dual boost")
SHORT = {"Unbiased MD": "Unbiased", "Protein-GaMD": "Protein-GaMD", "LiGaMD dual boost": "LiGaMD"}
COLORS = {"Unbiased MD": "#2878B5", "Protein-GaMD": "#E36B2C", "LiGaMD dual boost": "#149E79"}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def read_summary() -> dict[str, list[dict[str, float | int | None]]]:
    rows: dict[str, list[dict[str, float | int | None]]] = defaultdict(list)
    with (DATA / "replica_summary.csv").open(newline="") as handle:
        for raw in csv.DictReader(handle):
            rows[raw["method"]].append(
                {
                    "replica": int(raw["replica"]),
                    "crossed": int(raw["crossed_10A"]),
                    "first": float(raw["first_crossing_after_production_start_ns"])
                    if raw["first_crossing_after_production_start_ns"]
                    else None,
                    "occupancy": float(raw["occupancy_above_10A_percent"]),
                    "longest": float(raw["longest_continuous_above_10A_ns"]),
                    "max_contact": float(raw["max_min_ligand_protein_distance_A"]),
                    "max_pocket": float(raw["max_pocket_anchor_distance_A_reps1to3_only"])
                    if raw["max_pocket_anchor_distance_A_reps1to3_only"]
                    else None,
                }
            )
    return rows


def read_traces() -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    values: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    with (DATA / "contact_distance_traces_0p2ns.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            values[(row["method"], int(row["replica"]))].append(
                (float(row["production_time_ns"]), float(row["min_ligand_protein_distance_A"]))
            )
    return {key: (np.asarray([x[0] for x in val]), np.asarray([x[1] for x in val])) for key, val in values.items()}


def read_rows(filename: str) -> list[dict[str, str]]:
    with (DATA / filename).open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, stem: str) -> None:
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(FIGURES / f"{stem}.{extension}", bbox_inches="tight", pad_inches=0.04, **kwargs)
    plt.close(fig)


def mean_bar(ax, x: float, values: np.ndarray, width: float = 0.24) -> None:
    ax.hlines(float(np.mean(values)), x - width, x + width, color="#111111", lw=1.8, zorder=4)


def all_replica_traces(summary, traces) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(10.4, 5.45), sharex=True, sharey=True)
    for row, method in enumerate(METHODS):
        n_cross = sum(int(item["crossed"]) for item in summary[method])
        for column in range(6):
            ax = axes[row, column]
            time, distance = traces[(method, column + 1)]
            ax.axhspan(10, 26, color="#F0F0F0", zorder=0)
            ax.plot(time, distance, color=COLORS[method], lw=0.62, alpha=0.94)
            ax.axhline(10, color="#555555", lw=0.7, ls=(0, (3, 2)))
            ax.set_xlim(0, 430)
            ax.set_ylim(0, 26)
            ax.set_xticks((0, 200, 400))
            ax.set_yticks((0, 10, 20))
            if row == 0:
                ax.set_title(f"Replica {column + 1}", pad=4)
            if column == 0:
                ax.text(
                    0.03,
                    0.92,
                    f"{SHORT[method]} · {n_cross}/6 cross",
                    transform=ax.transAxes,
                    color=COLORS[method],
                    fontsize=8.2,
                    fontweight="bold",
                    va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.2},
                )
            ax.tick_params(labelsize=7.5, length=3)
    fig.text(0.51, 0.995, "Minimum ligand–protein distance", ha="center", va="top", fontsize=12.5, fontweight="bold")
    fig.text(0.008, 0.50, "Minimum distance (Å)", ha="left", va="center", rotation=90, fontsize=9, fontweight="bold")
    fig.text(0.51, 0.012, "Time after production start (ns)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.text(0.995, 0.014, "Shaded region: >10 Å • production window: 70–500 ns", ha="right", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0.025, 0.045, 1, 0.96), w_pad=0.65, h_pad=0.70)
    save(fig, "all_replica_traces")


def benchmark_summary(summary) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.1))
    x = np.arange(3)
    labels = [SHORT[m] for m in METHODS]
    jitter6 = np.linspace(-0.16, 0.16, 6)

    counts = np.asarray([sum(int(item["crossed"]) for item in summary[m]) for m in METHODS])
    proportions = counts / 6 * 100
    ci_low = np.asarray([0.0, 0.42, 54.1])
    ci_high = np.asarray([45.9, 64.1, 100.0])
    axes[0, 0].errorbar(
        x,
        proportions,
        yerr=np.vstack((proportions - ci_low, ci_high - proportions)),
        fmt="none",
        ecolor="#333333",
        capsize=3,
        lw=1.1,
        zorder=1,
    )
    for i, method in enumerate(METHODS):
        axes[0, 0].scatter(i, proportions[i], s=68, color=COLORS[method], edgecolor="white", lw=0.7, zorder=3)
        axes[0, 0].text(i, min(proportions[i] + 7, 105), f"{counts[i]}/6", ha="center", fontweight="bold")
    axes[0, 0].set_ylim(-3, 112)
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("Replicas crossing 10 Å (%)")
    axes[0, 0].set_title("A  Event incidence", loc="left")

    jitter3 = np.linspace(-0.11, 0.11, 3)
    for i, method in enumerate(METHODS):
        vals = np.asarray([item["max_pocket"] for item in summary[method] if item["max_pocket"] is not None], dtype=float)
        axes[0, 1].scatter(i + jitter3, vals, s=42, color=COLORS[method], edgecolor="white", lw=0.6, zorder=3)
        mean_bar(axes[0, 1], i, vals)
    axes[0, 1].axhline(25, color="#555555", lw=0.8, ls=(0, (4, 3)))
    axes[0, 1].text(2.48, 26.2, "25 Å pocket-egress reference", ha="right", fontsize=7.5, color="#444444")
    axes[0, 1].set_ylim(0, 57)
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("Maximum pocket distance (Å)")
    axes[0, 1].set_title("B  Pocket exit (replicas 1–3)", loc="left")

    for i, method in enumerate(METHODS):
        vals = np.asarray([item["max_contact"] for item in summary[method]], dtype=float)
        axes[1, 0].scatter(i + jitter6, vals, s=38, color=COLORS[method], edgecolor="white", lw=0.55, zorder=3)
        mean_bar(axes[1, 0], i, vals)
    axes[1, 0].axhline(10, color="#555555", lw=0.8, ls=(0, (4, 3)))
    axes[1, 0].set_ylim(0, 27)
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("Maximum minimum distance (Å)")
    axes[1, 0].set_title("C  Whole-protein detachment", loc="left")

    for i, method in enumerate(METHODS):
        vals = np.asarray([item["occupancy"] for item in summary[method]], dtype=float)
        axes[1, 1].scatter(i + jitter6, vals, s=38, color=COLORS[method], edgecolor="white", lw=0.55, zorder=3)
        mean_bar(axes[1, 1], i, vals)
        axes[1, 1].text(i, vals.mean() + 1.6, f"{vals.mean():.1f}%", ha="center", fontsize=7.5)
    axes[1, 1].set_ylim(0, 29)
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("Frames above 10 Å (%)")
    axes[1, 1].set_title("D  Separated-state occupancy", loc="left")

    fig.suptitle("Benzamidine–trypsin validation benchmark", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96), w_pad=2.1, h_pad=2.0)
    save(fig, "benchmark_summary")


def sensitivity_and_boost() -> None:
    sensitivity = read_rows("threshold_sensitivity.csv")
    boost = read_rows("boost_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))

    for method in METHODS:
        rows = [row for row in sensitivity if row["method"] == method]
        threshold = np.asarray([float(row["threshold_A"]) for row in rows])
        incidence = np.asarray([100 * float(row["replicas_crossing"]) / float(row["total_replicas"]) for row in rows])
        occupancy = np.asarray([float(row["mean_occupancy_percent"]) for row in rows])
        axes[0].plot(threshold, incidence, "o-", color=COLORS[method], lw=1.6, ms=4.5, label=SHORT[method])
        axes[1].plot(threshold, occupancy, "o-", color=COLORS[method], lw=1.6, ms=4.5)
    axes[0].set_xticks((6, 8, 10))
    axes[0].set_ylim(-3, 108)
    axes[0].set_xlabel("Distance threshold (Å)")
    axes[0].set_ylabel("Replicas crossing (%)")
    axes[0].set_title("A  Incidence", loc="left")
    axes[0].legend(loc="center left", fontsize=7.5)
    axes[1].set_xticks((6, 8, 10))
    axes[1].set_ylim(0, 16)
    axes[1].set_xlabel("Distance threshold (Å)")
    axes[1].set_ylabel("Mean occupancy (%)")
    axes[1].set_title("B  Threshold sensitivity", loc="left")

    for i, method in enumerate(("Protein-GaMD", "LiGaMD dual boost")):
        rows = [row for row in boost if row["method"] == method]
        offsets = np.linspace(-0.17, 0.17, 6)
        means = []
        for offset, row in zip(offsets, rows):
            mean = float(row["mean_kcal_per_mol"])
            q05 = float(row["q05_kcal_per_mol"])
            q95 = float(row["q95_kcal_per_mol"])
            means.append(mean)
            axes[2].vlines(i + offset, q05, q95, color=COLORS[method], lw=1.25)
            axes[2].scatter(i + offset, mean, s=27, color=COLORS[method], edgecolor="white", lw=0.45, zorder=3)
        mean_bar(axes[2], i, np.asarray(means), 0.26)
    axes[2].set_xlim(-0.55, 1.55)
    axes[2].set_ylim(38, 76)
    axes[2].set_xticks((0, 1), ("Protein-GaMD", "LiGaMD"))
    axes[2].set_ylabel("Total boost (kcal mol$^{-1}$)")
    axes[2].set_title("C  Boost magnitude", loc="left")
    axes[2].text(0.03, 0.04, "dot: replica mean\nline: 5th–95th percentile", transform=axes[2].transAxes, fontsize=7.2, color="#555555")

    fig.suptitle("Robustness and boost controls", fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.94), w_pad=2.1)
    save(fig, "robustness_controls")


def main() -> None:
    style()
    summary = read_summary()
    traces = read_traces()
    all_replica_traces(summary, traces)
    benchmark_summary(summary)
    sensitivity_and_boost()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
