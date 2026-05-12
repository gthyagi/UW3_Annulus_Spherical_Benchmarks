#!/usr/bin/env python3
"""Reproduce Kramer et al. spherical benchmark Figure 4 from UW3 metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

METRICS_ROOT = Path("/Volumes/seagate4_1/output/spherical/kramer/latest")
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "figure_4_kramer_spherical_convergence.pdf"

CASE_INFO = {
    "case1": {
        "velocity_slope": 1.5,
        "pressure_slope": 0.5,
    },
    "case2": {
        "velocity_slope": 3.0,
        "pressure_slope": 2.0,
    },
    "case3": {
        "velocity_slope": 1.5,
        "pressure_slope": 0.5,
    },
    "case4": {
        "velocity_slope": 3.0,
        "pressure_slope": 2.0,
    },
}

PANEL_LAYOUT = (
    ("case1", "v_l2_norm", r"$E_{L_2}^{\mathrm{rel}}(\mathbf{u})$", "a)"),
    ("case2", "v_l2_norm", r"$E_{L_2}^{\mathrm{rel}}(\mathbf{u})$", "b)"),
    ("case1", "p_l2_norm", r"$E_{L_2}^{\mathrm{rel}}(p)$", "c)"),
    ("case2", "p_l2_norm", r"$E_{L_2}^{\mathrm{rel}}(p)$", "d)"),
    ("case3", "v_l2_norm", r"$E_{L_2}^{\mathrm{rel}}(\mathbf{u})$", "e)"),
    ("case4", "v_l2_norm", r"$E_{L_2}^{\mathrm{rel}}(\mathbf{u})$", "f)"),
    ("case3", "p_l2_norm", r"$E_{L_2}^{\mathrm{rel}}(p)$", "g)"),
    ("case4", "p_l2_norm", r"$E_{L_2}^{\mathrm{rel}}(p)$", "h)"),
)
MARKERS = ("o", "s", "^", "D", "v", "P")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
H_VALUES = np.array([1 / 64, 1 / 32, 1 / 16, 1 / 8], dtype=float)
H_TICK_LABELS = ("1/64", "1/32", "1/16", "1/8")


@dataclass(frozen=True)
class MetricRecord:
    case: str
    inv_lc: int
    h: float
    l: int
    m: int
    k: int
    v_l2_norm: float
    p_l2_norm: float


DIR_PATTERN = re.compile(
    r"(?P<case>case\d+)_"
    r"inv_lc_(?P<inv_lc>\d+)_"
    r"l_(?P<l>\d+)_"
    r"m_(?P<m>\d+)_"
    r"k_(?P<k>\d+)_"
)


def read_scalar(h5f: h5py.File, name: str) -> float:
    """Read a scalar metric from a benchmark_metrics.h5 file."""

    value = h5f[name][()]
    if isinstance(value, bytes):
        return value.decode()
    return float(np.asarray(value))


def read_metrics(metrics_root: Path) -> list[MetricRecord]:
    """Read all complete Kramer spherical benchmark metric files."""

    records: list[MetricRecord] = []
    for metric_file in sorted(metrics_root.glob("case*_inv_lc_*/benchmark_metrics.h5")):
        match = DIR_PATTERN.search(metric_file.parent.name)
        if match is None:
            continue

        case = match.group("case")
        if case not in CASE_INFO:
            continue

        with h5py.File(metric_file, "r") as h5f:
            records.append(
                MetricRecord(
                    case=case,
                    inv_lc=int(read_scalar(h5f, "cellsize") ** -1),
                    h=read_scalar(h5f, "cellsize"),
                    l=int(read_scalar(h5f, "l")),
                    m=int(read_scalar(h5f, "m")),
                    k=int(read_scalar(h5f, "k")),
                    v_l2_norm=read_scalar(h5f, "v_l2_norm"),
                    p_l2_norm=read_scalar(h5f, "p_l2_norm"),
                )
            )

    return records


def grouped_lm(records: list[MetricRecord]) -> list[tuple[int, int, int]]:
    """Return stable (l, m, k) groups in benchmark order."""

    groups = sorted({(record.l, record.m, record.k) for record in records})
    return groups


def values_for_group(
    records: list[MetricRecord],
    case: str,
    group: tuple[int, int, int],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return h and metric arrays for one case/group pair."""

    l, m, k = group
    subset = [
        record
        for record in records
        if record.case == case and record.l == l and record.m == m and record.k == k
    ]
    subset = sorted(subset, key=lambda record: record.h)
    h = np.array([record.h for record in subset], dtype=float)
    y = np.array([getattr(record, metric_name) for record in subset], dtype=float)
    mask = np.isfinite(y) & (y > 0.0)
    return h[mask], y[mask]


def add_reference_slope(
    ax: plt.Axes,
    slope: float,
    y_values: list[np.ndarray],
    linestyle: str = "-",
) -> None:
    """Add a reference convergence line spanning the full h range."""

    positive_values = np.concatenate([values[np.isfinite(values) & (values > 0.0)] for values in y_values])
    if positive_values.size == 0:
        return

    h0 = H_VALUES[-1]
    h1 = H_VALUES[0]
    y0 = positive_values.max() * 0.8
    y1 = y0 * (h1 / h0) ** slope
    ax.plot(
        [h0, h1],
        [y0, y1],
        color="black",
        linestyle=linestyle,
        linewidth=1.3,
        zorder=5,
        label=rf"$\mathcal{{O}}(h^{{{slope:g}}})$",
    )


def reference_handles_labels(ax: plt.Axes) -> tuple[list, list]:
    """Return only reference-line legend entries for a panel."""

    handles, labels = ax.get_legend_handles_labels()
    refs = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if r"\mathcal{O}" in label
    ]
    if not refs:
        return [], []
    return [handle for handle, _ in refs], [label for _, label in refs]


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.04,
        0.94,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )


def plot_panel(
    ax: plt.Axes,
    records: list[MetricRecord],
    groups: list[tuple[int, int, int]],
    case: str,
    metric_name: str,
    ylabel: str | None,
    panel_label: str,
) -> None:
    plotted_values: list[np.ndarray] = []

    for idx, group in enumerate(groups):
        h, y = values_for_group(records, case, group, metric_name)
        if h.size == 0:
            continue

        l, m, k = group
        label = f"l{l}_m{m}"

        ax.loglog(
            h,
            y,
            marker=MARKERS[idx % len(MARKERS)],
            markersize=4.0,
            linewidth=1.4,
            color=COLORS[idx % len(COLORS)],
            label=label,
            zorder=6,
        )
        plotted_values.append(y)

    if case in ("case2", "case4") and metric_name == "v_l2_norm":
        add_reference_slope(ax, 2.0, plotted_values, linestyle="-")
        add_reference_slope(ax, 3.0, plotted_values, linestyle="--")
    else:
        slope_key = "velocity_slope" if metric_name == "v_l2_norm" else "pressure_slope"
        add_reference_slope(ax, CASE_INFO[case][slope_key], plotted_values)

    ax.set_xlim(H_VALUES[0] * 0.82, H_VALUES[-1] * 1.18)
    ax.set_xticks(H_VALUES)
    ax.set_xticklabels(H_TICK_LABELS)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", linewidth=0.45, color="0.84")
    ax.tick_params(axis="both", which="both", direction="in", labelsize=9)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    add_panel_label(ax, panel_label)


def main() -> None:
    records = read_metrics(METRICS_ROOT)
    if not records:
        raise FileNotFoundError(f"No benchmark_metrics.h5 files found in {METRICS_ROOT}")

    groups = grouped_lm(records)
    fig, axes = plt.subplots(4, 2, figsize=(7.4, 10.0), sharex=True)

    for ax, (case, metric_name, ylabel, panel_label) in zip(axes.flat, PANEL_LAYOUT):
        plot_panel(ax, records, groups, case, metric_name, ylabel, panel_label)

    axes[0, 0].set_title("Delta-Function", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Smooth", fontsize=12, fontweight="bold")
    axes[2, 0].set_title("Delta-Function", fontsize=12, fontweight="bold")
    axes[2, 1].set_title("Smooth", fontsize=12, fontweight="bold")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$h$")

    fig.text(0.055, 0.725, "Free-Slip", rotation=90, va="center", ha="center", fontsize=15, fontweight="bold")
    fig.text(0.055, 0.29, "Zero-Slip", rotation=90, va="center", ha="center", fontsize=15, fontweight="bold")

    data_handles, data_labels = axes[0, 0].get_legend_handles_labels()
    series_handles = [
        handle for handle, label in zip(data_handles, data_labels) if r"\mathcal{O}" not in label
    ]
    series_labels = [label for label in data_labels if r"\mathcal{O}" not in label]
    reference_handle, reference_label = reference_handles_labels(axes[0, 0])

    axes[0, 0].legend(
        reference_handle,
        reference_label,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.85",
        fontsize=7.5,
    )
    fig.legend(
        series_handles,
        series_labels,
        loc="lower center",
        ncol=6,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.85",
        fontsize=8.0,
        bbox_to_anchor=(0.58, -0.01),
    )

    for ax in list(axes.flat)[1:]:
        handles, labels = reference_handles_labels(ax)
        if handles:
            ax.legend(
                handles,
                labels,
                loc="lower right",
                frameon=True,
                framealpha=0.9,
                facecolor="white",
                edgecolor="0.85",
                fontsize=7.5,
            )

    fig.subplots_adjust(left=0.17, right=0.985, top=0.965, bottom=0.082, wspace=0.25)
    fig.canvas.draw()
    block_gap = 0.025
    for ax in axes[2:, :].flat:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - block_gap, pos.width, pos.height])
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
