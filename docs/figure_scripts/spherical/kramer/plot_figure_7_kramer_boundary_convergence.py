#!/usr/bin/env python3
"""Reproduce Kramer et al. spherical benchmark Figure 7 from UW3 metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

METRICS_ROOT = Path("/Volumes/seagate4_1/output/spherical/kramer/latest")
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "figure_7_kramer_boundary_convergence.pdf"

MARKERS = ("o", "s", "^", "D", "v", "P")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
H_VALUES = np.array([1 / 64, 1 / 32, 1 / 16, 1 / 8], dtype=float)
H_TICK_LABELS = ("1/64", "1/32", "1/16", "1/8")

DIR_PATTERN = re.compile(
    r"case1_"
    r"inv_lc_(?P<inv_lc>\d+)_"
    r"l_(?P<l>\d+)_"
    r"m_(?P<m>\d+)_"
    r"k_(?P<k>\d+)_"
)


@dataclass(frozen=True)
class MetricRecord:
    h: float
    l: int
    m: int
    k: int
    v_l2_norm_upper: float
    v_l2_norm_lower: float
    sigma_rr_l2_norm_upper: float
    sigma_rr_l2_norm_lower: float


def read_scalar(h5f: h5py.File, name: str) -> float:
    return float(np.asarray(h5f[name][()]))


def read_metrics() -> list[MetricRecord]:
    """Read case1 free-slip delta-function boundary metric files."""

    records: list[MetricRecord] = []
    for metric_file in sorted(METRICS_ROOT.glob("case1_inv_lc_*/benchmark_metrics.h5")):
        match = DIR_PATTERN.search(metric_file.parent.name)
        if match is None:
            continue

        with h5py.File(metric_file, "r") as h5f:
            records.append(
                MetricRecord(
                    h=read_scalar(h5f, "cellsize"),
                    l=int(read_scalar(h5f, "l")),
                    m=int(read_scalar(h5f, "m")),
                    k=int(read_scalar(h5f, "k")),
                    v_l2_norm_upper=read_scalar(h5f, "v_l2_norm_upper"),
                    v_l2_norm_lower=read_scalar(h5f, "v_l2_norm_lower"),
                    sigma_rr_l2_norm_upper=read_scalar(h5f, "sigma_rr_l2_norm_upper"),
                    sigma_rr_l2_norm_lower=read_scalar(h5f, "sigma_rr_l2_norm_lower"),
                )
            )

    if not records:
        raise FileNotFoundError(f"No case1 benchmark_metrics.h5 files found in {METRICS_ROOT}")

    return records


def grouped_lm(records: list[MetricRecord]) -> list[tuple[int, int, int]]:
    return sorted({(record.l, record.m, record.k) for record in records})


def values_for_group(
    records: list[MetricRecord],
    group: tuple[int, int, int],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    l_val, m_val, k_val = group
    subset = [
        record
        for record in records
        if record.l == l_val and record.m == m_val and record.k == k_val
    ]
    subset = sorted(subset, key=lambda record: record.h)
    h = np.array([record.h for record in subset], dtype=float)
    y = np.array([getattr(record, metric_name) for record in subset], dtype=float)
    mask = np.isfinite(y) & (y > 0.0)
    return h[mask], y[mask]


def add_reference_slope(ax: plt.Axes, slope: float, y_values: list[np.ndarray]) -> None:
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
        linestyle="-",
        linewidth=1.3,
        zorder=5,
        label=rf"$\mathcal{{O}}(h^{{{slope:g}}})$",
    )


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
    metric_name: str,
    slope: float,
    ylabel: str,
    panel_label: str,
) -> None:
    plotted_values: list[np.ndarray] = []

    for idx, group in enumerate(groups):
        h, y = values_for_group(records, group, metric_name)
        if h.size == 0:
            continue

        l_val, m_val, _ = group
        ax.loglog(
            h,
            y,
            marker=MARKERS[idx % len(MARKERS)],
            markersize=4.0,
            linewidth=1.4,
            color=COLORS[idx % len(COLORS)],
            label=f"l{l_val}_m{m_val}",
            zorder=6,
        )
        plotted_values.append(y)

    add_reference_slope(ax, slope, plotted_values)
    ax.set_xlim(H_VALUES[0] * 0.82, H_VALUES[-1] * 1.18)
    ax.set_xticks(H_VALUES)
    ax.set_xticklabels(H_TICK_LABELS)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", linewidth=0.45, color="0.84")
    ax.tick_params(axis="both", which="both", direction="in", labelsize=9)
    ax.set_ylabel(ylabel)
    add_panel_label(ax, panel_label)


def main() -> None:
    records = read_metrics()
    groups = grouped_lm(records)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), sharex=True)

    plot_panel(
        axes[0, 0],
        records,
        groups,
        "v_l2_norm_upper",
        3.0,
        r"$E_{L_2,\Gamma_{\mathrm{outer}}}^{\mathrm{rel}}(\mathbf{u})$",
        "a)",
    )
    plot_panel(
        axes[0, 1],
        records,
        groups,
        "v_l2_norm_lower",
        3.0,
        r"$E_{L_2,\Gamma_{\mathrm{inner}}}^{\mathrm{rel}}(\mathbf{u})$",
        "b)",
    )
    plot_panel(
        axes[1, 0],
        records,
        groups,
        "sigma_rr_l2_norm_upper",
        2.0,
        r"$E_{L_2,\Gamma_{\mathrm{outer}}}^{\mathrm{rel}}(\sigma_{rr})$",
        "c)",
    )
    plot_panel(
        axes[1, 1],
        records,
        groups,
        "sigma_rr_l2_norm_lower",
        2.0,
        r"$E_{L_2,\Gamma_{\mathrm{inner}}}^{\mathrm{rel}}(\sigma_{rr})$",
        "d)",
    )

    axes[0, 0].set_title("Outer boundary", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Inner boundary", fontsize=12, fontweight="bold")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$h$")

    data_handles, data_labels = axes[0, 0].get_legend_handles_labels()
    series_handles = data_handles[:-1]
    series_labels = data_labels[:-1]
    reference_handle = data_handles[-1:]
    reference_label = data_labels[-1:]

    axes[0, 0].legend(
        series_handles + reference_handle,
        series_labels + reference_label,
        loc="lower right",
        ncol=2,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.85",
        fontsize=7.5,
    )
    for ax in axes.flat[1:]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles[-1:],
                labels[-1:],
                loc="lower right",
                frameon=True,
                framealpha=0.9,
                facecolor="white",
                edgecolor="0.85",
                fontsize=7.5,
            )

    fig.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.11, wspace=0.25, hspace=0.22)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
