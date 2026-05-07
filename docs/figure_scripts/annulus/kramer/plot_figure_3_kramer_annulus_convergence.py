#!/usr/bin/env python3
"""Reproduce Kramer et al. annulus benchmark Figure 3 from UW3 metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


METRICS_ROOT = Path("/Volumes/seagate4_1/output/annulus/kramer/latest")
OUTPUT_FILE = Path(__file__).resolve().parent / "figure_3_kramer_annulus_convergence.pdf"

INV_LC_BASE = 8
INV_LC_VALUES = np.array([8, 16, 32, 64, 128, 256], dtype=int)
REFINEMENT_LEVELS = np.array([1 + int(np.log2(inv_lc // INV_LC_BASE)) for inv_lc in INV_LC_VALUES])
H_VALUES = 1.0 / INV_LC_VALUES.astype(float)
H_TICK_LABELS = tuple(f"1/{inv_lc}" for inv_lc in INV_LC_VALUES)

N_VALUES = (2, 8, 32)
MARKERS = {2: "o", 8: "*", 32: "s"}

DIR_PATTERN = re.compile(
    r"(?P<case>case\d+)_"
    r"inv_lc_(?P<inv_lc>\d+)_"
    r"n_(?P<n>\d+)_"
    r"(?:k_(?P<k>\d+)_)?"
)


@dataclass(frozen=True)
class MetricRecord:
    case: str
    inv_lc: int
    level: int
    n: int
    k: int | None
    v_l2_norm: float
    p_l2_norm: float


def read_scalar(h5f: h5py.File, name: str) -> float:
    value = h5f[name][()]
    if isinstance(value, bytes):
        return value.decode()
    return float(np.asarray(value))


def read_metrics() -> list[MetricRecord]:
    records: list[MetricRecord] = []

    for metric_file in sorted(METRICS_ROOT.glob("case*_inv_lc_*/benchmark_metrics.h5")):
        match = DIR_PATTERN.search(metric_file.parent.name)
        if match is None:
            continue

        case = match.group("case")
        if case not in {"case1", "case2", "case3", "case4"}:
            continue

        inv_lc = int(match.group("inv_lc"))
        if inv_lc not in set(INV_LC_VALUES):
            continue

        with h5py.File(metric_file, "r") as h5f:
            records.append(
                MetricRecord(
                    case=case,
                    inv_lc=inv_lc,
                    level=1 + int(np.log2(inv_lc // INV_LC_BASE)),
                    n=int(read_scalar(h5f, "n")),
                    k=None if match.group("k") is None else int(read_scalar(h5f, "k")),
                    v_l2_norm=read_scalar(h5f, "v_l2_norm"),
                    p_l2_norm=read_scalar(h5f, "p_l2_norm"),
                )
            )

    return records


def values_for_series(
    records: list[MetricRecord],
    *,
    case: str,
    n: int,
    k: int | None,
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    subset = [
        record
        for record in records
        if record.case == case and record.n == n and record.k == k
    ]
    subset = sorted(subset, key=lambda record: record.level)
    x = np.array([1.0 / record.inv_lc for record in subset], dtype=float)
    y = np.array([getattr(record, metric_name) for record in subset], dtype=float)
    mask = np.isfinite(y) & (y > 0.0)
    return x[mask], y[mask]


def add_reference_line(
    ax: plt.Axes,
    h_values: np.ndarray,
    slope: float,
    series: list[tuple[np.ndarray, np.ndarray]],
    linestyle: str = "-",
) -> None:
    values = [y for _, y in series]
    positive = [value[np.isfinite(value) & (value > 0.0)] for value in values if value.size]
    if not positive:
        return

    h0 = h_values[-1]
    lower_bounds = []
    upper_bounds = []
    for h_value in h_values:
        values_at_h = []
        for h_series, y_series in series:
            matches = np.isclose(h_series, h_value)
            values_at_h.extend(y_series[matches & np.isfinite(y_series) & (y_series > 0.0)])

        if not values_at_h:
            continue

        values_at_h = np.array(values_at_h, dtype=float)
        scale = (h_value / h0) ** slope
        lower_bounds.append(values_at_h.min() / scale)
        upper_bounds.append(values_at_h.max() / scale)

    if lower_bounds and max(lower_bounds) <= min(upper_bounds):
        y0 = np.exp(0.5 * (np.log(max(lower_bounds)) + np.log(min(upper_bounds))))
    else:
        implied_anchors = []
        for h_series, y_series in series:
            valid = np.isfinite(y_series) & (y_series > 0.0)
            implied_anchors.extend(y_series[valid] / (h_series[valid] / h0) ** slope)
        implied_anchors = np.array(implied_anchors, dtype=float)
        y0 = np.exp(np.median(np.log(implied_anchors)))

    y_ref = y0 * (h_values / h0) ** slope
    ax.loglog(
        h_values,
        y_ref,
        color="black",
        linestyle=linestyle,
        linewidth=1.0,
        label=rf"$\mathcal{{O}}(h^{{{slope:g}}})$",
        zorder=6,
    )


def plot_panel(
    ax: plt.Axes,
    records: list[MetricRecord],
    *,
    panel_label: str,
    case: str,
    k: int | None,
    metric_name: str,
    slope: float,
    extra_slopes: tuple[float, ...],
    show_xlabel: bool,
) -> None:
    plotted_values: list[np.ndarray] = []
    plotted_series: list[tuple[np.ndarray, np.ndarray]] = []

    for n in N_VALUES:
        x, y = values_for_series(records, case=case, n=n, k=k, metric_name=metric_name)
        if x.size == 0:
            continue

        ax.loglog(
            x,
            y,
            color="black",
            marker=MARKERS.get(n, "o"),
            linestyle="None",
            markersize=4.2,
            label="_nolegend_",
            zorder=7,
        )
        plotted_values.append(y)
        plotted_series.append((x, y))

    main_linestyle = "--" if metric_name == "v_l2_norm" and k is not None and slope == 3.0 else "-"
    add_reference_line(ax, H_VALUES, slope, plotted_series, linestyle=main_linestyle)
    for extra_slope in extra_slopes:
        add_reference_line(ax, H_VALUES, extra_slope, plotted_series, linestyle="-")

    ax.set_xlim(H_VALUES[-1] * 0.82, H_VALUES[0] * 1.18)
    ax.set_xticks(H_VALUES)
    if show_xlabel:
        ax.set_xticklabels(H_TICK_LABELS)
        ax.set_xlabel(r"$h$", labelpad=1.0)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    ax.grid(True, which="major", color="0.78", linewidth=0.45)
    ax.grid(True, which="minor", color="0.88", linewidth=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=8)
    ax.text(0.02, 0.98, f"({panel_label})", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.85",
        fontsize=6.6,
        handlelength=1.6,
        borderpad=0.25,
        labelspacing=0.25,
    )


def main() -> None:
    records = read_metrics()
    if not records:
        raise FileNotFoundError(f"No benchmark_metrics.h5 files found in {METRICS_ROOT}")

    fig, axes = plt.subplots(4, 3, figsize=(8.3, 11.0), sharex=False)

    columns = (
        ("Delta-Function", None),
        ("Smooth (k=2)", 2),
        ("Smooth (k=8)", 8),
    )
    panel_labels = np.array([["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"], ["j", "k", "l"]])

    row_specs = (
        ("case1", "v_l2_norm", 1.5, r"$\|e_\mathbf{u}\|_{L_2}$"),
        ("case1", "p_l2_norm", 0.5, r"$\|e_p\|_{L_2}$"),
        ("case3", "v_l2_norm", 1.5, r"$\|e_\mathbf{u}\|_{L_2}$"),
        ("case3", "p_l2_norm", 0.5, r"$\|e_p\|_{L_2}$"),
    )

    for row, (delta_case, metric_name, delta_slope, ylabel) in enumerate(row_specs):
        for col, (title, k) in enumerate(columns):
            if k is None:
                case = delta_case
                slope = delta_slope
            else:
                case = "case2" if delta_case == "case1" else "case4"
                slope = 3.0 if metric_name == "v_l2_norm" else 2.0
            extra_slopes = (2.0,) if k is not None and metric_name == "v_l2_norm" else ()

            plot_panel(
                axes[row, col],
                records,
                panel_label=panel_labels[row, col],
                case=case,
                k=k,
                metric_name=metric_name,
                slope=slope,
                extra_slopes=extra_slopes,
                show_xlabel=row == 3,
            )
            if col == 0:
                axes[row, col].set_ylabel(ylabel, fontsize=9)
            if row in (0, 2):
                axes[row, col].set_title(title, fontsize=10, pad=4)

    fig.text(0.53, 0.965, "Free-Slip", ha="center", va="center", fontsize=18)
    fig.text(0.53, 0.515, "Zero-Slip", ha="center", va="center", fontsize=18)

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.13,
        top=0.93,
        wspace=0.32,
        hspace=0.1,
    )

    # Open visual space between the two boundary-condition groups, matching the paper layout.
    for ax in axes[2:, :].flat:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.035, pos.width, pos.height])

    marker_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=MARKERS[n],
            linestyle="None",
            markersize=5,
            label=rf"$n={n}$",
        )
        for n in N_VALUES
    ]
    fig.legend(
        marker_handles,
        [handle.get_label() for handle in marker_handles],
        loc="lower center",
        ncol=3,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.85",
        fontsize=8.5,
        bbox_to_anchor=(0.545, 0.03),
    )

    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
