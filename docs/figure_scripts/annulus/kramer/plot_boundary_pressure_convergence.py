#!/usr/bin/env python3
"""Plot annulus Kramer boundary pressure convergence from UW3 metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


METRICS_ROOT = Path("/Volumes/seagate4_1/output/annulus/kramer/latest")
OUTPUT_FILE = Path(__file__).resolve().parent / "figure_boundary_pressure_convergence.pdf"

INV_LC_BASE = 8
INV_LC_VALUES = np.array([8, 16, 32, 64, 128, 256], dtype=int)
H_VALUES = 1.0 / INV_LC_VALUES.astype(float)
H_TICK_LABELS = tuple(f"1/{inv_lc}" for inv_lc in INV_LC_VALUES)

N_VALUES = (2, 8, 32)
COLORS = {2: "#0072B2", 8: "#D55E00", 32: "#009E73"}
MARKERS = {2: "o", 8: "*", 32: "s"}
BOUNDARIES = (
    ("p_l2_norm_lower", "inner", "-"),
    ("p_l2_norm_upper", "outer", "--"),
)

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
    p_l2_norm_lower: float
    p_l2_norm_upper: float


def read_scalar(h5f: h5py.File, name: str) -> float:
    value = h5f[name][()]
    if isinstance(value, bytes):
        return float(value.decode())
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
                    p_l2_norm_lower=read_scalar(h5f, "p_l2_norm_lower"),
                    p_l2_norm_upper=read_scalar(h5f, "p_l2_norm_upper"),
                )
            )

    return records


def validate_records(records: list[MetricRecord]) -> None:
    if not records:
        raise FileNotFoundError(f"No benchmark_metrics.h5 files found in {METRICS_ROOT}")

    required_panels = (
        ("case1", None, "free-slip delta"),
        ("case2", 2, "free-slip smooth k=2"),
        ("case2", 8, "free-slip smooth k=8"),
        ("case3", None, "zero-slip delta"),
        ("case4", 2, "zero-slip smooth k=2"),
        ("case4", 8, "zero-slip smooth k=8"),
    )

    missing: list[str] = []
    for case, k, label in required_panels:
        for n in N_VALUES:
            available = {
                record.inv_lc
                for record in records
                if record.case == case and record.k == k and record.n == n
            }
            missing_inv_lc = [inv_lc for inv_lc in INV_LC_VALUES if inv_lc not in available]
            if missing_inv_lc:
                missing.append(
                    f"{label}, n={n}: "
                    + ", ".join(f"1/{inv_lc}" for inv_lc in missing_inv_lc)
                )

    if missing:
        raise RuntimeError("Incomplete Kramer boundary pressure data: " + "; ".join(missing))


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
) -> None:
    positive = [
        y[np.isfinite(y) & (y > 0.0)]
        for _, y in series
        if y.size and np.any(np.isfinite(y) & (y > 0.0))
    ]
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
        linestyle=":",
        linewidth=1.15,
        zorder=5,
        label="_nolegend_",
    )


def plot_panel(
    ax: plt.Axes,
    records: list[MetricRecord],
    *,
    panel_label: str,
    case: str,
    k: int | None,
    slope: float,
    show_xlabel: bool,
) -> None:
    plotted_series: list[tuple[np.ndarray, np.ndarray]] = []

    for n in N_VALUES:
        for metric_name, _, linestyle in BOUNDARIES:
            x, y = values_for_series(records, case=case, n=n, k=k, metric_name=metric_name)
            if x.size == 0:
                continue

            ax.loglog(
                x,
                y,
                color=COLORS[n],
                marker=MARKERS[n],
                linestyle=linestyle,
                linewidth=1.15,
                markersize=4.2,
                zorder=8,
                label="_nolegend_",
            )
            plotted_series.append((x, y))

    add_reference_line(ax, H_VALUES, slope, plotted_series)

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
    ax.text(
        0.96,
        0.06,
        rf"$\mathcal{{O}}(h^{{{slope:g}}})$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )


def main() -> None:
    records = read_metrics()
    validate_records(records)

    fig, axes = plt.subplots(2, 3, figsize=(8.3, 5.7), sharex=False)

    columns = (
        ("Delta-Function", None, 0.5),
        ("Smooth (k=2)", 2, 2.0),
        ("Smooth (k=8)", 8, 2.0),
    )
    rows = (
        ("Free-Slip", "case1", "case2"),
        ("Zero-Slip", "case3", "case4"),
    )
    panel_labels = np.array([["a", "b", "c"], ["d", "e", "f"]])

    for row, (row_label, delta_case, smooth_case) in enumerate(rows):
        for col, (title, k, slope) in enumerate(columns):
            case = delta_case if k is None else smooth_case
            plot_panel(
                axes[row, col],
                records,
                panel_label=panel_labels[row, col],
                case=case,
                k=k,
                slope=slope,
                show_xlabel=row == 1,
            )
            if row == 0:
                axes[row, col].set_title(title, fontsize=10, pad=4)
            if col == 0:
                axes[row, col].set_ylabel(
                    row_label + "\n" + r"$E_{L_2,\Gamma}^{\mathrm{rel}}(p)$",
                    fontsize=9,
                )

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.22,
        top=0.925,
        wspace=0.30,
        hspace=0.14,
    )

    n_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[n],
            marker=MARKERS[n],
            linestyle="-",
            linewidth=1.15,
            markersize=5,
            label=rf"$n={n}$",
        )
        for n in N_VALUES
    ]
    boundary_handles = [
        Line2D([0], [0], color="black", linestyle=linestyle, linewidth=1.15, label=label)
        for _, label, linestyle in BOUNDARIES
    ]

    handles = n_handles + boundary_handles
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        ncol=5,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.85",
        fontsize=8.0,
        bbox_to_anchor=(0.545, 0.055),
        columnspacing=1.1,
        handlelength=1.8,
    )

    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
