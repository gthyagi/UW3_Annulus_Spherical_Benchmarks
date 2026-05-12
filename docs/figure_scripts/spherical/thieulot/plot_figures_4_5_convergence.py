#!/usr/bin/env python3
"""Create combined spherical Thieulot Figure 4/5 convergence plots."""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter


OUTPUT_ROOT = Path("/Volumes/seagate4_1/output/spherical/thieulot/latest")
SCRIPT_DIR = Path(__file__).resolve().parent
OUTFILE = SCRIPT_DIR / "figures_4_5_thieulot_convergence.pdf"

X_TICKS = [1.0 / value for value in (128, 64, 32, 16, 8)]
X_TICK_LABELS = [r"$1/128$", r"$1/64$", r"$1/32$", r"$1/16$", r"$1/8$"]
H_MIN = 1.0 / 128.0
H_MAX = 1.0 / 8.0

CASE_RE = re.compile(
    r"case_inv_lc_(?P<inv_lc>\d+)_"
    r"m_(?P<m>-?\d+)_"
    r"vdeg_2_pdeg_1_pcont_true_"
    r"stokes_tol_1e-09_ncpus_(?P<ncpus>\d+)_bc_essential$"
)


def read_metric(dataset: h5py.File, name: str) -> float:
    """Read a scalar metric from a benchmark HDF5 file."""

    value = dataset[name][()]
    return float(value.item() if hasattr(value, "item") else value)


def load_convergence_data(output_root: Path) -> dict[int, list[dict[str, float]]]:
    """Load available m=-1 and m=3 Q2Q1 convergence metrics."""

    data: dict[int, list[dict[str, float]]] = {-1: [], 3: []}

    for metrics_file in sorted(output_root.glob("case_inv_lc_*_m_*_*/benchmark_metrics.h5")):
        match = CASE_RE.match(metrics_file.parent.name)
        if match is None:
            continue

        m = int(match.group("m"))
        if m not in data:
            continue

        with h5py.File(metrics_file, "r") as h5:
            inv_lc = int(match.group("inv_lc"))

            data[m].append(
                {
                    "inv_lc": inv_lc,
                    "ncpus": int(match.group("ncpus")),
                    "h": 1.0 / inv_lc,
                    "v_l2_norm": read_metric(h5, "v_l2_norm"),
                    "p_l2_norm": read_metric(h5, "p_l2_norm"),
                }
            )

    for rows in data.values():
        rows.sort(key=lambda row: row["h"])

    return data


def add_reference_slope(
    ax: plt.Axes,
    exponent: float,
    x0: float,
    y0: float,
    x1: float,
    label: str,
) -> None:
    """Add a log-log reference slope anchored at (x0, y0)."""

    y1 = y0 * (x1 / x0) ** exponent
    ax.plot([x0, x1], [y0, y1], color="black", linewidth=1.4)
    ax.text(x1 * 1.05, y1, label, fontsize=12, va="center")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Add a panel label without interfering with the legend."""

    ax.text(
        -0.07,
        0.99,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
    )


def plot_metric(
    ax: plt.Axes,
    data: dict[int, list[dict[str, float]]],
    metric_name: str,
    ylabel: str,
    show_legend: bool,
) -> None:
    """Plot one convergence metric for m=-1 and m=3."""

    styles = {
        -1: {"marker": "o", "color": "#008080", "label": r"$P_2P_1$, $m=-1$"},
        3: {"marker": "^", "color": "#D55E00", "label": r"$P_2P_1$, $m=3$"},
    }

    for m, rows in data.items():
        if not rows:
            continue

        h_values = [row["h"] for row in rows]
        metric_values = [row[metric_name] for row in rows]
        ax.loglog(
            h_values,
            metric_values,
            color=styles[m]["color"],
            marker=styles[m]["marker"],
            linewidth=1.3,
            markersize=5.0,
            label=styles[m]["label"],
        )

    ax.set_xlabel(r"$h$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1.0 / 160.0, 1.0 / 6.0)
    ax.xaxis.set_major_locator(FixedLocator(X_TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter(X_TICK_LABELS))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="0.72", linewidth=0.7)
    ax.grid(True, which="minor", color="0.88", linewidth=0.45)
    if show_legend:
        ax.legend(
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="none",
            fontsize=10,
            loc="lower right",
        )


def main() -> None:
    data = load_convergence_data(OUTPUT_ROOT)

    missing = [m for m, rows in data.items() if not rows]
    if missing:
        raise RuntimeError(f"Missing convergence metrics for m values: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)

    plot_metric(
        axes[0],
        data,
        "v_l2_norm",
        r"$E_{L_2}^{\mathrm{rel}}(\mathbf{u})$",
        show_legend=False,
    )
    # axes[0].set_title("Velocity error")
    # add_reference_slope(axes[0], 2.0, H_MIN, 2.0e-5, H_MAX, r"$h^2$")
    add_reference_slope(axes[0], 3.0, H_MIN, 3.5e-6, H_MAX, r"$h^3$")
    add_panel_label(axes[0], "a)")

    plot_metric(
        axes[1],
        data,
        "p_l2_norm",
        r"$E_{L_2}^{\mathrm{rel}}(p)$",
        show_legend=True,
    )
    # axes[1].set_title("Pressure error")
    add_reference_slope(axes[1], 2.0, H_MIN, 6.0e-4, H_MAX, r"$h^2$")
    add_panel_label(axes[1], "b)")

    fig.savefig(OUTFILE, bbox_inches="tight")
    print(f"Wrote {OUTFILE}")


if __name__ == "__main__":
    main()
