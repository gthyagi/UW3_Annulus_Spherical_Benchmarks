#!/usr/bin/env python3
"""Plot spherical Thieulot sigma_rr boundary L2 convergence."""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter


OUTPUT_ROOT = Path("/Volumes/seagate4_1/output/spherical/thieulot/latest")
SCRIPT_DIR = Path(__file__).resolve().parent
OUTFILE = SCRIPT_DIR / "sigma_rr_boundary_convergence.pdf"

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


def load_sigma_rr_data(output_root: Path) -> dict[int, list[dict[str, float]]]:
    """Load sigma_rr boundary errors for m=-1 and m=3."""

    data: dict[int, list[dict[str, float]]] = {-1: [], 3: []}

    for metrics_file in sorted(output_root.glob("case_inv_lc_*_m_*_*/benchmark_metrics.h5")):
        match = CASE_RE.match(metrics_file.parent.name)
        if match is None:
            continue

        m = int(match.group("m"))
        if m not in data:
            continue

        inv_lc = int(match.group("inv_lc"))
        with h5py.File(metrics_file, "r") as h5:
            data[m].append(
                {
                    "inv_lc": inv_lc,
                    "ncpus": int(match.group("ncpus")),
                    "h": 1.0 / inv_lc,
                    "sigma_rr_l2_norm_lower": read_metric(
                        h5, "sigma_rr_l2_norm_lower"
                    ),
                    "sigma_rr_l2_norm_upper": read_metric(
                        h5, "sigma_rr_l2_norm_upper"
                    ),
                }
            )

    for rows in data.values():
        rows.sort(key=lambda row: row["h"])

    return data


def plot_boundary_metrics(
    ax: plt.Axes,
    data: dict[int, list[dict[str, float]]],
) -> None:
    """Plot inner and outer sigma_rr boundary metrics together."""

    styles = {
        -1: {"marker": "o", "color": "tab:blue", "label": r"$m=-1$"},
        3: {"marker": "^", "color": "tab:orange", "label": r"$m=3$"},
    }
    boundaries = {
        "sigma_rr_l2_norm_lower": {"linestyle": "-", "label": "inner"},
        "sigma_rr_l2_norm_upper": {"linestyle": "--", "label": "outer"},
    }

    for m, rows in data.items():
        if not rows:
            continue

        h_values = [row["h"] for row in rows]
        for metric_name, boundary_style in boundaries.items():
            metric_values = [row[metric_name] for row in rows]
            ax.loglog(
                h_values,
                metric_values,
                color=styles[m]["color"],
                linestyle=boundary_style["linestyle"],
                marker=styles[m]["marker"],
                linewidth=1.3,
                markersize=5.0,
                label=f"{styles[m]['label']}, {boundary_style['label']}",
            )

    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$|\sigma_{rr} - \sigma_{rr}^{ana}|_2$")
    ax.set_xlim(1.0 / 160.0, 1.0 / 6.0)
    ax.xaxis.set_major_locator(FixedLocator(X_TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter(X_TICK_LABELS))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="0.72", linewidth=0.7)
    ax.grid(True, which="minor", color="0.88", linewidth=0.45)
    ax.legend(frameon=False, fontsize=10, loc="lower right")


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
    ax.plot([x0, x1], [y0, y1], color="black", linewidth=1.2)
    ax.text(x1 * 1.05, y1, label, fontsize=12, va="center")


def main() -> None:
    data = load_sigma_rr_data(OUTPUT_ROOT)

    missing = [m for m, rows in data.items() if not rows]
    if missing:
        raise RuntimeError(f"Missing sigma_rr metrics for m values: {missing}")

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.4), constrained_layout=True)

    plot_boundary_metrics(ax, data)
    add_reference_slope(ax, 1.0, H_MIN, 7.0e-3, H_MAX, r"$h$")
    add_reference_slope(ax, 2.0, H_MIN, 2.0e-4, H_MAX, r"$h^2$")
    # ax.text(
    #     -0.07,
    #     0.99,
    #     "a)",
    #     transform=ax.transAxes,
    #     fontsize=14,
    #     fontweight="bold",
    #     va="top",
    #     ha="left",
    # )

    fig.savefig(OUTFILE, bbox_inches="tight")
    print(f"Wrote {OUTFILE}")


if __name__ == "__main__":
    main()
