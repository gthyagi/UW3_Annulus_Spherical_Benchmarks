#!/usr/bin/env python3
"""Plot h=1/128 boundary pressure absolute errors for tested element pairs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

METRICS_ROOT = Path("/Volumes/seagate4_1/output/annulus/thieulot/latest")
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "figure_boundary_pressure_error_by_element_h128.pdf"

H_TARGET = 1.0 / 128.0
K_VALUES = (1, 4, 8)
ELEMENTS = (
    (1, 0, False, r"$P_1\times P_0$"),
    (1, 1, True, r"$P_1\times P_1$"),
    (2, 0, False, r"$P_2\times P_0$"),
    (2, 1, False, r"$P_2\times P_{-1}$"),
    (2, 1, True, r"$P_2\times P_1$"),
    (3, 2, True, r"$P_3\times P_2$"),
)
BOUNDARIES = (
    ("p_l2_norm_lower_abs", "Inner boundary", "a)"),
    ("p_l2_norm_upper_abs", "Outer boundary", "b)"),
)

COLORS = {1: "#0072B2", 4: "#D55E00", 8: "#009E73"}

DIR_PATTERN = re.compile(
    r"model_inv_lc_(?P<inv_lc>\d+)_"
    r"k_(?P<k>\d+)_"
    r"vdeg_(?P<vdeg>\d+)_"
    r"pdeg_(?P<pdeg>\d+)_"
    r"pcont_(?P<pcont>true|false)_"
    r".*_bc_essential$"
)


@dataclass(frozen=True)
class MetricRecord:
    h: float
    k: int
    vdegree: int
    pdegree: int
    pcont: bool
    p_l2_norm_lower_abs: float
    p_l2_norm_upper_abs: float


def read_scalar(h5f: h5py.File, name: str) -> float:
    value = h5f[name][()]
    if isinstance(value, bytes):
        return float(value.decode())
    return float(np.asarray(value))


def read_records() -> dict[tuple[int, int, bool, int], MetricRecord]:
    valid_elements = {(vdegree, pdegree, pcont) for vdegree, pdegree, pcont, _ in ELEMENTS}
    records: dict[tuple[int, int, bool, int], MetricRecord] = {}

    for metrics_file in sorted(METRICS_ROOT.glob("model_inv_lc_*/benchmark_metrics.h5")):
        match = DIR_PATTERN.search(metrics_file.parent.name)
        if match is None:
            continue

        vdegree = int(match.group("vdeg"))
        pdegree = int(match.group("pdeg"))
        pcont = match.group("pcont") == "true"
        k = int(match.group("k"))
        if (vdegree, pdegree, pcont) not in valid_elements or k not in K_VALUES:
            continue

        with h5py.File(metrics_file, "r") as h5f:
            h = read_scalar(h5f, "cellsize")
            if not np.isclose(h, H_TARGET):
                continue

            records[(vdegree, pdegree, pcont, k)] = MetricRecord(
                h=h,
                k=k,
                vdegree=vdegree,
                pdegree=pdegree,
                pcont=pcont,
                p_l2_norm_lower_abs=read_scalar(h5f, "p_l2_norm_lower_abs"),
                p_l2_norm_upper_abs=read_scalar(h5f, "p_l2_norm_upper_abs"),
            )

    return records


def validate_records(records: dict[tuple[int, int, bool, int], MetricRecord]) -> None:
    missing = []
    for vdegree, pdegree, pcont, element_label in ELEMENTS:
        for k in K_VALUES:
            if (vdegree, pdegree, pcont, k) not in records:
                missing.append(f"{element_label}, k={k}")

    if missing:
        raise RuntimeError(
            "Incomplete h=1/128 boundary pressure data: "
            + "; ".join(missing)
            + f" in {METRICS_ROOT}"
        )


def plot_boundary_panel(
    ax: plt.Axes,
    records: dict[tuple[int, int, bool, int], MetricRecord],
    metric_name: str,
    boundary_label: str,
    panel_label: str,
) -> None:
    x = np.arange(len(ELEMENTS), dtype=float)
    width = 0.23
    offsets = np.linspace(-width, width, len(K_VALUES))

    for offset, k in zip(offsets, K_VALUES):
        values = [
            getattr(records[(vdegree, pdegree, pcont, k)], metric_name)
            for vdegree, pdegree, pcont, _ in ELEMENTS
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[k],
            edgecolor="black",
            linewidth=0.35,
            label=rf"$k={k}$",
            zorder=8,
        )

    ax.set_yscale("log")
    ax.set_ylabel(r"$E_{L_2}(p|_{\Gamma})$")
    ax.set_title(boundary_label, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, _, label in ELEMENTS], rotation=22, ha="right")
    ax.grid(True, which="both", axis="y", linewidth=0.45, color="0.84")
    ax.tick_params(axis="both", which="both", direction="in", labelsize=8)
    ax.set_axisbelow(True)
    ax.text(
        -0.105,
        0.99,
        panel_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )


def main() -> None:
    records = read_records()
    validate_records(records)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
    for ax, (metric_name, boundary_label, panel_label) in zip(axes, BOUNDARIES):
        plot_boundary_panel(ax, records, metric_name, boundary_label, panel_label)

    axes[1].set_xlabel(r"Element pair")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.81, 0.52),
        frameon=False,
        fontsize=8.5,
    )

    fig.subplots_adjust(left=0.12, right=0.79, top=0.96, bottom=0.14, hspace=0.16)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
