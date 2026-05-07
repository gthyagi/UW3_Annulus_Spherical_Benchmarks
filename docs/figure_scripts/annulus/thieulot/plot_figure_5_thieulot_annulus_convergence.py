#!/usr/bin/env python3
"""Plot annulus Thieulot velocity and pressure L2 convergence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

METRICS_ROOT = Path("/Volumes/seagate4_1/output/annulus/thieulot/latest")
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "figure_5_thieulot_annulus_convergence.pdf"

ELEMENTS = (
    (1, 0, False, r"$P_1\times P_0$"),
    (1, 1, True, r"$P_1\times P_1$"),
    (2, 1, True, r"$P_2\times P_1$"),
    (3, 2, True, r"$P_3\times P_2$"),
)
K_VALUES = (1, 4, 8)
H_TICKS = np.array([1 / 512, 1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8])
H_TICK_LABELS = ("1/512", "1/256", "1/128", "1/64", "1/32", "1/16", "1/8")
P3P2_EXCLUDED_H = 1 / 256

COLORS = {
    (1, 0, False): "#7A3E9D",
    (1, 1, True): "#C06C2B",
    (2, 1, True): "#008C72",
    (3, 2, True): "#3A8DDE",
}
MARKERS = {1: "s", 4: "o", 8: "^"}
REFERENCE_ELEMENTS = {
    "p1p1": (1, 1, True),
    "p2p1": (2, 1, True),
    "p3p2": (3, 2, True),
}

DIR_PATTERN = re.compile(
    r"model_inv_lc_(?P<inv_lc>\d+)_"
    r"k_(?P<k>\d+)_"
    r"vdeg_(?P<vdeg>\d+)_"
    r"pdeg_(?P<pdeg>\d+)_"
    r"pcont_(?P<pcont>true|false)_"
)


@dataclass(frozen=True)
class MetricRecord:
    inv_lc: int
    h: float
    k: int
    vdegree: int
    pdegree: int
    pcont: bool
    v_l2_norm: float
    p_l2_norm: float


def read_scalar(h5f: h5py.File, name: str) -> float:
    value = h5f[name][()]
    if isinstance(value, bytes):
        return float(value.decode())
    return float(np.asarray(value))


def read_records() -> list[MetricRecord]:
    records: list[MetricRecord] = []

    for metrics_file in sorted(METRICS_ROOT.glob("model_inv_lc_*/benchmark_metrics.h5")):
        match = DIR_PATTERN.search(metrics_file.parent.name)
        if match is None:
            continue

        vdegree = int(match.group("vdeg"))
        pdegree = int(match.group("pdeg"))
        pcont = match.group("pcont") == "true"
        k = int(match.group("k"))
        if (vdegree, pdegree, pcont, "") not in ELEMENT_LOOKUP or k not in K_VALUES:
            continue

        with h5py.File(metrics_file, "r") as h5f:
            h = read_scalar(h5f, "cellsize")
            if (
                vdegree == 3
                and pdegree == 2
                and pcont
                and np.isclose(h, P3P2_EXCLUDED_H)
            ):
                continue

            records.append(
                MetricRecord(
                    inv_lc=int(match.group("inv_lc")),
                    h=h,
                    k=k,
                    vdegree=vdegree,
                    pdegree=pdegree,
                    pcont=pcont,
                    v_l2_norm=read_scalar(h5f, "v_l2_norm"),
                    p_l2_norm=read_scalar(h5f, "p_l2_norm"),
                )
            )

    return records


ELEMENT_LOOKUP = {(vdegree, pdegree, pcont, "") for vdegree, pdegree, pcont, _ in ELEMENTS}


def values_for_series(
    records: list[MetricRecord],
    vdegree: int,
    pdegree: int,
    pcont: bool,
    k: int,
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    subset = [
        record
        for record in records
        if (
            record.vdegree == vdegree
            and record.pdegree == pdegree
            and record.pcont == pcont
            and record.k == k
        )
    ]
    subset = sorted(subset, key=lambda record: record.h)
    h = np.array([record.h for record in subset], dtype=float)
    y = np.array([getattr(record, metric_name) for record in subset], dtype=float)
    mask = np.isfinite(y) & (y > 0.0)
    return h[mask], y[mask]


def reference_anchor(h: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Choose a reference-line anchor through the selected element-family data."""

    h0 = np.nanmax(h)
    anchor_values = y[np.isclose(h, h0)]
    if anchor_values.size == 0:
        anchor_values = y
    y0 = np.exp(np.nanmean(np.log(anchor_values)))
    return h0, y0


def add_reference_line(
    ax: plt.Axes,
    records: list[MetricRecord],
    metric_name: str,
    element_key: tuple[int, int, bool],
    slope: float,
    linestyle: str,
) -> None:
    vdegree, pdegree, pcont = element_key
    h_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for k in K_VALUES:
        h_k, y_k = values_for_series(records, vdegree, pdegree, pcont, k, metric_name)
        if h_k.size:
            h_parts.append(h_k)
            y_parts.append(y_k)

    if not h_parts:
        return

    h = np.concatenate(h_parts)
    y = np.concatenate(y_parts)
    if h.size == 0 or y.size == 0:
        return

    h_range = np.array([np.nanmin(h), np.nanmax(h)])
    h0, y0 = reference_anchor(h, y)
    y_range = y0 * (h_range / h0) ** slope
    ax.loglog(
        h_range,
        y_range,
        color="black",
        linestyle=linestyle,
        linewidth=1.4,
        zorder=7,
        label=rf"$\mathcal{{O}}(h^{{{slope:g}}})$",
    )


def plot_metric(
    ax: plt.Axes,
    records: list[MetricRecord],
    metric_name: str,
    ylabel: str,
    reference_slopes: tuple[tuple[str, float, str], ...],
) -> None:
    for vdegree, pdegree, pcont, element_label in ELEMENTS:
        color = COLORS[(vdegree, pdegree, pcont)]
        for k in K_VALUES:
            h, y = values_for_series(records, vdegree, pdegree, pcont, k, metric_name)
            if h.size == 0:
                continue

            ax.loglog(
                h,
                y,
                marker=MARKERS[k],
                markersize=4.0,
                linewidth=1.2,
                color=color,
                zorder=8,
                label=rf"$k={k}$, {element_label}",
            )

    for reference_element, slope, linestyle in reference_slopes:
        add_reference_line(
            ax,
            records,
            metric_name,
            REFERENCE_ELEMENTS[reference_element],
            slope,
            linestyle,
        )

    ax.set_ylabel(ylabel)
    ax.set_xlim(H_TICKS[0] * 0.82, H_TICKS[-1] * 1.18)
    ax.set_xticks(H_TICKS)
    ax.set_xticklabels(H_TICK_LABELS)
    ax.grid(True, which="both", linewidth=0.45, color="0.84")
    ax.tick_params(axis="both", which="both", direction="in", labelsize=8)
    ax.set_axisbelow(True)


def main() -> None:
    records = read_records()
    if not records:
        raise FileNotFoundError(f"No matching benchmark_metrics.h5 files found in {METRICS_ROOT}")

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    plot_metric(
        axes[0],
        records,
        "v_l2_norm",
        r"$\|e_\mathbf{u}\|_{L_2}$",
        (("p1p1", 2.0, "-"), ("p2p1", 3.0, "--"), ("p3p2", 4.0, ":")),
    )
    # axes[0].set_title("velocity error", fontsize=11)

    plot_metric(
        axes[1],
        records,
        "p_l2_norm",
        r"$\|e_p\|_{L_2}$",
        (("p1p1", 1.0, "-"), ("p2p1", 2.0, "--"), ("p3p2", 3.0, ":")),
    )
    # axes[1].set_title("pressure error", fontsize=11)
    axes[1].set_xlabel(r"$h$")

    for label, ax in zip(("a)", "b)"), axes):
        ax.text(
            -0.125,
            0.99,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    data = [(h, l) for h, l in zip(handles, labels) if r"\mathcal{O}" not in l]
    refs_v = [(h, l) for h, l in zip(handles, labels) if r"\mathcal{O}" in l]
    refs_p = [
        (h, l)
        for h, l in zip(*axes[1].get_legend_handles_labels())
        if r"\mathcal{O}" in l
    ]

    axes[0].legend(
        [h for h, _ in refs_v],
        [l for _, l in refs_v],
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.85",
        fontsize=8,
    )
    axes[1].legend(
        [h for h, _ in refs_p],
        [l for _, l in refs_p],
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.85",
        fontsize=8,
    )

    fig.legend(
        [h for h, _ in data],
        [l for _, l in data],
        loc="center left",
        bbox_to_anchor=(0.795, 0.5),
        frameon=False,
        fontsize=8,
    )

    fig.subplots_adjust(left=0.12, right=0.78, top=0.96, bottom=0.11, hspace=0.12)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
