#!/usr/bin/env python3
"""Plot P2-P1 annulus Thieulot boundary pressure convergence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

METRICS_ROOT = Path("/Volumes/seagate4_1/output/annulus/thieulot/latest")
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "figure_p2p1_boundary_pressure_convergence.pdf"

K_VALUES = (1, 4, 8)
H_TICKS = np.array([1 / 512, 1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8])
H_TICK_LABELS = ("1/512", "1/256", "1/128", "1/64", "1/32", "1/16", "1/8")

COLORS = {1: "#0072B2", 4: "#D55E00", 8: "#009E73"}
MARKERS = {1: "s", 4: "o", 8: "^"}
BOUNDARIES = (
    ("p_l2_norm_lower_abs", "inner", "-"),
    ("p_l2_norm_upper_abs", "outer", "--"),
)

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
    inv_lc: int
    h: float
    k: int
    p_l2_norm_lower_abs: float
    p_l2_norm_upper_abs: float


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

        k = int(match.group("k"))
        vdegree = int(match.group("vdeg"))
        pdegree = int(match.group("pdeg"))
        pcont = match.group("pcont") == "true"
        if (vdegree, pdegree, pcont) != (2, 1, True) or k not in K_VALUES:
            continue

        with h5py.File(metrics_file, "r") as h5f:
            records.append(
                MetricRecord(
                    inv_lc=int(match.group("inv_lc")),
                    h=read_scalar(h5f, "cellsize"),
                    k=k,
                    p_l2_norm_lower_abs=read_scalar(h5f, "p_l2_norm_lower_abs"),
                    p_l2_norm_upper_abs=read_scalar(h5f, "p_l2_norm_upper_abs"),
                )
            )

    return sorted(records, key=lambda record: (record.k, record.h))


def values_for_series(
    records: list[MetricRecord],
    k: int,
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    subset = sorted((record for record in records if record.k == k), key=lambda record: record.h)
    h = np.array([record.h for record in subset], dtype=float)
    y = np.array([getattr(record, metric_name) for record in subset], dtype=float)
    mask = np.isfinite(y) & (y > 0.0)
    return h[mask], y[mask]


def reference_anchor(h: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    h0 = np.nanmax(h)
    anchor_values = y[np.isclose(h, h0)]
    if anchor_values.size == 0:
        anchor_values = y
    y0 = np.exp(np.nanmean(np.log(anchor_values)))
    return h0, y0


def add_reference_line(ax: plt.Axes, records: list[MetricRecord]) -> None:
    h_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for k in K_VALUES:
        for metric_name, _, _ in BOUNDARIES:
            h_k, y_k = values_for_series(records, k, metric_name)
            if h_k.size:
                h_parts.append(h_k)
                y_parts.append(y_k)

    if not h_parts:
        return

    h = np.concatenate(h_parts)
    y = np.concatenate(y_parts)
    h_range = np.array([np.nanmin(h), np.nanmax(h)])
    h0, y0 = reference_anchor(h, y)
    y_range = y0 * (h_range / h0) ** 2.0

    ax.loglog(
        h_range,
        y_range,
        color="black",
        linestyle=":",
        linewidth=1.4,
        zorder=7,
        label=r"$\mathcal{O}(h^2)$",
    )


def validate_records(records: list[MetricRecord]) -> None:
    if not records:
        raise FileNotFoundError(f"No P2-P1 boundary metric files found in {METRICS_ROOT}")

    missing = []
    for k in K_VALUES:
        h_values = {record.h for record in records if record.k == k}
        missing_h = [label for h, label in zip(H_TICKS, H_TICK_LABELS) if h not in h_values]
        if missing_h:
            missing.append(f"k={k}: {', '.join(missing_h)}")

    if missing:
        raise RuntimeError("Incomplete P2-P1 boundary pressure data: " + "; ".join(missing))


def main() -> None:
    records = read_records()
    validate_records(records)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for k in K_VALUES:
        color = COLORS[k]
        for metric_name, boundary_label, linestyle in BOUNDARIES:
            h, y = values_for_series(records, k, metric_name)
            ax.loglog(
                h,
                y,
                marker=MARKERS[k],
                markersize=4.5,
                linewidth=1.3,
                color=color,
                linestyle=linestyle,
                zorder=8,
                label=rf"$k={k}$, {boundary_label}",
            )

    add_reference_line(ax, records)

    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$E_{L_2,\Gamma}(p)$")
    ax.set_xlim(H_TICKS[0] * 0.82, H_TICKS[-1] * 1.18)
    ax.set_xticks(H_TICKS)
    ax.set_xticklabels(H_TICK_LABELS)
    ax.grid(True, which="both", linewidth=0.45, color="0.84")
    ax.tick_params(axis="both", which="both", direction="in", labelsize=8)
    ax.set_axisbelow(True)

    ax.legend(
        loc="lower right",
        ncol=2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.85",
        fontsize=8,
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.14)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
