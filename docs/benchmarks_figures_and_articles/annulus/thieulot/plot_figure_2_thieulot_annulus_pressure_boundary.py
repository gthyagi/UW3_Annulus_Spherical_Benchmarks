#!/usr/bin/env python3
"""Reproduce Thieulot annulus Figure 2 boundary pressure profiles."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_ROOT = Path("/Volumes/seagate4_1/output/annulus/thieulot/latest")
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "figure_2_thieulot_annulus_pressure_boundary.pdf"

INV_LC = 128
K_VALUES = (1, 2, 4)
R_INNER = 1.0
R_OUTER = 2.0
VDegree = 2
PDegree = 1
NCPUS = 8
STOKES_TOL = "1e-09"

COLORS = {1: "#7A3E9D", 2: "#008C72", 4: "#3A8DDE"}
THETA_TICKS = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi]
THETA_TICK_LABELS = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]


def case_dir(k: int) -> Path:
    return OUTPUT_ROOT / (
        f"model_inv_lc_{INV_LC}_k_{k}_"
        f"vdeg_{VDegree}_pdeg_{PDegree}_pcont_true_"
        f"stokes_tol_{STOKES_TOL}_ncpus_{NCPUS}_bc_essential"
    )


def analytical_pressure(theta: np.ndarray, radius: float, k: int) -> np.ndarray:
    """Analytical pressure from the annulus Thieulot manufactured solution."""

    c = -1.0
    denom = (R_OUTER**2) * np.log(R_INNER) - (R_INNER**2) * np.log(R_OUTER)
    a = -c * (2.0 * (np.log(R_INNER) - np.log(R_OUTER)) / denom)
    b = -c * ((R_OUTER**2 - R_INNER**2) / denom)

    f = a * radius + b / radius
    g = 0.5 * a * radius + (b / radius) * np.log(radius) + c / radius
    h = (2.0 * g - f) / radius
    return k * h * np.sin(k * theta)


def read_pressure_points(k: int) -> tuple[np.ndarray, np.ndarray]:
    pressure_file = case_dir(k) / "output.mesh.Pressure.00000.h5"
    if not pressure_file.is_file():
        raise FileNotFoundError(f"Missing pressure file: {pressure_file}")

    with h5py.File(pressure_file, "r") as h5f:
        coordinates = np.asarray(h5f["vertex_fields/coordinates"], dtype=np.float64)
        pressure = np.asarray(h5f["vertex_fields/Pressure_Pressure"], dtype=np.float64).reshape(-1)

    return coordinates, pressure


def boundary_profile(
    coordinates: np.ndarray,
    pressure: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.linalg.norm(coordinates, axis=1)
    mask = np.isclose(r, radius, atol=1.0e-8)
    if not np.any(mask):
        raise ValueError(f"No pressure points found on radius {radius}")

    theta = np.mod(np.arctan2(coordinates[mask, 1], coordinates[mask, 0]), 2.0 * np.pi)
    p = pressure[mask]
    order = np.argsort(theta)
    return theta[order], p[order]


def plot_boundary_panel(
    ax: plt.Axes,
    radius: float,
    title: str,
    ylabel: str,
    panel_label: str,
) -> None:
    theta_dense = np.linspace(0.0, 2.0 * np.pi, 1200)

    for k in K_VALUES:
        color = COLORS[k]
        coordinates, pressure = read_pressure_points(k)
        theta_uw, p_uw = boundary_profile(coordinates, pressure, radius)

        ax.plot(
            theta_dense,
            analytical_pressure(theta_dense, radius, k),
            color=color,
            linewidth=2.0,
            label=rf"$k={k}$ analytical",
            zorder=4,
        )
        ax.plot(
            theta_uw,
            p_uw,
            color=color,
            linestyle="--",
            linewidth=1.35,
            alpha=0.9,
            label=rf"$k={k}$ UW3",
            zorder=5,
        )

    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_xticks(THETA_TICKS)
    ax.set_xticklabels(THETA_TICK_LABELS)
    ax.grid(True, which="both", linewidth=0.45, color="0.84")
    ax.tick_params(axis="both", which="both", direction="in", labelsize=9)
    ax.set_axisbelow(True)
    ax.text(
        -0.125,
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
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True)

    plot_boundary_panel(axes[0], R_INNER, "", "Pressure (Inner Boundary)", "a)")
    plot_boundary_panel(axes[1], R_OUTER, "", "Pressure (Outer Boundary)", "b)")
    axes[1].set_xlabel(r"$\theta$")
    for ax in axes:
        ax.yaxis.set_label_coords(-0.10, 0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.80, 0.525),
        frameon=False,
        fontsize=8.5,
    )

    fig.subplots_adjust(left=0.12, right=0.78, top=0.96, bottom=0.10, hspace=0.08)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
