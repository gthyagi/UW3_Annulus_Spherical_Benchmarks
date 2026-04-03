#!/usr/bin/env python3
"""
Standalone spherical Thieulot radial-grading experiment driver.

This script keeps the analytical fields and Stokes formulation fixed and
applies a purely radial mesh deformation to cluster nodes toward the inner
boundary. It is intended to test how much inner-boundary-focused resolution
reduces pressure sensitivity without changing the benchmark equations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import sympy as sp
import underworld3 as uw
from underworld3.systems import Stokes

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value!r}")


def parse_cellsize(value: str | float) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return float(eval(str(value), {"__builtins__": {}}, {}))


def parse_csv_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def case_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.3g}"
    return value


def make_case_id(**parts) -> str:
    rendered = []
    for key, value in parts.items():
        value = case_value(value)
        if value is not None:
            rendered.append(f"{key}_{value}")
    return "_".join(rendered)


@dataclass
class GaugeMetrics:
    gauge: str
    pressure_volume_l2_analytic: float
    pressure_inner_l2_analytic: float
    pressure_outer_l2_analytic: float
    pressure_volume_mean: float
    pressure_inner_mean: float
    pressure_outer_mean: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone spherical Thieulot radial-grading comparison driver.",
        allow_abbrev=False,
    )
    parser.add_argument("-uw_cellsize", default="1/8")
    parser.add_argument("-uw_r_i", type=float, default=0.5)
    parser.add_argument("-uw_r_o", type=float, default=1.0)
    parser.add_argument("-uw_m", type=int, default=-1)
    parser.add_argument("-uw_vdegree", type=int, default=2)
    parser.add_argument("-uw_pdegree", type=int, default=1)
    parser.add_argument("-uw_pcont", type=parse_bool, default=True)
    parser.add_argument("-uw_qdegree", type=int, default=0)
    parser.add_argument("-uw_stokes_tol", type=float, default=1.0e-5)
    parser.add_argument(
        "-dbg_grading_ratio",
        type=float,
        default=1.0,
        help="Approximate outer/inner radial cell-size ratio. 1.0 means uniform.",
    )
    parser.add_argument(
        "-dbg_report_gauges",
        default="volume_mean,inner_surface_mean,outer_surface_mean",
        help="Comma-separated reporting-only gauge shifts.",
    )
    parser.add_argument(
        "-dbg_metrics_mode",
        choices=("full", "volume_only"),
        default="full",
        help="Compute all diagnostics or only volume L2 norms.",
    )
    parser.add_argument(
        "-dbg_output_root",
        default=None,
        help="Override the root output directory.",
    )
    parser.add_argument(
        "-dbg_write_output",
        type=parse_bool,
        default=False,
        help="Write mesh fields into the run directory.",
    )
    return parser


def analytic_solution(mesh, r_i, r_o, m, gamma=1.0, mu_0=1.0):
    if m == -4:
        raise ValueError("The Thieulot spherical benchmark is undefined for m = -4.")

    r = mesh.CoordinateSystem.xR[0]
    theta = mesh.CoordinateSystem.xR[1]
    phi_raw = mesh.CoordinateSystem.xR[2]
    phi = sp.Piecewise((2 * sp.pi + phi_raw, phi_raw < 0), (phi_raw, True))

    mu_expr = mu_0 * (r ** (m + 1))

    if m == -1:
        alpha = -gamma * (
            (r_o**3 - r_i**3)
            / ((r_o**3) * math.log(r_i) - (r_i**3) * math.log(r_o))
        )
        beta = -3.0 * gamma * (
            (math.log(r_o) - math.log(r_i))
            / ((r_i**3) * math.log(r_o) - (r_o**3) * math.log(r_i))
        )

        f = alpha * (r ** -(m + 3)) + beta * r
        g = (-2.0 / (r**2)) * (alpha * sp.log(r) + (beta / 3.0) * (r**3) + gamma)
        h = (2.0 / r) * mu_0 * g

        rho_expr = sp.simplify(
            (
                (alpha / r**4) * (8.0 * sp.log(r) - 6.0)
                + 8.0 * beta / (3.0 * r)
                + 8.0 * gamma / r**4
            )
            * sp.cos(theta)
        )
    else:
        alpha = gamma * (m + 1) * (
            (r_i**-3 - r_o**-3) / ((r_i ** -(m + 4)) - (r_o ** -(m + 4)))
        )
        beta = -3.0 * gamma * (
            ((r_i ** (m + 1)) - (r_o ** (m + 1)))
            / ((r_i ** (m + 4)) - (r_o ** (m + 4)))
        )

        f = alpha * (r ** -(m + 3)) + beta * r
        g = (-2.0 / (r**2)) * (
            (-alpha / (m + 1)) * r ** (-(m + 1))
            + (beta / 3.0) * (r**3)
            + gamma
        )
        h = ((m + 3) / r) * mu_expr * g

        rho_expr = sp.simplify(
            (r**m)
            * (
                2.0 * alpha * r ** (-(m + 4)) * ((m + 3) / (m + 1)) * (m - 1)
                - (2.0 * beta / 3.0) * (m - 1) * (m + 3)
                - m * (m + 5) * (2.0 * gamma / r**3)
            )
            * sp.cos(theta)
        )

    p_expr = h * sp.cos(theta)
    v_r = g * sp.cos(theta)
    v_theta = f * sp.sin(theta)
    v_phi = f * sp.sin(theta)

    v_x = (
        v_r * sp.sin(theta) * sp.cos(phi)
        + v_theta * sp.cos(theta) * sp.cos(phi)
        - v_phi * sp.sin(phi)
    )
    v_y = (
        v_r * sp.sin(theta) * sp.sin(phi)
        + v_theta * sp.cos(theta) * sp.sin(phi)
        + v_phi * sp.cos(phi)
    )
    v_z = v_r * sp.cos(theta) - v_theta * sp.sin(theta)

    v_expr = sp.Matrix([v_x, v_y, v_z])
    return v_expr, p_expr, rho_expr, mu_expr


def global_integral(mesh, fn):
    local = float(uw.maths.Integral(mesh, fn).evaluate())
    return float(uw.mpi.comm.allreduce(local))


def global_boundary_integral(mesh, fn, boundary_name):
    local = float(uw.maths.BdIntegral(mesh=mesh, fn=fn, boundary=boundary_name).evaluate())
    return float(uw.mpi.comm.allreduce(local))


def relative_l2_error(mesh, err_expr, ref_expr):
    if isinstance(err_expr, sp.MatrixBase):
        err_sq = err_expr.dot(err_expr)
        ref_sq = ref_expr.dot(ref_expr)
    else:
        err_sq = err_expr * err_expr
        ref_sq = ref_expr * ref_expr

    err_i = global_integral(mesh, err_sq)
    ref_i = global_integral(mesh, ref_sq)
    return float(np.sqrt(err_i / ref_i))


def boundary_relative_l2(mesh, err_expr, ref_expr, boundary_name):
    if isinstance(err_expr, sp.MatrixBase):
        err_sq = err_expr.dot(err_expr)
        ref_sq = ref_expr.dot(ref_expr)
    else:
        err_sq = err_expr * err_expr
        ref_sq = ref_expr * ref_expr

    err_i = global_boundary_integral(mesh, err_sq, boundary_name)
    ref_i = global_boundary_integral(mesh, ref_sq, boundary_name)
    return float(np.sqrt(err_i / ref_i))


def copy_variable_data(mesh, field):
    with mesh.access(field):
        return np.asarray(field.data, dtype=np.float64).copy()


def fill_variable_data(mesh, field, values):
    with mesh.access(field):
        field.data[...] = values


def subtract_constant(mesh, pressure_var, value):
    with mesh.access(pressure_var):
        pressure_var.data[:, 0] -= value


def pressure_mean(mesh, pressure_var):
    volume = global_integral(mesh, 1.0)
    return global_integral(mesh, pressure_var.sym[0]) / volume


def boundary_pressure_mean(mesh, pressure_var, boundary_name):
    measure = global_boundary_integral(mesh, 1.0, boundary_name)
    if np.isclose(measure, 0.0):
        return 0.0
    return global_boundary_integral(mesh, pressure_var.sym[0], boundary_name) / measure


def apply_pressure_gauge(mesh, pressure_var, gauge):
    if gauge == "none":
        return
    if gauge == "volume_mean":
        subtract_constant(mesh, pressure_var, pressure_mean(mesh, pressure_var))
        return
    if gauge == "inner_surface_mean":
        shift = boundary_pressure_mean(mesh, pressure_var, mesh.boundaries.Lower.name)
        subtract_constant(mesh, pressure_var, shift)
        return
    if gauge == "outer_surface_mean":
        shift = boundary_pressure_mean(mesh, pressure_var, mesh.boundaries.Upper.name)
        subtract_constant(mesh, pressure_var, shift)
        return
    raise ValueError(f"Unknown pressure gauge: {gauge}")


def grading_scales(grading_ratio):
    if np.isclose(grading_ratio, 1.0):
        return 1.0, 1.0
    a = math.log(grading_ratio)
    inner_scale = a / (math.exp(a) - 1.0)
    outer_scale = grading_ratio * inner_scale
    return inner_scale, outer_scale


def deform_mesh_radially(mesh, r_i, r_o, grading_ratio):
    if grading_ratio <= 0.0:
        raise ValueError("dbg_grading_ratio must be positive.")
    if np.isclose(grading_ratio, 1.0):
        return

    coords = np.asarray(mesh.X.coords, dtype=np.float64).copy()
    radii = np.sqrt(np.sum(coords**2, axis=1))
    thickness = r_o - r_i
    t = (radii - r_i) / thickness
    a = math.log(grading_ratio)
    mapped = (np.exp(a * t) - 1.0) / (math.exp(a) - 1.0)
    new_radii = r_i + thickness * mapped
    scale = new_radii / radii
    new_coords = coords * scale.reshape(-1, 1)
    mesh._deform_mesh(new_coords)


def gauge_metrics(mesh, pressure_var, p_ana_expr, gauge):
    original = copy_variable_data(mesh, pressure_var)
    try:
        apply_pressure_gauge(mesh, pressure_var, gauge)
        p_err_expr = pressure_var.sym[0] - p_ana_expr
        inner_l2 = boundary_relative_l2(
            mesh,
            p_err_expr,
            p_ana_expr,
            mesh.boundaries.Lower.name,
        )
        outer_l2 = boundary_relative_l2(
            mesh,
            p_err_expr,
            p_ana_expr,
            mesh.boundaries.Upper.name,
        )
        inner_mean = boundary_pressure_mean(
            mesh, pressure_var, mesh.boundaries.Lower.name
        )
        outer_mean = boundary_pressure_mean(
            mesh, pressure_var, mesh.boundaries.Upper.name
        )
        return GaugeMetrics(
            gauge=gauge,
            pressure_volume_l2_analytic=relative_l2_error(mesh, p_err_expr, p_ana_expr),
            pressure_inner_l2_analytic=inner_l2,
            pressure_outer_l2_analytic=outer_l2,
            pressure_volume_mean=pressure_mean(mesh, pressure_var),
            pressure_inner_mean=inner_mean,
            pressure_outer_mean=outer_mean,
        )
    finally:
        fill_variable_data(mesh, pressure_var, original)


def gauge_metrics_volume_only(mesh, pressure_var, p_ana_expr, gauge):
    original = copy_variable_data(mesh, pressure_var)
    try:
        apply_pressure_gauge(mesh, pressure_var, gauge)
        p_err_expr = pressure_var.sym[0] - p_ana_expr
        return GaugeMetrics(
            gauge=gauge,
            pressure_volume_l2_analytic=relative_l2_error(mesh, p_err_expr, p_ana_expr),
            pressure_inner_l2_analytic=float("nan"),
            pressure_outer_l2_analytic=float("nan"),
            pressure_volume_mean=pressure_mean(mesh, pressure_var),
            pressure_inner_mean=float("nan"),
            pressure_outer_mean=float("nan"),
        )
    finally:
        fill_variable_data(mesh, pressure_var, original)


def run(params):
    params.uw_cellsize = parse_cellsize(params.uw_cellsize)
    params.uw_pcont = params.uw_pcont if params.uw_pdegree > 0 else False
    report_gauges = parse_csv_strings(params.dbg_report_gauges)
    mesh_qdegree = (
        params.uw_qdegree if params.uw_qdegree > 0 else max(params.uw_vdegree, params.uw_pdegree)
    )
    inner_scale, outer_scale = grading_scales(params.dbg_grading_ratio)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_root = (
        params.dbg_output_root
        if params.dbg_output_root is not None
        else os.path.join(repo_root, "output", "sphere", "thieulot", "radial_grading")
    )
    run_id = make_case_id(
        inv_lc=int(round(1.0 / params.uw_cellsize)),
        m=params.uw_m,
        grade=params.dbg_grading_ratio,
        inner_scale=inner_scale,
        vdeg=params.uw_vdegree,
        pdeg=params.uw_pdegree,
        pcont=params.uw_pcont,
        qdeg=params.uw_qdegree if params.uw_qdegree > 0 else None,
        tol=params.uw_stokes_tol,
        np=uw.mpi.size,
    )
    run_dir = os.path.join(output_root, run_id)
    if uw.mpi.rank == 0:
        os.makedirs(run_dir, exist_ok=True)

    mesh = uw.meshing.SphericalShell(
        radiusInner=params.uw_r_i,
        radiusOuter=params.uw_r_o,
        cellSize=params.uw_cellsize,
        qdegree=mesh_qdegree,
        degree=1,
        filename=os.path.join(run_dir, "mesh.msh"),
    )
    deform_mesh_radially(mesh, params.uw_r_i, params.uw_r_o, params.dbg_grading_ratio)

    v_soln = uw.discretisation.MeshVariable(
        varname="Velocity",
        mesh=mesh,
        degree=params.uw_vdegree,
        vtype=uw.VarType.VECTOR,
        varsymbol=r"V",
    )
    p_soln = uw.discretisation.MeshVariable(
        varname="Pressure",
        mesh=mesh,
        degree=params.uw_pdegree,
        vtype=uw.VarType.SCALAR,
        varsymbol=r"P",
        continuous=params.uw_pcont,
    )

    v_ana_expr, p_ana_expr, rho_expr, mu_expr = analytic_solution(
        mesh,
        params.uw_r_i,
        params.uw_r_o,
        params.uw_m,
    )
    v_err_expr = sp.Matrix(v_soln.sym).T - v_ana_expr
    p_err_expr = p_soln.sym[0] - p_ana_expr

    stokes = Stokes(mesh, velocityField=v_soln, pressureField=p_soln)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = mu_expr
    stokes.saddle_preconditioner = 1.0 / mu_expr
    stokes.petsc_use_pressure_nullspace = True
    gravity_fn = -1.0 * mesh.CoordinateSystem.unit_e_0
    stokes.bodyforce = -rho_expr * gravity_fn
    stokes.add_essential_bc(v_ana_expr, mesh.boundaries.Upper.name)
    stokes.add_essential_bc(v_ana_expr, mesh.boundaries.Lower.name)
    stokes.tolerance = params.uw_stokes_tol
    stokes.petsc_options["ksp_monitor"] = None
    stokes.petsc_options["ksp_monitor_true_residual"] = None
    stokes.petsc_options["ksp_converged_reason"] = None
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.petsc_options["ksp_rtol"] = params.uw_stokes_tol
    stokes.petsc_options["ksp_atol"] = 0.0

    if uw.mpi.size == 1:
        stokes.petsc_options["ksp_type"] = "preonly"
        stokes.petsc_options["pc_type"] = "lu"
    else:
        stokes.petsc_options["ksp_type"] = "fgmres"
        stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
        stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
        stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
        stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
        stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
        stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
        stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None
        stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
        stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
        stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

    if uw.mpi.rank == 0:
        print(
            f"Running graded mesh case: m={params.uw_m}, cellsize={params.uw_cellsize}, "
            f"grading_ratio={params.dbg_grading_ratio}, inner_scale={inner_scale:.6g}",
            flush=True,
        )

    uw.timing.reset()
    uw.timing.start()
    stokes.solve(verbose=False, debug=False)
    uw.timing.stop()
    uw.timing.print_table(filename=os.path.join(run_dir, "stokes_timing.txt"))

    gauge_fn = gauge_metrics if params.dbg_metrics_mode == "full" else gauge_metrics_volume_only
    gauge_results = [
        gauge_fn(mesh, p_soln, p_ana_expr, gauge)
        for gauge in report_gauges
    ]
    velocity_l2 = relative_l2_error(mesh, v_err_expr, v_ana_expr)

    payload = {
        "params": {
            "uw_cellsize": params.uw_cellsize,
            "uw_r_i": params.uw_r_i,
            "uw_r_o": params.uw_r_o,
            "uw_m": params.uw_m,
            "uw_vdegree": params.uw_vdegree,
            "uw_pdegree": params.uw_pdegree,
            "uw_pcont": params.uw_pcont,
            "uw_qdegree": params.uw_qdegree,
            "uw_stokes_tol": params.uw_stokes_tol,
            "dbg_grading_ratio": params.dbg_grading_ratio,
            "dbg_report_gauges": report_gauges,
            "mpi_size": uw.mpi.size,
        },
        "mesh_grading": {
            "inner_scale_of_uniform": inner_scale,
            "outer_scale_of_uniform": outer_scale,
        },
        "solver": {
            "snes_reason": int(stokes.snes.getConvergedReason()),
            "ksp_reason": int(stokes.snes.ksp.getConvergedReason()),
        },
        "velocity_volume_l2_analytic": float(velocity_l2),
        "gauges": [asdict(result) for result in gauge_results],
    }

    if params.dbg_write_output:
        mesh.write_timestep(
            "output",
            index=0,
            meshVars=[v_soln, p_soln],
            outputPath=run_dir,
        )

    if uw.mpi.rank == 0:
        lines = []
        lines.append(
            f"m={params.uw_m}, cellsize={params.uw_cellsize}, grading_ratio={params.dbg_grading_ratio}, "
            f"inner_scale={inner_scale:.6g}, outer_scale={outer_scale:.6g}"
        )
        lines.append("gauge | v_vol_l2 | p_vol_l2 | p_inner_l2 | p_outer_l2 | p_mean")
        for metrics in gauge_results:
            lines.append(
                f"{metrics.gauge} | {velocity_l2:.6g} | "
                f"{metrics.pressure_volume_l2_analytic:.6g} | "
                f"{metrics.pressure_inner_l2_analytic:.6g} | "
                f"{metrics.pressure_outer_l2_analytic:.6g} | "
                f"{metrics.pressure_volume_mean:.6g}"
            )
        summary_text = "\n".join(lines)
        print(summary_text)
        with open(os.path.join(run_dir, "summary.txt"), "w", encoding="ascii") as stream:
            stream.write(summary_text + "\n")
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="ascii") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)

    return payload


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
