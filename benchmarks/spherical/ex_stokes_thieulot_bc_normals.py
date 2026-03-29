#!/usr/bin/env python3
"""
Standalone spherical Thieulot boundary-normal experiment driver.

This script is isolated from the main benchmark drivers. It is intended for
testing how weak normal-only boundary conditions affect the pressure field on
the spherical shell boundaries, with the essential-BC solution used as the
reference baseline.
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


def case_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.2g}"
    return value


def make_case_id(**parts) -> str:
    rendered = []
    for key, value in parts.items():
        value = case_value(value)
        if value is not None:
            rendered.append(f"{key}_{value}")
    return "_".join(rendered)


@dataclass
class CaseResult:
    case_name: str
    bc_form: str
    normal_type: str
    penalty: float
    tolerance: float
    velocity_volume_l2_analytic: float
    pressure_volume_l2_analytic: float
    pressure_inner_l2_analytic: float
    pressure_outer_l2_analytic: float
    pressure_volume_l2_baseline: float | None
    pressure_inner_l2_baseline: float | None
    pressure_outer_l2_baseline: float | None
    pressure_inner_mean: float
    pressure_outer_mean: float
    inner_normal_rms: float
    outer_normal_rms: float
    inner_tangential_rms: float
    outer_tangential_rms: float
    snes_reason: int
    ksp_reason: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone spherical Thieulot normal-BC comparison driver.",
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
    parser.add_argument("-uw_p_bc", type=parse_bool, default=False)
    parser.add_argument("-uw_stokes_tol", type=float, default=1.0e-5)
    parser.add_argument("-uw_vel_penalty", type=float, default=1.0e8)
    parser.add_argument(
        "-dbg_pressure_gauge",
        choices=("none", "volume_mean", "inner_surface_mean", "outer_surface_mean"),
        default="volume_mean",
    )
    parser.add_argument(
        "-dbg_snes_type",
        choices=("ksponly", "newtonls"),
        default="ksponly",
    )
    parser.add_argument(
        "-dbg_include_natural_full",
        type=parse_bool,
        default=True,
        help="Also run the current full-vector natural penalty as a control case.",
    )
    parser.add_argument(
        "-dbg_normal_types",
        default="petsc,analytic,projected",
        help="Comma-separated list of natural-normal BC normal choices to run.",
    )
    parser.add_argument(
        "-dbg_extra_penalties",
        default="",
        help="Optional comma-separated extra penalty values for analytic normal retests.",
    )
    parser.add_argument(
        "-dbg_extra_tolerances",
        default="",
        help="Optional comma-separated extra Stokes tolerances for analytic normal retests.",
    )
    parser.add_argument(
        "-dbg_write_output",
        type=parse_bool,
        default=False,
        help="Write mesh fields for each case into the run directory.",
    )
    parser.add_argument(
        "-dbg_output_root",
        default=None,
        help="Override the root output directory.",
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


def relative_l2_error(mesh, err_expr, ana_expr):
    if isinstance(err_expr, sp.MatrixBase):
        err_sq = err_expr.dot(err_expr)
        ana_sq = ana_expr.dot(ana_expr)
    else:
        err_sq = err_expr * err_expr
        ana_sq = ana_expr * ana_expr

    err_i = float(uw.maths.Integral(mesh, err_sq).evaluate())
    ana_i = float(uw.maths.Integral(mesh, ana_sq).evaluate())
    return float(np.sqrt(err_i / ana_i))


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


def global_integral(mesh, fn):
    local = float(uw.maths.Integral(mesh, fn).evaluate())
    return float(uw.mpi.comm.allreduce(local))


def global_boundary_integral(mesh, fn, boundary_name):
    local = float(uw.maths.BdIntegral(mesh=mesh, fn=fn, boundary=boundary_name).evaluate())
    return float(uw.mpi.comm.allreduce(local))


def subtract_constant(mesh, pressure_var, value):
    with mesh.access(pressure_var):
        pressure_var.data[:, 0] -= value


def pressure_mean(mesh, pressure_var):
    volume = global_integral(mesh, 1.0)
    return global_integral(mesh, pressure_var.sym[0]) / volume


def boundary_pressure_mean(mesh, pressure_var, boundary_name):
    measure = global_boundary_integral(mesh, 1.0, boundary_name)
    return global_boundary_integral(mesh, pressure_var.sym[0], boundary_name) / measure


def apply_pressure_gauge(mesh, pressure_var, gauge):
    if gauge == "none":
        return
    if gauge == "volume_mean":
        subtract_constant(mesh, pressure_var, pressure_mean(mesh, pressure_var))
        return
    if gauge == "inner_surface_mean":
        subtract_constant(
            mesh,
            pressure_var,
            boundary_pressure_mean(mesh, pressure_var, mesh.boundaries.Lower.name),
        )
        return
    if gauge == "outer_surface_mean":
        subtract_constant(
            mesh,
            pressure_var,
            boundary_pressure_mean(mesh, pressure_var, mesh.boundaries.Upper.name),
        )
        return
    raise ValueError(f"Unknown pressure gauge: {gauge}")


def parse_csv_floats(text):
    cleaned = text.strip()
    if not cleaned:
        return []
    return [float(item.strip()) for item in cleaned.split(",") if item.strip()]


def parse_csv_strings(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def build_projected_normals(mesh, degree):
    if uw.mpi.rank == 0:
        print("Building projected normals", flush=True)
    n_proj = uw.discretisation.MeshVariable(
        "GammaProj",
        mesh,
        mesh.dim,
        degree=degree,
        vtype=uw.VarType.VECTOR,
        varsymbol=r"\Gamma_p",
    )
    projector = uw.systems.Vector_Projection(mesh, n_proj)
    projector.uw_function = sp.Matrix([[0] * mesh.dim])
    projector.petsc_options["snes_type"] = "ksponly"
    projector.petsc_options["ksp_rtol"] = 1.0e-8
    if uw.mpi.size == 1:
        projector.petsc_options["ksp_type"] = "preonly"
        projector.petsc_options["pc_type"] = "lu"
    else:
        projector.petsc_options["ksp_type"] = "gmres"
        projector.petsc_options["pc_type"] = "gamg"

    unit_r = mesh.CoordinateSystem.unit_e_0
    gamma_norm = mesh.Gamma / sp.sqrt(mesh.Gamma.dot(mesh.Gamma))
    orientation = unit_r.dot(gamma_norm)

    projector.add_natural_bc(gamma_norm * orientation, mesh.boundaries.Upper.name)
    projector.add_natural_bc(gamma_norm * orientation, mesh.boundaries.Lower.name)
    projector.solve()

    with mesh.access(n_proj):
        mag = np.sqrt(np.sum(n_proj.data**2, axis=1, keepdims=True))
        mag[mag == 0.0] = 1.0
        n_proj.data[:] /= mag

    return n_proj


def make_stokes(mesh, v_soln, p_soln, mu_expr, bodyforce, tol, snes_type, p_bc):
    stokes = Stokes(mesh, velocityField=v_soln, pressureField=p_soln)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = mu_expr
    stokes.saddle_preconditioner = 1.0 / mu_expr

    if not p_bc:
        stokes.petsc_use_pressure_nullspace = True

    stokes.bodyforce = bodyforce
    stokes.tolerance = tol
    stokes.petsc_options["ksp_monitor"] = None
    stokes.petsc_options["ksp_monitor_true_residual"] = None
    stokes.petsc_options["ksp_converged_reason"] = None
    stokes.petsc_options["snes_type"] = snes_type
    stokes.petsc_options["ksp_rtol"] = tol
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

    if p_bc:
        stokes.add_condition(
            p_soln.field_id,
            "dirichlet",
            sp.Matrix([0]),
            mesh.boundaries.Lower.name,
            components=(0),
        )
        stokes.add_condition(
            p_soln.field_id,
            "dirichlet",
            sp.Matrix([0]),
            mesh.boundaries.Upper.name,
            components=(0),
        )

    return stokes


def surface_velocity_rms(mesh, velocity_expr, boundary_name, normal_expr):
    area = global_boundary_integral(mesh, 1.0, boundary_name)
    vn = normal_expr.dot(velocity_expr)
    speed_sq = velocity_expr.dot(velocity_expr)
    tangential_sq = sp.simplify(speed_sq - vn * vn)
    normal_rms = float(np.sqrt(global_boundary_integral(mesh, vn * vn, boundary_name) / area))
    tangential_rms = float(
        np.sqrt(global_boundary_integral(mesh, tangential_sq, boundary_name) / area)
    )
    return normal_rms, tangential_rms


def copy_variable_data(mesh, field):
    with mesh.access(field):
        return np.asarray(field.data, dtype=np.float64).copy()


def fill_variable_data(mesh, field, values):
    with mesh.access(field):
        field.data[...] = values


def as_column_matrix(expr):
    matrix = sp.Matrix(expr)
    if matrix.shape[0] == 1:
        matrix = matrix.T
    return matrix


def run_case(
    *,
    mesh,
    mesh_qdegree,
    params,
    run_dir,
    case_name,
    bc_form,
    normal_type,
    tol,
    penalty,
    projected_normals,
    p_baseline,
):
    if uw.mpi.rank == 0:
        print(
            f"Running case={case_name} bc_form={bc_form} normal={normal_type} "
            f"penalty={penalty:.3g} tol={tol:.3g}",
            flush=True,
        )
    v_soln = uw.discretisation.MeshVariable(
        varname=f"Velocity_{case_name}",
        mesh=mesh,
        degree=params.uw_vdegree,
        vtype=uw.VarType.VECTOR,
        varsymbol=r"V",
    )
    p_soln = uw.discretisation.MeshVariable(
        varname=f"Pressure_{case_name}",
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
    v_ana_expr = as_column_matrix(v_ana_expr)
    v_err_expr = as_column_matrix(v_soln.sym) - v_ana_expr
    p_err_expr = p_soln.sym[0] - p_ana_expr
    gravity_fn = -1.0 * mesh.CoordinateSystem.unit_e_0
    bodyforce = -rho_expr * gravity_fn

    stokes = make_stokes(
        mesh,
        v_soln,
        p_soln,
        mu_expr,
        bodyforce,
        tol,
        params.dbg_snes_type,
        params.uw_p_bc,
    )

    if bc_form == "essential":
        stokes.add_essential_bc(v_ana_expr, mesh.boundaries.Upper.name)
        stokes.add_essential_bc(v_ana_expr, mesh.boundaries.Lower.name)
    elif bc_form == "natural_full":
        stokes.add_natural_bc(penalty * v_err_expr, mesh.boundaries.Upper.name)
        stokes.add_natural_bc(penalty * v_err_expr, mesh.boundaries.Lower.name)
    elif bc_form == "natural_normal":
        if normal_type == "petsc":
            gamma = mesh.Gamma / sp.sqrt(mesh.Gamma.dot(mesh.Gamma))
        elif normal_type == "analytic":
            gamma = mesh.CoordinateSystem.unit_e_0
        elif normal_type == "projected":
            if projected_normals is None:
                raise ValueError("Projected normals were requested but not built.")
            gamma = projected_normals.sym
        else:
            raise ValueError(f"Unknown normal_type: {normal_type}")

        gamma_matrix = as_column_matrix(gamma)
        velocity_matrix = as_column_matrix(v_soln.sym)
        analytical_matrix = v_ana_expr
        normal_velocity_error = sp.simplify(
            gamma_matrix.dot(velocity_matrix) - gamma_matrix.dot(analytical_matrix)
        )
        bc_term = sp.Matrix(
            [penalty * normal_velocity_error * gamma_matrix[i, 0] for i in range(mesh.dim)]
        )
        stokes.add_natural_bc(bc_term, mesh.boundaries.Upper.name)
        stokes.add_natural_bc(bc_term, mesh.boundaries.Lower.name)
    else:
        raise ValueError(f"Unknown bc_form: {bc_form}")

    uw.timing.reset()
    uw.timing.start()
    stokes.solve(verbose=False, debug=False)
    uw.timing.stop()
    uw.timing.print_table(filename=os.path.join(run_dir, f"{case_name}_timing.txt"))

    if not params.uw_p_bc:
        apply_pressure_gauge(mesh, p_soln, params.dbg_pressure_gauge)

    unit_r = mesh.CoordinateSystem.unit_e_0
    v_vol_l2 = relative_l2_error(mesh, v_err_expr, v_ana_expr)
    p_vol_l2 = relative_l2_error(mesh, p_err_expr, p_ana_expr)
    p_inner_l2 = boundary_relative_l2(
        mesh,
        p_soln.sym[0] - p_ana_expr,
        p_ana_expr,
        mesh.boundaries.Lower.name,
    )
    p_outer_l2 = boundary_relative_l2(
        mesh,
        p_soln.sym[0] - p_ana_expr,
        p_ana_expr,
        mesh.boundaries.Upper.name,
    )

    p_vol_l2_baseline = None
    p_inner_l2_baseline = None
    p_outer_l2_baseline = None
    if p_baseline is not None:
        p_vol_l2_baseline = relative_l2_error(
            mesh,
            p_soln.sym[0] - p_baseline.sym[0],
            p_baseline.sym[0],
        )
        p_inner_l2_baseline = boundary_relative_l2(
            mesh,
            p_soln.sym[0] - p_baseline.sym[0],
            p_baseline.sym[0],
            mesh.boundaries.Lower.name,
        )
        p_outer_l2_baseline = boundary_relative_l2(
            mesh,
            p_soln.sym[0] - p_baseline.sym[0],
            p_baseline.sym[0],
            mesh.boundaries.Upper.name,
        )

    inner_normal_rms, inner_tangential_rms = surface_velocity_rms(
        mesh,
        sp.Matrix(v_soln.sym).T,
        mesh.boundaries.Lower.name,
        unit_r,
    )
    outer_normal_rms, outer_tangential_rms = surface_velocity_rms(
        mesh,
        sp.Matrix(v_soln.sym).T,
        mesh.boundaries.Upper.name,
        unit_r,
    )

    result = CaseResult(
        case_name=case_name,
        bc_form=bc_form,
        normal_type=normal_type,
        penalty=float(penalty),
        tolerance=float(tol),
        velocity_volume_l2_analytic=float(v_vol_l2),
        pressure_volume_l2_analytic=float(p_vol_l2),
        pressure_inner_l2_analytic=float(p_inner_l2),
        pressure_outer_l2_analytic=float(p_outer_l2),
        pressure_volume_l2_baseline=(
            None if p_vol_l2_baseline is None else float(p_vol_l2_baseline)
        ),
        pressure_inner_l2_baseline=(
            None if p_inner_l2_baseline is None else float(p_inner_l2_baseline)
        ),
        pressure_outer_l2_baseline=(
            None if p_outer_l2_baseline is None else float(p_outer_l2_baseline)
        ),
        pressure_inner_mean=float(
            boundary_pressure_mean(mesh, p_soln, mesh.boundaries.Lower.name)
        ),
        pressure_outer_mean=float(
            boundary_pressure_mean(mesh, p_soln, mesh.boundaries.Upper.name)
        ),
        inner_normal_rms=inner_normal_rms,
        outer_normal_rms=outer_normal_rms,
        inner_tangential_rms=inner_tangential_rms,
        outer_tangential_rms=outer_tangential_rms,
        snes_reason=int(stokes.snes.getConvergedReason()),
        ksp_reason=int(stokes.snes.ksp.getConvergedReason()),
    )

    if params.dbg_write_output:
        mesh.write_timestep(
            case_name,
            index=0,
            meshVars=[v_soln, p_soln],
            outputPath=run_dir,
        )

    return result, copy_variable_data(mesh, v_soln), copy_variable_data(mesh, p_soln)


def make_summary_text(params, results):
    lines = []
    lines.append(
        f"m={params.uw_m}, cellsize={params.uw_cellsize}, vdeg={params.uw_vdegree}, "
        f"pdeg={params.uw_pdegree}, qdeg={params.uw_qdegree if params.uw_qdegree > 0 else 'default'}"
    )
    lines.append(
        "case | p_vol_l2(ana) | p_inner_l2(ana) | p_inner_l2(base) | inner_tan_rms | penalty | tol"
    )
    for result in results:
        baseline_str = (
            "-"
            if result.pressure_inner_l2_baseline is None
            else f"{result.pressure_inner_l2_baseline:.6g}"
        )
        lines.append(
            f"{result.case_name} | "
            f"{result.pressure_volume_l2_analytic:.6g} | "
            f"{result.pressure_inner_l2_analytic:.6g} | "
            f"{baseline_str} | "
            f"{result.inner_tangential_rms:.6g} | "
            f"{result.penalty:.3g} | "
            f"{result.tolerance:.3g}"
        )
    return "\n".join(lines)


def run(params):
    params.uw_cellsize = parse_cellsize(params.uw_cellsize)
    params.uw_pcont = params.uw_pcont if params.uw_pdegree > 0 else False
    normal_types = parse_csv_strings(params.dbg_normal_types)
    extra_penalties = parse_csv_floats(params.dbg_extra_penalties)
    extra_tolerances = parse_csv_floats(params.dbg_extra_tolerances)
    mesh_qdegree = (
        params.uw_qdegree if params.uw_qdegree > 0 else max(params.uw_vdegree, params.uw_pdegree)
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_root = (
        params.dbg_output_root
        if params.dbg_output_root is not None
        else os.path.join(repo_root, "output", "spherical", "thieulot", "bc_normals")
    )
    run_id = make_case_id(
        inv_lc=int(round(1.0 / params.uw_cellsize)),
        m=params.uw_m,
        vdeg=params.uw_vdegree,
        pdeg=params.uw_pdegree,
        pcont=params.uw_pcont,
        qdeg=params.uw_qdegree if params.uw_qdegree > 0 else None,
        p_bc=params.uw_p_bc,
        penalty=params.uw_vel_penalty,
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

    projected_normals = None
    if "projected" in normal_types:
        projected_normals = build_projected_normals(mesh, degree=max(2, params.uw_vdegree))

    p_baseline = uw.discretisation.MeshVariable(
        "PressureBaseline",
        mesh,
        degree=params.uw_pdegree,
        vtype=uw.VarType.SCALAR,
        varsymbol=r"P_b",
        continuous=params.uw_pcont,
    )
    v_baseline = uw.discretisation.MeshVariable(
        "VelocityBaseline",
        mesh,
        degree=params.uw_vdegree,
        vtype=uw.VarType.VECTOR,
        varsymbol=r"V_b",
    )

    results = []

    if uw.mpi.rank == 0:
        print("Running case: essential", flush=True)
    essential_result, v_data, p_data = run_case(
        mesh=mesh,
        mesh_qdegree=mesh_qdegree,
        params=params,
        run_dir=run_dir,
        case_name="essential",
        bc_form="essential",
        normal_type="none",
        tol=params.uw_stokes_tol,
        penalty=params.uw_vel_penalty,
        projected_normals=projected_normals,
        p_baseline=None,
    )
    results.append(essential_result)
    fill_variable_data(mesh, v_baseline, v_data)
    fill_variable_data(mesh, p_baseline, p_data)

    if params.dbg_include_natural_full:
        if uw.mpi.rank == 0:
            print("Running case: natural_full", flush=True)
        result, _, _ = run_case(
            mesh=mesh,
            mesh_qdegree=mesh_qdegree,
            params=params,
            run_dir=run_dir,
            case_name="natural_full",
            bc_form="natural_full",
            normal_type="none",
            tol=params.uw_stokes_tol,
            penalty=params.uw_vel_penalty,
            projected_normals=projected_normals,
            p_baseline=p_baseline,
        )
        results.append(result)

    for normal_type in normal_types:
        if uw.mpi.rank == 0:
            print(f"Running case: natural_normal_{normal_type}", flush=True)
        result, _, _ = run_case(
            mesh=mesh,
            mesh_qdegree=mesh_qdegree,
            params=params,
            run_dir=run_dir,
            case_name=f"natural_normal_{normal_type}",
            bc_form="natural_normal",
            normal_type=normal_type,
            tol=params.uw_stokes_tol,
            penalty=params.uw_vel_penalty,
            projected_normals=projected_normals,
            p_baseline=p_baseline,
        )
        results.append(result)

    for penalty in extra_penalties:
        if uw.mpi.rank == 0:
            print(
                f"Running case: natural_normal_analytic_penalty_{case_value(penalty)}",
                flush=True,
            )
        result, _, _ = run_case(
            mesh=mesh,
            mesh_qdegree=mesh_qdegree,
            params=params,
            run_dir=run_dir,
            case_name=f"natural_normal_analytic_penalty_{case_value(penalty)}",
            bc_form="natural_normal",
            normal_type="analytic",
            tol=params.uw_stokes_tol,
            penalty=penalty,
            projected_normals=projected_normals,
            p_baseline=p_baseline,
        )
        results.append(result)

    for tol in extra_tolerances:
        if uw.mpi.rank == 0:
            print(
                f"Running case: natural_normal_analytic_tol_{case_value(tol)}",
                flush=True,
            )
        result, _, _ = run_case(
            mesh=mesh,
            mesh_qdegree=mesh_qdegree,
            params=params,
            run_dir=run_dir,
            case_name=f"natural_normal_analytic_tol_{case_value(tol)}",
            bc_form="natural_normal",
            normal_type="analytic",
            tol=tol,
            penalty=params.uw_vel_penalty,
            projected_normals=projected_normals,
            p_baseline=p_baseline,
        )
        results.append(result)

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
            "uw_p_bc": params.uw_p_bc,
            "uw_stokes_tol": params.uw_stokes_tol,
            "uw_vel_penalty": params.uw_vel_penalty,
            "dbg_pressure_gauge": params.dbg_pressure_gauge,
            "dbg_snes_type": params.dbg_snes_type,
            "mpi_size": uw.mpi.size,
        },
        "results": [asdict(result) for result in results],
    }

    if uw.mpi.rank == 0:
        summary_text = make_summary_text(params, results)
        print(summary_text)
        with open(os.path.join(run_dir, "summary.txt"), "w", encoding="ascii") as stream:
            stream.write(summary_text + "\n")
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="ascii") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)

    return payload


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
