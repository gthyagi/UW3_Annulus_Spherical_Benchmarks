#!/usr/bin/env python3
"""
Standalone spherical Thieulot benchmark debug driver.

This script is intentionally isolated from the main benchmark drivers so the
`m = -1` and `m = 3` cases can be compared with a minimal, transparent setup.
It preserves the same analytical solution and Stokes formulation as the current
benchmark path, while adding pressure-focused diagnostics that are useful for
explaining why a run can have a reasonable velocity field but a poor pressure
match.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import h5py
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
class FieldDiagnostics:
    relative_nodal_l2: float
    correlation: float
    projection_scale: float


@dataclass
class PressureDiagnostics:
    relative_integral_l2: float
    relative_nodal_l2: float
    relative_nodal_l2_affine: float
    correlation: float
    projection_scale: float
    affine_scale: float
    affine_offset: float
    numerical_mean: float
    analytical_mean: float
    numerical_std: float
    analytical_std: float
    numerical_inner_surface_mean: float
    numerical_outer_surface_mean: float


@dataclass
class ResidualDiagnostics:
    solved_function_norm: float
    analytical_function_norm: float
    analytical_to_solved_ratio: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone spherical Thieulot debug driver.",
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
    parser.add_argument("-uw_vel_penalty", type=float, default=1.0e8)
    parser.add_argument("-uw_bc_type", choices=("natural", "essential"), default="essential")
    parser.add_argument("-uw_p_bc", type=parse_bool, default=False)
    parser.add_argument(
        "-dbg_stokes_penalty",
        type=float,
        default=0.0,
        help="Optional augmented-Lagrangian penalty for reproducing older spherical solve paths.",
    )
    parser.add_argument(
        "-dbg_snes_type",
        choices=("ksponly", "newtonls"),
        default="ksponly",
        help="PETSc SNES mode for comparing current and older spherical solve paths.",
    )
    parser.add_argument(
        "-dbg_pressure_gauge",
        choices=("none", "volume_mean", "inner_surface_mean", "outer_surface_mean"),
        default="volume_mean",
        help="Reporting-only pressure gauge shift applied after the solve.",
    )
    parser.add_argument(
        "-dbg_pressure_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the analytical pressure for comparison and residual probing.",
    )
    parser.add_argument(
        "-dbg_bodyforce_sign",
        type=float,
        default=-1.0,
        help="Multiplier applied to rho before forming rho * gravity.",
    )
    parser.add_argument(
        "-dbg_bodyforce_scale",
        type=float,
        default=1.0,
        help="Additional scale applied to rho before forming rho * gravity.",
    )
    parser.add_argument(
        "-dbg_write_output",
        type=parse_bool,
        default=False,
        help="Write mesh fields to the debug output directory.",
    )
    parser.add_argument(
        "-dbg_output_root",
        default=None,
        help="Override the root output directory.",
    )
    return parser


def analytic_solution(mesh, r_i, r_o, m, gamma=1.0, mu_0=1.0):
    """
    Return spherical benchmark fields (velocity, pressure, density, body-force density, viscosity).
    """

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

    err_I = uw.maths.Integral(mesh, err_sq)
    ana_I = uw.maths.Integral(mesh, ana_sq)
    return float(np.sqrt(err_I.evaluate()) / np.sqrt(ana_I.evaluate()))


def subtract_constant(mesh, pressure_var, value):
    with mesh.access(pressure_var):
        pressure_var.data[:, 0] -= value


def global_integral(mesh, fn):
    local = float(uw.maths.Integral(mesh, fn).evaluate())
    return float(uw.mpi.comm.allreduce(local))


def global_boundary_integral(mesh, fn, boundary_name):
    local = float(uw.maths.BdIntegral(mesh=mesh, fn=fn, boundary=boundary_name).evaluate())
    return float(uw.mpi.comm.allreduce(local))


def pressure_mean(mesh, pressure_var):
    volume = global_integral(mesh, 1.0)
    if np.isclose(volume, 0.0):
        raise ValueError("Mesh has zero volume; cannot compute pressure mean.")
    return global_integral(mesh, pressure_var.sym[0]) / volume


def boundary_pressure_mean(mesh, pressure_var, boundary_name):
    measure = global_boundary_integral(mesh, 1.0, boundary_name)
    if np.isclose(measure, 0.0):
        return 0.0
    return global_boundary_integral(mesh, pressure_var.sym[0], boundary_name) / measure


def apply_pressure_gauge(mesh, pressure_var, gauge: str):
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


def safe_corrcoef(a, b):
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.allclose(np.std(a), 0.0) or np.allclose(np.std(b), 0.0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_projection_scale(reference, target):
    denom = float(np.dot(reference, reference))
    if np.isclose(denom, 0.0):
        return float("nan")
    return float(np.dot(target, reference) / denom)


def evaluate_scalar(expr, coords):
    values = np.asarray(uw.function.evaluate(expr, coords), dtype=np.float64)
    return values.reshape(-1)


def evaluate_vector(expr, coords):
    values = np.asarray(uw.function.evaluate(expr, coords), dtype=np.float64)
    return values.reshape(-1, 3)


def assign_field_from_expr(mesh, field, expr):
    coords = np.asarray(field.coords)
    values = np.asarray(uw.function.evaluate(expr, coords), dtype=np.float64)
    with mesh.access(field):
        field.data[...] = values.reshape(field.data.shape)


def copy_fields_to_global(stokes):
    gvec = stokes.dm.getGlobalVec()
    gvec.setArray(0.0)
    for name, var in stokes.fields.items():
        sgvec = gvec.getSubVector(stokes._subdict[name][0])
        subdm = stokes._subdict[name][1]
        subdm.localToGlobal(var.vec, sgvec)
        gvec.restoreSubVector(stokes._subdict[name][0], sgvec)
    return gvec


def compute_function_norm(stokes):
    gvec = copy_fields_to_global(stokes)
    fvec = stokes.dm.getGlobalVec()
    stokes.snes.computeFunction(gvec, fvec)
    return float(fvec.norm())


def field_diagnostics(numerical, analytical):
    num = np.asarray(numerical, dtype=np.float64).reshape(-1)
    ana = np.asarray(analytical, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(ana)
    rel = float(np.linalg.norm(num - ana) / denom) if not np.isclose(denom, 0.0) else float("nan")
    return FieldDiagnostics(
        relative_nodal_l2=rel,
        correlation=safe_corrcoef(num, ana),
        projection_scale=safe_projection_scale(ana, num),
    )


def pressure_diagnostics(mesh, pressure_var, p_ana_expr, relative_integral_l2):
    coords = np.asarray(pressure_var.coords)
    p_num = np.asarray(pressure_var.data[:, 0], dtype=np.float64).copy()
    p_ana = evaluate_scalar(p_ana_expr, coords)

    ref_norm = np.linalg.norm(p_ana)
    rel_nodal = float(np.linalg.norm(p_num - p_ana) / ref_norm)

    design = np.column_stack([p_ana, np.ones_like(p_ana)])
    affine_scale, affine_offset = np.linalg.lstsq(design, p_num, rcond=None)[0]
    p_affine = affine_scale * p_ana + affine_offset
    rel_affine = float(np.linalg.norm(p_num - p_affine) / ref_norm)

    return PressureDiagnostics(
        relative_integral_l2=float(relative_integral_l2),
        relative_nodal_l2=rel_nodal,
        relative_nodal_l2_affine=rel_affine,
        correlation=safe_corrcoef(p_num, p_ana),
        projection_scale=safe_projection_scale(p_ana, p_num),
        affine_scale=float(affine_scale),
        affine_offset=float(affine_offset),
        numerical_mean=float(np.mean(p_num)),
        analytical_mean=float(np.mean(p_ana)),
        numerical_std=float(np.std(p_num)),
        analytical_std=float(np.std(p_ana)),
        numerical_inner_surface_mean=float(
            boundary_pressure_mean(mesh, pressure_var, mesh.boundaries.Lower.name)
        ),
        numerical_outer_surface_mean=float(
            boundary_pressure_mean(mesh, pressure_var, mesh.boundaries.Upper.name)
        ),
    )


def residual_diagnostics(mesh, stokes, velocity_var, pressure_var, v_ana_expr, p_ana_expr):
    solved_velocity = np.asarray(velocity_var.data, dtype=np.float64).copy()
    solved_pressure = np.asarray(pressure_var.data, dtype=np.float64).copy()
    solved_norm = float(stokes.snes.getFunctionNorm())

    assign_field_from_expr(mesh, velocity_var, v_ana_expr)
    assign_field_from_expr(mesh, pressure_var, p_ana_expr)
    analytical_norm = compute_function_norm(stokes)

    with mesh.access(velocity_var, pressure_var):
        velocity_var.data[...] = solved_velocity
        pressure_var.data[...] = solved_pressure

    ratio = analytical_norm / solved_norm if not np.isclose(solved_norm, 0.0) else float("nan")
    return ResidualDiagnostics(
        solved_function_norm=solved_norm,
        analytical_function_norm=analytical_norm,
        analytical_to_solved_ratio=float(ratio),
    )


def run(params):
    params.uw_cellsize = parse_cellsize(params.uw_cellsize)
    pressure_is_continuous = params.uw_pcont if params.uw_pdegree > 0 else False
    is_p1p0 = params.uw_vdegree == 1 and params.uw_pdegree == 0
    mesh_qdegree = (
        params.uw_qdegree if params.uw_qdegree > 0 else max(params.uw_vdegree, params.uw_pdegree)
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_root = (
        params.dbg_output_root
        if params.dbg_output_root is not None
        else os.path.join(repo_root, "output", "sphere", "thieulot", "debug")
    )
    case_id = make_case_id(
        case="case",
        inv_lc=int(round(1.0 / params.uw_cellsize)),
        m=params.uw_m,
        vdeg=params.uw_vdegree,
        pdeg=params.uw_pdegree,
        pcont=pressure_is_continuous,
        qdeg=params.uw_qdegree if params.uw_qdegree > 0 else None,
        bc=params.uw_bc_type,
        p_bc=params.uw_p_bc,
        gauge=params.dbg_pressure_gauge,
        pscale=params.dbg_pressure_scale if not np.isclose(params.dbg_pressure_scale, 1.0) else None,
        bfsign=params.dbg_bodyforce_sign,
        bfscale=params.dbg_bodyforce_scale if not np.isclose(params.dbg_bodyforce_scale, 1.0) else None,
        np=uw.mpi.size,
    )
    output_dir = os.path.join(output_root, case_id)
    if uw.mpi.rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    mesh = uw.meshing.SphericalShell(
        radiusInner=params.uw_r_i,
        radiusOuter=params.uw_r_o,
        cellSize=params.uw_cellsize,
        qdegree=mesh_qdegree,
        degree=1,
        filename=os.path.join(output_dir, "mesh.msh"),
    )

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
        continuous=pressure_is_continuous,
    )

    v_ana_expr, p_ana_expr, rho_expr, mu_expr = analytic_solution(
        mesh,
        params.uw_r_i,
        params.uw_r_o,
        params.uw_m,
    )
    p_ana_expr = sp.simplify(params.dbg_pressure_scale * p_ana_expr)
    v_err_expr = sp.Matrix(v_soln.sym).T - v_ana_expr
    p_err_expr = p_soln.sym[0] - p_ana_expr

    stokes = Stokes(mesh, velocityField=v_soln, pressureField=p_soln)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = mu_expr
    stokes.penalty = params.dbg_stokes_penalty
    if np.isclose(params.dbg_stokes_penalty, 0.0):
        stokes.saddle_preconditioner = 1.0 / mu_expr
    else:
        stokes.saddle_preconditioner = 1.0 / (mu_expr + params.dbg_stokes_penalty)
    if not params.uw_p_bc:
        stokes.petsc_use_pressure_nullspace = True

    gravity_fn = -1.0 * mesh.CoordinateSystem.unit_e_0
    stokes.bodyforce = (
        params.dbg_bodyforce_scale * params.dbg_bodyforce_sign * rho_expr
    ) * gravity_fn

    if params.uw_bc_type == "natural":
        stokes.add_natural_bc(
            params.uw_vel_penalty * v_err_expr,
            mesh.boundaries.Upper.name,
        )
        stokes.add_natural_bc(
            params.uw_vel_penalty * v_err_expr,
            mesh.boundaries.Lower.name,
        )
    else:
        stokes.add_essential_bc(v_ana_expr, mesh.boundaries.Upper.name)
        stokes.add_essential_bc(v_ana_expr, mesh.boundaries.Lower.name)

    if params.uw_p_bc:
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

    stokes.tolerance = params.uw_stokes_tol
    stokes.petsc_options["ksp_monitor"] = None
    stokes.petsc_options["ksp_monitor_true_residual"] = None
    stokes.petsc_options["ksp_converged_reason"] = None
    stokes.petsc_options["snes_type"] = params.dbg_snes_type
    stokes.petsc_options["ksp_rtol"] = params.uw_stokes_tol
    stokes.petsc_options["ksp_atol"] = 0.0

    if is_p1p0:
        if uw.mpi.size == 1:
            stokes.petsc_options["ksp_type"] = "preonly"
            stokes.petsc_options["pc_type"] = "lu"
        else:
            stokes.petsc_options["ksp_type"] = "gmres"
            stokes.petsc_options["ksp_max_it"] = 500
            stokes.petsc_options["ksp_pc_side"] = "right"
            stokes.petsc_options["pc_type"] = "asm"
            stokes.petsc_options["pc_asm_type"] = "basic"
            stokes.petsc_options["sub_ksp_type"] = "preonly"
            stokes.petsc_options["sub_pc_type"] = "lu"
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

    uw.timing.reset()
    uw.timing.start()
    stokes.solve(verbose=False, debug=False)
    uw.timing.stop()
    uw.timing.print_table(filename=os.path.join(output_dir, "stokes_timing.txt"))

    if not params.uw_p_bc:
        apply_pressure_gauge(mesh, p_soln, params.dbg_pressure_gauge)

    v_err_l2 = relative_l2_error(mesh, v_err_expr, v_ana_expr)
    p_err_l2 = relative_l2_error(mesh, p_err_expr, p_ana_expr)

    v_coords = np.asarray(v_soln.coords)
    v_num = np.asarray(v_soln.data, dtype=np.float64).copy()
    v_ana = evaluate_vector(v_ana_expr, v_coords)
    velocity_diag = field_diagnostics(v_num.reshape(-1), v_ana.reshape(-1))
    pressure_diag = pressure_diagnostics(mesh, p_soln, p_ana_expr, p_err_l2)
    residual_diag = residual_diagnostics(mesh, stokes, v_soln, p_soln, v_ana_expr, p_ana_expr)

    diagnostics = {
        "case_id": case_id,
        "output_dir": output_dir,
        "params": {
            "uw_cellsize": params.uw_cellsize,
            "uw_r_i": params.uw_r_i,
            "uw_r_o": params.uw_r_o,
            "uw_m": params.uw_m,
            "uw_vdegree": params.uw_vdegree,
            "uw_pdegree": params.uw_pdegree,
            "uw_pcont": pressure_is_continuous,
            "uw_qdegree": params.uw_qdegree,
            "uw_stokes_tol": params.uw_stokes_tol,
            "uw_vel_penalty": params.uw_vel_penalty,
            "uw_bc_type": params.uw_bc_type,
            "uw_p_bc": params.uw_p_bc,
            "dbg_stokes_penalty": params.dbg_stokes_penalty,
            "dbg_snes_type": params.dbg_snes_type,
            "dbg_pressure_gauge": params.dbg_pressure_gauge,
            "dbg_pressure_scale": params.dbg_pressure_scale,
            "dbg_bodyforce_sign": params.dbg_bodyforce_sign,
            "dbg_bodyforce_scale": params.dbg_bodyforce_scale,
            "mpi_size": uw.mpi.size,
        },
        "solver": {
            "snes_reason": int(stokes.snes.getConvergedReason()),
            "ksp_reason": int(stokes.snes.ksp.getConvergedReason()),
        },
        "residuals": asdict(residual_diag),
        "velocity": {
            "relative_integral_l2": v_err_l2,
            **asdict(velocity_diag),
        },
        "pressure": asdict(pressure_diag),
        "rho_stats": {
            "domain_mean": global_integral(mesh, rho_expr) / global_integral(mesh, 1.0),
        },
    }

    if uw.mpi.rank == 0:
        print(f"Case: {case_id}")
        print(f"Output directory: {output_dir}")
        print(f"Relative velocity L2 error: {v_err_l2}")
        print(f"Relative pressure L2 error: {p_err_l2}")
        print(
            "Pressure nodal diagnostics: "
            f"corr={pressure_diag.correlation:.6g}, "
            f"proj_scale={pressure_diag.projection_scale:.6g}, "
            f"affine_scale={pressure_diag.affine_scale:.6g}, "
            f"affine_offset={pressure_diag.affine_offset:.6g}, "
            f"rel_l2_affine={pressure_diag.relative_nodal_l2_affine:.6g}"
        )
        print(
            "Residual diagnostics: "
            f"solved={residual_diag.solved_function_norm:.6g}, "
            f"analytical={residual_diag.analytical_function_norm:.6g}, "
            f"ratio={residual_diag.analytical_to_solved_ratio:.6g}"
        )

        with open(os.path.join(output_dir, "diagnostics.json"), "w", encoding="ascii") as stream:
            json.dump(diagnostics, stream, indent=2, sort_keys=True)

        err_h5 = os.path.join(output_dir, "error_norm.h5")
        if os.path.isfile(err_h5):
            os.remove(err_h5)
        with h5py.File(err_h5, "w") as h5f:
            h5f.create_dataset("m", data=params.uw_m)
            h5f.create_dataset("cellsize", data=params.uw_cellsize)
            h5f.create_dataset("v_l2_norm", data=v_err_l2)
            h5f.create_dataset("p_l2_norm", data=p_err_l2)

    if params.dbg_write_output:
        mesh.write_timestep(
            "output",
            index=0,
            meshVars=[v_soln, p_soln],
            outputPath=str(output_dir),
        )

    return diagnostics


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
