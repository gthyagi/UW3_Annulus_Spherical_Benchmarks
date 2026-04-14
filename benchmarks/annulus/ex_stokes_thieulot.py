# %% [markdown]
# ## Annulus Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark Paper](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-2765/) [ASPECT Results](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/annulus/doc/annulus.html)
#
# ### Authors
# Thyagarajulu Gollapalli ([GitHub](https://github.com/gthyagi)) <br>
# Underworld3 Development Team ([UW3 Repository](https://github.com/underworldcode/underworld3))
#
# ### Analytical solution
#
# This benchmark is based on a manufactured solution in which an analytical solution to the isoviscous incompressible Stokes equations is derived in an annulus geometry. The velocity and pressure fields are as follows:
#
# $$ v_{\theta}(r, \theta) = f(r) \cos(k\theta) $$
#
# $$ v_r(r, \theta) = g(r)k \sin(k\theta) $$
#
# $$ p(r, \theta) = kh(r) \sin(k\theta) + \rho_0g_r(R_2 - r) $$
#
# $$ \rho(r, \theta) = m(r)k \sin(k\theta) + \rho_0 $$
#
# with
#
# $$ f(r) = Ar + \frac{B}{r} $$
#
# $$ g(r) = \frac{A}{2}r + \frac{B}{r}\ln r + \frac{C}{r} $$
#
# $$ h(r) = \frac{2g(r) - f(r)}{r} $$
#
# $$ m(r) = g''(r) - \frac{g'(r)}{r} - \frac{g(r)}{r^2}(k^2 - 1) + \frac{f(r)}{r^2} + \frac{f'(r)}{r} $$
#
# $$ A = -C\frac{2(\ln R_1 - \ln R_2)}{R_2^2 \ln R_1 - R_1^2 \ln R_2} $$
#
# $$ B = -C\frac{R_2^2 - R_1^2}{R_2^2 \ln R_1 - R_1^2 \ln R_2} $$
#
#
# The parameters $A$ and $B$ are chosen so that $ v_r(R_1, \theta) = v_r(R_2, \theta) = 0 $ for all $\theta \in [0, 2\pi]$, i.e. the velocity is tangential to both inner and outer surfaces. The gravity vector is radial inward and of unit length.
#
# The parameter $k$ controls the number of convection cells present in the domain
#
# In the present case, we set $ R_1 = 1.0, R_2 = 2.0$ and $C = -1 $.

# %%
import os
import subprocess
import sys
from fractions import Fraction
import h5py
import numpy as np
import sympy as sp
import underworld3 as uw
from mpi4py import MPI
from underworld3.systems import Stokes

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

is_serial = (uw.mpi.size == 1)

# %% [markdown]
# ### Mesh Parameters

# %%
# Ensure CLI arguments are parsed into PETSc options so uw.Params picks them up reliably across environments
uw.parse_cmd_line_options()

params = uw.Params(
    uw_cellsize=uw.Param(
        "1.0 / 64.0",
        type=uw.ParamType.STRING,
        description="Target annulus mesh cell size",
    ),
    uw_r_i=uw.Param(
        1.0,
        type=uw.ParamType.FLOAT,
        description="Inner annulus radius",
    ),
    uw_r_o=uw.Param(
        2.0,
        type=uw.ParamType.FLOAT,
        description="Outer annulus radius",
    ),
    uw_k=uw.Param(
        2,
        type=uw.ParamType.INTEGER,
        description="Convection-cell wave number",
    ),
    uw_vdegree=uw.Param(
        2,
        type=uw.ParamType.INTEGER,
        description="Velocity polynomial degree",
    ),
    uw_pdegree=uw.Param(
        1,
        type=uw.ParamType.INTEGER,
        description="Pressure polynomial degree",
    ),
    uw_pcont=uw.Param(
        True,
        type=uw.ParamType.BOOLEAN,
        description="Pressure continuity flag",
    ),
    uw_stokes_tol=uw.Param(
        1e-9,
        type=uw.ParamType.FLOAT,
        description="Stokes solver tolerance",
    ),
    uw_vel_penalty=uw.Param(
        2.5e8,
        type=uw.ParamType.FLOAT,
        description="Penalty for curved-boundary tangential flow",
    ),
    uw_bc_type=uw.Param(
        "essential",
        type=uw.ParamType.STRING,
        description="Boundary-condition mode: natural or essential",
    ),
    uw_run_on_gadi=uw.Param(
        False,
        type=uw.ParamType.BOOLEAN,
        description="Use Gadi scratch paths for benchmark output",
    ),
)

if any(arg in ("--help", "-h", "-help", "-uw_help") for arg in sys.argv[1:]):
    print(params.cli_help())
    raise SystemExit(0)

# %%
def parse_float_fraction(value):
    """Parse a decimal or simple rational string deterministically."""

    text = str(value).strip().replace(" ", "")
    if text.count("/") > 1:
        raise ValueError(f"Unsupported rational format: {value}")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(Fraction(numerator) / Fraction(denominator))
    return float(Fraction(text))

# %%
params.uw_cellsize = parse_float_fraction(params.uw_cellsize)

# set pressure continuity based on velocity degree
pressure_is_continuous = params.uw_pcont if params.uw_pdegree > 0 else False
is_p1p0 = params.uw_vdegree == 1 and params.uw_pdegree == 0

if uw.mpi.rank == 0 and params.uw_pdegree == 0 and params.uw_pcont:
    print("Degree-0 pressure uses discontinuous storage; overriding uw_pcont to false.")

# %%
# For k = 0 (no body force), the flow must be driven by boundary tractions,
# so use natural BCs. Essential BCs only prescribe velocities on the boundary
# and do not drive the interior solution. For k > 0, both BC types are effective.
if params.uw_k == 0:
    params.uw_bc_type = "natural"

# %%
# set uw_vel_penalty to None for essential BCs
if params.uw_bc_type == "essential":
    params.uw_vel_penalty = None

# %% [markdown]
# ### Output Directory

# %%
def _case_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return np.format_float_scientific(value, unique=True, precision=12, trim="-")
    return value


def make_case_id(*, case, **kwargs):
    parts = [case]
    parts += [f"{key}_{_case_value(value)}" for key, value in kwargs.items() if value is not None]
    return "_".join(parts)

# --- repo root (for git SHA, code reference) ---
if "__file__" in globals():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
else:
    # fallback for Jupyter / interactive
    repo_root = os.getcwd()

# --- output location (runtime dependent) ---
if params.uw_run_on_gadi:
    output_base = "/scratch/m18/tg7098"
else:
    output_base = repo_root

output_root = os.path.join(output_base, "output", "annulus", "thieulot", "latest")
metrics_filename = "benchmark_metrics.h5"

case_id = make_case_id(
    case="model",
    inv_lc=int(1 / params.uw_cellsize),
    k=params.uw_k,
    vdeg=params.uw_vdegree,
    pdeg=params.uw_pdegree,
    pcont=pressure_is_continuous,
    stokes_tol=params.uw_stokes_tol,
    ncpus=uw.mpi.size,
    bc=params.uw_bc_type,
    vel_penalty=params.uw_vel_penalty,
)

output_dir = os.path.join(output_root, case_id)
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True,)


# %% [markdown]
# ### Analytical Solution Helpers

# %%
def analytic_solution(
    mesh,
    r_i,
    r_o,
    k,
    C=-1,
    rho0=0,
):
    """Return analytical annulus benchmark fields (v, p, rho) as UW expressions."""

    x, y = mesh.CoordinateSystem.X
    r, th = mesh.CoordinateSystem.xR
    unit_rvec = mesh.CoordinateSystem.unit_e_0

    denom = (r_o**2) * sp.log(r_i) - (r_i**2) * sp.log(r_o)

    A = -C * (2 * (sp.log(r_i) - sp.log(r_o)) / denom)
    B = -C * ((r_o**2 - r_i**2) / denom)

    f = A * r + B / r
    g = (A / 2) * r + (B / r) * sp.log(r) + C / r
    h = (2 * g - f) / r

    m = (
        sp.diff(g, r, 2)
        - sp.diff(g, r) / r
        - (g / r**2) * (k**2 - 1)
        + f / r**2
        + sp.diff(f, r) / r
    )

    v_r = g * k * sp.sin(k * th)
    v_th = f * sp.cos(k * th)

    if k == 0:
        v_uw = mesh.CoordinateSystem.rRotN.T * sp.Matrix([0, v_th])
        p_uw = sp.Integer(0)
        rho_uw = sp.Integer(0)
    else:
        v_uw = mesh.CoordinateSystem.rRotN.T * sp.Matrix([v_r, v_th])
        p_uw = k * h * sp.sin(k * th) + rho0 * (r_o - r)
        rho_uw = m * k * sp.sin(k * th) + rho0

    return v_uw, p_uw, rho_uw


# %% [markdown]
# ### Create Mesh

# %%
uw.pprint("Stage start: mesh creation/loading")

mesh = uw.meshing.Annulus(
    radiusOuter=params.uw_r_o,
    radiusInner=params.uw_r_i,
    cellSize=params.uw_cellsize,
    qdegree=max(params.uw_pdegree, params.uw_vdegree),
    degree=1,
    filename=os.path.join(output_dir, "mesh.msh"),
)

if is_serial:
    mesh.dm.view()

# %%
uw.pprint("Stage complete: mesh creation/loading")

x, y = mesh.CoordinateSystem.X
r, th = mesh.CoordinateSystem.xR
unit_rvec = mesh.CoordinateSystem.unit_e_0

# %% [markdown]
# ### Create Mesh Variables

# %%
v_soln = uw.discretisation.MeshVariable(
    varname="Velocity", 
    mesh=mesh, 
    degree=params.uw_vdegree, 
    vtype=uw.VarType.VECTOR, 
    varsymbol=r"V"
)

p_soln = uw.discretisation.MeshVariable(
    varname="Pressure",
    mesh=mesh,
    degree=params.uw_pdegree,
    vtype=uw.VarType.SCALAR,
    varsymbol=r"P",
    continuous=pressure_is_continuous,
)

# %%
# Analytical solution and error expressions
v_ana_expr, p_ana_expr, rho_ana_expr = analytic_solution(
    mesh,
    params.uw_r_i,
    params.uw_r_o,
    params.uw_k,
)
v_err_expr = sp.Matrix(v_soln.sym).T - v_ana_expr
p_err_expr = p_soln.sym[0] - p_ana_expr

# %% [markdown]
# ### Stokes

# %% [markdown]
# #### Stokes Setup

# %%
stokes = Stokes(mesh, velocityField=v_soln, pressureField=p_soln)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0
stokes.petsc_use_pressure_nullspace = True

gravity_fn = -1.0 * unit_rvec
stokes.bodyforce = rho_ana_expr * gravity_fn

# %% [markdown]
# #### Nullspace Handling
#
# The coupled Stokes system is solved with PETSc's constant-pressure nullspace
# enabled. This removes the additive pressure gauge freedom during the solve
# without imposing an artificial pressure Dirichlet condition on either annulus
# boundary.
#
# We still subtract the domain-average pressure after the solve so the reported
# pressure field has a unique zero-mean gauge for benchmark comparisons.
#
# We do not register or subtract a rigid-body rotation mode in this script.
# Although a centered annulus with exact free-slip shell boundary conditions can
# admit one rotational velocity null mode, this benchmark does not use those
# boundary conditions:
# - `essential` prescribes the full analytical velocity on both boundaries
# - `natural` penalizes the full velocity error `v - v_ana` on both boundaries
#
# Both branches therefore select a specific tangential boundary velocity, so a
# rigid rotation is not an exact null mode of the posed problem.
#
# %% [markdown]
# #### Tolerance And BC Type
#
# `stokes.tolerance` does not affect the two boundary-condition branches equally.
#
# - `essential` applies the analytical velocity strongly on `Upper` and `Lower`.
#   In this benchmark that branch is relatively insensitive to a looser solve
#   tolerance. For the 8-rank `k=0` tests here, `stokes.tolerance = 1e-5`
#   still gave a velocity L2 error of about `7.4e-7`.
# - `natural` applies the analytical velocity weakly through the penalty term
#   `uw_vel_penalty * (v - v_ana)`. This branch is much more tolerance-sensitive.
#   For the same 8-rank `k=0` tests, `stokes.tolerance = 1e-5` gave a larger
#   velocity L2 error of about `9e-5`.
#
# Practical choices for this script:
# - `essential`: `1e-5` is a good fast default. Use `1e-8` if you want a tighter
#   benchmark comparison.
# - `natural`: prefer `1e-8` as the practical default. Use `1e-10` for the
#   tightest comparison if the MPI solve remains affordable.
# - If a single value is needed for both branches, `1e-8` is a reasonable
#   compromise.
#
# In the current UW Stokes implementation, setting `stokes.tolerance` also sets
# the inner fieldsplit tolerances:
#
# - `fieldsplit_pressure_ksp_rtol = 0.1 * tolerance`
# - `fieldsplit_velocity_ksp_rtol = 0.033 * tolerance`
#
# This is important because `stokes.tolerance` is not only the outer Stokes
# solve target. It also controls how hard PETSc works inside the Schur-complement
# preconditioner. For example, `stokes.tolerance = 1e-10` drives the inner block
# solves to about `1e-11` for pressure and `3.3e-12` for velocity, which can
# increase runtime sharply under MPI. Conversely, if the tolerance is too loose,
# the weak-BC `natural` branch usually loses accuracy much faster than the
# strongly enforced `essential` branch.
#
# %% [markdown]
# #### Boundary Conditions

# %%
lower = mesh.boundaries.Lower.name
upper = mesh.boundaries.Upper.name

if params.uw_bc_type == "natural":
    stokes.add_natural_bc(params.uw_vel_penalty * v_err_expr, upper)
    stokes.add_natural_bc(params.uw_vel_penalty * v_err_expr, lower)
elif params.uw_bc_type == "essential":
    stokes.add_essential_bc(v_ana_expr, upper)
    stokes.add_essential_bc(v_ana_expr, lower)
else:
    raise ValueError(f"Unknown bc_type: {params.uw_bc_type}")

# %% [markdown]
# #### Solver Notes
#
# This benchmark is linear: the viscosity is prescribed, and both the
# `essential` and `natural` boundary conditions are linear in the unknown
# velocity and pressure fields. For a linear problem we use
# `snes_type = "ksponly"`, which bypasses Newton iterations and calls the
# PETSc linear solver (`KSP`) directly.
#
# If we instead use `snes_type = "newtonls"`, PETSc wraps the same linear
# system inside a nonlinear Newton solve. That is useful for genuinely
# nonlinear problems, but here it makes the reported `SNES` iteration count
# harder to interpret because the benchmark itself is still linear.
#
# `stokes.tolerance` is the UW-level solver tolerance. In this branch it sets
# the `SNES` relative tolerance and related defaults, but it does not set
# `ksp_rtol`. Because we use `ksponly`, the important stopping criterion is
# `ksp_rtol`, which controls the required relative reduction in the linear
# residual. `ksp_atol` is the absolute residual tolerance; we set it to `0.0`
# so convergence is controlled by the relative tolerance rather than by an
# absolute threshold.
#
# In short:
# - `newtonls`: nonlinear Newton solve
# - `ksponly`: direct linear solve through `KSP`
# - `ksp_rtol`: relative linear residual tolerance
# - `ksp_atol`: absolute linear residual tolerance
# - `stokes.tolerance`: UW convenience tolerance kept consistent with `ksp_rtol`
#
# `P1/P0` is treated separately. In serial we use a direct LU solve as the
# reference result. Under MPI we use an `asm_lu` branch (`PCASM` with local LU
# subsolves) because the multigrid fieldsplit settings used for `P2/P1` and
# `P3/P2` are not robust for `P1/P0` here. This MPI `P1/P0` path is not a
# global exact solve, so the result depends on `-np`: changing the number of
# ranks changes the subdomain partition and therefore changes the preconditioner
# and the final iterative answer. For benchmark-quality MPI comparisons, prefer
# `P2/P1` and `P3/P2`.
#
# %% [markdown]
# #### Solver Settings

# %%
stokes.tolerance = params.uw_stokes_tol

stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["ksp_monitor_true_residual"] = None
stokes.petsc_options["ksp_converged_reason"] = None

# stokes.petsc_options["snes_monitor"] = None
# stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["snes_type"] = "ksponly"
stokes.petsc_options["ksp_rtol"] = params.uw_stokes_tol
stokes.petsc_options["ksp_atol"] = 0.0

if is_p1p0:
    if uw.mpi.size == 1:
        stokes.petsc_options["ksp_type"] = "preonly"
        stokes.petsc_options["pc_type"] = "lu"
    else:
        if uw.mpi.rank == 0:
            print(
                "P1/P0 under MPI uses asm_lu (ASM with local LU subsolves). "
                "Results are -np dependent because ASM changes with the domain partition."
            )

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

# %% [markdown]
# #### Solve Stokes

# %%
uw.pprint("Stage start: stokes solve")

uw.timing.reset()
uw.timing.start()
stokes.solve()
uw.timing.stop()
uw.timing.print_table(filename=os.path.join(output_dir, "stokes_timing.txt"),)

snes_reason = int(stokes.snes.getConvergedReason())
ksp_reason = int(stokes.snes.ksp.getConvergedReason())
snes_iterations = int(stokes.snes.getIterationNumber())
ksp_iterations = int(stokes.snes.ksp.getIterationNumber())

if uw.mpi.rank == 0:
    print(snes_reason)
    print(ksp_reason)

uw.pprint("Stage complete: stokes solve")

# %% [markdown]
# ### Benchmark Calibrations

# %%
def subtract_pressure_mean(mesh, pressure_var):
    """
    Subtract the domain-average pressure from the numerical pressure field.

    Parameters
    ----------
    mesh : uw.discretisation.Mesh
        Mesh used to evaluate the pressure and volume integrals.
    pressure_var : uw.discretisation.MeshVariable
        Scalar pressure field to shift to zero mean.
    """
    p_int = uw.maths.Integral(mesh, pressure_var.sym[0]).evaluate()
    volume = uw.maths.Integral(mesh, 1.0).evaluate()
    pressure_var.data[:, 0] -= p_int / volume


subtract_pressure_mean(mesh, p_soln)

# %% [markdown]
# ### Save h5 Output

# %%
uw.pprint("Stage start: saving h5 output")

mesh.write_timestep(
    "output",
    index=0,
    meshVars=[v_soln, p_soln],
    outputPath=str(output_dir),
)

uw.pprint("Stage complete: saving h5 output")

# %% [markdown]
# ### Relative Error Norms

# %%
def _squared_norm(expr):
    """Return squared magnitude of scalar/vector expression."""
    expr = expr.sym if hasattr(expr, "sym") else expr
    return expr.dot(expr) if isinstance(expr, sp.MatrixBase) else expr**2


def relative_l2_error(mesh, err, ana, boundary=None):
    """Compute relative L2 error over domain or specified boundary."""
    err_fn = _squared_norm(err)
    ana_fn = _squared_norm(ana)

    if boundary is None:
        err_I = uw.maths.Integral(mesh, err_fn)
        ana_I = uw.maths.Integral(mesh, ana_fn)
    else:
        err_I = uw.maths.BdIntegral(mesh=mesh, fn=err_fn, boundary=boundary)
        ana_I = uw.maths.BdIntegral(mesh=mesh, fn=ana_fn, boundary=boundary)

    return np.sqrt(err_I.evaluate() / ana_I.evaluate())


def absolute_l2_error(mesh, err, boundary=None):
    """Compute absolute L2 error over domain or specified boundary."""

    err_fn = _squared_norm(err)
    if boundary is None:
        err_I = uw.maths.Integral(mesh, err_fn)
    else:
        err_I = uw.maths.BdIntegral(mesh=mesh, fn=err_fn, boundary=boundary)
    return np.sqrt(err_I.evaluate())

# %%
def gather_run_metadata(
    mesh,
    velocity_var,
    pressure_var,
    snes_reason,
    ksp_reason,
    snes_iterations,
    ksp_iterations,
):
    """Return solver, mesh, and per-rank partition metadata for this run."""
    comm = MPI.COMM_WORLD

    v_start, v_end = mesh.dm.getDepthStratum(0)
    c_start, c_end = mesh.dm.getHeightStratum(0)

    local_vertices = int(v_end - v_start)
    local_cells = int(c_end - c_start)
    local_velocity_dofs = int(velocity_var.data.size)
    local_pressure_dofs = int(pressure_var.data.size)

    vertices_by_rank = comm.gather(local_vertices, root=0)
    cells_by_rank = comm.gather(local_cells, root=0)
    velocity_dofs_by_rank = comm.gather(local_velocity_dofs, root=0)
    pressure_dofs_by_rank = comm.gather(local_pressure_dofs, root=0)

    metadata = {
        "mpi_size": int(uw.mpi.size),
        "mesh_dim": int(mesh.dim),
        "global_vertices": int(comm.allreduce(local_vertices, op=MPI.SUM)),
        "global_cells": int(comm.allreduce(local_cells, op=MPI.SUM)),
        "global_velocity_dofs": int(comm.allreduce(local_velocity_dofs, op=MPI.SUM)),
        "global_pressure_dofs": int(comm.allreduce(local_pressure_dofs, op=MPI.SUM)),
        "snes_converged_reason": int(snes_reason),
        "ksp_converged_reason": int(ksp_reason),
        "snes_iterations": int(snes_iterations),
        "ksp_iterations": int(ksp_iterations),
    }

    if uw.mpi.rank == 0:
        vertices_by_rank = np.asarray(vertices_by_rank, dtype=np.int64)
        cells_by_rank = np.asarray(cells_by_rank, dtype=np.int64)
        velocity_dofs_by_rank = np.asarray(velocity_dofs_by_rank, dtype=np.int64)
        pressure_dofs_by_rank = np.asarray(pressure_dofs_by_rank, dtype=np.int64)

        metadata.update(
            {
                "local_vertices_by_rank": vertices_by_rank,
                "local_cells_by_rank": cells_by_rank,
                "local_velocity_dofs_by_rank": velocity_dofs_by_rank,
                "local_pressure_dofs_by_rank": pressure_dofs_by_rank,
                "cell_imbalance_ratio": float(cells_by_rank.max() / cells_by_rank.mean()),
                "velocity_dof_imbalance_ratio": float(
                    velocity_dofs_by_rank.max() / velocity_dofs_by_rank.mean()
                ),
                "pressure_dof_imbalance_ratio": float(
                    pressure_dofs_by_rank.max() / pressure_dofs_by_rank.mean()
                ),
                "rank_index_note": np.bytes_("array index corresponds to MPI rank"),
            }
        )

    return metadata

# %%
def current_git_sha(repo_path):
    """Return current git SHA, or 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# %%
v_err_l2 = relative_l2_error(mesh, v_err_expr, v_ana_expr)
p_err_l2_abs = absolute_l2_error(mesh, p_err_expr)
p_err_l2 = np.nan if params.uw_k == 0 else relative_l2_error(mesh, p_err_expr, p_ana_expr)

v_err_l2_lower = relative_l2_error(mesh, v_err_expr, v_ana_expr, boundary=lower)
v_err_l2_upper = relative_l2_error(mesh, v_err_expr, v_ana_expr, boundary=upper)

p_err_l2_lower_abs = absolute_l2_error(mesh, p_err_expr, boundary=lower)
p_err_l2_upper_abs = absolute_l2_error(mesh, p_err_expr, boundary=upper)
p_err_l2_lower = (
    np.nan
    if params.uw_k == 0
    else relative_l2_error(mesh, p_err_expr, p_ana_expr, boundary=lower)
)
p_err_l2_upper = (
    np.nan
    if params.uw_k == 0
    else relative_l2_error(mesh, p_err_expr, p_ana_expr, boundary=upper)
)

u_dot_n_l2_lower_abs = absolute_l2_error(mesh, unit_rvec.dot(v_soln.sym), boundary=lower)
u_dot_n_l2_upper_abs = absolute_l2_error(mesh, unit_rvec.dot(v_soln.sym), boundary=upper)

run_metadata = gather_run_metadata(
    mesh,
    v_soln,
    p_soln,
    snes_reason,
    ksp_reason,
    snes_iterations,
    ksp_iterations,
)

git_sha = current_git_sha(repo_root)
cli_args = " ".join(sys.argv)

metrics = {
    "k": params.uw_k,
    "cellsize": params.uw_cellsize,
    "v_l2_norm": v_err_l2,
    "p_l2_norm": p_err_l2,
    "p_l2_norm_abs": p_err_l2_abs,
    "v_l2_norm_lower": v_err_l2_lower,
    "v_l2_norm_upper": v_err_l2_upper,
    "p_l2_norm_lower": p_err_l2_lower,
    "p_l2_norm_upper": p_err_l2_upper,
    "p_l2_norm_lower_abs": p_err_l2_lower_abs,
    "p_l2_norm_upper_abs": p_err_l2_upper_abs,
    "u_dot_n_l2_norm_lower_abs": u_dot_n_l2_lower_abs,
    "u_dot_n_l2_norm_upper_abs": u_dot_n_l2_upper_abs,
}

if uw.mpi.rank == 0:
    print("=== L2 Error Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

# %% [markdown]
# ### Save Outputs

# %%
uw.pprint("Stage start: saving metric output")

if uw.mpi.rank == 0:
    metrics_h5 = os.path.join(output_dir, metrics_filename)
    if os.path.isfile(metrics_h5):
        os.remove(metrics_h5)

    with h5py.File(metrics_h5, "w") as f_h5:
        for key, value in metrics.items():
            f_h5.create_dataset(key, data=value)

        f_h5.create_dataset("git_sha", data=np.bytes_(git_sha))
        f_h5.create_dataset("command", data=np.bytes_(cli_args))

        for key, value in run_metadata.items():
            f_h5.create_dataset(key, data=value)

uw.pprint("Stage complete: saving metric output")
