# %% [markdown]
# ## Spherical Benchmark: Viscous Incompressible Stokes
#
# #### [Benchmark Paper](https://se.copernicus.org/articles/8/1181/2017/) [ASPECT Results](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/hollow_sphere/doc/hollow_sphere.html)
#
# ### Authors
# Thyagarajulu Gollapalli ([GitHub](https://github.com/gthyagi)) <br>
# Underworld3 Development Team ([UW3 Repository](https://github.com/underworldcode/underworld3))
#
# ### Analytical solution
#
# This benchmark is based on [Thieulot](https://se.copernicus.org/articles/8/1181/2017/) in which an analytical solution to the isoviscous incompressible Stokes equations is derived in a spherical shell geometry. The velocity and pressure fields are as follows:
#
# $$ v_{\theta}(r, \theta) = f(r) \sin(\theta) $$
# $$ v_{\phi}(r, \theta) = f(r) \sin(\theta) $$
# $$ v_r(r, \theta) = g(r) \cos(\theta) $$
# $$ p(r, \theta) = h(r) \cos(\theta) $$
# $$ \mu(r) = \mu_{0}r^{m+1} $$
#
# where $m$ is an integer (positive or negative). Note that $m = -1$ yields a constant viscosity.
#
# $$ f(r) = {\alpha} r^{-(m+3)} + \beta r $$
#
# ##### Case $m = -1$
#
# $$ g(r) = -\frac{2}{r^2} \bigg(\alpha \ln r + \frac{\beta}{3}r^3 + \gamma \bigg) $$
# $$ h(r) = \frac{2}{r} \mu_{0} g(r) $$
# $$ \rho(r, \theta) = \bigg(\frac{\alpha}{r^4} (8\ln r - 6) + \frac{8\beta}{3r} + 8\frac{\gamma}{r^4} \bigg) \cos(\theta)$$
# $$ \alpha = -\gamma \frac{R_2^3 - R_1^3}{R_2^3 \ln R_1 - R_1^3 \ln R_2} $$
# $$ \beta = -3\gamma \frac{\ln R_2 - \ln R_1}{R_1^3 \ln R_2 - R_2^3 \ln R_1} $$
#
# ##### Case $m \neq -1$
#
# $$ g(r) = -\frac{2}{r^2} \bigg(-\frac{\alpha}{m+1} r^{-(m+1)} + \frac{\beta}{3}r^3 + \gamma \bigg) $$
# $$ h(r) = \frac{m+3}{r} \mu(r) g(r) $$
# $$ \rho(r, \theta) = \bigg[2\alpha r^{-(m+4)}\frac{m+3}{m+1}(m-1) - \frac{2\beta}{3}(m-1)(m+3) - m(m+5)\frac{2\gamma}{r^3} \bigg] \cos(\theta) $$
# $$ \alpha = \gamma (m+1) \frac{R_1^{-3} - R_2^{-3}}{R_1^{-(m+4)} - R_2^{-(m+4)}} $$
# $$ \beta = -3\gamma \frac{R_1^{m+1} - R_2^{m+1}}{R_1^{m+4} - R_2^{m+4}} $$
# Note that this imposes that $m \neq -4$.
#
# The radial component of the velocity is zero on the inside $r = R_1$ and outside $r = R_2$ of the domain, thereby ensuring a tangential flow on the boundaries, i.e.
# $$ v_r(R_1, \theta) = v_r(R_2, \theta) = 0 $$
#
# The gravity vector is radial and of unit length. We set $R_1 = 0.5$ and $R_2 = 1.0$.
#
# In this work, the following spherical coordinates conventions are used: $r$ is the radial distance, $\theta \in [0,\pi]$ is the polar angle and $\phi \in [0, 2\pi]$ is the azimuthal angle.

# %%
import os
import subprocess
import sys
from fractions import Fraction
import h5py
from mpi4py import MPI
import numpy as np
import sympy as sp
import underworld3 as uw
from underworld3.systems import Stokes

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

is_serial = (uw.mpi.size == 1)

# %% [markdown]
# ### Mesh Parameters

# %%
params = uw.Params(
    uw_cellsize=uw.Param(
        1.0 / 8.0,
        type=uw.ParamType.STRING,
        description="Target spherical-shell mesh cell size",
    ),
    uw_r_i=uw.Param(
        0.5,
        type=uw.ParamType.FLOAT,
        description="Inner spherical-shell radius",
    ),
    uw_r_o=uw.Param(
        1.0,
        type=uw.ParamType.FLOAT,
        description="Outer spherical-shell radius",
    ),
    uw_m=uw.Param(
        -1,
        type=uw.ParamType.INTEGER,
        description="Viscosity exponent in the analytical solution",
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
        1e-6,
        type=uw.ParamType.FLOAT,
        description="Stokes solver tolerance",
    ),
    uw_vel_penalty=uw.Param(
        1e8,
        type=uw.ParamType.FLOAT,
        description="Penalty for curved-boundary tangential flow",
    ),
    uw_bc_type=uw.Param(
        "essential",
        type=uw.ParamType.STRING,
        description="Boundary-condition mode: natural or essential",
    ),
    run_on_gadi=uw.Param(
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

if params.uw_m == -4:
    raise ValueError("The Thieulot spherical benchmark is undefined for m = -4.")

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
if params.run_on_gadi:
    output_base = "/scratch/m18/tg7098"
else:
    output_base = repo_root

output_root = os.path.join(output_base, "output", "spherical", "thieulot", "latest")
metrics_filename = "benchmark_metrics.h5"

case_id = make_case_id(
    case="case",
    inv_lc=int(1 / params.uw_cellsize),
    m=params.uw_m,
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
    os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ### Analytical Solution Helpers

# %%
def analytic_solution(  
    mesh,
    r_i,
    r_o,
    m,
    gamma=1.0,
    mu_0=1.0,
):
    """Return spherical benchmark fields (v, p, rho, bodyforce-rho, mu) as UW expressions."""

    r = mesh.CoordinateSystem.xR[0]
    theta = mesh.CoordinateSystem.xR[1]
    phi_raw = mesh.CoordinateSystem.xR[2]
    phi = sp.Piecewise(
        (2 * sp.pi + phi_raw, phi_raw < 0),
        (phi_raw, True),
    )

    mu_expr = mu_0 * (r ** (m + 1))

    if m == -1:
        alpha = -gamma * (
            (r_o**3 - r_i**3)
            / ((r_o**3) * np.log(r_i) - (r_i**3) * np.log(r_o))
        )
        beta = -3.0 * gamma * (
            (np.log(r_o) - np.log(r_i))
            / ((r_i**3) * np.log(r_o) - (r_o**3) * np.log(r_i))
        )

        f = alpha * (r ** -(m + 3)) + beta * r
        g = (-2.0 / (r**2)) * (alpha * sp.log(r) + (beta / 3.0) * (r**3) + gamma)
        h = (2.0 / r) * mu_0 * g

        force_term = (
            -(r * sp.diff(f, r, 3))
            - (3.0 * sp.diff(f, r, 2))
            + ((2.0 * sp.diff(f, r) / r) - sp.diff(g, r, 2))
            + 2.0 * ((f + g) / r**2)
        )
        rho_expr = sp.simplify(force_term * sp.cos(theta))
        rho_bodyforce_expr = -rho_expr
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

        # The implemented 3D Cartesian field requires an additional r**m factor
        # in the radial body-force coefficient for m != -1.
        rho_expr = sp.simplify(
            (r**m)
            * (
                2.0 * alpha * r ** (-(m + 4)) * ((m + 3) / (m + 1)) * (m - 1)
                - (2.0 * beta / 3.0) * (m - 1) * (m + 3)
                - m * (m + 5) * (2.0 * gamma / r**3)
            )
            * sp.cos(theta)
        )
        rho_bodyforce_expr = -rho_expr

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

    return v_expr, p_expr, rho_expr, rho_bodyforce_expr, mu_expr


# %% [markdown]
# ### Create Mesh

# %%
mesh = uw.meshing.SphericalShell(
    radiusInner=params.uw_r_i,
    radiusOuter=params.uw_r_o,
    cellSize=params.uw_cellsize,
    qdegree=max(params.uw_pdegree, params.uw_vdegree),
    degree=1,
    filename=os.path.join(output_dir, "mesh.msh"),
)

if is_serial:
    mesh.dm.view()

# %%
x, y, z = mesh.CoordinateSystem.X
unit_rvec = mesh.CoordinateSystem.unit_e_0

# %% [markdown]
# ### Create Mesh Variables

# %%
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

# %%
v_ana_expr, p_ana_expr, rho_ana_expr, rho_bodyforce_expr, mu_expr = analytic_solution(
    mesh,
    params.uw_r_i,
    params.uw_r_o,
    params.uw_m,
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
stokes.constitutive_model.Parameters.viscosity = mu_expr
stokes.saddle_preconditioner = 1.0 / mu_expr
stokes.petsc_use_pressure_nullspace = True

gravity_fn = -1.0 * unit_rvec
stokes.bodyforce = rho_bodyforce_expr * gravity_fn

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
#   tolerance. 
# - `natural` applies the analytical velocity weakly through the penalty term
#   `uw_vel_penalty * (v - v_ana)`. This branch is much more tolerance-sensitive.
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
uw.timing.reset()
uw.timing.start()
stokes.solve()
uw.timing.stop()
uw.timing.print_table(filename=os.path.join(output_dir, "stokes_timing.txt"))

snes_reason = int(stokes.snes.getConvergedReason())
ksp_reason = int(stokes.snes.ksp.getConvergedReason())
snes_iterations = int(stokes.snes.getIterationNumber())
ksp_iterations = int(stokes.snes.ksp.getIterationNumber())

if uw.mpi.rank == 0:
    print(snes_reason)
    print(ksp_reason)

# %% [markdown]
# ### Benchmark Calibrations
#
# Pressure is determined only up to a constant, so after the PETSc nullspace
# solve we shift the reported pressure field to the benchmark gauge used for
# comparison.

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
    p_int_local = float(uw.maths.Integral(mesh, pressure_var.sym[0]).evaluate())
    volume_local = float(uw.maths.Integral(mesh, 1.0).evaluate())

    p_int = float(uw.mpi.comm.allreduce(p_int_local))
    volume = float(uw.mpi.comm.allreduce(volume_local))

    if np.isclose(volume, 0.0):
        raise ValueError("The mesh has zero global volume; cannot normalize pressure.")

    pressure_var.data[:, 0] -= p_int / volume

def subtract_surface_pressure_mean(
    mesh,
    pressure_var,
    boundary_name,
):
    """
    Shift pressure so the average pressure on a named boundary is zero.
    """
    p_bd_int_local = float(
        uw.maths.BdIntegral(mesh=mesh, fn=pressure_var.sym[0], boundary=boundary_name).evaluate()
    )
    bd_measure_local = float(
        uw.maths.BdIntegral(mesh=mesh, fn=1.0, boundary=boundary_name).evaluate()
    )

    p_bd_int = float(uw.mpi.comm.allreduce(p_bd_int_local))
    bd_measure = float(uw.mpi.comm.allreduce(bd_measure_local))

    if np.isclose(bd_measure, 0.0):
        return
    
    pressure_var.data[:, 0] -= p_bd_int / bd_measure

# %%
subtract_pressure_mean(mesh, p_soln)
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


def gather_run_metadata(mesh, velocity_var, pressure_var):
    """Return machine-readable solver and mesh metadata for the current run."""

    v_start, v_end = mesh.dm.getDepthStratum(0)
    c_start, c_end = mesh.dm.getHeightStratum(0)

    local_vertices = int(v_end - v_start)
    local_cells = int(c_end - c_start)
    local_velocity_dofs = int(velocity_var.data.size)
    local_pressure_dofs = int(pressure_var.data.size)

    return {
        "mpi_size": int(uw.mpi.size),
        "mesh_dim": int(mesh.dim),
        "local_vertices": local_vertices,
        "global_vertices": int(MPI.COMM_WORLD.allreduce(local_vertices, op=MPI.SUM)),
        "local_cells": local_cells,
        "global_cells": int(MPI.COMM_WORLD.allreduce(local_cells, op=MPI.SUM)),
        "local_velocity_dofs": local_velocity_dofs,
        "global_velocity_dofs": int(MPI.COMM_WORLD.allreduce(local_velocity_dofs, op=MPI.SUM)),
        "local_pressure_dofs": local_pressure_dofs,
        "global_pressure_dofs": int(MPI.COMM_WORLD.allreduce(local_pressure_dofs, op=MPI.SUM)),
        "snes_converged_reason": snes_reason,
        "ksp_converged_reason": ksp_reason,
        "snes_iterations": snes_iterations,
        "ksp_iterations": ksp_iterations,
    }


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
p_err_l2 = relative_l2_error(mesh, p_err_expr, p_ana_expr)
v_err_l2_lower = relative_l2_error(mesh, v_err_expr, v_ana_expr, boundary=lower)
v_err_l2_upper = relative_l2_error(mesh, v_err_expr, v_ana_expr, boundary=upper)
p_err_l2_lower = relative_l2_error(mesh, p_err_expr, p_ana_expr, boundary=lower)
p_err_l2_upper = relative_l2_error(mesh, p_err_expr, p_ana_expr, boundary=upper)
u_dot_n_l2_lower_abs = absolute_l2_error(mesh, unit_rvec.dot(v_soln.sym), boundary=lower)
u_dot_n_l2_upper_abs = absolute_l2_error(mesh, unit_rvec.dot(v_soln.sym), boundary=upper)
run_metadata = gather_run_metadata(mesh, v_soln, p_soln)
git_sha = current_git_sha(repo_root)
cli_args = " ".join(sys.argv)

if uw.mpi.rank == 0:
    print("=== Relative L2 Errors ===")
    print(f"Velocity (domain): {v_err_l2}")
    print(f"Pressure (domain): {p_err_l2}")
    print(f"Velocity (lower):  {v_err_l2_lower}")
    print(f"Velocity (upper):  {v_err_l2_upper}")
    print(f"Pressure (lower):  {p_err_l2_lower}")
    print(f"Pressure (upper):  {p_err_l2_upper}")
    print(f"u.n abs (lower): {u_dot_n_l2_lower_abs}")
    print(f"u.n abs (upper): {u_dot_n_l2_upper_abs}")

# %% [markdown]
# ### Save Outputs

# %%
if uw.mpi.rank == 0:
    err_h5 = os.path.join(output_dir, metrics_filename)
    if os.path.isfile(err_h5):
        os.remove(err_h5)
    with h5py.File(err_h5, "w") as f_h5:
        f_h5.create_dataset("m", data=params.uw_m)
        f_h5.create_dataset("cellsize", data=params.uw_cellsize)
        f_h5.create_dataset("v_l2_norm", data=v_err_l2)
        f_h5.create_dataset("p_l2_norm", data=p_err_l2)
        f_h5.create_dataset("v_l2_norm_lower", data=v_err_l2_lower)
        f_h5.create_dataset("v_l2_norm_upper", data=v_err_l2_upper)
        f_h5.create_dataset("p_l2_norm_lower", data=p_err_l2_lower)
        f_h5.create_dataset("p_l2_norm_upper", data=p_err_l2_upper)
        f_h5.create_dataset("u_dot_n_l2_norm_lower_abs", data=u_dot_n_l2_lower_abs)
        f_h5.create_dataset("u_dot_n_l2_norm_upper_abs", data=u_dot_n_l2_upper_abs)
        f_h5.create_dataset("git_sha", data=np.bytes_(git_sha))
        f_h5.create_dataset("command", data=np.bytes_(cli_args))
        for key, value in run_metadata.items():
            f_h5.create_dataset(key, data=value)

# %%
mesh.write_timestep(
    'output',
    index=0,
    meshVars=[v_soln, p_soln],
    outputPath=str(output_dir),
)
