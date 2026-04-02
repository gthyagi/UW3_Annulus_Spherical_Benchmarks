# %% [markdown]
# ## Annulus Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark paper](https://gmd.copernicus.org/articles/14/1899/2021/)
#
# ### Authors
# Thyagarajulu Gollapalli ([GitHub](https://github.com/gthyagi)) <br>
# Underworld3 Development Team ([UW3 Repository](https://github.com/underworldcode/underworld3))
#
#
# ##### Case1: Freeslip boundaries and delta function density perturbation
# ##### Case2: Freeslip boundaries and smooth density distribution
# ##### Case3: Noslip boundaries and delta function density perturbation
# ##### Case4: Noslip boundaries and smooth density distribution

# %%
import os
import sys
import h5py
import numpy as np
import sympy as sp
import underworld3 as uw
from underworld3.systems import Stokes
from types import SimpleNamespace

os.environ["UW_TIMING_ENABLE"] = "1"

is_serial = (uw.mpi.size == 1)

# %% [markdown]
# ### Convection Parameters

# %%
params = uw.Params(
    uw_case=uw.Param(
        "case2",
        type=uw.ParamType.STRING,
        description="Benchmark case: case1, case2, case3, case4",
    ),
    uw_n=uw.Param(
        2,
        type=uw.ParamType.INTEGER,
        description="Wave number for density perturbation",
    ),
    uw_k=uw.Param(
        2,
        type=uw.ParamType.INTEGER,
        description="Power exponent for smooth density",
    ),
    uw_radius_inner=uw.Param(
        1.22,
        type=uw.ParamType.FLOAT,
        description="Inner radius",
    ),
    uw_radius_internal=uw.Param(
        2.0,
        type=uw.ParamType.FLOAT,
        description="Internal interface radius",
    ),
    uw_radius_outer=uw.Param(
        2.22,
        type=uw.ParamType.FLOAT,
        description="Outer radius",
    ),
    uw_cellsize=uw.Param(
        "1/32",
        type=uw.ParamType.STRING,
        description="Background mesh cell size",
    ),
    uw_cellsize_internal_boundary_factor=uw.Param(
        1,
        type=uw.ParamType.INTEGER,
        description="Internal-boundary cell-size refinement factor",
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
        1e-5,
        type=uw.ParamType.FLOAT,
        description="Stokes solver tolerance",
    ),
    uw_vel_penalty=uw.Param(
        2.5e8,
        type=uw.ParamType.FLOAT,
        description="Penalty for natural-BC velocity matching",
    ),
    uw_bc_type=uw.Param(
        None,
        type=uw.ParamType.STRING,
        description="Boundary-condition mode: natural or essential",
    ),
    uw_freeslip_type=uw.Param(
        'nitsche',
        type=uw.ParamType.STRING,
        description="Freeslip method: penalty or nitsche",
    ),
)

if any(arg in ("--help", "-h", "-help", "-uw_help") for arg in sys.argv[1:]):
    print(params.cli_help())
    raise SystemExit(0)

# %%
params.uw_cellsize = float(eval(str(params.uw_cellsize), {"__builtins__": {}}, {}))

pressure_is_continuous = params.uw_pcont if params.uw_pdegree > 0 else False
is_p1p0 = params.uw_vdegree == 1 and params.uw_pdegree == 0

if uw.mpi.rank == 0 and params.uw_pdegree == 0 and params.uw_pcont:
    print("Degree-0 pressure uses discontinuous storage; overriding uw_pcont to false.")

# %% [markdown]
# ### Convection Parameters

# %%
case = params.uw_case
n = params.uw_n
k = params.uw_k

# %% [markdown]
# ### Mesh Parameters

# %%
r_i = params.uw_radius_inner
r_int = params.uw_radius_internal
r_o = params.uw_radius_outer
cellsize = params.uw_cellsize
cellsize_int_bd_fac = params.uw_cellsize_internal_boundary_factor


# %% [markdown]
# ### Case Mapping

# %%
freeslip = False
noslip = False
delta_fn = False
smooth = False

if case in ("case1",):
    freeslip = True
    delta_fn = True
    params.uw_bc_type = f'natural_{params.uw_freeslip_type}'
    if params.uw_freeslip_type == "nitsche":
        params.uw_vel_penalty = None
elif case in ("case2",):
    freeslip = True
    smooth = True
    params.uw_bc_type = f'natural_{params.uw_freeslip_type}'
    if params.uw_freeslip_type == "nitsche":
        params.uw_vel_penalty = None
elif case in ("case3",):
    noslip = True
    delta_fn = True
    params.uw_bc_type = "essential"
    params.uw_vel_penalty = None
elif case in ("case4",):
    noslip = True
    smooth = True
    params.uw_bc_type = "essential"
    params.uw_vel_penalty = None
else:
    raise ValueError(f"Unknown case: {case}")

# %% [markdown]
# ### Output Directory

# %%
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
output_root = os.path.join(repo_root, "output", "annulus", "kramer", "latest")

def _case_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.2g}"
    return value


def make_case_id(*, case, **kwargs):
    parts = [case]
    parts += [f"{key}_{_case_value(value)}" for key, value in kwargs.items() if value is not None]
    return "_".join(parts)


case_id = make_case_id(
    case=case,
    inv_lc=int(1 / cellsize),
    n=n,
    k=k,
    vdeg=params.uw_vdegree,
    pdeg=params.uw_pdegree,
    pcont=pressure_is_continuous,
    vel_penalty=params.uw_vel_penalty,
    stokes_tol=params.uw_stokes_tol,
    ncpus=uw.mpi.size,
    bc=params.uw_bc_type,
)

output_dir = os.path.join(output_root, case_id)

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ### Analytical Solution Handles

# %%
def coefficients_cylinder_delta_fs(Rp, Rm, rp, n, g, nu, sign):
    alpha_pm, alpha_mp = [Rp / rp, Rm / rp][:: int(sign)]
    pm = sign

    A = -0.125 * (alpha_mp ** (2 * n - 2) - 1) * g * pm * rp ** (-n + 2) / (
        (alpha_mp ** (2 * n - 2) - alpha_pm ** (2 * n - 2)) * (n - 1) * nu
    )
    B = -0.125 * (alpha_mp ** (2 * n + 2) - 1) * alpha_pm ** (2 * n + 2) * g * pm * rp ** (n + 2) / (
        (alpha_mp ** (2 * n + 2) - alpha_pm ** (2 * n + 2)) * (n + 1) * nu
    )
    C = 0.125 * (alpha_mp ** (2 * n + 2) - 1) * g * pm / (
        (alpha_mp ** (2 * n + 2) - alpha_pm ** (2 * n + 2)) * (n + 1) * nu * rp**n
    )
    D = 0.125 * (alpha_mp ** (2 * n - 2) - 1) * alpha_pm ** (2 * n - 2) * g * pm * rp**n / (
        (alpha_mp ** (2 * n - 2) - alpha_pm ** (2 * n - 2)) * (n - 1) * nu
    )
    return A, B, C, D

# %%
def coefficients_cylinder_delta_ns(Rp, Rm, rp, n, g, nu, sign):
    alpha_p, alpha_m = [Rp / rp, Rm / rp]
    pm, mp = sign, -sign

    A = -0.125 * (
        ((alpha_m**2 - alpha_p**2) * n - (n + 1) * pm + 1 / alpha_m ** (2 * n) - 1 / alpha_p ** (2 * n)) * (n - 1)
        + (alpha_m**2 / alpha_p ** (2 * n) - alpha_p**2 / alpha_m ** (2 * n)) * n
        + (n**2 * (alpha_m / alpha_p) ** (2 * mp) - (alpha_m / alpha_p) ** (2 * n * pm)) * pm
    ) * g * rp ** (-n + 2) / (
        (n**2 * (alpha_m / alpha_p - alpha_p / alpha_m) ** 2 - ((alpha_m / alpha_p) ** n - 1 / (alpha_m / alpha_p) ** n) ** 2)
        * (n - 1)
        * nu
    )
    B = -0.125 * (
        ((alpha_m**2 - alpha_p**2) * n - (n - 1) * pm - alpha_m ** (2 * n) + alpha_p ** (2 * n)) * (n + 1)
        - (alpha_m**2 * alpha_p ** (2 * n) - alpha_m ** (2 * n) * alpha_p**2) * n
        + (n**2 * (alpha_m / alpha_p) ** (2 * mp) - (alpha_m / alpha_p) ** (2 * mp * n)) * pm
    ) * g * rp ** (n + 2) / (
        (n**2 * (alpha_m / alpha_p - alpha_p / alpha_m) ** 2 - ((alpha_m / alpha_p) ** n - 1 / (alpha_m / alpha_p) ** n) ** 2)
        * (n + 1)
        * nu
    )
    C = -0.125 * (
        (n**2 * (alpha_m / alpha_p) ** (2 * pm) - (alpha_m / alpha_p) ** (2 * n * pm)) * mp
        - (mp * (n - 1) + n * (1 / alpha_m**2 - 1 / alpha_p**2) - 1 / alpha_m ** (2 * n) + 1 / alpha_p ** (2 * n)) * (n + 1)
        - n * (1 / (alpha_m ** (2 * n) * alpha_p**2) - 1 / (alpha_m**2 * alpha_p ** (2 * n)))
    ) * g / (
        (n**2 * (alpha_m / alpha_p - alpha_p / alpha_m) ** 2 - ((alpha_m / alpha_p) ** n - 1 / (alpha_m / alpha_p) ** n) ** 2)
        * (n + 1)
        * nu
        * rp**n
    )
    D = -0.125 * (
        (n**2 * (alpha_m / alpha_p) ** (2 * pm) - (alpha_m / alpha_p) ** (2 * mp * n)) * mp
        - (mp * (n + 1) + n * (1 / alpha_m**2 - 1 / alpha_p**2) + alpha_m ** (2 * n) - alpha_p ** (2 * n)) * (n - 1)
        + n * (alpha_m ** (2 * n) / alpha_p**2 - alpha_p ** (2 * n) / alpha_m**2)
    ) * g * rp**n / (
        (n**2 * (alpha_m / alpha_p - alpha_p / alpha_m) ** 2 - ((alpha_m / alpha_p) ** n - 1 / (alpha_m / alpha_p) ** n) ** 2)
        * (n - 1)
        * nu
    )
    return A, B, C, D

# %%
def coefficients_cylinder_smooth_fs(Rp, Rm, k, n, g, nu):
    alpha = Rm / Rp
    A = -0.25 * (alpha**2 - alpha ** (k + n + 3)) * Rp ** (-n + 3) * g / (
        (alpha + alpha**n) * (alpha**n - alpha) * (k + n + 1) * (k - n + 3) * nu
    )
    B = 0.25 * Rp ** (n + 3) * (alpha ** (k + n + 3) - alpha ** (2 * n + 2)) * g / (
        (alpha ** (n + 1) + 1) * (alpha ** (n + 1) - 1) * (k + n + 3) * (k - n + 1) * nu
    )
    C = -0.25 * Rp ** (-n + 1) * (alpha ** (k + n + 3) - 1) * g / (
        (alpha ** (n + 1) + 1) * (alpha ** (n + 1) - 1) * (k + n + 3) * (k - n + 1) * nu
    )
    D = -0.25 * Rp ** (n + 1) * (alpha ** (k + n + 3) - alpha ** (2 * n)) * g / (
        (alpha + alpha**n) * (alpha**n - alpha) * (k + n + 1) * (k - n + 3) * nu
    )
    E = g * n / (((k + 3) ** 2 - n**2) * ((k + 1) ** 2 - n**2) * Rp**k * nu)
    return A, B, C, D, E

# %%
def coefficients_cylinder_smooth_ns(Rp, Rm, k, n, g, nu):
    alpha = Rm / Rp
    denom = ((alpha ** (n + 1) - alpha ** (n - 1)) ** 2 * n**2 - (alpha ** (2 * n) - 1) ** 2) * ((k + 3) ** 2 - n**2) * (
        (k + 1) ** 2 - n**2
    ) * nu
    A = 0.5 * (
        (alpha ** (k + n + 3) + alpha ** (2 * n)) * (k + n + 1) * (n + 1)
        - (alpha ** (k + n + 1) + alpha ** (2 * n + 2)) * (k + n + 3) * n
        - (alpha ** (k + 3 * n + 3) + 1) * (k - n + 1)
    ) * Rp ** (-n + 3) * g * n / denom
    B = -0.5 * (
        (alpha ** (k + 3 * n + 3) + alpha ** (2 * n)) * (k - n + 1) * (n - 1)
        - (alpha ** (k + 3 * n + 1) + alpha ** (2 * n + 2)) * (k - n + 3) * n
        + (alpha ** (k + n + 3) + alpha ** (4 * n)) * (k + n + 1)
    ) * Rp ** (n + 3) * g * n / denom
    C = 0.5 * (
        (alpha ** (k + n + 1) + alpha ** (2 * n)) * (k + n + 3) * (n - 1)
        - (alpha ** (k + n + 3) + alpha ** (2 * n - 2)) * (k + n + 1) * n
        + (alpha ** (k + 3 * n + 1) + 1) * (k - n + 3)
    ) * Rp ** (-n + 1) * g * n / denom
    D = -0.5 * (
        (alpha ** (k + 3 * n + 1) + alpha ** (2 * n)) * (k - n + 3) * (n + 1)
        - (alpha ** (k + 3 * n + 3) + alpha ** (2 * n - 2)) * (k - n + 1) * n
        - (alpha ** (k + n + 1) + alpha ** (4 * n)) * (k + n + 3)
    ) * Rp ** (n + 1) * g * n / denom
    E = g * n / (((k + 3) ** 2 - n**2) * ((k + 1) ** 2 - n**2) * Rp**k * nu)
    return A, B, C, D, E

# %%
def build_delta_solution(Rp, Rm, rp, n, g, nu, sign, no_slip):
    if no_slip:
        ABCD = coefficients_cylinder_delta_ns(Rp, Rm, rp, n, g, nu, sign)
    else:
        ABCD = coefficients_cylinder_delta_fs(Rp, Rm, rp, n, g, nu, sign)
    _, _, C, D = ABCD
    return SimpleNamespace(
        n=n,
        g=g,
        nu=nu,
        ABCD=ABCD,
        G=-4 * nu * C * (n + 1),
        H=-4 * nu * D * (n - 1),
    )

# %%
def build_smooth_solution(Rp, Rm, k, n, g, nu, no_slip):
    if abs(k + 3) == n or abs(k + 1) == n:
        raise NotImplementedError(f"Smooth solution not implemented for k={k}, n={n}")
    if no_slip:
        ABCDE = coefficients_cylinder_smooth_ns(Rp, Rm, k, n, g, nu)
    else:
        ABCDE = coefficients_cylinder_smooth_fs(Rp, Rm, k, n, g, nu)
    _, _, C, D, _ = ABCDE
    F = -g * (k + 1) * Rp ** (-k) / ((k + 1) ** 2 - n**2)
    return SimpleNamespace(
        n=n,
        k=k,
        g=g,
        nu=nu,
        ABCDE=ABCDE,
        G=-4 * nu * C * (n + 1),
        H=-4 * nu * D * (n - 1),
        F=F,
    )

# %%
if freeslip and delta_fn:
    soln_above = build_delta_solution(r_o, r_i, r_int, n, -1.0, 1.0, +1, no_slip=False)
    soln_below = build_delta_solution(r_o, r_i, r_int, n, -1.0, 1.0, -1, no_slip=False)
elif freeslip and smooth:
    soln_above = build_smooth_solution(r_o, r_i, k, n, 1.0, 1.0, no_slip=False)
    soln_below = build_smooth_solution(r_o, r_i, k, n, 1.0, 1.0, no_slip=False)
elif noslip and delta_fn:
    soln_above = build_delta_solution(r_o, r_i, r_int, n, -1.0, 1.0, +1, no_slip=True)
    soln_below = build_delta_solution(r_o, r_i, r_int, n, -1.0, 1.0, -1, no_slip=True)
elif noslip and smooth:
    soln_above = build_smooth_solution(r_o, r_i, k, n, 1.0, 1.0, no_slip=True)
    soln_below = build_smooth_solution(r_o, r_i, k, n, 1.0, 1.0, no_slip=True)

# %%
def analytical_velocity_cartesian_sympy(soln, r_sym, th_sym, rrotN):
    """Build a symbolic Cartesian velocity from analytical cylindrical coefficients."""
    n_sol = int(soln.n)

    if hasattr(soln, "ABCD"):
        A, B, C, D = soln.ABCD
        psi_r = A * r_sym**n_sol + B * r_sym**(-n_sol) + C * r_sym**(n_sol + 2) + D * r_sym**(-n_sol + 2)
        dpsi_rdr = (
            A * n_sol * r_sym**(n_sol - 1)
            + B * (-n_sol) * r_sym**(-n_sol - 1)
            + C * (n_sol + 2) * r_sym**(n_sol + 1)
            + D * (-n_sol + 2) * r_sym**(-n_sol + 1)
        )
    elif hasattr(soln, "ABCDE"):
        A, B, C, D, E = soln.ABCDE
        k_sol = int(soln.k)
        psi_r = (
            A * r_sym**n_sol
            + B * r_sym**(-n_sol)
            + C * r_sym**(n_sol + 2)
            + D * r_sym**(-n_sol + 2)
            + E * r_sym**(k_sol + 3)
        )
        dpsi_rdr = (
            A * n_sol * r_sym**(n_sol - 1)
            + B * (-n_sol) * r_sym**(-n_sol - 1)
            + C * (n_sol + 2) * r_sym**(n_sol + 1)
            + D * (-n_sol + 2) * r_sym**(-n_sol + 1)
            + E * (k_sol + 3) * r_sym**(k_sol + 2)
        )
    else:
        raise TypeError(f"Unsupported analytical solution type: {type(soln)}")

    u_r = -(n_sol * sp.cos(n_sol * th_sym) * psi_r) / r_sym
    u_theta = sp.sin(n_sol * th_sym) * dpsi_rdr
    return rrotN.T * sp.Matrix([u_r, u_theta])

# %%
def analytical_pressure_sympy(soln, r_sym, th_sym):
    """Build a symbolic pressure expression from analytical cylindrical coefficients."""
    n_sol = int(soln.n)
    p_expr = (soln.G * r_sym**n_sol + soln.H * r_sym**(-n_sol)) * sp.cos(n_sol * th_sym)
    if hasattr(soln, "F"):
        p_expr += soln.F * r_sym**(int(soln.k) + 1) * sp.cos(n_sol * th_sym)
    return p_expr

# %% [markdown]
# ### Create Mesh

# %%
if delta_fn:
    mesh = uw.meshing.AnnulusInternalBoundary(
        radiusOuter=r_o,
        radiusInternal=r_int,
        radiusInner=r_i,
        cellSize_Inner=cellsize,
        cellSize_Internal=cellsize / cellsize_int_bd_fac,
        cellSize_Outer=cellsize,
        filename=f"{output_dir}/mesh.msh",
    )
elif smooth:
    mesh = uw.meshing.Annulus(
        radiusOuter=r_o,
        radiusInner=r_i,
        cellSize=cellsize,
        qdegree=max(params.uw_pdegree, params.uw_vdegree),
        degree=1,
        filename=f"{output_dir}/mesh.msh",
        refinement=None,
    )

if is_serial:
    mesh.dm.view()

# %%
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR
v_theta_fn_xy = r_uw * mesh.CoordinateSystem.rRotN.T * sp.Matrix((0, 1))
v_ana_above_sym = analytical_velocity_cartesian_sympy(soln_above, r_uw, th_uw, mesh.CoordinateSystem.rRotN)
v_ana_below_sym = analytical_velocity_cartesian_sympy(soln_below, r_uw, th_uw, mesh.CoordinateSystem.rRotN)
p_ana_above_sym = analytical_pressure_sympy(soln_above, r_uw, th_uw)
p_ana_below_sym = analytical_pressure_sympy(soln_below, r_uw, th_uw)
v_ana_sym = sp.Matrix(
    [[
        sp.Piecewise(
            (v_ana_above_sym[i], r_uw > r_int),
            (v_ana_below_sym[i], True),
        )
        for i in range(v_ana_above_sym.rows)
    ]]
)
p_ana_sym = sp.Piecewise(
    (p_ana_above_sym, r_uw > r_int),
    (p_ana_below_sym, True),
)

# %% [markdown]
# ### Create Mesh Variables

# %%
v_uw = uw.discretisation.MeshVariable(
    varname="V_u",
    mesh=mesh,
    degree=params.uw_vdegree,
    vtype=uw.VarType.VECTOR,
    varsymbol=r"{V_u}",
)
p_uw = uw.discretisation.MeshVariable(
    varname="P_u",
    mesh=mesh,
    degree=params.uw_pdegree,
    vtype=uw.VarType.SCALAR,
    varsymbol=r"{P_u}",
    continuous=pressure_is_continuous,
)

# %% [markdown]
# ### Stokes

# %% [markdown]
# #### Stokes Setup

# %%
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0
stokes.petsc_use_pressure_nullspace = True
if freeslip:
    if hasattr(type(stokes), "petsc_use_nullspace"):
        stokes.petsc_use_nullspace = True
    else:
        stokes.petsc_velocity_nullspace_basis = [v_theta_fn_xy]

# %%
if delta_fn:
    rho = sp.cos(n * th_uw) * sp.exp(-1e5 * ((r_uw - r_int) ** 2))
    stokes.add_natural_bc(-rho * unit_rvec, "Internal")
    stokes.bodyforce = sp.Matrix([0.0, 0.0])
elif smooth:
    rho = ((r_uw / r_o) ** k) * sp.cos(n * th_uw)
    gravity_fn = -1.0 * unit_rvec
    stokes.bodyforce = rho * gravity_fn

# %% [markdown]
# #### Nullspace Handling
#
# The coupled Stokes system is solved with PETSc's constant-pressure nullspace
# enabled. This removes the additive pressure gauge freedom during the solve
# without imposing an artificial pressure Dirichlet condition on the annulus
# boundaries.
#
# We still subtract the domain-average pressure after the solve so the reported
# pressure field has a unique zero-mean gauge for benchmark comparisons.
#
# For the free-slip annulus cases, the rigid-body rotation is an exact null
# mode. We enable the PETSc velocity nullspace automatically in those cases.
# On newer UW branches this goes through `stokes.petsc_use_nullspace`; on older
# branches we fall back to the explicit annulus rotation basis.
#
# %% [markdown]
# #### Tolerance And BC Type
#
# `stokes.tolerance` does not affect the two Kramer case families equally.
#
# - free-slip cases use a penalty on the normal velocity component and are more
#   tolerance-sensitive.
# - no-slip cases use strong zero-velocity Dirichlet conditions and are less
#   sensitive to a looser tolerance.
#
# Practical choices for this script:
# - free-slip: `1e-8`
# - no-slip: `1e-5`
#
# In the current UW Stokes implementation, setting `stokes.tolerance` also sets
# the inner fieldsplit tolerances:
#
# - `fieldsplit_pressure_ksp_rtol = 0.1 * tolerance`
# - `fieldsplit_velocity_ksp_rtol = 0.033 * tolerance`
#
# This is important because `stokes.tolerance` is not only the outer Stokes
# solve target. It also controls how hard PETSc works inside the Schur-complement
# preconditioner. Very small tolerances can therefore increase runtime sharply,
# while too-loose tolerances usually degrade the weak-BC `natural` branch faster
# than the strongly enforced `essential` branch.
#
# %% [markdown]
# #### Boundary Conditions

# %%
inner = mesh.boundaries.Lower.name
outer = mesh.boundaries.Upper.name

#
# Legacy UW implementation kept for reference only. It imposed full analytical
# boundary velocity matching instead of the physical Kramer benchmark BCs.
#
# if freeslip:
#     if params.uw_bc_type == "natural":
#         v_diff = v_uw.sym - v_ana.sym
#         stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Upper.name)
#         stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Lower.name)
#     elif params.uw_bc_type == "essential":
#         stokes.add_essential_bc(v_ana_above_sym, mesh.boundaries.Upper.name)
#         stokes.add_essential_bc(v_ana_below_sym, mesh.boundaries.Lower.name)
# elif noslip:
#     if params.uw_bc_type == "natural":
#         v_diff = v_uw.sym - v_ana.sym
#         stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Upper.name)
#         stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Lower.name)
#     elif params.uw_bc_type == "essential":
#         stokes.add_essential_bc(sp.Matrix([0.0, 0.0]), mesh.boundaries.Upper.name)
#         stokes.add_essential_bc(sp.Matrix([0.0, 0.0]), mesh.boundaries.Lower.name)

if freeslip:
    if params.uw_freeslip_type == "penalty":
        # UW implements annulus free-slip through a penalty on the normal velocity
        # component. This matches the authors' physical condition u.n = 0 while
        # leaving tangential motion free on the shell boundaries.
        Gamma_N = mesh.CoordinateSystem.unit_e_0
        # Gamma_N = mesh.Gamma
        stokes.add_natural_bc(params.uw_vel_penalty * Gamma_N.dot(v_uw.sym) * Gamma_N, outer)
        stokes.add_natural_bc(params.uw_vel_penalty * Gamma_N.dot(v_uw.sym) * Gamma_N, inner)
    elif params.uw_freeslip_type == "nitsche":
        # Nitsche's method is more robust than the penalty method for free-slip
        # conditions, and it does not require tuning a penalty parameter. It
        # imposes the same physical condition u.n = 0 while leaving tangential
        # motion free on the shell boundaries.
        outer_normal = mesh.CoordinateSystem.unit_e_0
        inner_normal = -mesh.CoordinateSystem.unit_e_0
        stokes.add_nitsche_bc(outer, normal=outer_normal, gamma=10)
        stokes.add_nitsche_bc(inner, normal=inner_normal, gamma=10)
elif noslip:
    stokes.add_essential_bc(sp.Matrix([0.0, 0.0]), outer)
    stokes.add_essential_bc(sp.Matrix([0.0, 0.0]), inner)
else:
    raise ValueError(f"Unsupported case flags: freeslip={freeslip}, noslip={noslip}")

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
stokes.solve(verbose=False)
uw.timing.stop()
uw.timing.print_table(filename=f"{output_dir}/stokes_timing.txt")

if uw.mpi.rank == 0:
    print(stokes.snes.getConvergedReason())
    print(stokes.snes.ksp.getConvergedReason())

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
    p_mean = p_int / volume

    pressure_var.data[:, 0] -= p_mean


def subtract_rigid_rotation(mesh, velocity_var, rotation_mode):
    """
    Remove the rigid-body rotation component from the numerical velocity field.

    This matches the benchmark-paper postprocessing used for free-slip cases.
    """
    mode_int = uw.maths.Integral(mesh, rotation_mode.dot(velocity_var.sym)).evaluate()
    mode_norm = uw.maths.Integral(mesh, rotation_mode.dot(rotation_mode)).evaluate()
    coeff = mode_int / mode_norm

    dv = uw.function.evaluate(coeff * rotation_mode, velocity_var.coords)
    velocity_var.data[...] -= dv.reshape(velocity_var.data.shape)


subtract_pressure_mean(mesh, p_uw)

if freeslip:
    subtract_rigid_rotation(mesh, v_uw, v_theta_fn_xy)


# %% [markdown]
# ### Errors and L2 Norm

# %%
v_err_sym = v_uw.sym - v_ana_sym
p_err_sym = p_uw.sym[0] - p_ana_sym


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


# %%
v_err_l2 = relative_l2_error(mesh, v_err_sym, v_ana_sym)
p_err_l2 = relative_l2_error(mesh, p_err_sym, p_ana_sym)
v_err_l2_inner = relative_l2_error(mesh, v_err_sym, v_ana_sym, boundary=inner)
v_err_l2_outer = relative_l2_error(mesh, v_err_sym, v_ana_sym, boundary=outer)
p_err_l2_inner = relative_l2_error(mesh, p_err_sym, p_ana_sym, boundary=inner)
p_err_l2_outer = relative_l2_error(mesh, p_err_sym, p_ana_sym, boundary=outer)

if uw.mpi.rank == 0:
    print("=== Relative L2 Errors ===")
    print(f"Velocity (domain): {v_err_l2}")
    print(f"Pressure (domain): {p_err_l2}")
    print(f"Velocity (inner):  {v_err_l2_inner}")
    print(f"Velocity (outer):  {v_err_l2_outer}")
    print(f"Pressure (inner):  {p_err_l2_inner}")
    print(f"Pressure (outer):  {p_err_l2_outer}")

# %% [markdown]
# ### Save Outputs

# %%
if uw.mpi.rank == 0:
    err_h5 = os.path.join(output_dir, "error_norm.h5")
    if os.path.isfile(err_h5):
        os.remove(err_h5)
    with h5py.File(err_h5, "w") as f_h5:
        f_h5.create_dataset("case", data=np.bytes_(case))
        f_h5.create_dataset("n", data=n)
        f_h5.create_dataset("k", data=k)
        f_h5.create_dataset("cellsize", data=cellsize)
        f_h5.create_dataset("v_l2_norm", data=v_err_l2)
        f_h5.create_dataset("p_l2_norm", data=p_err_l2)
        f_h5.create_dataset("v_l2_norm_inner", data=v_err_l2_inner)
        f_h5.create_dataset("v_l2_norm_outer", data=v_err_l2_outer)
        f_h5.create_dataset("p_l2_norm_inner", data=p_err_l2_inner)
        f_h5.create_dataset("p_l2_norm_outer", data=p_err_l2_outer)

# %%
mesh.write_timestep(
    'output',
    index=0,
    meshVars=[v_uw, p_uw],
    outputPath=str(output_dir),
)

# %%
