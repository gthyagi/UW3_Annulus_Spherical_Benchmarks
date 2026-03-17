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
        3,
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
        1.0 / 32.0,
        type=uw.ParamType.FLOAT,
        description="Background mesh cell size",
    ),
    uw_cellsize_internal_boundary_factor=uw.Param(
        2,
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
        1e-10,
        type=uw.ParamType.FLOAT,
        description="Stokes solver tolerance",
    ),
    uw_vel_penalty=uw.Param(
        2.5e8,
        type=uw.ParamType.FLOAT,
        description="Penalty for natural-BC velocity matching",
    ),
    uw_bc_type=uw.Param(
        "natural",
        type=uw.ParamType.STRING,
        description="Boundary-condition mode: natural or essential",
    ),
    uw_ana_normal=uw.Param(
        False,
        type=uw.ParamType.BOOLEAN,
        description="Use analytical radial normal (legacy option)",
    ),
    uw_petsc_normal=uw.Param(
        True,
        type=uw.ParamType.BOOLEAN,
        description="Use PETSc Gamma normal (legacy option)",
    ),
)

if any(arg in ("--help", "-h", "-help", "-uw_help") for arg in sys.argv[1:]):
    print(params.cli_help())
    raise SystemExit(0)

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
# ### Boundary Condition Parameters

# %%
# which normals to use
ana_normal = params.uw_ana_normal  # mesh radial unit vectors
petsc_normal = params.uw_petsc_normal  # PETSc Gamma

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
elif case in ("case2",):
    freeslip = True
    smooth = True
elif case in ("case3",):
    noslip = True
    delta_fn = True
elif case in ("case4",):
    noslip = True
    smooth = True
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
    pcont=params.uw_pcont,
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
    continuous=params.uw_pcont,
)

v_ana = uw.discretisation.MeshVariable(
    varname="V_a",
    mesh=mesh,
    degree=params.uw_vdegree,
    vtype=uw.VarType.VECTOR,
    varsymbol=r"{V_a}",
)
p_ana = uw.discretisation.MeshVariable(
    varname="P_a",
    mesh=mesh,
    degree=params.uw_pdegree,
    vtype=uw.VarType.SCALAR,
    varsymbol=r"{P_a}",
    continuous=params.uw_pcont,
)
rho_ana = uw.discretisation.MeshVariable(
    varname="RHO_a",
    mesh=mesh,
    degree=params.uw_pdegree,
    vtype=uw.VarType.SCALAR,
    varsymbol=r"{RHO_a}",
    continuous=True,
)

v_err = uw.discretisation.MeshVariable(
    varname="V_e",
    mesh=mesh,
    degree=params.uw_vdegree,
    vtype=uw.VarType.VECTOR,
    varsymbol=r"{V_e}",
)
p_err = uw.discretisation.MeshVariable(
    varname="P_e",
    mesh=mesh,
    degree=params.uw_pdegree,
    vtype=uw.VarType.SCALAR,
    varsymbol=r"{P_e}",
    continuous=params.uw_pcont,
)


# %% [markdown]
# ### Analytical Field Fill

# %%
def analytical_values(var, r_int, fn_above, fn_below):
    coords = np.asarray(var.coords)
    mask_above = np.hypot(coords[:, 0], coords[:, 1]) > r_int

    ncomp = var.data.shape[1]
    values = np.empty_like(var.data)

    for mask, fn in ((mask_above, fn_above), (~mask_above, fn_below)):
        if np.any(mask):
            values[mask, :] = np.asarray(uw.function.evaluate(fn, coords[mask])).reshape(-1, ncomp)

    return values


# %%
v_ana_values = analytical_values(
    v_ana,
    r_int,
    v_ana_above_sym,
    v_ana_below_sym,
)
p_ana_values = analytical_values(
    p_ana,
    r_int,
    p_ana_above_sym,
    p_ana_below_sym,
)

with uw.synchronised_array_update():
    v_ana.data[...] = v_ana_values
    p_ana.data[...] = p_ana_values

# %% [markdown]
# ### Stokes

# %% [markdown]
# #### Stokes Setup

# %%
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# %%
if delta_fn:
    rho = sp.cos(n * th_uw) * sp.exp(-1e5 * ((r_uw - r_int) ** 2))
    stokes.add_natural_bc(-rho * unit_rvec, "Internal")
    stokes.bodyforce = sp.Matrix([0.0, 0.0])
elif smooth:
    rho = ((r_uw / r_o) ** k) * sp.cos(n * th_uw)
    gravity_fn = -1.0 * unit_rvec
    stokes.bodyforce = rho * gravity_fn

# %%
# Restore analytical density values into mesh variable
rho_ana.data[:] = np.asarray(uw.function.evaluate(rho, rho_ana.coords)).reshape(-1, 1)

# %% [markdown]
# #### Boundary Conditions

# %%
if freeslip:
    if ana_normal:
        Gamma = mesh.CoordinateSystem.unit_e_0
    elif petsc_normal:
        Gamma = mesh.Gamma

    if params.uw_bc_type == "natural":
        v_diff = v_uw.sym - v_ana.sym
        stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Upper.name)
        stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Lower.name)
    elif params.uw_bc_type == "essential":
        stokes.add_essential_bc(v_ana_above_sym, mesh.boundaries.Upper.name)
        stokes.add_essential_bc(v_ana_below_sym, mesh.boundaries.Lower.name)
elif noslip:
    if params.uw_bc_type == "natural":
        v_diff = v_uw.sym - v_ana.sym
        stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Upper.name)
        stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Lower.name)
    elif params.uw_bc_type == "essential":
        stokes.add_essential_bc(sp.Matrix([0.0, 0.0]), mesh.boundaries.Upper.name)
        stokes.add_essential_bc(sp.Matrix([0.0, 0.0]), mesh.boundaries.Lower.name)

# %% [markdown]
# #### Solver Settings

# %%
stokes.tolerance = params.uw_stokes_tol
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["ksp_monitor_true_residual"] = None
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["snes_type"] = "newtonls"
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
stokes.solve(verbose=False, debug=False)
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

    Parameters
    ----------
    mesh : uw.discretisation.Mesh
        Mesh used to evaluate the projection integrals.
    velocity_var : uw.discretisation.MeshVariable
        Vector velocity field to correct.
    rotation_mode : sympy.Matrix
        Rigid-body rotation null mode to project out, here `r * e_theta` in 2-D.
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
def compute_error(mesh_var, var_num, r_int, fn_above, fn_below):
    ana_values = analytical_values(mesh_var, r_int, fn_above, fn_below)
    return np.asarray(var_num.data) - ana_values

v_err_values = compute_error(v_err, v_uw, r_int, v_ana_above_sym, v_ana_below_sym)
p_err_values = compute_error(p_err, p_uw, r_int, p_ana_above_sym, p_ana_below_sym)

with uw.synchronised_array_update():
    v_err.data[...] = v_err_values
    p_err.data[...] = p_err_values


# %%
def relative_l2_error(mesh, err_var, ana_var):
    """Relative L2 error for scalar or vector expressions."""
    err_I = uw.maths.Integral(mesh, err_var.sym.dot(err_var.sym))
    ana_I = uw.maths.Integral(mesh, ana_var.sym.dot(ana_var.sym))
    return np.sqrt(err_I.evaluate()) / np.sqrt(ana_I.evaluate())


# %%
v_err_l2 = relative_l2_error(mesh, v_err, v_ana)
p_err_l2 = relative_l2_error(mesh, p_err, p_ana)

if uw.mpi.rank == 0:
    print("Relative velocity L2 error:", v_err_l2)
    print("Relative pressure L2 error:", p_err_l2)

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

# %%
mesh.write_timestep(
    'output',
    index=0,
    meshVars=[v_uw, p_uw, v_ana, p_ana, rho_ana, v_err, p_err],
    outputPath=str(output_dir),
)

# %%
