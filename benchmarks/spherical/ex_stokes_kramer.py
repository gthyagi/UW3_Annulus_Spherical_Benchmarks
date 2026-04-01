# %% [markdown]
# ## Spherical Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark paper](https://gmd.copernicus.org/articles/14/1899/2021/)
#
# ### Authors
# Thyagarajulu Gollapalli ([GitHub](https://github.com/gthyagi)) <br>
# Underworld3 Development Team ([UW3 Repository](https://github.com/underworldcode/underworld3))
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
from types import SimpleNamespace
from underworld3.systems import Stokes

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

is_serial = uw.mpi.size == 1

# %% [markdown]
# ### Convection Parameters

# %%
params = uw.Params(
    uw_case=uw.Param(
        "case2",
        type=uw.ParamType.STRING,
        description="Benchmark case: case1, case2, case3, case4",
    ),
    uw_l=uw.Param(
        2,
        type=uw.ParamType.INTEGER,
        description="Spherical harmonic degree",
    ),
    uw_m=uw.Param(
        1,
        type=uw.ParamType.INTEGER,
        description="Spherical harmonic order",
    ),
    uw_k=uw.Param(
        3,
        type=uw.ParamType.INTEGER,
        description="Power exponent for smooth density forcing",
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
        "1/8",
        type=uw.ParamType.STRING,
        description="Background mesh cell size",
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
        1e8,
        type=uw.ParamType.FLOAT,
        description="Penalty for natural-BC velocity matching",
    ),
    uw_bc_type=uw.Param(
        None,
        type=uw.ParamType.STRING,
        description="Boundary-condition mode: natural or essential",
    ),
    uw_pressure_pc_type=uw.Param(
        "mg",
        type=uw.ParamType.STRING,
        description="Pressure-block preconditioner type for debug runs",
    ),
    uw_pressure_pc_mg_type=uw.Param(
        "multiplicative",
        type=uw.ParamType.STRING,
        description="Pressure-block MG type when uw_pressure_pc_type=mg",
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
# ### Case Mapping

# %%
freeslip = False
noslip = False
delta_fn = False
smooth = False

if params.uw_case in ("case1",):
    freeslip = True
    delta_fn = True
elif params.uw_case in ("case2",):
    freeslip = True
    smooth = True
elif params.uw_case in ("case3",):
    noslip = True
    delta_fn = True
elif params.uw_case in ("case4",):
    noslip = True
    smooth = True
else:
    raise ValueError(f"Unknown case: {params.uw_case}")

# hard set the stokes tolerance based on the case bc type.
params.uw_stokes_tol = 1.0e-5 if freeslip else 1.0e-5

# %% [markdown]
# ### Output Directory

# %%
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


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
output_root = os.path.join(repo_root, "output", "spherical", "kramer", "latest")

case_id = make_case_id(
    case=params.uw_case,
    inv_lc=int(1 / params.uw_cellsize),
    l=params.uw_l,
    m=params.uw_m,
    k=params.uw_k,
    vdeg=params.uw_vdegree,
    pdeg=params.uw_pdegree,
    pcont=params.uw_pcont,
    vel_penalty=params.uw_vel_penalty,
    stokes_tol=params.uw_stokes_tol,
    ppc=params.uw_pressure_pc_type if params.uw_pressure_pc_type != "mg" else None,
    pmg=params.uw_pressure_pc_mg_type if params.uw_pressure_pc_mg_type != "multiplicative" else None,
    ncpus=uw.mpi.size,
    bc=params.uw_bc_type,
)

output_dir = os.path.join(output_root, case_id)

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ### Analytical Solution Helpers

# %%
def coefficients_sphere_delta_fs(Rp, Rm, rp, l, g, nu, sign):
    alpha_pm, alpha_mp = [Rp / rp, Rm / rp][:: int(sign)]
    pm = sign

    A = -0.5 * (alpha_mp ** (2 * l - 1) - 1) * g * pm * rp ** (-l + 2) / (
        (alpha_mp ** (2 * l - 1) - alpha_pm ** (2 * l - 1)) * (2 * l + 1) * (2 * l - 1) * nu
    )
    B = -0.5 * (alpha_mp ** (-2 * l - 3) - 1) * g * pm * rp ** (l + 3) / (
        (alpha_mp ** (-2 * l - 3) - alpha_pm ** (-2 * l - 3)) * (2 * l + 3) * (2 * l + 1) * nu
    )
    C = 0.5 * (alpha_mp ** (2 * l + 3) - 1) * g * pm / (
        (alpha_mp ** (2 * l + 3) - alpha_pm ** (2 * l + 3)) * (2 * l + 3) * (2 * l + 1) * nu * rp**l
    )
    D = 0.5 * (alpha_mp ** (-2 * l + 1) - 1) * g * pm * rp ** (l + 1) / (
        (alpha_mp ** (-2 * l + 1) - alpha_pm ** (-2 * l + 1)) * (2 * l + 1) * (2 * l - 1) * nu
    )
    return A, B, C, D

# %%
def coefficients_sphere_delta_ns(Rp, Rm, rp, l, g, nu, sign):
    alpha_p, alpha_m = [Rp / rp, Rm / rp]
    alpha_pm, alpha_mp = [Rp / rp, Rm / rp][:: int(sign)]
    pm, mp = sign, -sign

    denom = (
        (2 * l + 1) ** 2 * (alpha_m**2 / alpha_p**2 + alpha_p**2 / alpha_m**2)
        - 2 * (2 * l + 3) * (2 * l - 1)
        - 4 * (alpha_m / alpha_p) ** (2 * l + 1)
        - 4 * (alpha_m / alpha_p) ** (-2 * l - 1)
    ) * nu

    A = -0.5 * (
        alpha_m**2
        - alpha_p**2
        - (2 * l + 1) * mp * (alpha_m / alpha_p) ** (2 * mp) / (2 * l - 1)
        - (2 * l + 3) * pm / (2 * l + 1)
        + 2 * (alpha_m ** (-2 * l - 1) - alpha_p ** (-2 * l - 1)) / (2 * l + 1)
        + 2 * (alpha_m**2 * alpha_p ** (-2 * l - 1) - alpha_m ** (-2 * l - 1) * alpha_p**2) / (2 * l - 1)
        - 4 * pm * (alpha_m / alpha_p) ** ((2 * l + 1) * pm) / ((2 * l + 1) * (2 * l - 1))
    ) * g * rp ** (-l + 2) / denom
    B = -0.5 * (
        alpha_m**2
        - alpha_p**2
        - (2 * l + 1) * mp * (alpha_m / alpha_p) ** (2 * mp) / (2 * l + 3)
        - (2 * l - 1) * pm / (2 * l + 1)
        - 2 * (alpha_m**2 * alpha_p ** (2 * l + 1) - alpha_m ** (2 * l + 1) * alpha_p**2) / (2 * l + 3)
        - 2 * (alpha_m ** (2 * l + 1) - alpha_p ** (2 * l + 1)) / (2 * l + 1)
        - 4 * pm * (alpha_m / alpha_p) ** ((2 * l + 1) * mp) / ((2 * l + 3) * (2 * l + 1))
    ) * g * rp ** (l + 3) / denom
    C = 0.5 * (
        (2 * l + 1) * pm * (alpha_m / alpha_p) ** (2 * pm) / (2 * l + 3)
        + (2 * l - 1) * mp / (2 * l + 1)
        - 2 * (alpha_m ** (-2 * l - 1) - alpha_p ** (-2 * l - 1)) / (2 * l + 1)
        + 4 * mp * (alpha_m / alpha_p) ** ((2 * l + 1) * pm) / ((2 * l + 3) * (2 * l + 1))
        + 2 * (alpha_m ** (-2 * l - 1) / alpha_p**2 - alpha_p ** (-2 * l - 1) / alpha_m**2) / (2 * l + 3)
        + 1 / alpha_m**2
        - 1 / alpha_p**2
    ) * g / (denom * rp**l)
    D = 0.5 * (
        (2 * l + 1) * pm * (alpha_m / alpha_p) ** (2 * pm) / (2 * l - 1)
        + (2 * l + 3) * mp / (2 * l + 1)
        + 2 * (alpha_m ** (2 * l + 1) - alpha_p ** (2 * l + 1)) / (2 * l + 1)
        + 4 * mp * (alpha_m / alpha_p) ** ((2 * l + 1) * mp) / ((2 * l + 1) * (2 * l - 1))
        - 2 * (alpha_m ** (2 * l + 1) / alpha_p**2 - alpha_p ** (2 * l + 1) / alpha_m**2) / (2 * l - 1)
        + 1 / alpha_m**2
        - 1 / alpha_p**2
    ) * g * rp ** (l + 1) / denom
    return A, B, C, D

# %%
def coefficients_sphere_smooth_fs(Rp, Rm, k, l, g, nu):
    alpha = Rm / Rp
    A = 0.5 * Rp ** (-l + 3) * (alpha ** (k + 3) - alpha ** (-l + 1)) * g / (
        (alpha**l - alpha ** (-l + 1)) * (k + l + 2) * (k - l + 3) * (2 * l + 1) * nu
    )
    B = 0.5 * Rp ** (l + 4) * (alpha ** (k + 4) - alpha ** (l + 3)) * g / (
        (alpha ** (l + 3) - 1 / alpha**l) * (k + l + 4) * (k - l + 1) * (2 * l + 1) * nu
    )
    C = -0.5 * Rp ** (-l + 1) * (alpha ** (k + 4) - 1 / alpha**l) * g / (
        (alpha ** (l + 3) - 1 / alpha**l) * (k + l + 4) * (k - l + 1) * (2 * l + 1) * nu
    )
    D = -0.5 * Rp ** (l + 2) * (alpha ** (k + 3) - alpha**l) * g / (
        (alpha**l - alpha ** (-l + 1)) * (k + l + 2) * (k - l + 3) * (2 * l + 1) * nu
    )
    E = g / (Rp**k * (k + l + 4) * (k + l + 2) * (k - l + 3) * (k - l + 1) * nu)
    return A, B, C, D, E

# %%
def coefficients_sphere_smooth_ns(Rp, Rm, k, l, g, nu):
    alpha = Rm / Rp
    gamma = (
        (alpha ** (l + 1) + alpha ** (l - 3)) * (2 * l + 1) ** 2
        - 2 * alpha ** (l - 1) * (2 * l + 3) * (2 * l - 1)
        - 4 * alpha ** (3 * l)
        - 4 * alpha ** (-l - 2)
    ) * (k + l + 4) * (k + l + 2) * (k - l + 3) * (k - l + 1)

    A = (
        (alpha ** (k + 2) + alpha ** (l - 1)) * (k + l + 2) * (2 * l + 3)
        - (alpha**k + alpha ** (l + 1)) * (k + l + 4) * (2 * l + 1)
        - 2 * (alpha ** (k + 2 * l + 3) + alpha ** (-l - 2)) * (k - l + 1)
    ) * Rp ** (-l + 3) * g / (gamma * nu)
    B = (
        (alpha ** (k + 2 * l + 1) + alpha ** (l + 1)) * (k - l + 3) * (2 * l + 1)
        - (alpha ** (k + 2 * l + 3) + alpha ** (l - 1)) * (k - l + 1) * (2 * l - 1)
        - 2 * (alpha ** (k + 2) + alpha ** (3 * l)) * (k + l + 2)
    ) * Rp ** (l + 4) * g / (gamma * nu)
    C = -(
        (alpha ** (k + 2) + alpha ** (l - 3)) * (k + l + 2) * (2 * l + 1)
        - (alpha**k + alpha ** (l - 1)) * (k + l + 4) * (2 * l - 1)
        - 2 * (alpha ** (k + 2 * l + 1) + alpha ** (-l - 2)) * (k - l + 3)
    ) * Rp ** (-l + 1) * g / (gamma * nu)
    D = -(
        (alpha ** (k + 2 * l + 1) + alpha ** (l - 1)) * (k - l + 3) * (2 * l + 3)
        - (alpha ** (k + 2 * l + 3) + alpha ** (l - 3)) * (k - l + 1) * (2 * l + 1)
        - 2 * (alpha**k + alpha ** (3 * l)) * (k + l + 4)
    ) * Rp ** (l + 2) * g / (gamma * nu)
    E = g / (Rp**k * (k + l + 4) * (k + l + 2) * (k - l + 3) * (k - l + 1) * nu)
    return A, B, C, D, E

# %%
def build_delta_solution(Rp, Rm, rp, l, m, g, nu, sign, no_slip):
    coeffs = coefficients_sphere_delta_ns(Rp, Rm, rp, l, g, nu, sign) if no_slip else coefficients_sphere_delta_fs(Rp, Rm, rp, l, g, nu, sign)
    _, _, C, D = coeffs
    return SimpleNamespace(
        l=l,
        m=m,
        g=g,
        nu=nu,
        ABCD=coeffs,
        G=-2 * nu * (l + 1) * (2 * l + 3) * C,
        H=-2 * nu * l * (2 * l - 1) * D,
    )

# %%
def build_smooth_solution(Rp, Rm, k, l, m, g, nu, no_slip):
    if (k + 1) * (k + 2) == l * (l + 1) or (k + 3) * (k + 4) == l * (l + 1):
        raise NotImplementedError(f"Smooth solution not implemented for k={k}, l={l}")
    coeffs = coefficients_sphere_smooth_ns(Rp, Rm, k, l, g, nu) if no_slip else coefficients_sphere_smooth_fs(Rp, Rm, k, l, g, nu)
    _, _, C, D, _ = coeffs
    return SimpleNamespace(
        l=l,
        m=m,
        k=k,
        g=g,
        nu=nu,
        ABCDE=coeffs,
        G=-2 * nu * (l + 1) * (2 * l + 3) * C,
        H=-2 * nu * l * (2 * l - 1) * D,
        K=-g * (k + 2) / ((k + 1) * (k + 2) - l * (l + 1)) / Rp**k,
    )

# %%
if freeslip and delta_fn:
    soln_above = build_delta_solution(params.uw_radius_outer, params.uw_radius_inner, params.uw_radius_internal, params.uw_l, params.uw_m, -1.0, 1.0, +1, no_slip=False)
    soln_below = build_delta_solution(params.uw_radius_outer, params.uw_radius_inner, params.uw_radius_internal, params.uw_l, params.uw_m, -1.0, 1.0, -1, no_slip=False)
elif freeslip and smooth:
    soln_above = build_smooth_solution(params.uw_radius_outer, params.uw_radius_inner, params.uw_k, params.uw_l, params.uw_m, 1.0, 1.0, no_slip=False)
    soln_below = build_smooth_solution(params.uw_radius_outer, params.uw_radius_inner, params.uw_k, params.uw_l, params.uw_m, 1.0, 1.0, no_slip=False)
elif noslip and delta_fn:
    soln_above = build_delta_solution(params.uw_radius_outer, params.uw_radius_inner, params.uw_radius_internal, params.uw_l, params.uw_m, -1.0, 1.0, +1, no_slip=True)
    soln_below = build_delta_solution(params.uw_radius_outer, params.uw_radius_inner, params.uw_radius_internal, params.uw_l, params.uw_m, -1.0, 1.0, -1, no_slip=True)
elif noslip and smooth:
    soln_above = build_smooth_solution(params.uw_radius_outer, params.uw_radius_inner, params.uw_k, params.uw_l, params.uw_m, 1.0, 1.0, no_slip=True)
    soln_below = build_smooth_solution(params.uw_radius_outer, params.uw_radius_inner, params.uw_k, params.uw_l, params.uw_m, 1.0, 1.0, no_slip=True)

# %% [markdown]
# ### Create Mesh

# %%
if delta_fn:
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=params.uw_radius_outer,
        radiusInternal=params.uw_radius_internal,
        radiusInner=params.uw_radius_inner,
        cellSize=params.uw_cellsize,
        qdegree=max(params.uw_pdegree, params.uw_vdegree),
        degree=1,
        filename=f"{output_dir}/mesh.msh",
        refinement=None,
    )
else:
    mesh = uw.meshing.SphericalShell(
        radiusOuter=params.uw_radius_outer,
        radiusInner=params.uw_radius_inner,
        cellSize=params.uw_cellsize,
        qdegree=max(params.uw_pdegree, params.uw_vdegree),
        degree=1,
        filename=f"{output_dir}/mesh.msh",
        refinement=None,
    )

if is_serial:
    mesh.dm.view()

# %%
unit_rvec = mesh.CoordinateSystem.unit_e_0
x_uw, y_uw, z_uw = mesh.X
r_uw = mesh.CoordinateSystem.xR[0]
th_uw = mesh.CoordinateSystem.xR[1]
phi_raw = mesh.CoordinateSystem.xR[2]
phi_uw = sp.Piecewise(
    (2 * sp.pi + phi_raw, phi_raw < 0),
    (phi_raw, True),
)
velocity_nullspace_basis = [
    sp.Matrix([0, -z_uw, y_uw]),
    sp.Matrix([z_uw, 0, -x_uw]),
    sp.Matrix([-y_uw, x_uw, 0]),
]
# Match the legacy Kramer postprocessing mode for free-slip shells.
v_theta_phi_fn_xyz = sp.Matrix(((0, 1, 1), (-1, 0, 1), (-1, -1, 0))) * mesh.CoordinateSystem.N.T
y_lm_sym = (
    sp.sqrt(
        (2 * params.uw_l + 1)
        / (4 * sp.pi)
        * sp.factorial(params.uw_l - params.uw_m)
        / sp.factorial(params.uw_l + params.uw_m)
    )
    * sp.cos(params.uw_m * phi_uw)
    * sp.assoc_legendre(params.uw_l, params.uw_m, sp.cos(th_uw))
)

# %%
def analytical_velocity_cartesian_sympy(soln, r_sym, y_sym):
    if hasattr(soln, "ABCD"):
        A, B, C, D = soln.ABCD
        P_l = A * r_sym**soln.l + B * r_sym ** (-soln.l - 1) + C * r_sym ** (soln.l + 2) + D * r_sym ** (-soln.l + 1)
    elif hasattr(soln, "ABCDE"):
        A, B, C, D, E = soln.ABCDE
        P_l = A * r_sym**soln.l + B * r_sym ** (-soln.l - 1) + C * r_sym ** (soln.l + 2) + D * r_sym ** (-soln.l + 1) + E * r_sym ** (soln.k + 3)
    else:
        raise TypeError(f"Unsupported analytical solution type: {type(soln)}")

    dPldr = sp.diff(P_l, r_sym)
    prefactor = -(P_l / r_sym + dPldr)
    u_r = -soln.l * (soln.l + 1) * P_l * y_sym / r_sym
    u_theta = prefactor * sp.diff(y_sym, th_uw)
    u_phi = prefactor * sp.diff(y_sym, phi_raw) / sp.sin(th_uw)
    return mesh.CoordinateSystem.rRotN.T * sp.Matrix([u_r, u_theta, u_phi])

# %%
def analytical_pressure_sympy(soln, r_sym, y_sym):
    p_sym = (soln.G * r_sym**soln.l + soln.H * r_sym ** (-soln.l - 1)) * y_sym
    if hasattr(soln, "K"):
        p_sym += soln.K * r_sym ** (soln.k + 1) * y_sym
    return p_sym

# %%
v_ana_above_sym = analytical_velocity_cartesian_sympy(soln_above, r_uw, y_lm_sym)
v_ana_below_sym = analytical_velocity_cartesian_sympy(soln_below, r_uw, y_lm_sym)
p_ana_above_sym = analytical_pressure_sympy(soln_above, r_uw, y_lm_sym)
p_ana_below_sym = analytical_pressure_sympy(soln_below, r_uw, y_lm_sym)
v_ana_sym = sp.Matrix(
    [[
        sp.Piecewise(
            (v_ana_above_sym[i], r_uw > params.uw_radius_internal),
            (v_ana_below_sym[i], True),
        )
        for i in range(v_ana_above_sym.rows)
    ]]
)
p_ana_sym = sp.Piecewise(
    (p_ana_above_sym, r_uw > params.uw_radius_internal),
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
# ### Field Materialisation

# %%
def fill_from_expression_chunked(var, fn, chunk_size=20000):
    coords = np.asarray(var.coords)

    with uw.synchronised_array_update():
        for start in range(0, coords.shape[0], chunk_size):
            chunk = slice(start, start + chunk_size)
            values = np.asarray(uw.function.evaluate(fn, coords[chunk])).reshape(
                -1, var.data.shape[1]
            )
            var.data[chunk, :] = values

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
        stokes.petsc_velocity_nullspace_basis = velocity_nullspace_basis


if delta_fn:
    rho = sp.exp(-1e5 * ((r_uw - params.uw_radius_internal) ** 2)) * y_lm_sym
    stokes.add_natural_bc(-rho * unit_rvec, mesh.boundaries.Internal.name)
    stokes.bodyforce = sp.Matrix([0.0, 0.0, 0.0])
else:
    rho = ((r_uw / params.uw_radius_outer) ** params.uw_k) * y_lm_sym
    gravity_fn = -1.0 * unit_rvec
    stokes.bodyforce = rho * gravity_fn

# %% [markdown]
# #### Nullspace Handling
#
# The coupled Stokes system is solved with PETSc's constant-pressure nullspace
# enabled. This removes the additive pressure gauge freedom during the solve
# without imposing an artificial pressure Dirichlet condition on the spherical
# boundaries.
#
# We still subtract the domain-average pressure after the solve so the reported
# pressure field has a unique zero-mean gauge for benchmark comparisons.
#
# This benchmark driver follows the annulus Kramer path and newer UW examples:
# for free-slip shells, enable the PETSc rigid-rotation nullspace during the
# solve, then subtract the benchmark's selected rigid rotation mode from the
# reported free-slip velocity field after the solve.
#
# %% [markdown]
# #### Tolerance And BC Type
#
# `stokes.tolerance` does not affect the two Kramer case families equally.
#
# - free-slip cases use weak analytical-velocity matching on the shell
#   boundaries and are more tolerance-sensitive.
# - no-slip cases can use strong zero-velocity Dirichlet conditions and are less
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

# %% [markdown]
# #### Boundary Conditions

# %%
# if freeslip:
#     if params.uw_bc_type == "natural":
#         v_diff = v_uw.sym - v_ana.sym
#         stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Upper.name)
#         stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Lower.name)
#     elif params.uw_bc_type == "essential":
#         stokes.add_essential_bc(v_ana.sym, mesh.boundaries.Upper.name)
#         stokes.add_essential_bc(v_ana.sym, mesh.boundaries.Lower.name)
#     else:
#         raise ValueError(f"Unknown bc_type: {params.uw_bc_type}")
# elif noslip:
#     if params.uw_bc_type == "natural":
#         v_diff = v_uw.sym - v_ana.sym
#         stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Upper.name)
#         stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Lower.name)
#     elif params.uw_bc_type == "essential":
#         stokes.add_essential_bc(sp.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.Upper.name)
#         stokes.add_essential_bc(sp.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.Lower.name)
#     else:
#         raise ValueError(f"Unknown bc_type: {params.uw_bc_type}")
# else:
#     raise ValueError(f"Unsupported case flags: freeslip={freeslip}, noslip={noslip}")

if freeslip:
    # UW implements annulus free-slip through a penalty on the normal velocity
    # component. This matches the authors' physical condition u.n = 0 while
    # leaving tangential motion free on the shell boundaries.
    Gamma_N = mesh.CoordinateSystem.unit_e_0
    stokes.add_natural_bc(params.uw_vel_penalty * Gamma_N.dot(v_uw.sym) * Gamma_N, mesh.boundaries.Upper.name)
    stokes.add_natural_bc(params.uw_vel_penalty * Gamma_N.dot(v_uw.sym) * Gamma_N, mesh.boundaries.Lower.name)
elif noslip:
    stokes.add_essential_bc(sp.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.Upper.name)
    stokes.add_essential_bc(sp.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.Lower.name)
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

    stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", params.uw_pressure_pc_type)
    if params.uw_pressure_pc_type == "mg":
        stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", params.uw_pressure_pc_mg_type)
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
    subtract_rigid_rotation(mesh, v_uw, v_theta_phi_fn_xyz)

# %% [markdown]
# ### Errors and L2 Norm

# %%
v_err_sym = v_uw.sym - v_ana_sym
p_err_sym = p_uw.sym[0] - p_ana_sym

# %%
def relative_l2_error(mesh, err_var, ana_var):
    """Relative L2 error for scalar or vector mesh variables / expressions."""
    err_expr = err_var.sym if hasattr(err_var, "sym") else err_var
    ana_expr = ana_var.sym if hasattr(ana_var, "sym") else ana_var

    if isinstance(err_expr, sp.MatrixBase):
        err_expr = err_expr.dot(err_expr)
        ana_expr = ana_expr.dot(ana_expr)
    else:
        err_expr = err_expr * err_expr
        ana_expr = ana_expr * ana_expr

    err_I = uw.maths.Integral(mesh, err_expr)
    ana_I = uw.maths.Integral(mesh, ana_expr)
    return np.sqrt(err_I.evaluate()) / np.sqrt(ana_I.evaluate())


# %%
v_err_l2 = relative_l2_error(mesh, v_err_sym, v_ana_sym)
p_err_l2 = relative_l2_error(mesh, p_err_sym, p_ana_sym)

if uw.mpi.rank == 0:
    print("Relative velocity L2 error:", v_err_l2)
    print("Relative pressure L2 error:", p_err_l2)

# %% [markdown]
# ### Save Outputs

# %%
if uw.mpi.rank == 0:
    with h5py.File(os.path.join(output_dir, "error_norm.h5"), "w") as f_h5:
        f_h5.create_dataset("l", data=params.uw_l)
        f_h5.create_dataset("m", data=params.uw_m)
        f_h5.create_dataset("k", data=params.uw_k)
        f_h5.create_dataset("cellsize", data=params.uw_cellsize)
        f_h5.create_dataset("v_l2_norm", data=v_err_l2)
        f_h5.create_dataset("p_l2_norm", data=p_err_l2)

# %%
fill_from_expression_chunked(v_ana, v_ana_sym)
fill_from_expression_chunked(p_ana, p_ana_sym)
fill_from_expression_chunked(rho_ana, rho)
fill_from_expression_chunked(v_err, v_err_sym)
fill_from_expression_chunked(p_err, p_err_sym)

# %%
mesh.write_timestep(
    "output",
    index=0,
    meshVars=[v_uw, p_uw, v_ana, p_ana, rho_ana, v_err, p_err],
    outputPath=str(output_dir),
)

# %%
# Serial post-processing plots should be generated in a separate script.
