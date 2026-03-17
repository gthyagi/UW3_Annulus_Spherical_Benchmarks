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
        1.0 / 8.0,
        type=uw.ParamType.FLOAT,
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
        1e-10,
        type=uw.ParamType.FLOAT,
        description="Stokes solver tolerance",
    ),
    uw_vel_penalty=uw.Param(
        1e8,
        type=uw.ParamType.FLOAT,
        description="Penalty for natural-BC velocity matching",
    ),
    uw_bc_type=uw.Param(
        "natural",
        type=uw.ParamType.STRING,
        description="Boundary-condition mode: natural or essential",
    ),
)

if any(arg in ("--help", "-h", "-help", "-uw_help") for arg in sys.argv[1:]):
    print(params.cli_help())
    raise SystemExit(0)

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
r_uw = mesh.CoordinateSystem.xR[0]
th_uw = mesh.CoordinateSystem.xR[1]
phi_raw = mesh.CoordinateSystem.xR[2]
phi_uw = sp.Piecewise(
    (2 * sp.pi + phi_raw, phi_raw < 0),
    (phi_raw, True),
)
null_mode_expr = sp.Matrix(((0, 1, 1), (-1, 0, 1), (-1, -1, 0))) * mesh.CoordinateSystem.N.T
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


def analytical_pressure_sympy(soln, r_sym, y_sym):
    p_sym = (soln.G * r_sym**soln.l + soln.H * r_sym ** (-soln.l - 1)) * y_sym
    if hasattr(soln, "K"):
        p_sym += soln.K * r_sym ** (soln.k + 1) * y_sym
    return p_sym


v_ana_above_sym = analytical_velocity_cartesian_sympy(soln_above, r_uw, y_lm_sym)
v_ana_below_sym = analytical_velocity_cartesian_sympy(soln_below, r_uw, y_lm_sym)
p_ana_above_sym = analytical_pressure_sympy(soln_above, r_uw, y_lm_sym)
p_ana_below_sym = analytical_pressure_sympy(soln_below, r_uw, y_lm_sym)

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
def analytical_values(var, fn_above, fn_below):
    coords = np.asarray(var.coords)
    radii = np.linalg.norm(coords, axis=1)
    mask_above = radii > params.uw_radius_internal
    values = np.empty_like(var.data)

    for mask, fn in ((mask_above, fn_above), (~mask_above, fn_below)):
        if np.any(mask):
            values[mask, :] = np.asarray(uw.function.evaluate(fn, coords[mask])).reshape(-1, var.data.shape[1])

    return values


# %%
v_ana_values = analytical_values(v_ana, v_ana_above_sym, v_ana_below_sym)
p_ana_values = analytical_values(p_ana, p_ana_above_sym, p_ana_below_sym)

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

gravity_fn = -1.0 * unit_rvec

if delta_fn:
    rho = sp.exp(-1e5 * ((r_uw - params.uw_radius_internal) ** 2)) * y_lm_sym
    stokes.add_natural_bc(-rho * unit_rvec, mesh.boundaries.Internal.name)
    stokes.bodyforce = sp.Matrix([0.0, 0.0, 0.0])
else:
    rho = ((r_uw / params.uw_radius_outer) ** params.uw_k) * y_lm_sym
    stokes.bodyforce = rho * gravity_fn

rho_ana.data[:] = np.asarray(uw.function.evaluate(rho, rho_ana.coords)).reshape(-1, 1)

# %% [markdown]
# #### Boundary Conditions

# %%
if freeslip:
    if params.uw_bc_type == "natural":
        v_diff = v_uw.sym - v_ana.sym
        stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Upper.name)
        stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Lower.name)
    elif params.uw_bc_type == "essential":
        stokes.add_essential_bc(v_ana.sym, mesh.boundaries.Upper.name)
        stokes.add_essential_bc(v_ana.sym, mesh.boundaries.Lower.name)
    else:
        raise ValueError(f"Unknown bc_type: {params.uw_bc_type}")
elif noslip:
    if params.uw_bc_type == "natural":
        v_diff = v_uw.sym - v_ana.sym
        stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Upper.name)
        stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Lower.name)
    elif params.uw_bc_type == "essential":
        stokes.add_essential_bc(sp.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.Upper.name)
        stokes.add_essential_bc(sp.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.Lower.name)
    else:
        raise ValueError(f"Unknown bc_type: {params.uw_bc_type}")

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
# ### Remove Null Mode

# %%
I0 = uw.maths.Integral(mesh, null_mode_expr.dot(v_uw.sym))
norm = I0.evaluate()
I0.fn = null_mode_expr.dot(null_mode_expr)
vnorm = I0.evaluate()

dv = uw.function.evaluate(norm * null_mode_expr, v_uw.coords) / vnorm
v_uw.data[...] -= np.asarray(dv).reshape(v_uw.data.shape)


# %% [markdown]
# ### Errors And L2 Norm

# %%
with uw.synchronised_array_update():
    v_err.data[...] = v_uw.data - v_ana.data
    p_err.data[...] = p_uw.data - p_ana.data


# %%
def relative_l2_error(mesh, err_expr, ana_expr):
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
v_err_l2 = relative_l2_error(mesh, v_err.sym, v_ana.sym)
p_err_l2 = relative_l2_error(mesh, p_err.sym[0], p_ana.sym[0])

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
mesh.write_timestep(
    "output",
    index=0,
    meshVars=[v_uw, p_uw, v_ana, p_ana, rho_ana, v_err, p_err],
    outputPath=str(output_dir),
)

# %%
# Serial post-processing plots should be generated in a separate script.
