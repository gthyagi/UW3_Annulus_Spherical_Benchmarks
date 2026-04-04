# %% [markdown]
# ## Spherical Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark paper](https://gmd.copernicus.org/articles/14/1899/2021/)
#
# ### Authors
# Thyagarajulu Gollapalli ([GitHub](https://github.com/gthyagi)) <br>
# Underworld3 Development Team ([UW3 Repository](https://github.com/underworldcode/underworld3))
#
# ##### Case1: Free-slip boundaries and delta function density perturbation
# ##### Case2: Free-slip boundaries and smooth density distribution
# ##### Case3: Zero-slip boundaries and delta function density perturbation
# ##### Case4: Zero-slip boundaries and smooth density distribution

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
    uw_freeslip_type=uw.Param(
        'nitsche',
        type=uw.ParamType.STRING,
        description="Free-slip method: penalty or nitsche",
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


params.uw_cellsize = parse_float_fraction(params.uw_cellsize)

pressure_is_continuous = params.uw_pcont if params.uw_pdegree > 0 else False
is_p1p0 = params.uw_vdegree == 1 and params.uw_pdegree == 0

if uw.mpi.rank == 0 and params.uw_pdegree == 0 and params.uw_pcont:
    print("Degree-0 pressure uses discontinuous storage; overriding uw_pcont to false.")

# benchmark only run k=l+1 for all cases.
params.uw_k = params.uw_l + 1

# %% [markdown]
# ### Case Mapping
#
# “zero-slip” and “no-slip” are used interchangeably in geodynamics.

# %%
freeslip = False
zeroslip = False
delta_fn = False
smooth = False

if params.uw_case in ("case1",):
    freeslip = True
    delta_fn = True
    params.uw_bc_type = f'natural_{params.uw_freeslip_type}'
    if params.uw_freeslip_type == "nitsche":
        params.uw_vel_penalty = None
elif params.uw_case in ("case2",):
    freeslip = True
    smooth = True
    params.uw_bc_type = f'natural_{params.uw_freeslip_type}'
    if params.uw_freeslip_type == "nitsche":
        params.uw_vel_penalty = None
elif params.uw_case in ("case3",):
    zeroslip = True
    delta_fn = True
    params.uw_bc_type = "essential"
    params.uw_vel_penalty = None
elif params.uw_case in ("case4",):
    zeroslip = True
    smooth = True
    params.uw_bc_type = "essential"
    params.uw_vel_penalty = None
else:
    raise ValueError(f"Unknown case: {params.uw_case}")

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

output_root = os.path.join(output_base, "output", "spherical", "kramer", "latest")
metrics_filename = "benchmark_metrics.h5"

case_id = make_case_id(
    case=params.uw_case,
    inv_lc=int(1 / params.uw_cellsize),
    l=params.uw_l,
    m=params.uw_m,
    k=params.uw_k,
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
elif zeroslip and delta_fn:
    soln_above = build_delta_solution(params.uw_radius_outer, params.uw_radius_inner, params.uw_radius_internal, params.uw_l, params.uw_m, -1.0, 1.0, +1, no_slip=True)
    soln_below = build_delta_solution(params.uw_radius_outer, params.uw_radius_inner, params.uw_radius_internal, params.uw_l, params.uw_m, -1.0, 1.0, -1, no_slip=True)
elif zeroslip and smooth:
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
# - zero-slip cases can use strong zero-velocity Dirichlet conditions and are less
#   sensitive to a looser tolerance.
#
# Practical choices for this script:
# - free-slip: `1e-8`
# - zero-slip: `1e-5`
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
inner = mesh.boundaries.Lower.name
outer = mesh.boundaries.Upper.name

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
# elif zeroslip:
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
#     raise ValueError(f"Unsupported case flags: freeslip={freeslip}, zeroslip={zeroslip}")

if freeslip:
    if params.uw_freeslip_type == "penalty":
        # UW implements annulus free-slip through a penalty on the normal velocity
        # component. This matches the authors' physical condition u.n = 0 while
        # leaving tangential motion free on the shell boundaries.
        Gamma_N = mesh.CoordinateSystem.unit_e_0
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
elif zeroslip:
    stokes.add_essential_bc(sp.Matrix([0.0, 0.0, 0.0]), outer)
    stokes.add_essential_bc(sp.Matrix([0.0, 0.0, 0.0]), inner)
else:
    raise ValueError(f"Unsupported case flags: freeslip={freeslip}, zeroslip={zeroslip}")

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

snes_reason = int(stokes.snes.getConvergedReason())
ksp_reason = int(stokes.snes.ksp.getConvergedReason())
snes_iterations = int(stokes.snes.getIterationNumber())
ksp_iterations = int(stokes.snes.ksp.getIterationNumber())

if uw.mpi.rank == 0:
    print(snes_reason)
    print(ksp_reason)

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


def subtract_rigid_rotations(mesh, velocity_var, rotation_modes, passes=2):
    """
    Remove rigid-body rotation components from the numerical velocity field.

    Spherical free-slip has a 3D rotational nullspace. We project the solved
    velocity onto each rigid-rotation basis mode and subtract those components
    before computing error norms.
    """
    for _ in range(passes):
        for rotation_mode in rotation_modes:
            mode_int = uw.maths.Integral(
                mesh, rotation_mode.dot(velocity_var.sym)
            ).evaluate()
            mode_norm = uw.maths.Integral(
                mesh, rotation_mode.dot(rotation_mode)
            ).evaluate()
            coeff = mode_int / mode_norm

            dv = uw.function.evaluate(coeff * rotation_mode, velocity_var.coords)
            velocity_var.data[...] -= dv.reshape(velocity_var.data.shape)


subtract_pressure_mean(mesh, p_uw)

if freeslip:
    subtract_rigid_rotations(mesh, v_uw, velocity_nullspace_basis)

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
v_err_l2 = relative_l2_error(mesh, v_err_sym, v_ana_sym)
p_err_l2 = relative_l2_error(mesh, p_err_sym, p_ana_sym)
p_err_l2_inner = relative_l2_error(mesh, p_err_sym, p_ana_sym, boundary=inner)
p_err_l2_outer = relative_l2_error(mesh, p_err_sym, p_ana_sym, boundary=outer)
v_err_l2_inner_abs = absolute_l2_error(mesh, v_err_sym, boundary=inner)
v_err_l2_outer_abs = absolute_l2_error(mesh, v_err_sym, boundary=outer)

if zeroslip:
    v_err_l2_inner = np.nan
    v_err_l2_outer = np.nan
else:
    v_err_l2_inner = relative_l2_error(mesh, v_err_sym, v_ana_sym, boundary=inner)
    v_err_l2_outer = relative_l2_error(mesh, v_err_sym, v_ana_sym, boundary=outer)

u_dot_n_l2_inner_abs = absolute_l2_error(mesh, unit_rvec.dot(v_uw.sym), boundary=inner)
u_dot_n_l2_outer_abs = absolute_l2_error(mesh, unit_rvec.dot(v_uw.sym), boundary=outer)
run_metadata = gather_run_metadata(mesh, v_uw, p_uw)
git_sha = current_git_sha(repo_root)
cli_args = " ".join(sys.argv)

if uw.mpi.rank == 0:
    print("=== Relative L2 Errors ===")
    print(f"Velocity (domain): {v_err_l2}")
    print(f"Pressure (domain): {p_err_l2}")
    print(f"Velocity (inner):  {v_err_l2_inner}")
    print(f"Velocity (outer):  {v_err_l2_outer}")
    print(f"Velocity abs (inner): {v_err_l2_inner_abs}")
    print(f"Velocity abs (outer): {v_err_l2_outer_abs}")
    print(f"Pressure (inner):  {p_err_l2_inner}")
    print(f"Pressure (outer):  {p_err_l2_outer}")
    print(f"u.n abs (inner): {u_dot_n_l2_inner_abs}")
    print(f"u.n abs (outer): {u_dot_n_l2_outer_abs}")

# %% [markdown]
# ### Save Outputs

# %%
if uw.mpi.rank == 0:
    with h5py.File(os.path.join(output_dir, metrics_filename), "w") as f_h5:
        f_h5.create_dataset("l", data=params.uw_l)
        f_h5.create_dataset("m", data=params.uw_m)
        f_h5.create_dataset("k", data=params.uw_k)
        f_h5.create_dataset("cellsize", data=params.uw_cellsize)
        f_h5.create_dataset("v_l2_norm", data=v_err_l2)
        f_h5.create_dataset("p_l2_norm", data=p_err_l2)
        f_h5.create_dataset("v_l2_norm_inner", data=v_err_l2_inner)
        f_h5.create_dataset("v_l2_norm_outer", data=v_err_l2_outer)
        f_h5.create_dataset("v_l2_norm_inner_abs", data=v_err_l2_inner_abs)
        f_h5.create_dataset("v_l2_norm_outer_abs", data=v_err_l2_outer_abs)
        f_h5.create_dataset("p_l2_norm_inner", data=p_err_l2_inner)
        f_h5.create_dataset("p_l2_norm_outer", data=p_err_l2_outer)
        f_h5.create_dataset("u_dot_n_l2_norm_inner_abs", data=u_dot_n_l2_inner_abs)
        f_h5.create_dataset("u_dot_n_l2_norm_outer_abs", data=u_dot_n_l2_outer_abs)
        f_h5.create_dataset("git_sha", data=np.bytes_(git_sha))
        f_h5.create_dataset("command", data=np.bytes_(cli_args))
        for key, value in run_metadata.items():
            f_h5.create_dataset(key, data=value)

mesh.write_timestep(
    "output",
    index=0,
    meshVars=[v_uw, p_uw],
    outputPath=str(output_dir),
)
