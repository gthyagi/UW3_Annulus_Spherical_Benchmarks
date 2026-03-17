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
import scipy.special
import sympy as sp
import underworld3 as uw
from math import acos, atan2, cos, pi, sin, sqrt, tan
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
        description="Benchmark case: case1, case2, case3, case4",
    ),
    uw_l=uw.Param(
        2,
        description="Spherical harmonic degree",
    ),
    uw_m=uw.Param(
        1,
        description="Spherical harmonic order",
    ),
    uw_k=uw.Param(
        3,
        description="Power exponent for smooth density forcing",
    ),
    uw_radius_inner=uw.Param(
        1.22,
        description="Inner radius",
    ),
    uw_radius_internal=uw.Param(
        2.0,
        description="Internal interface radius",
    ),
    uw_radius_outer=uw.Param(
        2.22,
        description="Outer radius",
    ),
    uw_cellsize=uw.Param(
        1.0 / 8.0,
        description="Background mesh cell size",
    ),
    uw_vdegree=uw.Param(
        2,
        description="Velocity polynomial degree",
    ),
    uw_pdegree=uw.Param(
        1,
        description="Pressure polynomial degree",
    ),
    uw_pcont=uw.Param(
        True,
        description="Pressure continuity flag",
    ),
    uw_stokes_tol=uw.Param(
        1e-10,
        description="Stokes solver tolerance",
    ),
    uw_vel_penalty=uw.Param(
        1e8,
        description="Penalty for natural-BC velocity matching",
    ),
    uw_bc_type=uw.Param(
        "natural",
        description="Boundary-condition mode: natural or essential",
    ),
)

if any(arg in ("--help", "-h", "-help", "-uw_help") for arg in sys.argv[1:]):
    print(params.cli_help())
    raise SystemExit(0)

# %% [markdown]
# ### Convection Parameters

# %%
case = params.uw_case
l = int(params.uw_l)
m = int(params.uw_m)
k = int(params.uw_k)

# %% [markdown]
# ### Mesh Parameters

# %%
r_i = float(params.uw_radius_inner)
r_int = float(params.uw_radius_internal)
r_o = float(params.uw_radius_outer)
cellsize = float(params.uw_cellsize)

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
output_root = os.path.join(repo_root, "output", "spherical", "kramer", "latest")

output_dir = os.path.join(
    output_root,
    (
        f"{case}_inv_lc_{int(1/cellsize)}_l_{l}_m_{m}_k_{k}_vdeg_{params.uw_vdegree}_pdeg_{params.uw_pdegree}"
        f"_pcont_{str(params.uw_pcont).lower()}_vel_penalty_{params.uw_vel_penalty:.2g}_stokes_tol_{params.uw_stokes_tol:.2g}"
        f"_ncpus_{uw.mpi.size}_bc_{params.uw_bc_type}/"
    ),
)

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


def sph_harm_real_compat(l, m, theta, phi):
    sph_harm_y = getattr(scipy.special, "sph_harm_y", None)
    if sph_harm_y is not None:
        return sph_harm_y(l, m, theta, phi)
    return scipy.special.sph_harm(m, l, phi, theta)


def Y(l, m, theta, phi):
    return sph_harm_real_compat(l, m, theta, phi).real


def dYdphi(l, m, theta, phi):
    return -m * sph_harm_real_compat(l, m, theta, phi).imag


def dYdtheta(l, m, theta, phi):
    dydt = m / tan(theta) * Y(l, m, theta, phi)
    if m < l:
        dydt += sqrt((l - m) * (l + m + 1)) * Y(l, m + 1, theta, phi * m / (m + 1))
    return dydt


def to_spherical(X):
    r = sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)
    theta = acos(X[2] / r)
    phi = atan2(X[1], X[0])
    return r, theta, phi


class SphericalStokesSolutionBase:
    def __init__(self, l, m, Rp=2.22, Rm=1.22, nu=1.0, g=1.0):
        if l < 0:
            raise ValueError("Spherical order l cannot be negative.")
        if abs(m) > l:
            raise ValueError("Absolute value of spherical degree m must be <= l.")
        self.l = l
        self.m = m
        self.Rp = Rp
        self.Rm = Rm
        self.nu = nu
        self.g = g

    def Pl(self, r):
        raise NotImplementedError

    def dPldr(self, r):
        raise NotImplementedError

    def p(self, r, theta, phi):
        raise NotImplementedError

    def u_theta(self, r, theta, phi):
        return -(self.Pl(r) / r + self.dPldr(r)) * dYdtheta(self.l, self.m, theta, phi)

    def u_phi(self, r, theta, phi):
        return -(self.Pl(r) / r + self.dPldr(r)) / sin(theta) * dYdphi(self.l, self.m, theta, phi)

    def u_r(self, r, theta, phi):
        return -self.l * (self.l + 1) * self.Pl(r) * Y(self.l, self.m, theta, phi) / r

    def velocity_cartesian(self, X):
        r, theta, phi = to_spherical(X)
        if theta < 1e-7 * pi or theta > pi * (1.0 - 1e-7):
            dx = 1e-6 * r
            return tuple(
                np.mean(
                    [
                        self.velocity_cartesian((x, y, X[2]))
                        for x, y in ((dx, dx), (-dx, dx), (dx, -dx), (-dx, -dx))
                    ],
                    axis=0,
                )
            )

        ur = self.u_r(r, theta, phi)
        uth = self.u_theta(r, theta, phi)
        uph = self.u_phi(r, theta, phi)
        req = sqrt(X[0] ** 2 + X[1] ** 2)
        costh = cos(theta)
        return (
            X[0] / r * ur + X[0] / req * costh * uth - X[1] / req * uph,
            X[1] / r * ur + X[1] / req * costh * uth + X[0] / req * uph,
            X[2] / r * ur - sin(theta) * uth,
        )

    def pressure_cartesian(self, X):
        return self.p(*to_spherical(X))


class SphericalDeltaSolution(SphericalStokesSolutionBase):
    def __init__(self, ABCD, l, m, Rp=2.22, Rm=1.22, nu=1.0, g=1.0):
        super().__init__(l, m, Rp=Rp, Rm=Rm, nu=nu, g=g)
        self.ABCD = ABCD
        _, _, C, D = ABCD
        self.G = -2 * nu * (l + 1) * (2 * l + 3) * C
        self.H = -2 * nu * l * (2 * l - 1) * D

    def Pl(self, r):
        A, B, C, D = self.ABCD
        l = self.l
        return A * r**l + B * r ** (-l - 1) + C * r ** (l + 2) + D * r ** (-l + 1)

    def dPldr(self, r):
        A, B, C, D = self.ABCD
        l = self.l
        return l * A * r ** (l - 1) + (-l - 1) * B * r ** (-l - 2) + (l + 2) * C * r ** (l + 1) + (-l + 1) * D * r**-l

    def p(self, r, theta, phi):
        return (self.G * r**self.l + self.H * r ** (-self.l - 1)) * Y(self.l, self.m, theta, phi)


class SphericalSmoothSolution(SphericalStokesSolutionBase):
    def __init__(self, ABCDE, k, l, m, Rp=2.22, Rm=1.22, nu=1.0, g=1.0):
        super().__init__(l, m, Rp=Rp, Rm=Rm, nu=nu, g=g)
        self.k = k
        self.ABCDE = ABCDE
        _, _, C, D, _ = ABCDE
        self.G = -2 * nu * (l + 1) * (2 * l + 3) * C
        self.H = -2 * nu * l * (2 * l - 1) * D
        self.K = -g * (k + 2) / ((k + 1) * (k + 2) - l * (l + 1)) / Rp**k

    def Pl(self, r):
        A, B, C, D, E = self.ABCDE
        l, k = self.l, self.k
        return A * r**l + B * r ** (-l - 1) + C * r ** (l + 2) + D * r ** (-l + 1) + E * r ** (k + 3)

    def dPldr(self, r):
        A, B, C, D, E = self.ABCDE
        l, k = self.l, self.k
        return (
            l * A * r ** (l - 1)
            + (-l - 1) * B * r ** (-l - 2)
            + (l + 2) * C * r ** (l + 1)
            + (-l + 1) * D * r**-l
            + (k + 3) * E * r ** (k + 2)
        )

    def p(self, r, theta, phi):
        return (
            self.G * r**self.l + self.H * r ** (-self.l - 1) + self.K * r ** (self.k + 1)
        ) * Y(self.l, self.m, theta, phi)


def build_delta_solution(Rp, Rm, rp, l, m, g, nu, sign, no_slip):
    coeffs = coefficients_sphere_delta_ns(Rp, Rm, rp, l, g, nu, sign) if no_slip else coefficients_sphere_delta_fs(Rp, Rm, rp, l, g, nu, sign)
    return SphericalDeltaSolution(coeffs, l, m, Rp=Rp, Rm=Rm, nu=nu, g=g)


def build_smooth_solution(Rp, Rm, k, l, m, g, nu, no_slip):
    if (k + 1) * (k + 2) == l * (l + 1) or (k + 3) * (k + 4) == l * (l + 1):
        raise NotImplementedError(f"Smooth solution not implemented for k={k}, l={l}")
    coeffs = coefficients_sphere_smooth_ns(Rp, Rm, k, l, g, nu) if no_slip else coefficients_sphere_smooth_fs(Rp, Rm, k, l, g, nu)
    return SphericalSmoothSolution(coeffs, k, l, m, Rp=Rp, Rm=Rm, nu=nu, g=g)


if freeslip and delta_fn:
    soln_above = build_delta_solution(r_o, r_i, r_int, l, m, -1.0, 1.0, +1, no_slip=False)
    soln_below = build_delta_solution(r_o, r_i, r_int, l, m, -1.0, 1.0, -1, no_slip=False)
elif freeslip and smooth:
    soln_above = build_smooth_solution(r_o, r_i, k, l, m, 1.0, 1.0, no_slip=False)
    soln_below = build_smooth_solution(r_o, r_i, k, l, m, 1.0, 1.0, no_slip=False)
elif noslip and delta_fn:
    soln_above = build_delta_solution(r_o, r_i, r_int, l, m, -1.0, 1.0, +1, no_slip=True)
    soln_below = build_delta_solution(r_o, r_i, r_int, l, m, -1.0, 1.0, -1, no_slip=True)
elif noslip and smooth:
    soln_above = build_smooth_solution(r_o, r_i, k, l, m, 1.0, 1.0, no_slip=True)
    soln_below = build_smooth_solution(r_o, r_i, k, l, m, 1.0, 1.0, no_slip=True)

# %% [markdown]
# ### Create Mesh

# %%
if delta_fn:
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=r_o,
        radiusInternal=r_int,
        radiusInner=r_i,
        cellSize=cellsize,
        qdegree=max(params.uw_pdegree, params.uw_vdegree),
        degree=1,
        filename=f"{output_dir}/mesh.msh",
        refinement=None,
    )
else:
    mesh = uw.meshing.SphericalShell(
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
r_uw = mesh.CoordinateSystem.xR[0]
th_uw = mesh.CoordinateSystem.xR[1]
phi_raw = mesh.CoordinateSystem.xR[2]
phi_uw = sp.Piecewise(
    (2 * sp.pi + phi_raw, phi_raw < 0),
    (phi_raw, True),
)
null_mode_expr = sp.Matrix(((0, 1, 1), (-1, 0, 1), (-1, -1, 0))) * mesh.CoordinateSystem.N.T

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
def fill_piecewise(var, fn_above, fn_below):
    radii = uw.function.evaluate(r_uw, var.coords)

    for i, coord in enumerate(var.coords):
        fn = fn_above if radii[i] > r_int else fn_below
        var.data[i] = fn(coord)


# %%
fill_piecewise(v_ana, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
fill_piecewise(p_ana, soln_above.pressure_cartesian, soln_below.pressure_cartesian)

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
y_lm_real = (
    sp.sqrt((2 * l + 1) / (4 * sp.pi) * sp.factorial(l - m) / sp.factorial(l + m))
    * sp.cos(m * phi_uw)
    * sp.assoc_legendre(l, m, sp.cos(th_uw))
)

gravity_fn = -1.0 * unit_rvec

if delta_fn:
    rho = sp.exp(-1e5 * ((r_uw - r_int) ** 2)) * y_lm_real
    stokes.add_natural_bc(-rho * unit_rvec, mesh.boundaries.Internal.name)
    stokes.bodyforce = sp.Matrix([0.0, 0.0, 0.0])
else:
    rho = ((r_uw / r_o) ** k) * y_lm_real
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
def fill_error(var_err, var_num, fn_above, fn_below):
    radii = uw.function.evaluate(r_uw, var_err.coords)

    for i, coord in enumerate(var_err.coords):
        fn = fn_above if radii[i] > r_int else fn_below
        var_err.data[i] = var_num.data[i] - fn(coord)


# %%
fill_error(v_err, v_uw, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
fill_error(p_err, p_uw, soln_above.pressure_cartesian, soln_below.pressure_cartesian)


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
        f_h5.create_dataset("l", data=l)
        f_h5.create_dataset("m", data=m)
        f_h5.create_dataset("k", data=k)
        f_h5.create_dataset("cellsize", data=cellsize)
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
