# %% [markdown]
# ## Spherical Kramer Latest Post-Processing
#
# Edit `dirname` below, then run this script to recreate the spherical Kramer
# field plots from the latest split checkpoint output.

# %%
# to fix trame issue
import nest_asyncio

nest_asyncio.apply()

# %%
import os
import re
import sys
from math import factorial
from types import SimpleNamespace

import cmcrameri.cm as cmc
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy import special

try:
    from IPython.display import display
except ImportError:
    display = None

IS_INTERACTIVE = (
    hasattr(sys, "ps1") or sys.flags.interactive or "ipykernel" in sys.modules
)
JUPYTER_BACKEND = "html"

if IS_INTERACTIVE:
    pv.global_theme.jupyter_backend = JUPYTER_BACKEND

# %% [markdown]
# ### Parameters And Paths

# %%
dirname = "case2_inv_lc_8_l_2_m_1_k_3_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-05_ncpus_8_bc_natural_nitsche"

# %%
output_dir = os.path.join("../../output/spherical/kramer/latest/", f"{dirname}/")

# %%
pattern = (
    r"(?P<case>case\d+)_"
    r"inv_lc_(?P<inv_lc>\d+)_"
    r"l_(?P<l>\d+)_"
    r"m_(?P<m>-?\d+)_"
    r"k_(?P<k>\d+)_"
    r"vdeg_(?P<vdeg>\d+)_"
    r"pdeg_(?P<pdeg>\d+)_"
    r"pcont_(?P<pcont>true|false)_"
    r"(?:vel_penalty_(?P<vel_penalty>[0-9.eE+\-]+)_)?"
    r"stokes_tol_(?P<stokes_tol>[0-9.eE+\-]+)_"
    r"ncpus_(?P<ncpus>\d+)_"
    r"bc_(?P<bc_type>[A-Za-z]+)"
    r"(?:_(?P<freeslip_type>nitsche|penalty))?$"
)

match = re.search(pattern, dirname)
if match is None:
    raise ValueError(f"Could not parse dirname: {dirname}")

params = match.groupdict()

# %%
case = params["case"]
inv_lc = int(params["inv_lc"])
cellsize = 1.0 / inv_lc
l = int(params["l"])
m = int(params["m"])
k = int(params["k"])
vdegree = int(params["vdeg"])
pdegree = int(params["pdeg"])
pcont = params["pcont"] == "true"
pcont_str = str(pcont).lower()
vel_penalty = None if params["vel_penalty"] is None else float(params["vel_penalty"])
stokes_tol = float(params["stokes_tol"])
ncpus = int(params["ncpus"])
bc_type = params["bc_type"]
freeslip_type = params["freeslip_type"]

r_i = 1.22
r_int = 2.0
r_o = 2.22
clip_angle = 135.0
cpos = "yz"

freeslip = case in ("case1", "case2")
noslip = case in ("case3", "case4")
delta_fn = case in ("case1", "case3")
smooth = case in ("case2", "case4")


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


def build_case_solutions():
    if freeslip and delta_fn:
        return (
            build_delta_solution(r_o, r_i, r_int, l, m, -1.0, 1.0, +1, no_slip=False),
            build_delta_solution(r_o, r_i, r_int, l, m, -1.0, 1.0, -1, no_slip=False),
        )
    if freeslip and smooth:
        soln = build_smooth_solution(r_o, r_i, k, l, m, 1.0, 1.0, no_slip=False)
        return soln, soln
    if noslip and delta_fn:
        return (
            build_delta_solution(r_o, r_i, r_int, l, m, -1.0, 1.0, +1, no_slip=True),
            build_delta_solution(r_o, r_i, r_int, l, m, -1.0, 1.0, -1, no_slip=True),
        )
    if noslip and smooth:
        soln = build_smooth_solution(r_o, r_i, k, l, m, 1.0, 1.0, no_slip=True)
        return soln, soln
    raise ValueError(f"Unsupported case: {case}")


def real_spherical_harmonic_and_derivatives(l_val, m_val, theta, phi):
    x = np.cos(theta)
    p_lm = special.lpmv(m_val, l_val, x)
    if l_val - 1 >= m_val:
        p_lm_prev = special.lpmv(m_val, l_val - 1, x)
    else:
        p_lm_prev = np.zeros_like(theta)

    norm = np.sqrt(
        (2 * l_val + 1)
        / (4 * np.pi)
        * factorial(l_val - m_val)
        / factorial(l_val + m_val)
    )
    cos_mphi = np.cos(m_val * phi)
    sin_mphi = np.sin(m_val * phi)
    y_lm = norm * cos_mphi * p_lm

    sin_theta = np.sin(theta)
    safe_sin = np.where(np.abs(sin_theta) > 1.0e-12, sin_theta, np.inf)
    dplm_dtheta = (l_val * x * p_lm - (l_val + m_val) * p_lm_prev) / safe_sin
    dy_dtheta = norm * cos_mphi * dplm_dtheta
    dy_dphi = -m_val * norm * sin_mphi * p_lm
    return y_lm, dy_dtheta, dy_dphi


def analytical_velocity_cartesian(soln, rr, theta, phi, y_lm, dy_dtheta, dy_dphi):
    l_val = int(soln.l)
    if hasattr(soln, "ABCD"):
        A, B, C, D = soln.ABCD
        P_l = A * rr**l_val + B * rr ** (-l_val - 1) + C * rr ** (l_val + 2) + D * rr ** (-l_val + 1)
        dPldr = (
            A * l_val * rr ** (l_val - 1)
            + B * (-l_val - 1) * rr ** (-l_val - 2)
            + C * (l_val + 2) * rr ** (l_val + 1)
            + D * (-l_val + 1) * rr ** (-l_val)
        )
    else:
        A, B, C, D, E = soln.ABCDE
        k_val = int(soln.k)
        P_l = A * rr**l_val + B * rr ** (-l_val - 1) + C * rr ** (l_val + 2) + D * rr ** (-l_val + 1) + E * rr ** (k_val + 3)
        dPldr = (
            A * l_val * rr ** (l_val - 1)
            + B * (-l_val - 1) * rr ** (-l_val - 2)
            + C * (l_val + 2) * rr ** (l_val + 1)
            + D * (-l_val + 1) * rr ** (-l_val)
            + E * (k_val + 3) * rr ** (k_val + 2)
        )

    prefactor = -(P_l / rr + dPldr)
    u_r = -l_val * (l_val + 1) * P_l * y_lm / rr
    u_theta = prefactor * dy_dtheta
    sin_theta = np.sin(theta)
    safe_sin = np.where(np.abs(sin_theta) > 1.0e-12, sin_theta, np.inf)
    u_phi = prefactor * dy_dphi / safe_sin

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    u_x = sin_theta * cos_phi * u_r + cos_theta * cos_phi * u_theta - sin_phi * u_phi
    u_y = sin_theta * sin_phi * u_r + cos_theta * sin_phi * u_theta + cos_phi * u_phi
    u_z = cos_theta * u_r - sin_theta * u_theta
    return np.column_stack([u_x, u_y, u_z])


def analytical_pressure(soln, rr, y_lm):
    p_expr = (soln.G * rr**soln.l + soln.H * rr ** (-soln.l - 1)) * y_lm
    if hasattr(soln, "K"):
        p_expr += soln.K * rr ** (int(soln.k) + 1) * y_lm
    return p_expr


def analytical_solution(points_xyz):
    soln_above, soln_below = build_case_solutions()
    rr = np.linalg.norm(points_xyz, axis=1)
    theta = np.arccos(np.clip(points_xyz[:, 2] / rr, -1.0, 1.0))
    phi = np.mod(np.arctan2(points_xyz[:, 1], points_xyz[:, 0]), 2 * np.pi)
    y_lm, dy_dtheta, dy_dphi = real_spherical_harmonic_and_derivatives(l, m, theta, phi)
    mask_above = rr > r_int

    v_ana = np.empty((points_xyz.shape[0], 3), dtype=np.float64)
    p_ana = np.empty(points_xyz.shape[0], dtype=np.float64)

    for mask, soln in ((mask_above, soln_above), (~mask_above, soln_below)):
        if np.any(mask):
            v_ana[mask] = analytical_velocity_cartesian(
                soln,
                rr[mask],
                theta[mask],
                phi[mask],
                y_lm[mask],
                dy_dtheta[mask],
                dy_dphi[mask],
            )
            p_ana[mask] = analytical_pressure(soln, rr[mask], y_lm[mask])

    if delta_fn:
        rho_ana = np.exp(-1e5 * ((rr - r_int) ** 2)) * y_lm
    else:
        rho_ana = ((rr / r_o) ** k) * y_lm

    return v_ana, p_ana, rho_ana

# %%
SCREENSHOT_WINDOW_SIZE = (750, 750)
mesh_h5_path = os.path.join(output_dir, "output.mesh.00000.h5")

if not os.path.isfile(mesh_h5_path):
    raise FileNotFoundError(f"Missing mesh H5 file: {mesh_h5_path}")

# %% [markdown]
# ### Checkpoint Read

# %%
def list_h5_datasets(h5f):
    paths = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    h5f.visititems(visitor)
    return paths


def find_field_path(h5f, field_name):
    candidates = (
        f"vertex_fields/{field_name}_{field_name}",
        f"vertex_fields/{field_name}",
        f"fields/{field_name}",
        field_name,
    )
    for cand in candidates:
        if cand in h5f:
            return cand

    datasets = list_h5_datasets(h5f)
    for dset in datasets:
        if dset.split("/")[-1].lower() == field_name.lower():
            return dset

    raise KeyError(f"Field '{field_name}' not found in checkpoint.")


def read_field(h5f, field_name, n_points):
    path = find_field_path(h5f, field_name)
    arr = np.asarray(h5f[path], dtype=np.float64)

    if arr.ndim == 2 and arr.shape[0] != n_points and arr.shape[1] == n_points:
        arr = arr.T

    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)

    if arr.shape[0] != n_points:
        raise ValueError(f"Field {field_name} has shape {arr.shape}, expected first dim {n_points}.")

    return arr


def read_split_field(mesh_h5_file, field_name, n_points):
    split_file = mesh_h5_file.replace(".00000.h5", f".{field_name}.00000.h5")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Missing split field file: {split_file}")

    with h5py.File(split_file, "r") as h5f:
        return read_field(h5f, field_name, n_points)


with h5py.File(mesh_h5_path, "r") as h5f:
    points = np.asarray(h5f["geometry/vertices"], dtype=np.float64)
    cells = np.asarray(h5f["viz/topology/cells"], dtype=np.int64)

v_u = read_split_field(mesh_h5_path, "V_u", points.shape[0])
p_u = read_split_field(mesh_h5_path, "P_u", points.shape[0]).reshape(-1)
v_a, p_a, rho_a = analytical_solution(points)
v_e = np.asarray(v_u, dtype=np.float64) - v_a
p_e = np.asarray(p_u, dtype=np.float64).reshape(-1) - p_a

cells_flat = np.hstack([np.full((cells.shape[0], 1), cells.shape[1], dtype=np.int64), cells]).ravel()
celltypes = np.full(cells.shape[0], pv.CellType.TETRA, dtype=np.uint8)

grid = pv.UnstructuredGrid(cells_flat, celltypes, points)
grid.point_data["V_u"] = np.asarray(v_u, dtype=np.float64)
grid.point_data["P_u"] = np.asarray(p_u, dtype=np.float64).reshape(-1)
grid.point_data["V_a"] = np.asarray(v_a, dtype=np.float64)
grid.point_data["P_a"] = np.asarray(p_a, dtype=np.float64).reshape(-1)
grid.point_data["V_e"] = np.asarray(v_e, dtype=np.float64)
grid.point_data["P_e"] = np.asarray(p_e, dtype=np.float64).reshape(-1)
grid.point_data["RHO_a"] = np.asarray(rho_a, dtype=np.float64).reshape(-1)
grid.point_data["RHO_plot"] = -np.asarray(rho_a, dtype=np.float64).reshape(-1)

v_ana_mag = np.linalg.norm(v_a, axis=1)
v_err_mag = np.linalg.norm(v_e, axis=1)

with np.errstate(divide="ignore", invalid="ignore"):
    v_err_pct = np.where(v_ana_mag > 1.0e-14, 100.0 * v_err_mag / v_ana_mag, 0.0)
    p_err_pct = np.where(np.abs(p_a) > 1.0e-14, 100.0 * p_e / p_a, 0.0)

grid.point_data["V_err_pct"] = np.nan_to_num(v_err_pct)
grid.point_data["P_err_pct"] = np.nan_to_num(p_err_pct)

# %%
case_limits = {
    "case1": {
        "velocity": [0.0, 0.015],
        "pressure": [-0.25, 0.25],
        "rho": [-0.4, 0.4],
        "velocity_error": [0.0, 0.005],
        "velocity_pct": [0.0, 100.0],
        "pressure_error": [-0.065, 0.065],
        "pressure_pct": [-100.0, 100.0],
    },
    "case2": {
        "velocity": [0.0, 0.007],
        "pressure": [-0.1, 0.1],
        "rho": [-0.4, 0.4],
        "velocity_error": [0.0, 1.0e-4],
        "velocity_pct": [0.0, 100.0],
        "pressure_error": [-0.01, 0.01],
        "pressure_pct": [-100.0, 100.0],
    },
    "case3": {
        "velocity": [0.0, 0.003],
        "pressure": [-0.3, 0.3],
        "rho": [-0.4, 0.4],
        "velocity_error": [0.0, 6.0e-3],
        "velocity_pct": [0.0, 100.0],
        "pressure_error": [-0.065, 0.065],
        "pressure_pct": [-100.0, 100.0],
    },
    "case4": {
        "velocity": [0.0, 0.001],
        "pressure": [-0.1, 0.1],
        "rho": [-0.4, 0.4],
        "velocity_error": [0.0, 1.0e-4],
        "velocity_pct": [0.0, 100.0],
        "pressure_error": [-0.01, 0.01],
        "pressure_pct": [-100.0, 100.0],
    },
}

limits = case_limits[case]

# %% [markdown]
# ### Plot Helpers

# %%
def clip_grid(grid, clip_angle, crinkle=False):
    normal_1 = (
        np.cos(np.deg2rad(clip_angle)),
        np.cos(np.deg2rad(clip_angle)),
        0.0,
    )
    normal_2 = (
        np.cos(np.deg2rad(clip_angle)),
        -np.cos(np.deg2rad(clip_angle)),
        0.0,
    )

    clip_1 = grid.clip(origin=(0.0, 0.0, 0.0), normal=normal_1, invert=False, crinkle=crinkle)
    clip_2 = grid.clip(origin=(0.0, 0.0, 0.0), normal=normal_2, invert=False, crinkle=crinkle)

    return [clip_1, clip_2]


def save_colorbar(colormap, clim, label, fname, label_y):
    fig = plt.figure(figsize=(5, 5))
    plt.rc("font", size=18)
    image = plt.imshow(np.array([[clim[0], clim[1]]]), cmap=colormap)
    plt.gca().set_visible(False)

    cax = plt.axes([0.1, 0.2, 1.15, 0.06])
    cb = plt.colorbar(image, orientation="horizontal", cax=cax)
    cb.ax.set_title(label, fontsize=18, x=0.5, y=label_y)

    fig.savefig(os.path.join(output_dir, f"{fname}_cbhorz.pdf"), dpi=150, bbox_inches="tight")
    if IS_INTERACTIVE and display is not None:
        display(fig)
    plt.close(fig)


def make_plotter(off_screen, window_size=None):
    kwargs = {"off_screen": off_screen}
    if window_size is not None:
        kwargs["window_size"] = window_size

    plotter = pv.Plotter(**kwargs)
    plotter.image_scale = 3.5
    plotter.set_background("white")
    return plotter


def configure_camera(plotter):
    plotter.camera_position = cpos
    plotter.render()
    plotter.camera.zoom(1.4)


def add_field_meshes(plotter, work, scalars, colormap, clim):
    for clipped in clip_grid(work, clip_angle):
        plotter.add_mesh(
            clipped,
            scalars=scalars,
            cmap=colormap,
            clim=clim,
            edge_color="k",
            show_edges=False,
            show_scalar_bar=False,
        )


def add_mesh_scene(plotter):
    for clipped in clip_grid(grid, clip_angle, crinkle=True):
        plotter.add_mesh(
            clipped,
            color="white",
            edge_color="black",
            show_edges=True,
            show_scalar_bar=False,
        )


def save_field_plot(field_name, png_name, colormap, clim, cb_label, cb_name, label_y, vector=False):
    work = grid.copy(deep=True)

    if vector:
        values = np.asarray(work.point_data[field_name], dtype=np.float64)
        work.point_data[f"{field_name}_mag"] = np.linalg.norm(values, axis=1)
        scalars = f"{field_name}_mag"
    else:
        scalars = field_name

    if IS_INTERACTIVE:
        display_plotter = make_plotter(off_screen=False)
        add_field_meshes(display_plotter, work, scalars, colormap, clim)
        configure_camera(display_plotter)
        display_plotter.show(auto_close=False)

    save_plotter = make_plotter(
        off_screen=True,
        window_size=SCREENSHOT_WINDOW_SIZE,
    )
    add_field_meshes(save_plotter, work, scalars, colormap, clim)
    configure_camera(save_plotter)
    save_plotter.screenshot(os.path.join(output_dir, png_name))
    save_plotter.close()

    save_colorbar(colormap, clim, cb_label, cb_name, label_y)

# %% [markdown]
# ### Mesh Plot

# %%
if IS_INTERACTIVE:
    mesh_display_plotter = make_plotter(off_screen=False)
    add_mesh_scene(mesh_display_plotter)
    configure_camera(mesh_display_plotter)
    mesh_display_plotter.show(auto_close=False)

mesh_save_plotter = make_plotter(
    off_screen=True,
    window_size=SCREENSHOT_WINDOW_SIZE,
)
add_mesh_scene(mesh_save_plotter)
configure_camera(mesh_save_plotter)
mesh_save_plotter.screenshot(os.path.join(output_dir, "mesh.png"))
mesh_save_plotter.close()

# %% [markdown]
# ### Field Plots

# %%
save_field_plot("V_a", "vel_ana.png", cmc.lapaz.resampled(21), limits["velocity"], "Velocity", "v_ana", -2.05, vector=True)
save_field_plot("P_a", "p_ana.png", cmc.vik.resampled(41), limits["pressure"], "Pressure", "p_ana", -2.0)
save_field_plot("RHO_plot", "rho_ana.png", cmc.roma.resampled(31), limits["rho"], "Rho", "rho_ana", -2.0)

save_field_plot("V_u", "vel_uw.png", cmc.lapaz.resampled(21), limits["velocity"], "Velocity", "v_uw", -2.05, vector=True)
save_field_plot("V_e", "vel_r_err.png", cmc.lapaz.resampled(11), limits["velocity_error"], "Velocity Error (relative)", "v_err_rel", -2.05, vector=True)
save_field_plot("V_err_pct", "vel_p_err.png", cmc.oslo_r.resampled(21), limits["velocity_pct"], "Velocity Error (%)", "v_err_perc", -2.05)

save_field_plot("P_u", "p_uw.png", cmc.vik.resampled(41), limits["pressure"], "Pressure", "p_uw", -2.0)
save_field_plot("P_e", "p_r_err.png", cmc.vik.resampled(41), limits["pressure_error"], "Pressure Error (relative)", "p_err_rel", -2.0)
save_field_plot("P_err_pct", "p_p_err.png", cmc.vik.resampled(41), limits["pressure_pct"], "Pressure Error (%)", "p_err_perc", -2.0)
