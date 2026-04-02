# %% [markdown]
# ## Kramer Legacy Field Plots (PyVista Only)
#
# Reads checkpoint output and reproduces legacy scalar/vector figures.

# %%
# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# %%
import os
import re
import cmcrameri.cm as cmc
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from types import SimpleNamespace

# %% [markdown]
# ### Parameters And Paths

# %%
dirname = f'case2_inv_lc_32_n_2_k_2_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-05_ncpus_8_bc_natural_nitsche'

# %%
output_dir = os.path.join("../../output/annulus/kramer/latest/", f'{dirname}/')

# %%
pattern = (
    r"(?P<case>case\d+)_"
    r"inv_lc_(?P<inv_lc>\d+)_"
    r"n_(?P<n>\d+)_"
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

m = re.search(pattern, dirname)
if m is None:
    raise ValueError(f"Could not parse dirname: {dirname}")
params = m.groupdict()

# %%
# parsed values
inv_lc = int(params["inv_lc"])
cellsize = 1 / inv_lc

case = params["case"]
n = int(params["n"])
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

# constants (not encoded in name)
r_o = 2.22
r_int = 2.0
r_i = 1.22

freeslip = case in ("case1", "case2")
noslip = case in ("case3", "case4")
delta_fn = case in ("case1", "case3")
smooth = case in ("case2", "case4")

# %%
plot_size = (750, 750)


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
    coeffs = coefficients_cylinder_delta_ns(Rp, Rm, rp, n, g, nu, sign) if no_slip else coefficients_cylinder_delta_fs(Rp, Rm, rp, n, g, nu, sign)
    _, _, C, D = coeffs
    return SimpleNamespace(
        n=n,
        g=g,
        nu=nu,
        ABCD=coeffs,
        G=-4 * nu * C * (n + 1),
        H=-4 * nu * D * (n - 1),
    )


def build_smooth_solution(Rp, Rm, k, n, g, nu, no_slip):
    if abs(k + 3) == n or abs(k + 1) == n:
        raise NotImplementedError(f"Smooth solution not implemented for k={k}, n={n}")
    coeffs = coefficients_cylinder_smooth_ns(Rp, Rm, k, n, g, nu) if no_slip else coefficients_cylinder_smooth_fs(Rp, Rm, k, n, g, nu)
    _, _, C, D, _ = coeffs
    F = -g * (k + 1) * Rp ** (-k) / ((k + 1) ** 2 - n**2)
    return SimpleNamespace(
        n=n,
        k=k,
        g=g,
        nu=nu,
        ABCDE=coeffs,
        G=-4 * nu * C * (n + 1),
        H=-4 * nu * D * (n - 1),
        F=F,
    )


def build_case_solutions():
    if freeslip and delta_fn:
        return (
            build_delta_solution(r_o, r_i, r_int, n, -1.0, 1.0, +1, no_slip=False),
            build_delta_solution(r_o, r_i, r_int, n, -1.0, 1.0, -1, no_slip=False),
        )
    if freeslip and smooth:
        soln = build_smooth_solution(r_o, r_i, k, n, 1.0, 1.0, no_slip=False)
        return soln, soln
    if noslip and delta_fn:
        return (
            build_delta_solution(r_o, r_i, r_int, n, -1.0, 1.0, +1, no_slip=True),
            build_delta_solution(r_o, r_i, r_int, n, -1.0, 1.0, -1, no_slip=True),
        )
    if noslip and smooth:
        soln = build_smooth_solution(r_o, r_i, k, n, 1.0, 1.0, no_slip=True)
        return soln, soln
    raise ValueError(f"Unsupported case: {case}")


def analytical_velocity_cartesian(soln, rr, tt):
    n_sol = int(soln.n)
    if hasattr(soln, "ABCD"):
        A, B, C, D = soln.ABCD
        psi_r = A * rr**n_sol + B * rr**(-n_sol) + C * rr ** (n_sol + 2) + D * rr ** (-n_sol + 2)
        dpsi_rdr = (
            A * n_sol * rr ** (n_sol - 1)
            + B * (-n_sol) * rr ** (-n_sol - 1)
            + C * (n_sol + 2) * rr ** (n_sol + 1)
            + D * (-n_sol + 2) * rr ** (-n_sol + 1)
        )
    else:
        A, B, C, D, E = soln.ABCDE
        k_sol = int(soln.k)
        psi_r = A * rr**n_sol + B * rr**(-n_sol) + C * rr ** (n_sol + 2) + D * rr ** (-n_sol + 2) + E * rr ** (k_sol + 3)
        dpsi_rdr = (
            A * n_sol * rr ** (n_sol - 1)
            + B * (-n_sol) * rr ** (-n_sol - 1)
            + C * (n_sol + 2) * rr ** (n_sol + 1)
            + D * (-n_sol + 2) * rr ** (-n_sol + 1)
            + E * (k_sol + 3) * rr ** (k_sol + 2)
        )

    u_r = -(n_sol * np.cos(n_sol * tt) * psi_r) / rr
    u_theta = np.sin(n_sol * tt) * dpsi_rdr
    v_x = u_r * np.cos(tt) - u_theta * np.sin(tt)
    v_y = u_r * np.sin(tt) + u_theta * np.cos(tt)
    return np.column_stack([v_x, v_y, np.zeros_like(v_x)])


def analytical_pressure(soln, rr, tt):
    n_sol = int(soln.n)
    p_expr = (soln.G * rr**n_sol + soln.H * rr**(-n_sol)) * np.cos(n_sol * tt)
    if hasattr(soln, "F"):
        p_expr += soln.F * rr ** (int(soln.k) + 1) * np.cos(n_sol * tt)
    return p_expr


def analytical_solution(points_xy):
    soln_above, soln_below = build_case_solutions()
    rr = np.sqrt(points_xy[:, 0] ** 2 + points_xy[:, 1] ** 2)
    tt = np.arctan2(points_xy[:, 1], points_xy[:, 0])
    mask_above = rr > r_int

    v_ana = np.empty((points_xy.shape[0], 3), dtype=np.float64)
    p_ana = np.empty(points_xy.shape[0], dtype=np.float64)

    for mask, soln in ((mask_above, soln_above), (~mask_above, soln_below)):
        if np.any(mask):
            v_ana[mask] = analytical_velocity_cartesian(soln, rr[mask], tt[mask])
            p_ana[mask] = analytical_pressure(soln, rr[mask], tt[mask])

    if delta_fn:
        rho_ana = np.cos(n * tt) * np.exp(-1e5 * ((rr - r_int) ** 2))
    else:
        rho_ana = ((rr / r_o) ** k) * np.cos(n * tt)

    return v_ana, p_ana, rho_ana

# %%
def resolve_checkpoint_paths(output_dir):
    """Pick an available checkpoint naming scheme in the output directory."""
    preferred = (
        ("output.mesh.00000.xdmf", "output.mesh.00000.h5"),
        ("output_step_00000.xdmf", "output_step_00000.h5"),
    )
    for xdmf_name, h5_name in preferred:
        xdmf_file = os.path.join(output_dir, xdmf_name)
        h5_file = os.path.join(output_dir, h5_name)
        if os.path.isfile(h5_file):
            return xdmf_file, h5_file

    h5_candidates = sorted(
        [f for f in os.listdir(output_dir) if f.endswith(".h5")]
    )
    raise FileNotFoundError(
        f"No checkpoint mesh H5 found in {output_dir}. H5 files: {h5_candidates}"
    )


# %%
xdmf_path, h5_path = resolve_checkpoint_paths(output_dir)
if not os.path.isfile(xdmf_path):
    print(f"Warning: XDMF not found: {xdmf_path}. Using H5-only reconstruction.")

# %%
def list_h5_datasets(h5f):
    """Return all dataset paths in an H5 file."""
    paths = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    h5f.visititems(visitor)
    return paths

# %%
def find_field_path(
    h5f,
    field_name,
):
    """Find best-matching dataset path for a logical field name."""
    candidates = [
        f"vertex_fields/{field_name}_{field_name}",
        f"vertex_fields/{field_name}",
        f"fields/{field_name}",
        field_name,
    ]
    for cand in candidates:
        if cand in h5f:
            return cand

    datasets = list_h5_datasets(h5f)

    def norm(txt):
        return txt.replace("{", "").replace("}", "").replace("\\", "").lower()

    fname_norm = norm(field_name)

    exact = []
    contains = []
    for dset in datasets:
        tail = dset.split("/")[-1]
        tail_norm = norm(tail)
        if tail_norm == fname_norm or tail_norm == f"{fname_norm}_{fname_norm}":
            exact.append(dset)
        elif fname_norm in tail_norm:
            contains.append(dset)

    if exact:
        return sorted(exact, key=len)[0]
    if contains:
        return sorted(contains, key=len)[0]

    raise KeyError(f"Field '{field_name}' not found in H5 datasets.")

# %%
def read_field(
    h5f,
    field_name,
    n_points,
):
    """Read field array and align shape to [N] or [N, C]."""
    path = find_field_path(
        h5f,
        field_name,
    )
    arr = np.asarray(
        h5f[path],
        dtype=np.float64,
    )

    if arr.ndim == 2 and arr.shape[0] != n_points and arr.shape[1] == n_points:
        arr = arr.T

    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)

    if arr.shape[0] != n_points:
        raise ValueError(
            f"Field {field_name} has shape {arr.shape}, expected first dim {n_points}."
        )

    return arr


# %%
def read_field_from_files(
    mesh_h5_file,
    mesh_h5f,
    field_name,
    n_points,
):
    """Read field from the mesh H5, then fallback to per-field split files."""
    try:
        return read_field(
            mesh_h5f,
            field_name,
            n_points,
        )
    except Exception:
        pass

    split_file = mesh_h5_file.replace(
        ".00000.h5",
        f".{field_name}.00000.h5",
    )
    if os.path.isfile(split_file):
        with h5py.File(split_file, "r") as field_h5f:
            return read_field(
                field_h5f,
                field_name,
                n_points,
            )

    raise KeyError(
        f"Field '{field_name}' not found in {mesh_h5_file} or split file {split_file}"
    )

# %%
def load_grid_and_fields(
    xdmf_file,
    h5_file,
):
    """Load mesh plus reconstruct analytical and error fields from checkpoint files."""
    grid = None
    if os.path.isfile(xdmf_file):
        try:
            xobj = pv.read(xdmf_file)
            if isinstance(xobj, pv.MultiBlock):
                for blk in xobj:
                    if isinstance(blk, pv.DataSet):
                        grid = blk
                        break
            elif isinstance(xobj, pv.DataSet):
                grid = xobj
        except Exception:
            grid = None

    with h5py.File(h5_file, "r") as h5f:
        vertices = np.asarray(
            h5f["geometry/vertices"],
            dtype=np.float64,
        )
        triangles = np.asarray(
            h5f["viz/topology/cells"],
            dtype=np.int64,
        )

        if grid is None:
            points3 = np.c_[vertices, np.zeros(vertices.shape[0], dtype=np.float64)]
            n_cells = triangles.shape[0]
            cells = np.hstack(
                [
                    np.full((n_cells, 1), 3, dtype=np.int64),
                    triangles,
                ]
            ).ravel()
            celltypes = np.full(
                n_cells,
                pv.CellType.TRIANGLE,
                dtype=np.uint8,
            )
            grid = pv.UnstructuredGrid(
                cells,
                celltypes,
                points3,
            )

        n_points = grid.n_points

        v_u = read_field_from_files(
            h5_file,
            h5f,
            "V_u",
            n_points,
        )
        p_u = read_field_from_files(
            h5_file,
            h5f,
            "P_u",
            n_points,
        )

    if v_u.ndim == 2 and v_u.shape[1] == 2:
        v_u = np.c_[v_u, np.zeros(v_u.shape[0], dtype=np.float64)]
    if p_u.ndim == 2 and p_u.shape[1] == 1:
        p_u = p_u.reshape(-1)

    pts = np.asarray(
        grid.points[:, :2],
        dtype=np.float64,
    )
    v_a, p_a, rho_a = analytical_solution(pts)
    v_e = np.asarray(v_u, dtype=np.float64) - v_a
    p_e = np.asarray(p_u, dtype=np.float64).reshape(-1) - p_a

    grid.point_data["V_u"] = np.asarray(v_u, dtype=np.float64)
    grid.point_data["P_u"] = np.asarray(p_u, dtype=np.float64).reshape(-1)
    grid.point_data["V_a"] = np.asarray(v_a, dtype=np.float64)
    grid.point_data["P_a"] = np.asarray(p_a, dtype=np.float64).reshape(-1)
    grid.point_data["V_e"] = np.asarray(v_e, dtype=np.float64)
    grid.point_data["P_e"] = np.asarray(p_e, dtype=np.float64).reshape(-1)
    grid.point_data["RHO_a"] = np.asarray(
        rho_a,
        dtype=np.float64,
    ).reshape(-1)

    return grid

# %%
def save_colorbar(
    colormap="",
    cb_bounds=None,
    vmin=None,
    vmax=None,
    figsize_cb=(6, 1),
    primary_fs=18,
    cb_orient="vertical",
    cb_axis_label="",
    cb_label_xpos=0.5,
    cb_label_ypos=0.5,
    fformat="png",
    output_path="",
    fname="",
):
    """Save a standalone colorbar image."""
    plt.figure(figsize=figsize_cb)
    plt.rc("font", size=primary_fs)

    if cb_bounds is not None:
        bounds_np = np.array([cb_bounds])
        plt.imshow(bounds_np, cmap=colormap)
    else:
        v_min_max_np = np.array([[vmin, vmax]])
        plt.imshow(v_min_max_np, cmap=colormap)

    plt.gca().set_visible(False)

    if cb_orient == "vertical":
        cax = plt.axes([0.1, 0.2, 0.06, 1.15])
        cb = plt.colorbar(
            orientation="vertical",
            cax=cax,
        )
        cb.ax.set_title(
            cb_axis_label,
            fontsize=primary_fs,
            x=cb_label_xpos,
            y=cb_label_ypos,
            rotation=90,
        )
        plt.savefig(
            f"{output_path}{fname}_cbvert.{fformat}",
            dpi=150,
            bbox_inches="tight",
        )

    elif cb_orient == "horizontal":
        cax = plt.axes([0.1, 0.2, 1.15, 0.06])
        cb = plt.colorbar(
            orientation="horizontal",
            cax=cax,
        )
        cb.ax.set_title(
            cb_axis_label,
            fontsize=primary_fs,
            x=cb_label_xpos,
            y=cb_label_ypos,
        )
        plt.savefig(
            f"{output_path}{fname}_cbhorz.{fformat}",
            dpi=150,
            bbox_inches="tight",
        )

    return

# %%
def plot_vector(
    grid,
    vector_name="vector",
    cmap=None,
    clim=None,
    vmag=1.0,
    vfreq=10,
    show_arrows=False,
    clip_angle=0.0,
    show_edges=False,
    cpos="xy",
    window_size=(750, 750),
    save_png=False,
    dir_fname=None,
    image_scale=3.5,
):
    """Legacy-style vector magnitude plot."""
    vec = np.asarray(
        grid.point_data[vector_name],
        dtype=np.float64,
    )
    vec_mag = np.linalg.norm(
        vec[:, :2],
        axis=1,
    )

    work = grid.copy(deep=True)
    mag_name = f"{vector_name}_mag"
    work.point_data[mag_name] = vec_mag

    pl = pv.Plotter(
        window_size=window_size,
        off_screen=bool(save_png),
    )
    pl.image_scale = image_scale

    pl.add_mesh(
        work,
        scalars=mag_name,
        cmap=cmap,
        clim=clim,
        edge_color="k",
        show_edges=show_edges,
        opacity=1.0,
        show_scalar_bar=False,
    )

    if show_arrows:
        idx = np.arange(0, work.n_points, max(1, vfreq), dtype=int)
        if idx.size:
            pl.add_arrows(
                work.points[idx],
                vec[idx],
                mag=vmag,
                color="k",
            )

    if save_png and dir_fname:
        pl.camera_position = cpos
        pl.camera.zoom(1.4)
        pl.render()
        pl.screenshot(dir_fname)
    
    pl.show(cpos=cpos)

    return dir_fname

# %%
def plot_scalar(
    grid,
    scalar_name="scalar",
    cmap=None,
    clim=None,
    window_size=(750, 750),
    title=None,
    clip_angle=0.0,
    cpos="xy",
    show_edges=False,
    save_png=False,
    dir_fname=None,
    image_scale=3.5,
):
    """Legacy-style scalar plot."""
    work = grid.copy(deep=True)

    pl = pv.Plotter(
        window_size=window_size,
        off_screen=bool(save_png),
    )
    pl.image_scale = image_scale

    pl.add_mesh(
        work,
        scalars=scalar_name,
        cmap=cmap,
        clim=clim,
        edge_color="k",
        show_edges=show_edges,
        opacity=1.0,
        show_scalar_bar=False,
    )

    if title:
        pl.add_text(
            title,
            font_size=18,
        )

    if save_png and dir_fname:
        pl.camera_position = cpos
        pl.camera.zoom(1.4)
        pl.render()
        pl.screenshot(dir_fname)
    
    pl.show(cpos=cpos)

    return dir_fname

# %%
def plot_scalar_with_colorbar(
    grid,
    scalar_name,
    png_name,
    cmap,
    clim,
    cb_label,
    cb_name,
    cb_label_ypos=-2.0,
):
    """Plot scalar field and save matching colorbar."""
    png = plot_scalar(
        grid,
        scalar_name=scalar_name,
        cmap=cmap,
        clim=clim,
        clip_angle=0.0,
        cpos="xy",
        window_size=plot_size,
        save_png=True,
        dir_fname=os.path.join(output_dir, png_name),
    )

    save_colorbar(
        colormap=cmap,
        cb_bounds=None,
        vmin=clim[0],
        vmax=clim[1],
        figsize_cb=(5, 5),
        primary_fs=18,
        cb_orient="horizontal",
        cb_axis_label=cb_label,
        cb_label_xpos=0.5,
        cb_label_ypos=cb_label_ypos,
        fformat="pdf",
        output_path=output_dir,
        fname=cb_name,
    )

    return png

# %%
def plot_vector_with_colorbar(
    grid,
    vector_name,
    png_name,
    cmap,
    clim,
    cb_label,
    cb_name,
    vmag,
    vfreq,
    show_arrows,
    cb_label_ypos=-2.05,
):
    """Plot vector field and save matching colorbar."""
    png = plot_vector(
        grid,
        vector_name=vector_name,
        cmap=cmap,
        clim=clim,
        vmag=vmag,
        vfreq=vfreq,
        show_arrows=show_arrows,
        clip_angle=0.0,
        cpos="xy",
        window_size=plot_size,
        save_png=True,
        dir_fname=os.path.join(output_dir, png_name),
    )

    save_colorbar(
        colormap=cmap,
        cb_bounds=None,
        vmin=clim[0],
        vmax=clim[1],
        figsize_cb=(5, 5),
        primary_fs=18,
        cb_orient="horizontal",
        cb_axis_label=cb_label,
        cb_label_xpos=0.5,
        cb_label_ypos=cb_label_ypos,
        fformat="pdf",
        output_path=output_dir,
        fname=cb_name,
    )

    return png

# %% [markdown]
# ### Load Fields And Build Derived Scalars

# %%
grid = load_grid_and_fields(
    xdmf_path,
    h5_path,
)

v_a_mag = np.linalg.norm(
    np.asarray(grid.point_data["V_a"])[:, :2],
    axis=1,
)
v_e_mag = np.linalg.norm(
    np.asarray(grid.point_data["V_e"])[:, :2],
    axis=1,
)

with np.errstate(divide="ignore", invalid="ignore"):
    v_err_pct = np.where(v_a_mag > 1.0e-14, (v_e_mag / v_a_mag) * 100.0, 0.0)
    p_err_pct = np.where(
        np.abs(np.asarray(grid.point_data["P_a"])) > 1.0e-14,
        (np.asarray(grid.point_data["P_e"]) / np.asarray(grid.point_data["P_a"])) * 100.0,
        0.0,
    )

grid.point_data["V_err_pct"] = np.nan_to_num(v_err_pct)
grid.point_data["P_err_pct"] = np.nan_to_num(p_err_pct)
grid.point_data["RHO_a_neg"] = -np.asarray(grid.point_data["RHO_a"]).reshape(-1)

# %% [markdown]
# ### Case Color Limits

# %%
vel_clim = {
    "case1": [0.0, 0.05],
    "case2": [0.0, 0.04],
    "case3": [0.0, 0.01],
    "case4": [0.0, 0.00925],
}
vel_err_clim = {
    "case1": [0.0, 0.005],
    "case2": [0.0, 7e-4],
    "case3": [0.0, 1e-4],
    "case4": [0.0, 1e-5],
}
vel_pct_clim = {
    "case1": [0.0, 20.0],
    "case2": [0.0, 20.0],
    "case3": [0.0, 5.0],
    "case4": [0.0, 1.0],
}
p_clim = {
    "case1": [-0.65, 0.65],
    "case2": [-0.5, 0.5],
    "case3": [-0.65, 0.65],
    "case4": [-0.5, 0.5],
}
p_err_clim = {
    "case1": [-0.065, 0.065],
    "case2": [-0.003, 0.003],
    "case3": [-0.0065, 0.0065],
    "case4": [-0.0045, 0.0045],
}

if case not in vel_clim:
    raise ValueError(f"Unknown case: {case}")

# %% [markdown]
# ### Analytical Velocity

# %%
vel_ana_png = plot_vector_with_colorbar(
    grid,
    vector_name="V_a",
    png_name="vel_ana.png",
    cmap=cmc.lapaz.resampled(11),
    clim=vel_clim[case],
    cb_label="Velocity",
    cb_name="v_ana",
    vmag=1.0,
    vfreq=75,
    show_arrows=False,
    cb_label_ypos=-2.05,
)

# %% [markdown]
# ### Analytical Pressure

# %%
p_ana_png = plot_scalar_with_colorbar(
    grid,
    scalar_name="P_a",
    png_name="p_ana.png",
    cmap=cmc.vik.resampled(41),
    clim=p_clim[case],
    cb_label="Pressure",
    cb_name="p_ana",
)

# %% [markdown]
# ### Analytical Density

# %%
rho_ana_png = plot_scalar_with_colorbar(
    grid,
    scalar_name="RHO_a_neg",
    png_name="rho_ana.png",
    cmap=cmc.roma.resampled(31),
    clim=[-1.0, 1.0],
    cb_label="Rho",
    cb_name="rho_ana",
)

# %% [markdown]
# ### Solution Velocity

# %%
vel_u_png = plot_vector_with_colorbar(
    grid,
    vector_name="V_u",
    png_name="vel_uw.png",
    cmap=cmc.lapaz.resampled(11),
    clim=vel_clim[case],
    cb_label="Velocity",
    cb_name="v_uw",
    vmag=1.0,
    vfreq=75,
    show_arrows=False,
    cb_label_ypos=-2.05,
)

# %% [markdown]
# ### Relative Velocity Error

# %%
vel_err_png = plot_vector_with_colorbar(
    grid,
    vector_name="V_e",
    png_name="vel_r_err.png",
    cmap=cmc.lapaz.resampled(11),
    clim=vel_err_clim[case],
    cb_label="Velocity Error (relative)",
    cb_name="v_err_rel",
    vmag=1.0,
    vfreq=75,
    show_arrows=False,
    cb_label_ypos=-2.05,
)

# %% [markdown]
# ### Velocity Error (%)

# %%
vel_pct_png = plot_scalar_with_colorbar(
    grid,
    scalar_name="V_err_pct",
    png_name="vel_p_err.png",
    cmap=cmc.oslo_r.resampled(21),
    clim=vel_pct_clim[case],
    cb_label="Velocity Error (%)",
    cb_name="v_err_perc",
    cb_label_ypos=-2.05,
)

# %% [markdown]
# ### Solution Pressure

# %%
p_u_png = plot_scalar_with_colorbar(
    grid,
    scalar_name="P_u",
    png_name="p_uw.png",
    cmap=cmc.vik.resampled(41),
    clim=p_clim[case],
    cb_label="Pressure",
    cb_name="p_uw",
)

# %% [markdown]
# ### Relative Pressure Error

# %%
p_err_png = plot_scalar_with_colorbar(
    grid,
    scalar_name="P_e",
    png_name="p_r_err.png",
    cmap=cmc.vik.resampled(41),
    clim=p_err_clim[case],
    cb_label="Pressure Error (relative)",
    cb_name="p_err_rel",
)

# %% [markdown]
# ### Pressure Error (%)

# %%
p_pct_png = plot_scalar_with_colorbar(
    grid,
    scalar_name="P_err_pct",
    png_name="p_p_err.png",
    cmap=cmc.vik.resampled(41),
    clim=[-100.0, 100.0],
    cb_label="Pressure Error (%)",
    cb_name="p_err_perc",
)

# %%
