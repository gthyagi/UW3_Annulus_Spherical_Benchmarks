# %% [markdown]
# ## Thieulot Latest Additional Analysis (PyVista Only)
#
# Boundary and radial-profile comparison between analytical and numerical
# fields from the latest annulus checkpoint output.

# %%
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# %% [markdown]
# ### Parameters And Paths

# %%
dirname = "model_inv_lc_2_k_2_vdeg_2_pdeg_1_pcont_true_vel_penalty_2.5e+08_stokes_tol_1e-10_ncpus_4_bc_natural"

# %%
pattern = (
    r"inv_lc_(?P<inv_lc>\d+)_"
    r"k_(?P<k>\d+)_"
    r"vdeg_(?P<vdeg>\d+)_"
    r"pdeg_(?P<pdeg>\d+)_"
    r"pcont_(?P<pcont>true|false)_"
    r"vel_penalty_(?P<vel_penalty>[0-9.eE+\-]+)_"
    r"stokes_tol_(?P<stokes_tol>[0-9.eE+\-]+)_"
    r"ncpus_(?P<ncpus>\d+)"
    r"(?:_bc_(?P<bc_type>\w+))?"
)

match = re.search(pattern, dirname)
if match is None:
    raise ValueError(f"Could not parse dirname: {dirname}")

params = match.groupdict()

# %%
inv_lc = int(params["inv_lc"])
cellsize = 1.0 / inv_lc

k = int(params["k"])
vdegree = int(params["vdeg"])
pdegree = int(params["pdeg"])
pcont = params["pcont"] == "true"
pcont_str = str(pcont).lower()

vel_penalty = float(params["vel_penalty"])
stokes_tol = float(params["stokes_tol"])
ncpus = int(params["ncpus"])
bc_type = params["bc_type"]

r_i = 1.0
r_o = 2.0

additional_analysis = True
analysis_save_pdf = True
analysis_n_theta = 1000
analysis_radii = np.linspace(r_i, r_o - 1.0e-3, 10)

# %%
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
output_dir = os.path.join(repo_root, "output", "annulus", "thieulot", "latest", dirname)


# %%
def resolve_checkpoint_paths(output_path):
    """Pick an available latest checkpoint naming scheme."""

    preferred = (
        (
            os.path.join(output_path, "output.mesh.00000.xdmf"),
            os.path.join(output_path, "output.mesh.00000.h5"),
            os.path.join(output_path, "output.mesh.Velocity.00000.h5"),
            os.path.join(output_path, "output.mesh.Pressure.00000.h5"),
        ),
        (
            os.path.join(output_path, "output_step_00000.xdmf"),
            os.path.join(output_path, "output_step_00000.h5"),
            None,
            None,
        ),
    )

    for xdmf_file, mesh_h5_file, velocity_h5_file, pressure_h5_file in preferred:
        if os.path.isfile(mesh_h5_file):
            return xdmf_file, mesh_h5_file, velocity_h5_file, pressure_h5_file

    h5_candidates = sorted(
        [name for name in os.listdir(output_path) if name.endswith(".h5")]
    )
    raise FileNotFoundError(
        f"No checkpoint mesh H5 found in {output_path}. H5 files: {h5_candidates}"
    )


# %%
xdmf_path, mesh_h5_path, velocity_h5_path, pressure_h5_path = resolve_checkpoint_paths(output_dir)
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
def normalize_name(text):
    """Normalize field names for relaxed H5 matching."""

    return text.replace("{", "").replace("}", "").replace("\\", "").lower()


# %%
def find_field_path(h5f, field_names):
    """Find the best-matching dataset path for any logical field alias."""

    for field_name in field_names:
        direct_candidates = (
            f"vertex_fields/{field_name}_{field_name}",
            f"vertex_fields/{field_name}",
            f"fields/{field_name}",
            field_name,
        )
        for candidate in direct_candidates:
            if candidate in h5f:
                return candidate

    datasets = list_h5_datasets(h5f)
    wanted = [normalize_name(name) for name in field_names]

    exact = []
    contains = []
    for dset in datasets:
        tail = dset.split("/")[-1]
        tail_norm = normalize_name(tail)
        for name_norm in wanted:
            if tail_norm == name_norm or tail_norm == f"{name_norm}_{name_norm}":
                exact.append(dset)
            elif name_norm in tail_norm:
                contains.append(dset)

    if exact:
        return sorted(set(exact), key=len)[0]
    if contains:
        return sorted(set(contains), key=len)[0]

    raise KeyError(f"Fields {field_names} not found in H5 datasets.")


# %%
def read_field(h5f, field_names, n_points):
    """Read a field array and align it to point-major storage."""

    path = find_field_path(h5f, field_names)
    array = np.asarray(h5f[path], dtype=np.float64)

    if array.ndim == 2 and array.shape[0] != n_points and array.shape[1] == n_points:
        array = array.T

    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)

    if array.shape[0] != n_points:
        raise ValueError(
            f"Field {field_names} has shape {array.shape}, expected first dim {n_points}."
        )

    return array


# %%
def read_field_from_files(mesh_h5_file, mesh_h5f, field_names, n_points, split_h5_file=None):
    """Read a field from the mesh H5 first, then fallback to a split field file."""

    mesh_read_error = None
    try:
        return read_field(mesh_h5f, field_names, n_points)
    except Exception as exc:
        mesh_read_error = exc

    if split_h5_file and os.path.isfile(split_h5_file):
        with h5py.File(split_h5_file, "r") as field_h5f:
            return read_field(field_h5f, field_names, n_points)

    if split_h5_file is None:
        split_h5_candidates = [
            mesh_h5_file.replace(".00000.h5", f".{field_name}.00000.h5")
            for field_name in field_names
        ]
    else:
        split_h5_candidates = [split_h5_file]

    raise ValueError(
        f"Could not read point field {field_names} from {mesh_h5_file}. "
        "If this is a latest serial output_step checkpoint, rerun or post-process a split "
        "checkpoint directory with output.mesh.*.h5 files."
    ) from mesh_read_error


# %%
def load_grid_and_fields(xdmf_file, mesh_h5_file, velocity_file=None, pressure_file=None):
    """Load solution grid and velocity/pressure point fields."""

    grid = None
    if os.path.isfile(xdmf_file):
        try:
            xobj = pv.read(xdmf_file)
            if isinstance(xobj, pv.MultiBlock):
                for block in xobj:
                    if isinstance(block, pv.DataSet):
                        grid = block
                        break
            elif isinstance(xobj, pv.DataSet):
                grid = xobj
        except Exception:
            grid = None

    with h5py.File(mesh_h5_file, "r") as mesh_h5f:
        vertices = np.asarray(mesh_h5f["geometry/vertices"], dtype=np.float64)
        triangles = np.asarray(mesh_h5f["viz/topology/cells"], dtype=np.int64)

        if grid is None:
            points3 = np.c_[vertices, np.zeros(vertices.shape[0], dtype=np.float64)]
            n_cells = triangles.shape[0]
            cells = np.hstack(
                [np.full((n_cells, 1), 3, dtype=np.int64), triangles]
            ).ravel()
            celltypes = np.full(n_cells, pv.CellType.TRIANGLE, dtype=np.uint8)
            grid = pv.UnstructuredGrid(cells, celltypes, points3)

        n_points = grid.n_points

        velocity = read_field_from_files(
            mesh_h5_file,
            mesh_h5f,
            ["Velocity", "V_u"],
            n_points,
            split_h5_file=velocity_file,
        )
        pressure = read_field_from_files(
            mesh_h5_file,
            mesh_h5f,
            ["Pressure", "P_u"],
            n_points,
            split_h5_file=pressure_file,
        )

    if velocity.ndim == 1:
        velocity = velocity.reshape(-1, 1)

    if velocity.shape[1] == 2:
        velocity = np.c_[velocity, np.zeros(velocity.shape[0], dtype=np.float64)]

    grid.point_data["V_u"] = np.asarray(velocity, dtype=np.float64)
    grid.point_data["P_u"] = np.asarray(pressure, dtype=np.float64).reshape(-1)

    return grid


# %%
def analytical_solution(points_xy, r_inner, r_outer, wavemode, C=-1.0, rho0=0.0):
    """Return analytical velocity, pressure, and density on XY points."""

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    denom = (r_outer * r_outer) * np.log(r_inner) - (r_inner * r_inner) * np.log(r_outer)
    A = -C * (2.0 * (np.log(r_inner) - np.log(r_outer)) / denom)
    B = -C * ((r_outer * r_outer - r_inner * r_inner) / denom)

    log_r = np.log(radius)
    f = A * radius + B / radius
    g = 0.5 * A * radius + (B / radius) * log_r + C / radius

    f_r = A - B / (radius * radius)
    g_r = 0.5 * A + B * (1.0 - log_r) / (radius * radius) - C / (radius * radius)
    g_rr = B * (2.0 * log_r - 3.0) / (radius * radius * radius) + 2.0 * C / (radius * radius * radius)

    h = (2.0 * g - f) / radius
    m = g_rr - g_r / radius - g * (wavemode * wavemode - 1.0) / (radius * radius) + f / (radius * radius) + f_r / radius

    sin_kth = np.sin(wavemode * theta)
    cos_kth = np.cos(wavemode * theta)

    v_r = g * wavemode * sin_kth
    v_th = f * cos_kth

    v_x = v_r * np.cos(theta) - v_th * np.sin(theta)
    v_y = v_r * np.sin(theta) + v_th * np.cos(theta)

    v_ana = np.c_[v_x, v_y, np.zeros_like(v_x)]
    p_ana = wavemode * h * sin_kth + rho0 * (r_outer - radius)
    rho_ana = m * wavemode * sin_kth + rho0

    return v_ana, p_ana, rho_ana


# %%
def get_boundary_indices(points_xy, r_inner, r_outer, radius_tol):
    """Get lower/upper boundary point indices sorted by theta."""

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    theta[theta < 0.0] += 2.0 * np.pi

    lower_idx = np.where(np.abs(radius - r_inner) <= radius_tol)[0]
    upper_idx = np.where(np.abs(radius - r_outer) <= radius_tol)[0]

    if lower_idx.size == 0:
        lower_idx = np.argsort(np.abs(radius - r_inner))[: max(16, radius.size // 50)]
    if upper_idx.size == 0:
        upper_idx = np.argsort(np.abs(radius - r_outer))[: max(16, radius.size // 50)]

    lower_order = np.argsort(theta[lower_idx])
    upper_order = np.argsort(theta[upper_idx])

    return lower_idx[lower_order], upper_idx[upper_order], theta


# %%
def plot_stats(
    data_list=None,
    label_list=None,
    line_style=None,
    xlabel="",
    ylabel="",
    xlim=None,
    ylim=None,
    mod_xticks=False,
    save_pdf=False,
    output_path="",
    fname="",
):
    """Plot line statistics and optionally save PDF."""

    if data_list is None:
        data_list = []
    if label_list is None:
        label_list = []
    if line_style is None:
        line_style = []

    fig, ax = plt.subplots()
    for i, data in enumerate(data_list):
        ax.plot(
            data[:, 0],
            data[:, 1],
            label=label_list[i],
            linestyle=line_style[i],
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle="--")
    ax.legend(loc=(1.01, 0.60), fontsize=14)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
        if mod_xticks:
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.01, np.pi / 2.0))
            ax.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if save_pdf:
        plt.savefig(
            os.path.join(output_path, f"{fname}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

    plt.close(fig)


# %%
def get_magnitude(array_xy):
    """Compute vector magnitude from XY columns."""

    sqrd_sum = np.zeros((array_xy.shape[0], 1), dtype=np.float64)
    for i in range(array_xy.shape[1]):
        sqrd_sum += array_xy[:, i : i + 1] ** 2
    return np.sqrt(sqrd_sum)


# %%
def add_polar_velocity_fields(grid, vector_name="V_u"):
    """Add polar velocity components at mesh points."""

    pts = np.asarray(grid.points[:, :2], dtype=np.float64)
    vec = np.asarray(grid.point_data[vector_name], dtype=np.float64)

    theta = np.arctan2(pts[:, 1], pts[:, 0])
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    v_r = vec[:, 0] * cos_th + vec[:, 1] * sin_th
    v_th = -vec[:, 0] * sin_th + vec[:, 1] * cos_th

    grid.point_data["theta"] = theta
    grid.point_data["V_u_r"] = v_r
    grid.point_data["V_u_th"] = v_th


# %%
def sample_fields_on_circle(grid, radius, n_theta, field_names):
    """Sample point fields on a circle of constant radius."""

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=True)
    points = np.c_[
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros_like(theta),
    ]
    probe = pv.PolyData(points).sample(grid)

    sampled = {}
    for field_name in field_names:
        if field_name in probe.point_data:
            sampled[field_name] = np.asarray(
                probe.point_data[field_name],
                dtype=np.float64,
            ).reshape(-1)
        else:
            sampled[field_name] = np.full(theta.shape[0], np.nan, dtype=np.float64)

    return theta, sampled


# %%
def analytical_fg(r_values, r_inner, r_outer, C=-1.0):
    """Analytical f(r), g(r) used for RMS velocity comparison."""

    denom = (r_outer * r_outer) * np.log(r_inner) - (r_inner * r_inner) * np.log(r_outer)
    A = -C * (2.0 * (np.log(r_inner) - np.log(r_outer)) / denom)
    B = -C * ((r_outer * r_outer - r_inner * r_inner) / denom)
    f = A * r_values + B / r_values
    g = 0.5 * A * r_values + (B / r_values) * np.log(r_values) + C / r_values
    return f, g


# %%
def finite_concat(*arrays):
    """Return finite values from one or more arrays as a single vector."""

    parts = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            parts.append(values)

    if not parts:
        return np.array([], dtype=np.float64)

    return np.concatenate(parts)


# %%
def padded_limits(*arrays, symmetric=False):
    """Return padded plot limits from finite data."""

    values = finite_concat(*arrays)
    if values.size == 0:
        return None

    if symmetric:
        vmax = float(np.max(np.abs(values)))
        vmax = max(vmax, 1.0e-12)
        return [-1.05 * vmax, 1.05 * vmax]

    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if np.isclose(vmin, vmax):
        span = max(abs(vmin), 1.0) * 0.05
        return [vmin - span, vmax + span]

    pad = 0.05 * (vmax - vmin)
    return [vmin - pad, vmax + pad]


# %% [markdown]
# ### Load Data And Build Fields

# %%
grid = load_grid_and_fields(
    xdmf_path,
    mesh_h5_path,
    velocity_file=velocity_h5_path,
    pressure_file=pressure_h5_path,
)
points_xy = np.asarray(grid.points[:, :2], dtype=np.float64)

v_ana, p_ana, rho_ana = analytical_solution(points_xy, r_i, r_o, k)
v_u = np.asarray(grid.point_data["V_u"], dtype=np.float64)
p_u = np.asarray(grid.point_data["P_u"], dtype=np.float64).reshape(-1)


# %% [markdown]
# ### Run Additional Analysis

# %%
if additional_analysis:
    boundary_tol = max(3.0 * cellsize, 1.0e-3)
    lower_idx, upper_idx, theta_all = get_boundary_indices(points_xy, r_i, r_o, boundary_tol)

    lower_theta = theta_all[lower_idx]
    upper_theta = theta_all[upper_idx]

    p_ana_lower = p_ana[lower_idx].reshape(-1, 1)
    p_ana_upper = p_ana[upper_idx].reshape(-1, 1)
    p_u_lower = p_u[lower_idx].reshape(-1, 1)
    p_u_upper = p_u[upper_idx].reshape(-1, 1)

    v_ana_lower = v_ana[lower_idx, :2]
    v_ana_upper = v_ana[upper_idx, :2]
    v_u_lower = v_u[lower_idx, :2]
    v_u_upper = v_u[upper_idx, :2]

    plot_stats(
        data_list=[
            np.hstack((np.c_[lower_theta], p_ana_lower)),
            np.hstack((np.c_[upper_theta], p_ana_upper)),
            np.hstack((np.c_[lower_theta], p_u_lower)),
            np.hstack((np.c_[upper_theta], p_u_upper)),
        ],
        label_list=[
            f"k={k} (analy.), " + r"$r=R_{1}$",
            f"k={k} (analy.), " + r"$r=R_{2}$",
            f"k={k} (UW), " + r"$r=R_{1}$",
            f"k={k} (UW), " + r"$r=R_{2}$",
        ],
        line_style=["-", "-", "--", "--"],
        xlabel=r"$\theta$",
        ylabel="Pressure",
        xlim=[0.0, 2.0 * np.pi],
        ylim=padded_limits(p_ana_lower, p_ana_upper, p_u_lower, p_u_upper, symmetric=True),
        mod_xticks=True,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="boundary_pressure",
    )

    v_ana_lower_mag = get_magnitude(v_ana_lower)
    v_ana_upper_mag = get_magnitude(v_ana_upper)
    v_u_lower_mag = get_magnitude(v_u_lower)
    v_u_upper_mag = get_magnitude(v_u_upper)

    plot_stats(
        data_list=[
            np.hstack((np.c_[lower_theta], v_ana_lower_mag)),
            np.hstack((np.c_[upper_theta], v_ana_upper_mag)),
            np.hstack((np.c_[lower_theta], v_u_lower_mag)),
            np.hstack((np.c_[upper_theta], v_u_upper_mag)),
        ],
        label_list=[
            f"k={k} (analy.), " + r"$r=R_{1}$",
            f"k={k} (analy.), " + r"$r=R_{2}$",
            f"k={k} (UW), " + r"$r=R_{1}$",
            f"k={k} (UW), " + r"$r=R_{2}$",
        ],
        line_style=["-", "-", "--", "--"],
        xlabel=r"$\theta$",
        ylabel="Velocity Magnitude",
        xlim=[0.0, 2.0 * np.pi],
        ylim=padded_limits(v_ana_lower_mag, v_ana_upper_mag, v_u_lower_mag, v_u_upper_mag),
        mod_xticks=True,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="boundary_velocity_magnitude",
    )

    add_polar_velocity_fields(grid, vector_name="V_u")

    if grid.cells.size % 4 == 0:
        tri = grid.cells.reshape(-1, 4)[:, 1:4]
        pts2d = np.asarray(grid.points[:, :2], dtype=np.float64)
        tri_pts = pts2d[tri]
        cross_z = (
            (tri_pts[:, 1, 0] - tri_pts[:, 0, 0]) * (tri_pts[:, 2, 1] - tri_pts[:, 0, 1])
            - (tri_pts[:, 1, 1] - tri_pts[:, 0, 1]) * (tri_pts[:, 2, 0] - tri_pts[:, 0, 0])
        )
        tri_area = 0.5 * np.abs(cross_z)
        tri_centers = np.mean(tri_pts, axis=1)
        center_points = np.c_[tri_centers, np.zeros(tri_centers.shape[0])]
        center_probe = pv.PolyData(center_points).sample(grid)
        vr_center = np.asarray(center_probe.point_data["V_u_r"], dtype=np.float64)
        vth_center = np.asarray(center_probe.point_data["V_u_th"], dtype=np.float64)
        v_r_int = np.sum(vr_center * tri_area)
        v_th_int = np.sum(vth_center * tri_area)
        print((1.0 / (2.0 * np.pi)) * v_r_int)
        print((1.0 / (2.0 * np.pi)) * v_th_int)

    vr_avg = np.zeros_like(analysis_radii, dtype=np.float64)
    vth_avg = np.zeros_like(analysis_radii, dtype=np.float64)
    vr_rms = np.zeros_like(analysis_radii, dtype=np.float64)
    vth_rms = np.zeros_like(analysis_radii, dtype=np.float64)

    for i, radius in enumerate(analysis_radii):
        theta_s, sampled = sample_fields_on_circle(
            grid,
            radius=radius,
            n_theta=analysis_n_theta,
            field_names=["V_u_r", "V_u_th"],
        )
        vr_s = sampled["V_u_r"]
        vth_s = sampled["V_u_th"]

        vr_mask = np.isfinite(vr_s)
        vth_mask = np.isfinite(vth_s)

        if np.count_nonzero(vr_mask) > 1:
            vr_avg[i] = np.trapz(vr_s[vr_mask], x=theta_s[vr_mask]) / (2.0 * np.pi)
            vr_rms[i] = np.sqrt(np.trapz(vr_s[vr_mask] ** 2, x=theta_s[vr_mask]) / (2.0 * np.pi))
        else:
            vr_avg[i] = np.nan
            vr_rms[i] = np.nan

        if np.count_nonzero(vth_mask) > 1:
            vth_avg[i] = np.trapz(vth_s[vth_mask], x=theta_s[vth_mask]) / (2.0 * np.pi)
            vth_rms[i] = np.sqrt(np.trapz(vth_s[vth_mask] ** 2, x=theta_s[vth_mask]) / (2.0 * np.pi))
        else:
            vth_avg[i] = np.nan
            vth_rms[i] = np.nan

    f_r, g_r = analytical_fg(analysis_radii, r_i, r_o)
    vr_rms_ana = k * np.abs(g_r) / np.sqrt(2.0)
    vth_rms_ana = np.abs(f_r) / np.sqrt(2.0)

    zero_line = np.zeros_like(analysis_radii)

    plot_stats(
        data_list=[
            np.c_[analysis_radii, zero_line],
            np.c_[analysis_radii, vr_avg],
        ],
        label_list=[f"k={k} (analy.)", f"k={k} (UW)"],
        line_style=["-", "--"],
        xlabel="r",
        ylabel=r"$<v_{r}>$",
        xlim=[r_i, r_o],
        ylim=padded_limits(zero_line, vr_avg, symmetric=True),
        mod_xticks=False,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="vr_average",
    )

    plot_stats(
        data_list=[
            np.c_[analysis_radii, zero_line],
            np.c_[analysis_radii, vth_avg],
        ],
        label_list=[f"k={k} (analy.)", f"k={k} (UW)"],
        line_style=["-", "--"],
        xlabel="r",
        ylabel=r"$<v_{\theta}>$",
        xlim=[r_i, r_o],
        ylim=padded_limits(zero_line, vth_avg, symmetric=True),
        mod_xticks=False,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="vtheta_average",
    )

    plot_stats(
        data_list=[
            np.c_[analysis_radii, vr_rms_ana],
            np.c_[analysis_radii, vr_rms],
        ],
        label_list=[f"k={k} (analy.)", f"k={k} (UW)"],
        line_style=["-", "--"],
        xlabel="r",
        ylabel=r"$<v_{r}>_{rms}$",
        xlim=[r_i, r_o],
        ylim=padded_limits(vr_rms_ana, vr_rms),
        mod_xticks=False,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="vr_rms",
    )

    plot_stats(
        data_list=[
            np.c_[analysis_radii, vth_rms_ana],
            np.c_[analysis_radii, vth_rms],
        ],
        label_list=[f"k={k} (analy.)", f"k={k} (UW)"],
        line_style=["-", "--"],
        xlabel="r",
        ylabel=r"$<v_{\theta}>_{rms}$",
        xlim=[r_i, r_o],
        ylim=padded_limits(vth_rms_ana, vth_rms),
        mod_xticks=False,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="vtheta_rms",
    )

    print("Additional analysis complete.")
