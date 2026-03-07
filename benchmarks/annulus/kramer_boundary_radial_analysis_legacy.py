# %% [markdown]
# ## Kramer Legacy Boundary/Radial Analysis (PyVista Only)
#
# Reads checkpoint output and reproduces boundary/radial comparison statistics.

# %%
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# %% [markdown]
# ### Parameters And Paths

# %%
r_o = 2.22
r_int = 2.0
r_i = 1.22

res = 16
cellsize = 1 / res

vdegree = 2
pdegree = 1
pcont = True
pcont_str = str(pcont).lower()

vel_penalty = 2.5e8
stokes_tol = 1e-10

case = "case2"
n = 2
k = 3

parallel_output = False

additional_analysis = True
analysis_save_pdf = True
analysis_n_theta = 1000
analysis_radii = np.linspace(
    r_i,
    r_o - 1.0e-3,
    11,
    endpoint=True,
)

# %%
vel_penalty_str = str("{:.1e}".format(vel_penalty))
stokes_tol_str = str("{:.1e}".format(stokes_tol))

suffix = "_parallel" if parallel_output else ""
output_dir = os.path.join(
    "../../output/annulus/kramer/legacy/",
    (
        f"{case}_n_{n}_k_{k}_res_{res}_vdeg_{vdegree}_pdeg_{pdegree}"
        f"_pcont_{pcont_str}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}{suffix}/"
    ),
)

xdmf_path = os.path.join(output_dir, "output_step_00000.xdmf")
h5_path = os.path.join(output_dir, "output_step_00000.h5")

if not os.path.isfile(h5_path):
    raise FileNotFoundError(f"Missing H5 file: {h5_path}")
if not os.path.isfile(xdmf_path):
    print(f"Warning: XDMF not found: {xdmf_path}. Using H5-only reconstruction.")

# %% [markdown]
# ### Def: H5 Dataset Listing

# %%
def list_h5_datasets(h5f):
    """Return all dataset paths in an H5 file."""
    paths = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    h5f.visititems(visitor)
    return paths

# %% [markdown]
# ### Def: Find Field Path

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

# %% [markdown]
# ### Def: Read Field

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

# %% [markdown]
# ### Def: Load Grid And Fields

# %%
def load_grid_and_fields(
    xdmf_file,
    h5_file,
):
    """Load mesh plus boundary/radial analysis fields."""
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

        v_u = read_field(
            h5f,
            "V_u",
            n_points,
        )
        p_u = read_field(
            h5f,
            "P_u",
            n_points,
        )
        v_a = read_field(
            h5f,
            "V_a",
            n_points,
        )
        p_a = read_field(
            h5f,
            "P_a",
            n_points,
        )

    if v_u.ndim == 2 and v_u.shape[1] == 2:
        v_u = np.c_[v_u, np.zeros(v_u.shape[0], dtype=np.float64)]
    if v_a.ndim == 2 and v_a.shape[1] == 2:
        v_a = np.c_[v_a, np.zeros(v_a.shape[0], dtype=np.float64)]

    grid.point_data["V_u"] = v_u
    grid.point_data["P_u"] = p_u.reshape(-1)
    grid.point_data["V_a"] = v_a
    grid.point_data["P_a"] = p_a.reshape(-1)

    return grid

# %% [markdown]
# ### Def: Boundary Indices

# %%
def get_boundary_indices(
    points_xy,
    r_inner,
    r_outer,
    radius_tol,
):
    """Get lower/upper boundary point indices sorted by theta."""
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    theta[theta < 0.0] += 2.0 * np.pi

    lower_idx = np.where(np.abs(r - r_inner) <= radius_tol)[0]
    upper_idx = np.where(np.abs(r - r_outer) <= radius_tol)[0]

    if lower_idx.size == 0:
        lower_idx = np.argsort(np.abs(r - r_inner))[: max(16, r.size // 50)]
    if upper_idx.size == 0:
        upper_idx = np.argsort(np.abs(r - r_outer))[: max(16, r.size // 50)]

    lower_order = np.argsort(theta[lower_idx])
    upper_order = np.argsort(theta[upper_idx])

    return lower_idx[lower_order], upper_idx[upper_order], theta

# %% [markdown]
# ### Def: Plot Stats

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
    ax.legend(
        loc=(1.01, 0.60),
        fontsize=14,
    )

    if xlim is not None:
        ax.set_xlim(
            xlim[0],
            xlim[1],
        )
        if mod_xticks:
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.01, np.pi / 2.0))
            ax.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    if ylim is not None:
        ax.set_ylim(
            ylim[0],
            ylim[1],
        )

    if save_pdf:
        plt.savefig(
            os.path.join(output_path, f"{fname}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

# %% [markdown]
# ### Def: Magnitude

# %%
def get_magnitude(array_xy):
    """Compute vector magnitude from XY columns."""
    sqrd_sum = np.zeros((array_xy.shape[0], 1), dtype=np.float64)
    for i in range(array_xy.shape[1]):
        sqrd_sum += array_xy[:, i : i + 1] ** 2
    return np.sqrt(sqrd_sum)

# %% [markdown]
# ### Def: Polar Components

# %%
def add_polar_velocity_fields(
    grid,
    vector_name="V_u",
):
    """Add polar velocity components at mesh points."""
    pts = np.asarray(
        grid.points[:, :2],
        dtype=np.float64,
    )
    vec = np.asarray(
        grid.point_data[vector_name],
        dtype=np.float64,
    )

    theta = np.arctan2(pts[:, 1], pts[:, 0])
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    v_r = vec[:, 0] * cos_th + vec[:, 1] * sin_th
    v_th = -vec[:, 0] * sin_th + vec[:, 1] * cos_th

    grid.point_data["V_u_r"] = v_r
    grid.point_data["V_u_th"] = v_th

# %% [markdown]
# ### Def: Sample Circle

# %%
def sample_fields_on_circle(
    grid,
    radius,
    n_theta,
    field_names,
):
    """Sample point fields on a circle of constant radius."""
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        n_theta,
        endpoint=True,
    )
    points = np.c_[
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros_like(theta),
    ]

    probe = pv.PolyData(points).sample(grid)
    sampled = {}
    for fname in field_names:
        if fname in probe.point_data:
            sampled[fname] = np.asarray(
                probe.point_data[fname],
                dtype=np.float64,
            ).reshape(-1)
        else:
            sampled[fname] = np.full(theta.shape[0], np.nan, dtype=np.float64)

    return theta, sampled

# %% [markdown]
# ### Load Data

# %%
grid = load_grid_and_fields(
    xdmf_path,
    h5_path,
)

points_xy = np.asarray(
    grid.points[:, :2],
    dtype=np.float64,
)

v_u = np.asarray(
    grid.point_data["V_u"],
    dtype=np.float64,
)
p_u = np.asarray(
    grid.point_data["P_u"],
    dtype=np.float64,
).reshape(-1)

v_a = np.asarray(
    grid.point_data["V_a"],
    dtype=np.float64,
)
p_a = np.asarray(
    grid.point_data["P_a"],
    dtype=np.float64,
).reshape(-1)

# %% [markdown]
# ### Run Additional Analysis

# %%
if additional_analysis:
    boundary_tol = max(3.0 * cellsize, 1.0e-3)
    lower_idx, upper_idx, theta_all = get_boundary_indices(
        points_xy,
        r_i,
        r_o,
        boundary_tol,
    )

    lower_theta = theta_all[lower_idx]
    upper_theta = theta_all[upper_idx]

    p_ana_lower = p_a[lower_idx].reshape(-1, 1)
    p_ana_upper = p_a[upper_idx].reshape(-1, 1)
    p_uw_lower = p_u[lower_idx].reshape(-1, 1)
    p_uw_upper = p_u[upper_idx].reshape(-1, 1)

    v_ana_lower = v_a[lower_idx, :2]
    v_ana_upper = v_a[upper_idx, :2]
    v_uw_lower = v_u[lower_idx, :2]
    v_uw_upper = v_u[upper_idx, :2]

    p_ylim = {
        "case1": [-0.75, 0.75],
        "case2": [-0.65, 0.65],
        "case3": [-0.95, 0.95],
        "case4": [-0.65, 0.65],
    }[case]

    p_data_list = [
        np.hstack((np.c_[lower_theta], p_ana_lower)),
        np.hstack((np.c_[upper_theta], p_ana_upper)),
        np.hstack((np.c_[lower_theta], p_uw_lower)),
        np.hstack((np.c_[upper_theta], p_uw_upper)),
    ]
    p_label_list = [
        f"k={k} (analy.), " + r"$r=R_{1}$",
        f"k={k} (analy.), " + r"$r=R_{2}$",
        f"k={k} (UW), " + r"$r=R_{1}$",
        f"k={k} (UW), " + r"$r=R_{2}$",
    ]
    plot_stats(
        data_list=p_data_list,
        label_list=p_label_list,
        line_style=["-", "-", "--", "--"],
        xlabel=r"$\theta$",
        ylabel="Pressure",
        xlim=[0.0, 2.0 * np.pi],
        ylim=p_ylim,
        mod_xticks=True,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="p_r_i_o",
    )

    v_ana_lower_mag = get_magnitude(v_ana_lower)
    v_ana_upper_mag = get_magnitude(v_ana_upper)
    v_uw_lower_mag = get_magnitude(v_uw_lower)
    v_uw_upper_mag = get_magnitude(v_uw_upper)

    vmag_ylim = {
        "case1": [0.0, 5e-2],
        "case2": [0.0, 4e-2],
        "case3": [-1e-10, 1e-8],
        "case4": [-1e-10, 6e-9],
    }[case]

    vmag_data_list = [
        np.hstack((np.c_[lower_theta], v_ana_lower_mag)),
        np.hstack((np.c_[upper_theta], v_ana_upper_mag)),
        np.hstack((np.c_[lower_theta], v_uw_lower_mag)),
        np.hstack((np.c_[upper_theta], v_uw_upper_mag)),
    ]
    vmag_label_list = [
        f"k={k} (analy.), " + r"$r=R_{1}$",
        f"k={k} (analy.), " + r"$r=R_{2}$",
        f"k={k} (UW), " + r"$r=R_{1}$",
        f"k={k} (UW), " + r"$r=R_{2}$",
    ]
    plot_stats(
        data_list=vmag_data_list,
        label_list=vmag_label_list,
        line_style=["-", "-", "--", "--"],
        xlabel=r"$\theta$",
        ylabel="Velocity Magnitude",
        xlim=[0.0, 2.0 * np.pi],
        ylim=vmag_ylim,
        mod_xticks=True,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="vel_r_i_o",
    )

    add_polar_velocity_fields(
        grid,
        vector_name="V_u",
    )

    if grid.cells.size % 4 == 0:
        tri = grid.cells.reshape(-1, 4)[:, 1:4]
        pts2d = np.asarray(
            grid.points[:, :2],
            dtype=np.float64,
        )
        tri_pts = pts2d[tri]
        cross_z = (
            (tri_pts[:, 1, 0] - tri_pts[:, 0, 0]) * (tri_pts[:, 2, 1] - tri_pts[:, 0, 1])
            - (tri_pts[:, 1, 1] - tri_pts[:, 0, 1]) * (tri_pts[:, 2, 0] - tri_pts[:, 0, 0])
        )
        tri_area = 0.5 * np.abs(cross_z)
        tri_centers = np.mean(
            tri_pts,
            axis=1,
        )
        center_points = np.c_[tri_centers, np.zeros(tri_centers.shape[0])]
        center_probe = pv.PolyData(center_points).sample(grid)

        vr_center = np.asarray(center_probe.point_data["V_u_r"], dtype=np.float64)
        vth_center = np.asarray(center_probe.point_data["V_u_th"], dtype=np.float64)

        v_r_int = np.sum(vr_center * tri_area)
        v_th_int = np.sum(vth_center * tri_area)
        print((1.0 / (2.0 * np.pi)) * v_r_int)
        print((1.0 / (2.0 * np.pi)) * v_th_int)

    vr_avg = np.zeros_like(
        analysis_radii,
        dtype=np.float64,
    )
    vth_avg = np.zeros_like(
        analysis_radii,
        dtype=np.float64,
    )

    for i, rval in enumerate(analysis_radii):
        theta_s, sampled = sample_fields_on_circle(
            grid,
            radius=rval,
            n_theta=analysis_n_theta,
            field_names=["V_u_r", "V_u_th"],
        )

        vr_s = sampled["V_u_r"]
        vth_s = sampled["V_u_th"]

        vr_mask = np.isfinite(vr_s)
        vth_mask = np.isfinite(vth_s)

        if np.count_nonzero(vr_mask) > 1:
            vr_avg[i] = np.trapz(vr_s[vr_mask], x=theta_s[vr_mask]) / (2.0 * np.pi)
        else:
            vr_avg[i] = np.nan

        if np.count_nonzero(vth_mask) > 1:
            vth_avg[i] = np.trapz(vth_s[vth_mask], x=theta_s[vth_mask]) / (2.0 * np.pi)
        else:
            vth_avg[i] = np.nan

    plot_stats(
        data_list=[
            np.c_[analysis_radii, np.zeros_like(analysis_radii)],
            np.c_[analysis_radii, vr_avg],
        ],
        label_list=[f"k={k} (analy.)", f"k={k} (UW)"],
        line_style=["-", "--"],
        xlabel="r",
        ylabel=r"$<v_{r}>$",
        xlim=[r_i, r_o],
        ylim=[np.nanmin(vr_avg), np.nanmax(vr_avg)],
        mod_xticks=False,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="vel_r_avg",
    )

    plot_stats(
        data_list=[
            np.c_[analysis_radii, np.zeros_like(analysis_radii)],
            np.c_[analysis_radii, vth_avg],
        ],
        label_list=[f"k={k} (analy.)", f"k={k} (UW)"],
        line_style=["-", "--"],
        xlabel="r",
        ylabel=r"$<v_{\theta}>$",
        xlim=[r_i, r_o],
        ylim=[np.nanmin(vth_avg), np.nanmax(vth_avg)],
        mod_xticks=False,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="vel_th_avg",
    )

    print("Kramer legacy boundary/radial analysis complete.")
    print(f"Loaded: {xdmf_path}")
    print(f"Loaded: {h5_path}")
