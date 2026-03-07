# %% [markdown]
# ## Thieulot Legacy Additional Analysis (PyVista Only)
#
# Boundary and radial-profile comparison between analytical and numerical fields.

# %%
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# %% [markdown]
# ### Parameters And Paths

# %%
cellsize = 1 / 64
r_i = 1.0
r_o = 2.0
k = 2
vdegree = 2
pdegree = 1
pcont = True
pcont_str = str(pcont).lower()
stokes_tol = 1e-10
vel_penalty = 2.5e8

additional_analysis = True
analysis_save_pdf = True
analysis_n_theta = 1000
analysis_radii = np.hstack(
    (
        np.linspace(
            r_i,
            np.pi / 2.0,
            7,
            endpoint=True,
        ),
        np.linspace(
            np.pi / 1.92,
            r_o - 1.0e-3,
            4,
            endpoint=True,
        ),
    )
)

# %%
output_dir = os.path.join(
    "../../output/annulus/thieulot/legacy/",
    (
        f"model_inv_lc_{int(1/cellsize)}_k_{k}_vdeg_{vdegree}_pdeg_{pdegree}"
        f"_pcont_{pcont_str}_vel_penalty_{vel_penalty:.2g}_stokes_tol_{stokes_tol:.2g}/"
    ),
)
xdmf_path = os.path.join(
    output_dir,
    "output_step_00000.xdmf",
)
h5_path = os.path.join(
    output_dir,
    "output_step_00000.h5",
)

if not os.path.isfile(h5_path):
    raise FileNotFoundError(f"Missing H5 file: {h5_path}")
if not os.path.isfile(xdmf_path):
    print(f"Warning: XDMF not found: {xdmf_path}. Using H5-only reconstruction.")

# %% [markdown]
# ### Def: Load Grid And Fields

# %%
def load_grid_and_fields(
    xdmf_file,
    h5_file,
):
    """Load solution grid and velocity/pressure point fields."""
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

        if "vertex_fields/V_u_V_u" in h5f:
            v_u = np.asarray(
                h5f["vertex_fields/V_u_V_u"],
                dtype=np.float64,
            )
        else:
            v_u = np.asarray(
                h5f["fields/V_u"],
                dtype=np.float64,
            )

        if "vertex_fields/P_u_P_u" in h5f:
            p_u = np.asarray(
                h5f["vertex_fields/P_u_P_u"],
                dtype=np.float64,
            )
        else:
            p_u = np.asarray(
                h5f["fields/P_u"],
                dtype=np.float64,
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

    p_u = p_u.reshape(-1)
    if grid.n_points != v_u.shape[0]:
        raise ValueError(
            f"Velocity size mismatch: grid={grid.n_points}, field={v_u.shape[0]}"
        )
    if grid.n_points != p_u.shape[0]:
        raise ValueError(
            f"Pressure size mismatch: grid={grid.n_points}, field={p_u.shape[0]}"
        )

    v_u3 = np.c_[v_u[:, :2], np.zeros(v_u.shape[0], dtype=np.float64)]
    grid.point_data["V_u"] = v_u3
    grid.point_data["P_u"] = p_u

    return grid

# %% [markdown]
# ### Def: Analytical Solution

# %%
def analytical_solution(
    points_xy,
    r_inner,
    r_outer,
    wavemode,
    C=-1.0,
    rho0=0.0,
):
    """Return analytical velocity, pressure, and density on XY points."""
    x = points_xy[:, 0]
    y = points_xy[:, 1]

    r = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)

    denom = (r_outer * r_outer) * np.log(r_inner) - (r_inner * r_inner) * np.log(r_outer)
    A = -C * (2.0 * (np.log(r_inner) - np.log(r_outer)) / denom)
    B = -C * ((r_outer * r_outer - r_inner * r_inner) / denom)

    log_r = np.log(r)
    f = A * r + B / r
    g = 0.5 * A * r + (B / r) * log_r + C / r

    f_r = A - B / (r * r)
    g_r = 0.5 * A + B * (1.0 - log_r) / (r * r) - C / (r * r)
    g_rr = B * (2.0 * log_r - 3.0) / (r * r * r) + 2.0 * C / (r * r * r)

    h = (2.0 * g - f) / r
    m = g_rr - g_r / r - g * (wavemode * wavemode - 1.0) / (r * r) + f / (r * r) + f_r / r

    sin_kth = np.sin(wavemode * th)
    cos_kth = np.cos(wavemode * th)

    v_r = g * wavemode * sin_kth
    v_th = f * cos_kth

    v_x = v_r * np.cos(th) - v_th * np.sin(th)
    v_y = v_r * np.sin(th) + v_th * np.cos(th)

    v_ana = np.c_[v_x, v_y, np.zeros_like(v_x)]
    p_ana = wavemode * h * sin_kth + rho0 * (r_outer - r)
    rho_ana = m * wavemode * sin_kth + rho0

    return v_ana, p_ana, rho_ana

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

    grid.point_data["theta"] = theta
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
# ### Def: Analytical f(r), g(r)

# %%
def analytical_fg(
    r_values,
    r_inner,
    r_outer,
    C=-1.0,
):
    """Analytical f(r), g(r) used for RMS velocity comparison."""
    denom = (r_outer * r_outer) * np.log(r_inner) - (r_inner * r_inner) * np.log(r_outer)
    A = -C * (2.0 * (np.log(r_inner) - np.log(r_outer)) / denom)
    B = -C * ((r_outer * r_outer - r_inner * r_inner) / denom)
    f = A * r_values + B / r_values
    g = 0.5 * A * r_values + (B / r_values) * np.log(r_values) + C / r_values
    return f, g

# %% [markdown]
# ### Load Data And Build Fields

# %%
grid = load_grid_and_fields(
    xdmf_path,
    h5_path,
)
points_xy = np.asarray(
    grid.points[:, :2],
    dtype=np.float64,
)

v_ana, p_ana, rho_ana = analytical_solution(
    points_xy,
    r_i,
    r_o,
    k,
)
v_u = np.asarray(
    grid.point_data["V_u"],
    dtype=np.float64,
)
p_u = np.asarray(
    grid.point_data["P_u"],
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

    p_ana_lower = p_ana[lower_idx].reshape(-1, 1)
    p_ana_upper = p_ana[upper_idx].reshape(-1, 1)
    p_u_lower = p_u[lower_idx].reshape(-1, 1)
    p_u_upper = p_u[upper_idx].reshape(-1, 1)

    v_ana_lower = v_ana[lower_idx, :2]
    v_ana_upper = v_ana[upper_idx, :2]
    v_u_lower = v_u[lower_idx, :2]
    v_u_upper = v_u[upper_idx, :2]

    p_data_list = [
        np.hstack((np.c_[lower_theta], p_ana_lower)),
        np.hstack((np.c_[upper_theta], p_ana_upper)),
        np.hstack((np.c_[lower_theta], p_u_lower)),
        np.hstack((np.c_[upper_theta], p_u_upper)),
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
        ylim=[-2.5, 2.5],
        mod_xticks=True,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="boundary_pressure",
    )

    v_ana_lower_mag = get_magnitude(v_ana_lower)
    v_ana_upper_mag = get_magnitude(v_ana_upper)
    v_u_lower_mag = get_magnitude(v_u_lower)
    v_u_upper_mag = get_magnitude(v_u_upper)

    vmag_data_list = [
        np.hstack((np.c_[lower_theta], v_ana_lower_mag)),
        np.hstack((np.c_[upper_theta], v_ana_upper_mag)),
        np.hstack((np.c_[lower_theta], v_u_lower_mag)),
        np.hstack((np.c_[upper_theta], v_u_upper_mag)),
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
        ylim=[0.0, 2.5],
        mod_xticks=True,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="boundary_velocity_magnitude",
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
        vr_center = np.asarray(
            center_probe.point_data["V_u_r"],
            dtype=np.float64,
        )
        vth_center = np.asarray(
            center_probe.point_data["V_u_th"],
            dtype=np.float64,
        )
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
    vr_rms = np.zeros_like(
        analysis_radii,
        dtype=np.float64,
    )
    vth_rms = np.zeros_like(
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
            vr_rms[i] = np.sqrt(
                np.trapz((vr_s[vr_mask] ** 2), x=theta_s[vr_mask]) / (2.0 * np.pi)
            )
        else:
            vr_avg[i] = np.nan
            vr_rms[i] = np.nan

        if np.count_nonzero(vth_mask) > 1:
            vth_avg[i] = np.trapz(vth_s[vth_mask], x=theta_s[vth_mask]) / (2.0 * np.pi)
            vth_rms[i] = np.sqrt(
                np.trapz((vth_s[vth_mask] ** 2), x=theta_s[vth_mask]) / (2.0 * np.pi)
            )
        else:
            vth_avg[i] = np.nan
            vth_rms[i] = np.nan

    f_r, g_r = analytical_fg(
        analysis_radii,
        r_i,
        r_o,
    )
    vr_rms_ana = k * np.abs(g_r) / np.sqrt(2.0)
    vth_rms_ana = np.abs(f_r) / np.sqrt(2.0)

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
        fname="vr_average",
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
        ylim=[0.0, 2.1],
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
        ylim=[0.0, 2.0],
        mod_xticks=False,
        save_pdf=analysis_save_pdf,
        output_path=output_dir,
        fname="vtheta_rms",
    )

    print("Additional analysis complete.")
