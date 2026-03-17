# %% [markdown]
# ## Thieulot Legacy Post-Processing (PyVista Only)
#
# Uses only `pyvista` + `h5py` + `numpy` + `matplotlib`.

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

# %% [markdown]
# ### Parameters And Paths

# %%
dirname = f'model_inv_lc_64_k_2_vdeg_2_pdeg_1_pcont_true_vel_penalty_2.5e+08_stokes_tol_1e-10_ncpus_1'

# %%
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
output_dir = os.path.join(repo_root, "output", "annulus", "thieulot", "legacy", dirname)

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
)

m = re.search(pattern, dirname)
params = m.groupdict()

# %%
# convert types
inv_lc = int(params["inv_lc"])
cellsize = 1 / inv_lc

k = int(params["k"])
vdegree = int(params["vdeg"])
pdegree = int(params["pdeg"])
pcont = params["pcont"] == "true"

vel_penalty = float(params["vel_penalty"])
stokes_tol = float(params["stokes_tol"])
ncpus = int(params["ncpus"])

# constants
r_i = 1.0
r_o = 2.0

# %%
xdmf_path = os.path.join(output_dir, "output_step_00000.xdmf")
h5_path = os.path.join(output_dir, "output_step_00000.h5")

if not os.path.isfile(h5_path):
    raise FileNotFoundError(f"Missing H5 file: {h5_path}")
if not os.path.isfile(xdmf_path):
    print(f"Warning: XDMF not found: {xdmf_path}. Using H5-only reconstruction.")

# %%
# Performance control for arrows only.
arrow_target = 1400
plot_size = (750, 750)

# %%
def load_grid_and_fields(
    xdmf_file,
    h5_file,
):
    """Load solution grid and fields (try XDMF, fallback to H5 mesh/topology)."""
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
        vertices = np.asarray(h5f["geometry/vertices"], dtype=np.float64)
        triangles = np.asarray(h5f["viz/topology/cells"], dtype=np.int64)

        if "vertex_fields/V_u_V_u" in h5f:
            v_u = np.asarray(h5f["vertex_fields/V_u_V_u"], dtype=np.float64)
        else:
            v_u = np.asarray(h5f["fields/V_u"], dtype=np.float64)

        if "vertex_fields/P_u_P_u" in h5f:
            p_u = np.asarray(h5f["vertex_fields/P_u_P_u"], dtype=np.float64)
        else:
            p_u = np.asarray(h5f["fields/P_u"], dtype=np.float64)

    if grid is None:
        points3 = np.c_[vertices, np.zeros(vertices.shape[0], dtype=np.float64)]
        n_cells = triangles.shape[0]
        cells = np.hstack([np.full((n_cells, 1), 3, dtype=np.int64), triangles]).ravel()
        celltypes = np.full(n_cells, pv.CellType.TRIANGLE, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cells, celltypes, points3)

    p_u = p_u.reshape(-1)
    if grid.n_points != v_u.shape[0]:
        raise ValueError(
            f"Velocity size mismatch: grid={grid.n_points}, field={v_u.shape[0]}"
        )
    if grid.n_points != p_u.shape[0]:
        raise ValueError(
            f"Pressure size mismatch: grid={grid.n_points}, field={p_u.shape[0]}"
        )

    v_u2 = v_u[:, :2]
    v_u3 = np.c_[v_u2, np.zeros(v_u2.shape[0], dtype=np.float64)]
    grid.point_data["V_u"] = v_u3
    grid.point_data["P_u"] = p_u

    return grid

# %%
def analytical_solution(
    points_xy,
    r_i,
    r_o,
    k,
    C=-1.0,
    rho0=0.0,
):
    """Return analytical velocity, pressure, and density on XY points."""

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y, x)

    denom = (r_o**2) * np.log(r_i) - (r_i**2) * np.log(r_o)

    A = -C * (2.0 * (np.log(r_i) - np.log(r_o)) / denom)
    B = -C * ((r_o**2 - r_i**2) / denom)

    log_r = np.log(r)

    f = A * r + B / r
    g = 0.5 * A * r + (B / r) * log_r + C / r

    f_r = A - B / r**2
    g_r = 0.5 * A + B * (1.0 - log_r) / r**2 - C / r**2
    g_rr = B * (2.0 * log_r - 3.0) / r**3 + 2.0 * C / r**3

    h = (2.0 * g - f) / r
    m = g_rr - g_r / r - g * (k**2 - 1.0) / r**2 + f / r**2 + f_r / r

    sin_kth = np.sin(k * th)
    cos_kth = np.cos(k * th)

    v_r = g * k * sin_kth
    v_th = f * cos_kth

    cos_th = np.cos(th)
    sin_th = np.sin(th)

    v_x = v_r * cos_th - v_th * sin_th
    v_y = v_r * sin_th + v_th * cos_th

    v_ana = np.c_[v_x, v_y, np.zeros_like(v_x)]
    p_ana = k * h * sin_kth + rho0 * (r_o - r)
    rho_ana = m * k * sin_kth + rho0

    return v_ana, p_ana, rho_ana

# %%
def uniform_arrow_indices(
    points_xy,
    n_target,
):
    """Uniform XY bin sampling: one arrow per occupied bin."""
    pts = np.asarray(points_xy)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.array([], dtype=int)

    n_side = max(2, int(np.sqrt(max(1, n_target))))

    xmin = np.min(pts[:, 0])
    xmax = np.max(pts[:, 0])
    ymin = np.min(pts[:, 1])
    ymax = np.max(pts[:, 1])

    dx = (xmax - xmin) / n_side if xmax > xmin else 1.0
    dy = (ymax - ymin) / n_side if ymax > ymin else 1.0

    chosen = {}
    for i, (xv, yv) in enumerate(pts[:, :2]):
        ix = min(n_side - 1, int((xv - xmin) / (dx + 1e-15)))
        iy = min(n_side - 1, int((yv - ymin) / (dy + 1e-15)))
        key = (ix, iy)
        if key not in chosen:
            chosen[key] = i

    return np.array(list(chosen.values()), dtype=int)

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
    """Save a standalone colorbar image.

    Uses either `cb_bounds` or `[vmin, vmax]` for scale, supports vertical
    and horizontal layouts, and writes to `{output_path}{fname}_cb*.{fformat}`.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize_cb)
    plt.rc("font", size=primary_fs)  # Set font size
    if cb_bounds is not None:
        bounds_np = np.array([cb_bounds])
        img = plt.imshow(bounds_np, cmap=colormap)
    else:
        v_min_max_np = np.array([[vmin, vmax]])
        img = plt.imshow(v_min_max_np, cmap=colormap)

    plt.gca().set_visible(False)

    if cb_orient == "vertical":
        cax = plt.axes([0.1, 0.2, 0.06, 1.15])
        cb = plt.colorbar(orientation="vertical", cax=cax)
        cb.ax.set_title(
            cb_axis_label,
            fontsize=primary_fs,
            x=cb_label_xpos,
            y=cb_label_ypos,
            rotation=90,
        )
        plt.savefig(
            f"{output_path}{fname}_cbvert.{fformat}", dpi=150, bbox_inches="tight"
        )

    elif cb_orient == "horizontal":
        cax = plt.axes([0.1, 0.2, 1.15, 0.06])
        cb = plt.colorbar(orientation="horizontal", cax=cax)
        cb.ax.set_title(
            cb_axis_label, fontsize=primary_fs, x=cb_label_xpos, y=cb_label_ypos
        )
        plt.savefig(
            f"{output_path}{fname}_cbhorz.{fformat}", dpi=150, bbox_inches="tight"
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
    n_arrows=None,
):
    """Legacy-style vector plot, with uniform arrow sampling."""
    vec = np.asarray(grid.point_data[vector_name], dtype=np.float64)
    vec_mag = np.linalg.norm(vec[:, :2], axis=1)

    work = grid.copy(deep=True)
    mag_name = f"{vector_name}_mag"
    work.point_data[mag_name] = vec_mag

    pl = pv.Plotter(window_size=window_size, off_screen=bool(save_png))
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
        target = n_arrows
        if target is None:
            target = max(10, work.n_points // max(1, vfreq))
        idx = uniform_arrow_indices(
            work.points[:, :2],
            target,
        )
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

    pl = pv.Plotter(window_size=window_size, off_screen=bool(save_png))
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
        pl.add_text(title, font_size=18)

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
    """Plot scalar field and save a matching horizontal colorbar."""
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
    n_arrows,
    cb_label_ypos=-2.05,
):
    """Plot vector field and save a matching horizontal colorbar."""
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
        n_arrows=n_arrows,
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
# ### Load Grid And Build Derived Fields

# %%
grid = load_grid_and_fields(xdmf_path, h5_path)
points_xy = np.asarray(grid.points[:, :2], dtype=np.float64)

v_ana, p_ana, rho_ana = analytical_solution(points_xy, r_i, r_o, k)

grid.point_data["V_ana"] = v_ana
grid.point_data["P_ana"] = p_ana
grid.point_data["Rho_ana"] = rho_ana

v_u = np.asarray(grid.point_data["V_u"], dtype=np.float64)
p_u = np.asarray(grid.point_data["P_u"], dtype=np.float64).reshape(-1)

v_err = v_u - v_ana
p_err = p_u - p_ana

grid.point_data["V_err"] = v_err
grid.point_data["P_err"] = p_err

v_ana_mag = np.linalg.norm(v_ana[:, :2], axis=1)
v_err_mag = np.linalg.norm(v_err[:, :2], axis=1)

with np.errstate(divide="ignore", invalid="ignore"):
    v_err_pct = np.where(v_ana_mag > 1.0e-14, (v_err_mag / v_ana_mag) * 100.0, 0.0)
    p_err_pct = np.where(np.abs(p_ana) > 1.0e-14, (p_err / p_ana) * 100.0, 0.0)

grid.point_data["V_err_pct"] = np.nan_to_num(v_err_pct)
grid.point_data["P_err_pct"] = np.nan_to_num(p_err_pct)

# %% [markdown]
# ### Plot Analytical Velocity

# %%
vel_ana_png = plot_vector_with_colorbar(
    grid,
    vector_name="V_ana",
    png_name="vel_ana.png",
    cmap=cmc.lapaz.resampled(11),
    clim=[0.0, 2.5],
    cb_label="Velocity",
    cb_name="v_ana",
    vmag=1.0e-1,
    vfreq=40,
    show_arrows=False,
    n_arrows=arrow_target,
    cb_label_ypos=-2.05,
)

# %% [markdown]
# ### Plot Analytical Pressure

# %%
p_ana_png = plot_scalar_with_colorbar(
    grid,
    scalar_name="P_ana",
    png_name="p_ana.png",
    cmap=cmc.vik.resampled(41),
    clim=[-8.5, 8.5],
    cb_label="Pressure",
    cb_name="p_ana",
)

# %% [markdown]
# ### Plot Analytical Density

# %%
grid.point_data["Rho_ana_neg"] = -np.asarray(
    grid.point_data["Rho_ana"],
)
rho_ana_png = plot_scalar_with_colorbar(
    grid,
    scalar_name="Rho_ana_neg",
    png_name="rho_ana.png",
    cmap=cmc.roma.resampled(31),
    clim=[-67.5, 67.5],
    cb_label="Rho",
    cb_name="rho_ana",
)

# %% [markdown]
# ### Plot Solution Velocity

# %%
vel_u_png = plot_vector_with_colorbar(
    grid,
    vector_name="V_u",
    png_name="vel_uw.png",
    cmap=cmc.lapaz.resampled(11),
    clim=[0.0, 2.5],
    cb_label="Velocity",
    cb_name="v_uw",
    vmag=1.0e-1,
    vfreq=40,
    show_arrows=True,
    n_arrows=arrow_target,
    cb_label_ypos=-2.05,
)

# %% [markdown]
# ### Plot Relative Velocity Error Vector

# %%
vel_err_png = plot_vector_with_colorbar(
    grid,
    vector_name="V_err",
    png_name="vel_r_err.png",
    cmap=cmc.lapaz.resampled(11),
    clim=[0.0, 2.5e-4],
    cb_label="Velocity Error (relative)",
    cb_name="v_err_rel",
    vmag=10.0,
    vfreq=20,
    show_arrows=False,
    n_arrows=max(80, arrow_target // 2),
    cb_label_ypos=-2.05,
)

# %% [markdown]
# ### Plot Velocity Error (%)

# %%
vel_pct_png = plot_scalar_with_colorbar(
    grid,
    scalar_name="V_err_pct",
    png_name="vel_p_err.png",
    cmap=cmc.oslo_r.resampled(21),
    clim=[0.0, 1.0],
    cb_label="Velocity Error (%)",
    cb_name="v_err_perc",
    cb_label_ypos=-2.05,
)

# %% [markdown]
# ### Plot Solution Pressure

# %%
p_u_png = plot_scalar_with_colorbar(
    grid,
    scalar_name="P_u",
    png_name="p_uw.png",
    cmap=cmc.vik.resampled(41),
    clim=[-8.5, 8.5],
    cb_label="Pressure",
    cb_name="p_uw",
)

# %% [markdown]
# ### Plot Relative Pressure Error

# %%
p_err_png = plot_scalar_with_colorbar(
    grid,
    scalar_name="P_err",
    png_name="p_r_err.png",
    cmap=cmc.vik.resampled(41),
    clim=[-0.006, 0.006],
    cb_label="Pressure Error (relative)",
    cb_name="p_err_rel",
)

# %% [markdown]
# ### Plot Pressure Error (%)

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
