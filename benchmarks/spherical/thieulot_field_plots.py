# %% [markdown]
# ## Spherical Thieulot Latest Post-Processing
#
# Edit `dirname` below, then run this script to create field plots from the
# latest split checkpoint files.

# %%
# to fix trame issue
import nest_asyncio

nest_asyncio.apply()

# %%
import os
import re
import sys

import cmcrameri.cm as cmc
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import ticker

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
dirname = "case_inv_lc_64_m_3_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-09_ncpus_144_bc_essential"

# %%
# output_dir = os.path.join("../../output/spherical/thieulot/latest/", f"{dirname}/")
output_dir = os.path.join("/Volumes/seagate4_1/output/spherical/thieulot/latest/", f"{dirname}/")

# %%
pattern = (
    r"case_inv_lc_(?P<inv_lc>\d+)_"
    r"m_(?P<m>-?\d+)_"
    r"vdeg_(?P<vdeg>\d+)_"
    r"pdeg_(?P<pdeg>\d+)_"
    r"pcont_(?P<pcont>true|false)_"
    r"(?:vel_penalty_(?P<vel_penalty>[0-9.eE+\-]+)_)?"
    r"stokes_tol_(?P<stokes_tol>[0-9.eE+\-]+)_"
    r"(?:stokes_pen_(?P<stokes_pen>[0-9.eE+\-]+)_)?"
    r"ncpus_(?P<ncpus>\d+)"
)

match = re.search(pattern, dirname)
if match is None:
    raise ValueError(f"Could not parse dirname: {dirname}")

params = match.groupdict()

# %%
inv_lc = int(params["inv_lc"])
cellsize = 1.0 / inv_lc

m = int(params["m"])
vdegree = int(params["vdeg"])
pdegree = int(params["pdeg"])
pcont = params["pcont"] == "true"
pcont_str = str(pcont).lower()

vel_penalty = None if params["vel_penalty"] is None else float(params["vel_penalty"])
stokes_tol = float(params["stokes_tol"])
stokes_pen = None if params["stokes_pen"] is None else float(params["stokes_pen"])
ncpus = int(params["ncpus"])

r_i = 0.5
r_o = 1.0
clip_angle = 135.0
cpos = "yz"
SCREENSHOT_WINDOW_SIZE = (750, 750)

# %%
mesh_h5_path = os.path.join(output_dir, "output.mesh.00000.h5")
velocity_h5_path = os.path.join(output_dir, "output.mesh.Velocity.00000.h5")
pressure_h5_path = os.path.join(output_dir, "output.mesh.Pressure.00000.h5")

for path in (mesh_h5_path, velocity_h5_path, pressure_h5_path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing latest checkpoint file: {path}")

# %% [markdown]
# ### Analytical Solution

# %%
def thieulot_coefficients(
    m,
    r_i,
    r_o,
    gamma,
):
    """Return alpha and beta for the spherical Thieulot benchmark."""

    if m == -1:
        alpha = -gamma * (
            (r_o**3 - r_i**3)
            / ((r_o**3) * np.log(r_i) - (r_i**3) * np.log(r_o))
        )
        beta = -3.0 * gamma * (
            (np.log(r_o) - np.log(r_i))
            / ((r_i**3) * np.log(r_o) - (r_o**3) * np.log(r_i))
        )
    else:
        alpha = gamma * (m + 1) * (
            (r_i**-3 - r_o**-3) / ((r_i ** (-(m + 4))) - (r_o ** (-(m + 4))))
        )
        beta = -3.0 * gamma * (
            ((r_i ** (m + 1)) - (r_o ** (m + 1)))
            / ((r_i ** (m + 4)) - (r_o ** (m + 4)))
        )

    return alpha, beta


def thieulot_radial_functions(
    radius,
    m,
    r_i,
    r_o,
    gamma,
    mu_0,
):
    """Return f(r), g(r), h(r), and mu(r)."""

    alpha, beta = thieulot_coefficients(m, r_i, r_o, gamma)
    mu = mu_0 * radius ** (m + 1)
    f = alpha * radius ** (-(m + 3)) + beta * radius

    if m == -1:
        g = (-2.0 / radius**2) * (
            alpha * np.log(radius) + (beta / 3.0) * radius**3 + gamma
        )
        h = (2.0 / radius) * mu_0 * g
    else:
        g = (-2.0 / radius**2) * (
            (-alpha / (m + 1)) * radius ** (-(m + 1))
            + (beta / 3.0) * radius**3
            + gamma
        )
        h = ((m + 3) / radius) * mu * g

    return f, g, h, mu


def analytical_solution(
    points_xyz,
    m,
    r_i,
    r_o,
):
    """Return analytical velocity, pressure, and density at mesh points."""

    gamma = 1.0
    mu_0 = 1.0

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    radius = np.sqrt(x**2 + y**2 + z**2)
    xy_norm = np.sqrt(x**2 + y**2)
    theta = np.arccos(np.clip(z / radius, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    pole_epsilon = 1.0e-7
    pole_mask = xy_norm <= pole_epsilon * radius
    if np.any(pole_mask):
        phi[pole_mask] = np.mod(np.arctan2(y[pole_mask], x[pole_mask]), 2.0 * np.pi)
        phi[np.isnan(phi)] = 0.0
        north_mask = pole_mask & (z >= 0.0)
        south_mask = pole_mask & (z < 0.0)
        theta[north_mask] = pole_epsilon
        theta[south_mask] = np.pi - pole_epsilon

    alpha, beta = thieulot_coefficients(m, r_i, r_o, gamma)
    f, g, h, _ = thieulot_radial_functions(radius, m, r_i, r_o, gamma, mu_0)

    if m == -1:
        rho = (
            (alpha / radius**4) * (8.0 * np.log(radius) - 6.0)
            + 8.0 * beta / (3.0 * radius)
            + 8.0 * gamma / radius**4
        ) * np.cos(theta)
    else:
        rho = (
            radius**m
            * (
                2.0
                * alpha
                * radius ** (-(m + 4))
                * ((m + 3) / (m + 1))
                * (m - 1)
                - (2.0 * beta / 3.0) * (m - 1) * (m + 3)
                - m * (m + 5) * (2.0 * gamma / radius**3)
            )
        ) * np.cos(theta)

    p_ana = h * np.cos(theta)

    v_r = g * np.cos(theta)
    v_theta = f * np.sin(theta)
    v_phi = f * np.sin(theta)

    v_x = (
        v_r * np.sin(theta) * np.cos(phi)
        + v_theta * np.cos(theta) * np.cos(phi)
        - v_phi * np.sin(phi)
    )
    v_y = (
        v_r * np.sin(theta) * np.sin(phi)
        + v_theta * np.cos(theta) * np.sin(phi)
        + v_phi * np.cos(phi)
    )
    v_z = v_r * np.cos(theta) - v_theta * np.sin(theta)

    v_ana = np.c_[v_x, v_y, v_z]

    return v_ana, p_ana, rho

# %% [markdown]
# ### Read Latest Checkpoint

# %%
def read_vertex_field(
    h5_path,
    field_name,
    n_points,
):
    """Read a point field from a latest split checkpoint H5 file."""

    with h5py.File(h5_path, "r") as h5f:
        preferred = (
            f"vertex_fields/{field_name}_{field_name}",
            f"vertex_fields/{field_name}",
            f"fields/{field_name}",
            field_name,
        )

        array = None
        for path in preferred:
            if path in h5f:
                array = np.asarray(h5f[path], dtype=np.float64)
                break

    if array is None:
        raise KeyError(f"Field '{field_name}' not found in {h5_path}")

    if array.ndim == 2 and array.shape[0] != n_points and array.shape[1] == n_points:
        array = array.T

    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)

    if array.shape[0] != n_points:
        raise ValueError(
            f"Field {field_name} has shape {array.shape}, expected first dim {n_points}."
        )

    return array


with h5py.File(mesh_h5_path, "r") as h5f:
    points = np.asarray(h5f["geometry/vertices"], dtype=np.float64)
    cells = np.asarray(h5f["viz/topology/cells"], dtype=np.int64)

v_u = read_vertex_field(velocity_h5_path, "Velocity", points.shape[0])
p_u = read_vertex_field(pressure_h5_path, "Pressure", points.shape[0]).reshape(-1)

cells_flat = np.hstack(
    [np.full((cells.shape[0], 1), cells.shape[1], dtype=np.int64), cells]
).ravel()
celltypes = np.full(cells.shape[0], pv.CellType.TETRA, dtype=np.uint8)

grid = pv.UnstructuredGrid(cells_flat, celltypes, points)
grid.point_data["Velocity"] = v_u
grid.point_data["Pressure"] = p_u

# %% [markdown]
# ### Analytical And Error Fields

# %%
v_ana, p_ana, rho_ana = analytical_solution(grid.points, m, r_i, r_o)
v_err = v_u - v_ana
p_err = p_u - p_ana

grid.point_data["V_a"] = v_ana
grid.point_data["P_a"] = p_ana
grid.point_data["RHO_a"] = rho_ana
grid.point_data["V_e"] = v_err
grid.point_data["P_e"] = p_err

# %%
reference_limits = {
    -1: {
        "velocity": [0.0, 5.0],
        "pressure": [-2.5, 2.5],
        "rho": [-110.0, 110.0],
        "velocity_error": [0.0, 0.05],
        "pressure_error": [-0.5, 0.5],
    },
    3: {
        "velocity": [0.0, 20.0],
        "pressure": [-4.0, 4.0],
        "rho": [-60.0, 60.0],
        "velocity_error": [0.0, 4.0],
        "pressure_error": [-0.5, 0.5],
    },
}

if m in reference_limits:
    limits = reference_limits[m]
else:
    limits = {
        "velocity": [0.0, float(np.max(np.linalg.norm(v_u, axis=1)))],
        "pressure": [float(np.min(p_u)), float(np.max(p_u))],
        "rho": [float(np.min(rho_ana)), float(np.max(rho_ana))],
        "velocity_error": [0.0, float(np.max(np.linalg.norm(v_err, axis=1)))],
        "pressure_error": [float(np.min(p_err)), float(np.max(p_err))],
    }

# %% [markdown]
# ### Plot Helpers

# %%
def clip_grid(grid, clip_angle, crinkle=False):
    """Match the legacy two-plane clip used by the original plotting helpers."""

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

    clip_1 = grid.clip(
        origin=(0.0, 0.0, 0.0),
        normal=normal_1,
        invert=False,
        crinkle=crinkle,
    )
    clip_2 = grid.clip(
        origin=(0.0, 0.0, 0.0),
        normal=normal_2,
        invert=False,
        crinkle=crinkle,
    )

    return [clip_1, clip_2]


def save_colorbar(colormap, clim, label, fname, label_y):
    """Save a horizontal colorbar using the benchmark layout."""

    fig = plt.figure(figsize=(5, 5))
    plt.rc("font", size=18)
    image = plt.imshow(np.array([[clim[0], clim[1]]]), cmap=colormap)
    plt.gca().set_visible(False)

    cax = plt.axes([0.1, 0.2, 1.15, 0.06])
    cb = plt.colorbar(image, orientation="horizontal", cax=cax)
    cb.locator = ticker.MaxNLocator(nbins=5, min_n_ticks=3)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    cb.formatter = formatter
    cb.update_ticks()
    cb.ax.tick_params(labelsize=16)
    cb.ax.xaxis.get_offset_text().set_size(16)
    cb.ax.set_title(label, fontsize=18, x=0.5, y=label_y)

    fig.savefig(
        os.path.join(output_dir, f"{fname}_cbhorz.pdf"),
        dpi=150,
        bbox_inches="tight",
    )
    if IS_INTERACTIVE and display is not None:
        display(fig)
    plt.close(fig)


def save_vertical_colorbar(
    colormap,
    output_path,
    fname,
    cb_axis_label,
    vmin,
    vmax,
    figsize_cb=(4, 2.25),
    primary_fs=18,
    cb_label_xpos=3.7,
    cb_label_ypos=0.3,
):
    """Save a standalone vertical colorbar without displaying it."""

    fig = plt.figure(figsize=figsize_cb)
    plt.rc("font", size=primary_fs)
    image = plt.imshow(np.array([[vmin, vmax]]), cmap=colormap)
    plt.gca().set_visible(False)

    cax = plt.axes([0.1, 0.2, 0.06, 1.15])
    cb = plt.colorbar(image, orientation="vertical", cax=cax)
    cb.locator = ticker.MaxNLocator(nbins=5, min_n_ticks=3)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    cb.formatter = formatter
    cb.update_ticks()
    cb.ax.tick_params(labelsize=max(primary_fs - 2, 10))
    cb.ax.yaxis.get_offset_text().set_size(max(primary_fs - 2, 10))
    cb.ax.set_title(
        cb_axis_label,
        fontsize=primary_fs,
        x=cb_label_xpos,
        y=cb_label_ypos,
        rotation=90,
    )

    fig.savefig(
        os.path.join(output_dir, f"{fname}_cbvert.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def make_plotter(off_screen, window_size=None):
    """Create a white-background plotter for display or screenshots."""

    kwargs = {"off_screen": off_screen}
    if window_size is not None:
        kwargs["window_size"] = window_size

    plotter = pv.Plotter(**kwargs)
    plotter.image_scale = 3.5
    plotter.set_background("white")
    return plotter


def configure_camera(plotter):
    """Apply the common camera settings used by all plots."""

    plotter.camera_position = cpos
    plotter.render()
    plotter.camera.zoom(1.4)


def add_field_meshes(plotter, work, scalars, colormap, clim):
    """Add the clipped field scene to a plotter."""

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
    """Add the clipped mesh-only scene to a plotter."""

    for clipped in clip_grid(grid, clip_angle, crinkle=True):
        plotter.add_mesh(
            clipped,
            color="white",
            edge_color="black",
            show_edges=True,
            show_scalar_bar=False,
        )


def save_field_plot(
    field_name,
    png_name,
    colormap,
    clim,
    cb_label,
    cb_name,
    label_y,
    vector=False,
):
    """Save a clipped field plot."""

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
    save_vertical_colorbar(colormap, output_dir, cb_name, cb_label, clim[0], clim[1])

# %% [markdown]
# ### Analytical Function Plot

# %%
radius = np.linspace(r_i, r_o, 200)
f_r, g_r, h_r, mu_r = thieulot_radial_functions(radius, m, r_i, r_o, 1.0, 1.0)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))

fn_list = [f_r, g_r, h_r, mu_r]
ylabel_list = [r"$f(r)$", r"$g(r)$", r"$h(r)$", "Viscosity"]

for idx, ax in enumerate(axs.flatten()):
    ax.plot(radius, fn_list[idx], color="green", linewidth=1.0)
    ax.set_xlim(r_i, r_o)
    ax.grid(linewidth=0.7)
    ax.set_xlabel("r")
    ax.set_ylabel(ylabel_list[idx])
    ax.tick_params(axis="both", direction="in", pad=8)

axs[1, 1].set_yscale("log")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "analy_fns.pdf"), format="pdf", bbox_inches="tight")
plt.close(fig)

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
# ### Analytical Velocity

# %%
print("Plotting: analytical velocity")
save_field_plot("V_a", "vel_ana.png", cmc.lapaz.resampled(21), limits["velocity"], "Velocity", "v_ana", -2.05, vector=True)

# %% [markdown]
# ### Analytical Pressure

# %%
print("Plotting: analytical pressure")
save_field_plot("P_a", "p_ana.png", cmc.vik.resampled(41), limits["pressure"], "Pressure", "p_ana", -2.0)

# %% [markdown]
# ### Analytical Density

# %%
print("Plotting: analytical density")
save_field_plot("RHO_a", "rho_ana.png", cmc.roma_r.resampled(31), limits["rho"], "Rho", "rho_ana", -2.0)

# %% [markdown]
# ### Numerical Velocity

# %%
print("Plotting: numerical velocity")
save_field_plot("Velocity", "vel_uw.png", cmc.lapaz.resampled(21), limits["velocity"], "Velocity", "v_uw", -2.05, vector=True)

# %% [markdown]
# ### Absolute Velocity Error

# %%
print("Plotting: absolute velocity error")
save_field_plot("V_e", "vel_abs_err.png", cmc.lapaz.resampled(11), limits["velocity_error"], "Velocity Error (absolute)", "v_err_abs", -2.05, vector=True)

# %% [markdown]
# ### Numerical Pressure

# %%
print("Plotting: numerical pressure")
save_field_plot("Pressure", "p_uw.png", cmc.vik.resampled(41), limits["pressure"], "Pressure", "p_uw", -2.0)

# %% [markdown]
# ### Absolute Pressure Error

# %%
print("Plotting: absolute pressure error")
save_field_plot("P_e", "p_abs_err.png", cmc.vik.resampled(41), limits["pressure_error"], "Pressure Error (absolute)", "p_err_abs", -2.0)
