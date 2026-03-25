# %% [markdown]
# ## Spherical Kramer Latest Post-Processing
#
# Edit `dirname` below, then run this script to recreate the spherical Kramer
# field plots from the latest split checkpoint output.

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
dirname = "case2_inv_lc_4_l_2_m_1_k_3_vdeg_2_pdeg_1_pcont_true_vel_penalty_1e+08_stokes_tol_1e-10_ncpus_8_bc_natural"

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
    r"vel_penalty_(?P<vel_penalty>[0-9.eE+\-]+)_"
    r"stokes_tol_(?P<stokes_tol>[0-9.eE+\-]+)_"
    r"ncpus_(?P<ncpus>\d+)_"
    r"bc_(?P<bc_type>\w+)"
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
vel_penalty = float(params["vel_penalty"])
stokes_tol = float(params["stokes_tol"])
ncpus = int(params["ncpus"])
bc_type = params["bc_type"]

r_i = 1.22
r_int = 2.0
r_o = 2.22
clip_angle = 135.0
cpos = "yz"

# %%
plot_size = (750, 750)
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
v_a = read_split_field(mesh_h5_path, "V_a", points.shape[0])
p_a = read_split_field(mesh_h5_path, "P_a", points.shape[0]).reshape(-1)
v_e = read_split_field(mesh_h5_path, "V_e", points.shape[0])
p_e = read_split_field(mesh_h5_path, "P_e", points.shape[0]).reshape(-1)
rho_a = read_split_field(mesh_h5_path, "RHO_a", points.shape[0]).reshape(-1)

cells_flat = np.hstack([np.full((cells.shape[0], 1), cells.shape[1], dtype=np.int64), cells]).ravel()
celltypes = np.full(cells.shape[0], pv.CellType.TETRA, dtype=np.uint8)

grid = pv.UnstructuredGrid(cells_flat, celltypes, points)
grid.point_data["V_u"] = v_u
grid.point_data["P_u"] = p_u
grid.point_data["V_a"] = v_a
grid.point_data["P_a"] = p_a
grid.point_data["V_e"] = v_e
grid.point_data["P_e"] = p_e
grid.point_data["RHO_a"] = rho_a
grid.point_data["RHO_plot"] = -rho_a

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
    plt.close(fig)


def save_field_plot(field_name, png_name, colormap, clim, cb_label, cb_name, label_y, vector=False):
    work = grid.copy(deep=True)

    if vector:
        values = np.asarray(work.point_data[field_name], dtype=np.float64)
        work.point_data[f"{field_name}_mag"] = np.linalg.norm(values, axis=1)
        scalars = f"{field_name}_mag"
    else:
        scalars = field_name

    plotter = pv.Plotter(window_size=plot_size, off_screen=True)
    plotter.image_scale = 3.5
    plotter.set_background("white")

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

    plotter.camera_position = cpos
    plotter.render()
    plotter.camera.zoom(1.4)
    plotter.screenshot(os.path.join(output_dir, png_name))
    plotter.close()

    save_colorbar(colormap, clim, cb_label, cb_name, label_y)

# %% [markdown]
# ### Mesh Plot

# %%
mesh_plotter = pv.Plotter(window_size=plot_size, off_screen=True)
mesh_plotter.image_scale = 3.5
mesh_plotter.set_background("white")

for clipped in clip_grid(grid, clip_angle, crinkle=True):
    mesh_plotter.add_mesh(
        clipped,
        color="white",
        edge_color="black",
        show_edges=True,
        show_scalar_bar=False,
    )

mesh_plotter.camera_position = cpos
mesh_plotter.render()
mesh_plotter.camera.zoom(1.4)
mesh_plotter.screenshot(os.path.join(output_dir, "mesh.png"))
mesh_plotter.close()

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
