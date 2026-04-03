# %% [markdown]
# ## Thieulot Latest Post-Processing (PyVista Only)
#
# Uses latest annulus checkpoint output and reconstructs analytical/error
# fields for plotting.

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
dirname = "model_inv_lc_64_k_2_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-09_ncpus_8_bc_essential"

# %%
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
output_dir = os.path.join(repo_root, "output", "annulus", "thieulot", "latest", dirname)

# %%
pattern = (
    r"inv_lc_(?P<inv_lc>\d+)_"
    r"k_(?P<k>\d+)_"
    r"vdeg_(?P<vdeg>\d+)_"
    r"pdeg_(?P<pdeg>\d+)_"
    r"pcont_(?P<pcont>true|false)_"
    r"(?:vel_penalty_(?P<vel_penalty>[0-9.eE+\-]+)_)?"
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

vel_penalty = None if params["vel_penalty"] is None else float(params["vel_penalty"])
stokes_tol = float(params["stokes_tol"])
ncpus = int(params["ncpus"])
bc_type = params["bc_type"]

r_i = 1.0
r_o = 2.0

arrow_target = 1400
plot_size = (750, 750)
SCREENSHOT_WINDOW_SIZE = (750, 750)


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
def find_field_path(h5f, field_names, association_preference=None):
    """Find the best-matching dataset path for any logical field alias."""

    for field_name in field_names:
        direct_candidates = []
        if association_preference != "cell":
            direct_candidates.extend(
                (
                    f"vertex_fields/{field_name}_{field_name}",
                    f"vertex_fields/{field_name}",
                )
            )
        if association_preference != "point":
            direct_candidates.extend(
                (
                    f"cell_fields/{field_name}_{field_name}",
                    f"cell_fields/{field_name}",
                )
            )
        direct_candidates.extend((f"fields/{field_name}", field_name))
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
def read_field(
    h5f,
    field_names,
    n_points,
    n_cells=None,
    association_preference=None,
):
    """Read a field array and detect whether it is point- or cell-associated."""

    path = find_field_path(
        h5f,
        field_names,
        association_preference=association_preference,
    )
    array = np.asarray(h5f[path], dtype=np.float64)
    valid_sizes = tuple(size for size in (n_points, n_cells) if size is not None)

    if (
        array.ndim == 2
        and array.shape[0] not in valid_sizes
        and array.shape[1] in valid_sizes
    ):
        array = array.T

    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)

    if array.shape[0] == n_points:
        return array, "point"
    if n_cells is not None and array.shape[0] == n_cells:
        return array, "cell"

    raise ValueError(
        f"Field {field_names} has shape {array.shape}, expected first dim in {valid_sizes}."
    )


# %%
def read_field_from_files(
    mesh_h5_file,
    mesh_h5f,
    field_names,
    n_points,
    n_cells=None,
    association_preference=None,
    split_h5_file=None,
):
    """Read a field from the mesh H5 first, then fallback to a split field file."""

    mesh_read_error = None
    try:
        return read_field(
            mesh_h5f,
            field_names,
            n_points,
            n_cells=n_cells,
            association_preference=association_preference,
        )
    except Exception as exc:
        mesh_read_error = exc

    if split_h5_file and os.path.isfile(split_h5_file):
        with h5py.File(split_h5_file, "r") as field_h5f:
            return read_field(
                field_h5f,
                field_names,
                n_points,
                n_cells=n_cells,
                association_preference=association_preference,
            )

    if split_h5_file is None:
        split_h5_candidates = [
            mesh_h5_file.replace(".00000.h5", f".{field_name}.00000.h5")
            for field_name in field_names
        ]
    else:
        split_h5_candidates = [split_h5_file]

    raise ValueError(
        f"Could not read field {field_names} from {mesh_h5_file}. "
        "If this is a latest serial output_step checkpoint, rerun or post-process a split "
        "checkpoint directory with output.mesh.*.h5 files."
    ) from mesh_read_error


# %%
def load_grid_and_fields(
    xdmf_file,
    mesh_h5_file,
    velocity_file=None,
    pressure_file=None,
    pressure_degree=None,
):
    """Load mesh plus velocity/pressure fields from latest checkpoint files."""

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
        n_cells = grid.n_cells
        pressure_preference = None
        if pressure_degree is not None:
            pressure_preference = "cell" if pressure_degree == 0 else "point"

        velocity, velocity_assoc = read_field_from_files(
            mesh_h5_file,
            mesh_h5f,
            ["Velocity", "V_u"],
            n_points,
            n_cells=n_cells,
            association_preference="point",
            split_h5_file=velocity_file,
        )
        pressure, pressure_assoc = read_field_from_files(
            mesh_h5_file,
            mesh_h5f,
            ["Pressure", "P_u"],
            n_points,
            n_cells=n_cells,
            association_preference=pressure_preference,
            split_h5_file=pressure_file,
        )

    if velocity.ndim == 1:
        velocity = velocity.reshape(-1, 1)

    if velocity.shape[1] == 2:
        velocity = np.c_[velocity, np.zeros(velocity.shape[0], dtype=np.float64)]

    if velocity_assoc != "point":
        raise ValueError(
            f"Velocity field is {velocity_assoc}-associated, expected point-associated data."
        )

    grid.point_data["V_u"] = np.asarray(velocity, dtype=np.float64)

    pressure_array = np.asarray(pressure, dtype=np.float64).reshape(-1)
    if pressure_assoc == "point":
        grid.point_data["P_u"] = pressure_array
    else:
        grid.cell_data["P_u"] = pressure_array

    return grid


# %%
def get_field_association(grid, field_name):
    """Return whether a named field lives on points or cells."""

    if field_name in grid.point_data:
        return "point"
    if field_name in grid.cell_data:
        return "cell"

    raise KeyError(f"Field {field_name} not found in point_data or cell_data.")


# %%
def get_field_array(grid, field_name):
    """Return a field array plus its association."""

    association = get_field_association(grid, field_name)
    source = grid.point_data if association == "point" else grid.cell_data
    return np.asarray(source[field_name], dtype=np.float64), association


# %%
def set_field_array(grid, field_name, values, association):
    """Store a field array on either points or cells."""

    target = grid.point_data if association == "point" else grid.cell_data
    target[field_name] = np.asarray(values, dtype=np.float64)


# %%
def support_points_xy(grid, association):
    """Return XY coordinates for point- or cell-associated data."""

    if association == "point":
        return np.asarray(grid.points[:, :2], dtype=np.float64)
    if association == "cell":
        return np.asarray(grid.cell_centers().points[:, :2], dtype=np.float64)

    raise ValueError(f"Unsupported field association: {association}")


# %%
def analytical_solution(points_xy, r_inner, r_outer, wavemode, C=-1.0, rho0=0.0):
    """Return analytical velocity, pressure, and density on XY points."""

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    denom = (r_outer**2) * np.log(r_inner) - (r_inner**2) * np.log(r_outer)
    A = -C * (2.0 * (np.log(r_inner) - np.log(r_outer)) / denom)
    B = -C * ((r_outer**2 - r_inner**2) / denom)

    log_r = np.log(radius)

    f = A * radius + B / radius
    g = 0.5 * A * radius + (B / radius) * log_r + C / radius

    f_r = A - B / radius**2
    g_r = 0.5 * A + B * (1.0 - log_r) / radius**2 - C / radius**2
    g_rr = B * (2.0 * log_r - 3.0) / radius**3 + 2.0 * C / radius**3

    h = (2.0 * g - f) / radius
    m = g_rr - g_r / radius - g * (wavemode**2 - 1.0) / radius**2 + f / radius**2 + f_r / radius

    sin_kth = np.sin(wavemode * theta)
    cos_kth = np.cos(wavemode * theta)

    v_r = g * wavemode * sin_kth
    v_th = f * cos_kth

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    v_x = v_r * cos_th - v_th * sin_th
    v_y = v_r * sin_th + v_th * cos_th

    v_ana = np.c_[v_x, v_y, np.zeros_like(v_x)]
    p_ana = wavemode * h * sin_kth + rho0 * (r_outer - radius)
    rho_ana = m * wavemode * sin_kth + rho0

    return v_ana, p_ana, rho_ana


# %%
def regular_arrow_points(r_inner, r_outer, n_target):
    """Return regular annulus arrow points and nearby sample points."""

    n_r = max(3, int(np.sqrt(max(1, n_target)) / 2))
    n_theta = max(8, int(max(1, n_target) / n_r))
    sample_offset = max(1.0e-6, 1.0e-3 * (r_outer - r_inner))

    radius_inner = np.array([r_inner], dtype=np.float64)
    radius_outer = np.array([r_outer], dtype=np.float64)
    radius_mid = np.linspace(r_inner, r_outer, n_r)[1:-1]
    radius_plot = np.concatenate((radius_inner, radius_mid, radius_outer))
    radius_sample = radius_plot.copy()
    radius_sample[0] = min(r_outer, r_inner + sample_offset)
    radius_sample[-1] = max(r_inner, r_outer - sample_offset)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    radius_plot_grid, theta_plot_grid = np.meshgrid(radius_plot, theta, indexing="ij")
    radius_sample_grid, theta_sample_grid = np.meshgrid(
        radius_sample, theta, indexing="ij"
    )

    x_plot = radius_plot_grid * np.cos(theta_plot_grid)
    y_plot = radius_plot_grid * np.sin(theta_plot_grid)
    x_sample = radius_sample_grid * np.cos(theta_sample_grid)
    y_sample = radius_sample_grid * np.sin(theta_sample_grid)

    plot_points = np.c_[
        x_plot.ravel(),
        y_plot.ravel(),
        np.zeros(x_plot.size, dtype=np.float64),
    ]
    sample_points = np.c_[
        x_sample.ravel(),
        y_sample.ravel(),
        np.zeros(x_sample.size, dtype=np.float64),
    ]

    return plot_points, sample_points


# %%
def save_colorbar(
    colormap,
    output_path,
    fname,
    cb_axis_label,
    vmin,
    vmax,
    figsize_cb=(5, 5),
    primary_fs=18,
    cb_label_xpos=0.5,
    cb_label_ypos=-2.0,
):
    """Save a standalone horizontal colorbar."""

    fig = plt.figure(figsize=figsize_cb)
    plt.rc("font", size=primary_fs)
    img = plt.imshow(np.array([[vmin, vmax]]), cmap=colormap)
    plt.gca().set_visible(False)

    cax = plt.axes([0.1, 0.2, 1.15, 0.06])
    cb = plt.colorbar(img, orientation="horizontal", cax=cax)
    cb.locator = ticker.MaxNLocator(nbins=5, min_n_ticks=3)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    cb.formatter = formatter
    cb.update_ticks()
    cb.ax.tick_params(labelsize=max(primary_fs - 2, 10))
    cb.ax.xaxis.get_offset_text().set_size(max(primary_fs - 2, 10))
    cb.ax.set_title(cb_axis_label, fontsize=primary_fs, x=cb_label_xpos, y=cb_label_ypos)
    fig.savefig(
        os.path.join(output_path, f"{fname}_cbhorz.pdf"),
        dpi=150,
        bbox_inches="tight",
    )
    if IS_INTERACTIVE and display is not None:
        display(fig)
    plt.close(fig)


# %%
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
    img = plt.imshow(np.array([[vmin, vmax]]), cmap=colormap)
    plt.gca().set_visible(False)

    cax = plt.axes([0.1, 0.2, 0.06, 1.15])
    cb = plt.colorbar(img, orientation="vertical", cax=cax)
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
        os.path.join(output_path, f"{fname}_cbvert.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# %%
def plot_vector(
    grid,
    vector_name,
    cmap,
    clim,
    vmag,
    show_arrows,
    window_size,
    dir_fname,
    image_scale=3.5,
    n_arrows=None,
):
    """Plot a vector magnitude field with optional arrows."""

    vector = np.asarray(grid.point_data[vector_name], dtype=np.float64)
    vector_mag = np.linalg.norm(vector[:, :2], axis=1)

    work = grid.copy(deep=True)
    mag_name = f"{vector_name}_mag"
    work.point_data[mag_name] = vector_mag

    arrow_points = None
    arrow_vectors = None
    if show_arrows:
        arrow_points, sample_points = regular_arrow_points(r_i, r_o, n_arrows)
        if arrow_points.size:
            arrow_cloud = pv.PolyData(sample_points).sample(work)
            sampled = np.asarray(arrow_cloud.point_data[vector_name], dtype=np.float64)
            valid = np.all(np.isfinite(sampled), axis=1)
            arrow_points = arrow_points[valid]
            arrow_vectors = sampled[valid]

    def add_scene(plotter):
        plotter.add_mesh(
            work,
            scalars=mag_name,
            cmap=cmap,
            clim=clim,
            edge_color="k",
            show_edges=False,
            opacity=1.0,
            show_scalar_bar=False,
        )
        if arrow_points is not None and arrow_points.size:
            plotter.add_arrows(
                arrow_points,
                arrow_vectors,
                mag=vmag,
                color="k",
            )

    if IS_INTERACTIVE:
        display_plotter = pv.Plotter(off_screen=False)
        display_plotter.image_scale = image_scale
        display_plotter.set_background("white")
        add_scene(display_plotter)
        display_plotter.camera_position = "xy"
        display_plotter.render()
        display_plotter.camera.zoom(1.4)
        display_plotter.show(jupyter_backend=JUPYTER_BACKEND, auto_close=False)

    save_plotter = pv.Plotter(window_size=window_size, off_screen=True)
    save_plotter.image_scale = image_scale
    save_plotter.set_background("white")
    add_scene(save_plotter)
    save_plotter.camera_position = "xy"
    save_plotter.render()
    save_plotter.camera.zoom(1.4)
    save_plotter.screenshot(dir_fname)
    save_plotter.close()


# %%
def plot_scalar(
    grid,
    scalar_name,
    cmap,
    clim,
    window_size,
    dir_fname,
    image_scale=3.5,
):
    """Plot a scalar field."""
    association = get_field_association(grid, scalar_name)

    def add_scene(plotter):
        plotter.add_mesh(
            grid,
            scalars=scalar_name,
            preference=association,
            cmap=cmap,
            clim=clim,
            edge_color="k",
            show_edges=False,
            opacity=1.0,
            show_scalar_bar=False,
        )

    if IS_INTERACTIVE:
        display_plotter = pv.Plotter(off_screen=False)
        display_plotter.image_scale = image_scale
        display_plotter.set_background("white")
        add_scene(display_plotter)
        display_plotter.camera_position = "xy"
        display_plotter.render()
        display_plotter.camera.zoom(1.4)
        display_plotter.show(jupyter_backend=JUPYTER_BACKEND, auto_close=False)

    save_plotter = pv.Plotter(window_size=window_size, off_screen=True)
    save_plotter.image_scale = image_scale
    save_plotter.set_background("white")
    add_scene(save_plotter)
    save_plotter.camera_position = "xy"
    save_plotter.render()
    save_plotter.camera.zoom(1.4)
    save_plotter.screenshot(dir_fname)
    save_plotter.close()


# %%
def plot_scalar_with_colorbar(grid, scalar_name, png_name, cmap, clim, cb_label, cb_name):
    """Plot a scalar field and save a matching horizontal colorbar."""

    plot_scalar(
        grid,
        scalar_name=scalar_name,
        cmap=cmap,
        clim=clim,
        window_size=SCREENSHOT_WINDOW_SIZE,
        dir_fname=os.path.join(output_dir, png_name),
    )
    save_colorbar(
        colormap=cmap,
        output_path=output_dir,
        fname=cb_name,
        cb_axis_label=cb_label,
        vmin=clim[0],
        vmax=clim[1],
    )


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
    show_arrows,
    n_arrows,
):
    """Plot a vector field and save a matching horizontal colorbar."""

    plot_vector(
        grid,
        vector_name=vector_name,
        cmap=cmap,
        clim=clim,
        vmag=vmag,
        show_arrows=show_arrows,
        window_size=SCREENSHOT_WINDOW_SIZE,
        dir_fname=os.path.join(output_dir, png_name),
        n_arrows=n_arrows,
    )
    save_colorbar(
        colormap=cmap,
        output_path=output_dir,
        fname=cb_name,
        cb_axis_label=cb_label,
        vmin=clim[0],
        vmax=clim[1],
    )


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
def positive_clim(*arrays, percentile=100.0, default=1.0):
    """Return [0, vmax] from finite data."""

    values = finite_concat(*arrays)
    if values.size == 0:
        return [0.0, default]

    vmax = float(np.percentile(values, percentile))
    if vmax <= 0.0:
        vmax = default

    return [0.0, vmax]


# %%
def symmetric_clim(*arrays, percentile=100.0, default=1.0):
    """Return [-vmax, vmax] from finite data."""

    values = finite_concat(*arrays)
    if values.size == 0:
        return [-default, default]

    vmax = float(np.percentile(np.abs(values), percentile))
    if vmax <= 0.0:
        vmax = default

    return [-vmax, vmax]


# %% [markdown]
# ### Load Grid And Build Derived Fields

# %%
grid = load_grid_and_fields(
    xdmf_path,
    mesh_h5_path,
    velocity_file=velocity_h5_path,
    pressure_file=pressure_h5_path,
    pressure_degree=pdegree,
)
points_xy = np.asarray(grid.points[:, :2], dtype=np.float64)

v_ana, p_ana, rho_ana = analytical_solution(points_xy, r_i, r_o, k)
grid.point_data["V_ana"] = v_ana
grid.point_data["P_ana"] = p_ana
grid.point_data["Rho_ana"] = rho_ana

v_u, velocity_assoc = get_field_array(grid, "V_u")
if velocity_assoc != "point":
    raise ValueError(f"Velocity field is {velocity_assoc}-associated, expected point data.")

p_u, pressure_assoc = get_field_array(grid, "P_u")
p_u = p_u.reshape(-1)
pressure_points_xy = support_points_xy(grid, pressure_assoc)
_, p_ana_support, _ = analytical_solution(pressure_points_xy, r_i, r_o, k)

v_err = v_u - v_ana
p_err = p_u - p_ana_support

grid.point_data["V_err"] = v_err
set_field_array(grid, "P_err", p_err, pressure_assoc)

v_err_mag = np.linalg.norm(v_err[:, :2], axis=1)
grid.point_data["Rho_ana_neg"] = -rho_ana

velocity_clim = [0.0, 2.3]
pressure_clim = [-8.5, 8.5]
density_clim = [-67.5, 67.5]
velocity_error_clim = positive_clim(v_err_mag, percentile=99.0, default=1.0e-12)
pressure_error_clim = symmetric_clim(p_err, percentile=99.0, default=1.0e-12)

save_vertical_colorbar(
    colormap=cmc.lapaz.resampled(11),
    output_path=output_dir,
    fname="v_ana",
    cb_axis_label="Velocity",
    vmin=velocity_clim[0],
    vmax=velocity_clim[1],
)
save_vertical_colorbar(
    colormap=cmc.vik.resampled(41),
    output_path=output_dir,
    fname="p_ana",
    cb_axis_label="Pressure",
    vmin=pressure_clim[0],
    vmax=pressure_clim[1],
)
save_vertical_colorbar(
    colormap=cmc.roma.resampled(31),
    output_path=output_dir,
    fname="rho_ana",
    cb_axis_label="Rho",
    vmin=density_clim[0],
    vmax=density_clim[1],
)


# %% [markdown]
# ### Plot Analytical Velocity

# %%
print("Plotting: analytical velocity")
plot_vector_with_colorbar(
    grid,
    vector_name="V_ana",
    png_name="vel_ana.png",
    cmap=cmc.lapaz.resampled(11),
    clim=velocity_clim,
    cb_label="Velocity",
    cb_name="v_ana",
    vmag=1.0e-1,
    show_arrows=False,
    n_arrows=arrow_target,
)


# %% [markdown]
# ### Plot Analytical Pressure

# %%
print("Plotting: analytical pressure")
plot_scalar_with_colorbar(
    grid,
    scalar_name="P_ana",
    png_name="p_ana.png",
    cmap=cmc.vik.resampled(41),
    clim=pressure_clim,
    cb_label="Pressure",
    cb_name="p_ana",
)


# %% [markdown]
# ### Plot Analytical Density

# %%
print("Plotting: analytical density")
plot_scalar_with_colorbar(
    grid,
    scalar_name="Rho_ana_neg",
    png_name="rho_ana.png",
    cmap=cmc.roma.resampled(31),
    clim=density_clim,
    cb_label="Rho",
    cb_name="rho_ana",
)


# %% [markdown]
# ### Plot Solution Velocity

# %%
print("Plotting: numerical velocity")
plot_vector_with_colorbar(
    grid,
    vector_name="V_u",
    png_name="vel_uw.png",
    cmap=cmc.lapaz.resampled(11),
    clim=velocity_clim,
    cb_label="Velocity",
    cb_name="v_uw",
    vmag=1.0e-1,
    show_arrows=True,
    n_arrows=arrow_target,
)


# %% [markdown]
# ### Plot Absolute Velocity Error Vector

# %%
print("Plotting: absolute velocity error")
plot_vector_with_colorbar(
    grid,
    vector_name="V_err",
    png_name="vel_abs_err.png",
    cmap=cmc.lapaz.resampled(11),
    clim=velocity_error_clim,
    cb_label="Velocity Error (absolute)",
    cb_name="v_err_abs",
    vmag=10.0,
    show_arrows=False,
    n_arrows=max(80, arrow_target // 2),
)


# %% [markdown]
# ### Plot Solution Pressure

# %%
print("Plotting: numerical pressure")
plot_scalar_with_colorbar(
    grid,
    scalar_name="P_u",
    png_name="p_uw.png",
    cmap=cmc.vik.resampled(41),
    clim=pressure_clim,
    cb_label="Pressure",
    cb_name="p_uw",
)


# %% [markdown]
# ### Plot Absolute Pressure Error

# %%
print("Plotting: absolute pressure error")
plot_scalar_with_colorbar(
    grid,
    scalar_name="P_err",
    png_name="p_abs_err.png",
    cmap=cmc.vik.resampled(41),
    clim=pressure_error_clim,
    cb_label="Pressure Error (absolute)",
    cb_name="p_err_abs",
)
