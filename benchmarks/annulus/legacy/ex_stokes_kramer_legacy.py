# %% [markdown]
# ## Annulus Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark paper](https://gmd.copernicus.org/articles/14/1899/2021/) 
#
# ### Authors
# Thyagarajulu Gollapalli ([GitHub](https://github.com/gthyagi)) <br>
# Underworld3 Development Team ([UW3 Repository](https://github.com/underworldcode/underworld3))
#
#
# ##### Case1: Freeslip boundaries and delta function density perturbation
# <!--    1. Works fine (i.e., bc produce results) -->
# ##### Case2: Freeslip boundaries and smooth density distribution
# <!--
#     1. Works fine (i.e., bc produce results)
#     2. Output contains null space (for normals = unit radial vector)
# -->
# ##### Case3: Noslip boundaries and delta function density perturbation
# <!--
#     1. Works fine (i.e., bc produce results)
# -->
# ##### Case4: Noslip boundaries and smooth density distribution
# <!--
#     1. Works fine (i.e., bc produce results)
# -->

# %%
import os
import assess
import h5py
import numpy as np
import sympy as sp
import underworld3 as uw
from underworld3 import timing
from underworld3.systems import Stokes

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

is_serial = (uw.mpi.size == 1)

# %% [markdown]
# ### Convection Parameters

# %%
case = uw.options.getString(
    "case",
    default="case2",
)
n = uw.options.getInt(
    "n",
    default=2,
)
k = uw.options.getInt(
    "k",
    default=3,
)

# %% [markdown]
# ### Mesh Parameters

# %%
r_i = uw.options.getReal(
    "radius_inner",
    default=1.22,
)
r_int = uw.options.getReal(
    "radius_internal",
    default=2.0,
)
r_o = uw.options.getReal(
    "radius_outer",
    default=2.22,
)
cellsize = uw.options.getReal(
    "cellsize",
    default=1 / 32,
)
cellsize_int_bd_fac = uw.options.getInt(
    "cellsize_internal_boundary_factor",
    default=2,
)

# %% [markdown]
# ### Mesh Variable Parameters

# %%
vdegree = uw.options.getInt(
    "vdegree",
    default=2,
)
pdegree = uw.options.getInt(
    "pdegree",
    default=1,
)
pcont = uw.options.getBool(
    "pcont",
    default=True,
)
pcont_str = str(pcont).lower()

# %% [markdown]
# ### Solver Parameters

# %%
stokes_tol = uw.options.getReal(
    "stokes_tol",
    default=1e-10,
)

# %% [markdown]
# ### Boundary Condition Parameters

# %%
vel_penalty = uw.options.getReal(
    "vel_penalty",
    default=2.5e8,
)
bc_type = uw.options.getString(
    "bc_type",
    default="natural",
)

# which normals to use
ana_normal = False # mesh radial unit vectors
petsc_normal = True # petsc Gamma

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
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
output_dir = os.path.join(
    repo_root,
    "output",
    "annulus",
    "kramer",
    "legacy",
    (
        f"{case}_inv_lc_{int(1/cellsize)}_n_{n}_k_{k}_vdeg_{vdegree}_pdeg_{pdegree}"
        f"_pcont_{pcont_str}_vel_penalty_{vel_penalty:.2g}_stokes_tol_{stokes_tol:.2g}_ncpus_{uw.mpi.size}"
    ),
)

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ### Analytical Solution Handles

# %%
if freeslip and delta_fn:
    soln_above = assess.CylindricalStokesSolutionDeltaFreeSlip(n, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
    soln_below = assess.CylindricalStokesSolutionDeltaFreeSlip(n, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
elif freeslip and smooth:
    soln_above = assess.CylindricalStokesSolutionSmoothFreeSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
    soln_below = assess.CylindricalStokesSolutionSmoothFreeSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
elif noslip and delta_fn:
    soln_above = assess.CylindricalStokesSolutionDeltaZeroSlip(n, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
    soln_below = assess.CylindricalStokesSolutionDeltaZeroSlip(n, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
elif noslip and smooth:
    soln_above = assess.CylindricalStokesSolutionSmoothZeroSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
    soln_below = assess.CylindricalStokesSolutionSmoothZeroSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)

# %% [markdown]
# ### Create Mesh

# %%
if delta_fn:
    mesh = uw.meshing.AnnulusInternalBoundary(
        radiusOuter=r_o,
        radiusInternal=r_int,
        radiusInner=r_i,
        cellSize_Inner=cellsize,
        cellSize_Internal=cellsize / cellsize_int_bd_fac,
        cellSize_Outer=cellsize,
        filename=f"{output_dir}/mesh.msh",
    )
elif smooth:
    mesh = uw.meshing.Annulus(
        radiusOuter=r_o,
        radiusInner=r_i,
        cellSize=cellsize,
        qdegree=max(pdegree, vdegree),
        degree=1,
        filename=f"{output_dir}/mesh.msh",
        refinement=None,
    )

if is_serial:
    mesh.dm.view()

# %%
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR
v_theta_fn_xy = r_uw * mesh.CoordinateSystem.rRotN.T * sp.Matrix((0, 1))

# %% [markdown]
# ### Create Mesh Variables

# %%
v_uw = uw.discretisation.MeshVariable("V_u", mesh, mesh.data.shape[1], degree=vdegree)
p_uw = uw.discretisation.MeshVariable("P_u", mesh, 1, degree=pdegree, continuous=pcont)

v_ana = uw.discretisation.MeshVariable("V_a", mesh, mesh.data.shape[1], degree=vdegree)
p_ana = uw.discretisation.MeshVariable("P_a", mesh, 1, degree=pdegree, continuous=pcont)
rho_ana = uw.discretisation.MeshVariable("RHO_a", mesh, 1, degree=pdegree, continuous=True)

v_err = uw.discretisation.MeshVariable("V_e", mesh, mesh.data.shape[1], degree=vdegree)
p_err = uw.discretisation.MeshVariable("P_e", mesh, 1, degree=pdegree, continuous=pcont)


# %% [markdown]
# ### Analytical Field Fill

# %%
def assign_analytical_solution(var, r_int, fn_above, fn_below):
    radii = uw.function.evaluate(r_uw, var.coords)

    for i, coord in enumerate(var.coords):
        fn = fn_above if radii[i] > r_int else fn_below
        var.data[i] = fn(coord)


# %%
with mesh.access(v_ana, p_ana):
    assign_analytical_solution(v_ana, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
    assign_analytical_solution(p_ana, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)

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
if delta_fn:
    rho = sp.cos(n * th_uw) * sp.exp(-1e5 * ((r_uw - r_int) ** 2))
    stokes.add_natural_bc(-rho * unit_rvec, "Internal")
    stokes.bodyforce = sp.Matrix([0.0, 0.0])
elif smooth:
    rho = ((r_uw / r_o) ** k) * sp.cos(n * th_uw)
    gravity_fn = -1.0 * unit_rvec
    stokes.bodyforce = rho * gravity_fn

# %%
# restoring rho analytical values into mesh variable 
with mesh.access(rho_ana):
    rho_ana.data[:] = np.c_[uw.function.evaluate(rho, rho_ana.coords)]

# %% [markdown]
# #### Boundary Conditions

# %%
if freeslip:
    if ana_normal:
        Gamma = mesh.CoordinateSystem.unit_e_0
    elif petsc_normal:
        Gamma = mesh.Gamma

    if bc_type == "natural":
        v_diff = v_uw.sym - v_ana.sym
        stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Upper.name)
        stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Lower.name)
    elif bc_type == "essential":
        stokes.add_essential_bc(soln_above.velocity_cartesian, mesh.boundaries.Upper.name)
        stokes.add_essential_bc(soln_below.velocity_cartesian, mesh.boundaries.Lower.name)
elif noslip:
    if bc_type == "natural":
        v_diff = v_uw.sym - v_ana.sym
        stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Upper.name)
        stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Lower.name)
    elif bc_type == "essential":
        stokes.add_essential_bc(sp.Matrix([0.0, 0.0]), mesh.boundaries.Upper.name)
        stokes.add_essential_bc(sp.Matrix([0.0, 0.0]), mesh.boundaries.Lower.name)

# %% [markdown]
# #### Solver Settings

# %%
stokes.tolerance = stokes_tol
stokes.petsc_options["ksp_monitor"] = None
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
timing.reset()
timing.start()
stokes.solve(verbose=False, debug=False)
timing.stop()
timing.print_table(display_fraction=0.999, output_file=f"{output_dir}/stokes_timing.txt")

if uw.mpi.rank == 0:
    print(stokes.snes.getConvergedReason())
    print(stokes.snes.ksp.getConvergedReason())

# %% [markdown]
# ### Remove Null Mode

# %%
I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v_uw.sym))
norm = I0.evaluate()
I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
vnorm = I0.evaluate()

with mesh.access(v_uw):
    dv = uw.function.evaluate(norm * v_theta_fn_xy, v_uw.coords) / vnorm
    v_uw.data[...] -= dv


# %% [markdown]
# ### Errors and L2 Norm

# %%
def compute_error(mesh_var, var_num, r_in, fn_above, fn_below):
    r_vals = uw.function.evaluate(r_uw, mesh_var.coords)

    for i, coord in enumerate(mesh_var.coords):
        if r_vals[i] > r_in:
            mesh_var.data[i] = var_num.data[i] - fn_above(coord)
        else:
            mesh_var.data[i] = var_num.data[i] - fn_below(coord)


# %%
with mesh.access(v_uw, p_uw, v_err, p_err):
    compute_error(v_err, v_uw, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
    compute_error(p_err, p_uw, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)


# %%
def relative_l2_error(mesh, err_var, ana_var):
    """Relative L2 error for scalar or vector expressions."""
    err_I = uw.maths.Integral(mesh, err_var.sym.dot(err_var.sym))
    ana_I = uw.maths.Integral(mesh, ana_var.sym.dot(ana_var.sym))
    return np.sqrt(err_I.evaluate()) / np.sqrt(ana_I.evaluate())


# %%
v_err_l2 = relative_l2_error(mesh, v_err, v_ana)
p_err_l2 = relative_l2_error(mesh, p_err, p_ana)

if uw.mpi.rank == 0:
    print("Relative velocity L2 error:", v_err_l2)
    print("Relative pressure L2 error:", p_err_l2)

# %% [markdown]
# ### Save Outputs

# %%
if uw.mpi.size == 1 and os.path.isfile(output_dir + "error_norm.h5"):
    os.remove(output_dir + "error_norm.h5")
    print("Old file removed")

if uw.mpi.rank == 0:
    print("Creating new h5 file")
    with h5py.File(output_dir + "error_norm.h5", "w") as f_h5:
        f_h5.create_dataset("k", data=k)
        f_h5.create_dataset("cellsize", data=cellsize)
        f_h5.create_dataset("v_l2_norm", data=v_err_l2)
        f_h5.create_dataset("p_l2_norm", data=p_err_l2)

# %%
mesh.petsc_save_checkpoint(
    index=0,
    meshVars=[v_uw, p_uw, v_ana, p_ana, rho_ana, v_err, p_err],
    outputPath=os.path.join(os.path.relpath(output_dir), "output"),
)

# %%
