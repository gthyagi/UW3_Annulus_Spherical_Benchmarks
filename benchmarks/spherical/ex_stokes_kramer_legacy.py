# %% [markdown]
# ## Spherical Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark paper](https://gmd.copernicus.org/articles/14/1899/2021/)
#
# ### Authors
# Thyagarajulu Gollapalli ([GitHub](https://github.com/gthyagi)) <br>
# Underworld3 Development Team ([UW3 Repository](https://github.com/underworldcode/underworld3))
#
# ##### Case1: Freeslip boundaries and delta function density perturbation
# ##### Case2: Freeslip boundaries and smooth density distribution
# ##### Case3: Noslip boundaries and delta function density perturbation
# ##### Case4: Noslip boundaries and smooth density distribution

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

is_serial = uw.mpi.size == 1

# %% [markdown]
# ### Convection Parameters

# %%
case = uw.options.getString("case", default="case2")
l = uw.options.getInt("l", default=2)
m = uw.options.getInt("m", default=1)
k = uw.options.getInt("k", default=l + 1)

# %% [markdown]
# ### Mesh Parameters

# %%
r_i = uw.options.getReal("radius_inner", default=1.22)
r_int = uw.options.getReal("radius_internal", default=2.0)
r_o = uw.options.getReal("radius_outer", default=2.22)
cellsize = uw.options.getReal("cellsize", default=1.0 / 8.0)

# %% [markdown]
# ### Mesh Variable Parameters

# %%
vdegree = uw.options.getInt("vdegree", default=2)
pdegree = uw.options.getInt("pdegree", default=1)
pcont = uw.options.getBool("pcont", default=True)
pcont_str = str(pcont).lower()

# %% [markdown]
# ### Solver Parameters

# %%
vel_penalty = uw.options.getReal("vel_penalty", default=1.0e8)
stokes_tol = uw.options.getReal("stokes_tol", default=1.0e-10)
bc_type = uw.options.getString("bc_type", default="natural")

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
output_dir = os.path.join(
    "../../output/spherical/kramer/legacy/",
    (
        f"{case}_inv_lc_{int(1/cellsize)}_l_{l}_m_{m}_k_{k}_vdeg_{vdegree}_pdeg_{pdegree}"
        f"_pcont_{pcont_str}_vel_penalty_{vel_penalty:.2g}_stokes_tol_{stokes_tol:.2g}_ncpus_{uw.mpi.size}/"
    ),
)

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ### Analytical Solution Handles

# %%
if freeslip and delta_fn:
    soln_above = assess.SphericalStokesSolutionDeltaFreeSlip(l, m, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
    soln_below = assess.SphericalStokesSolutionDeltaFreeSlip(l, m, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
elif freeslip and smooth:
    soln_above = assess.SphericalStokesSolutionSmoothFreeSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
    soln_below = assess.SphericalStokesSolutionSmoothFreeSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
elif noslip and delta_fn:
    soln_above = assess.SphericalStokesSolutionDeltaZeroSlip(l, m, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
    soln_below = assess.SphericalStokesSolutionDeltaZeroSlip(l, m, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
elif noslip and smooth:
    soln_above = assess.SphericalStokesSolutionSmoothZeroSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
    soln_below = assess.SphericalStokesSolutionSmoothZeroSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)

# %% [markdown]
# ### Create Mesh

# %%
if delta_fn:
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=r_o,
        radiusInternal=r_int,
        radiusInner=r_i,
        cellSize=cellsize,
        qdegree=max(pdegree, vdegree),
        degree=1,
        filename=f"{output_dir}/mesh.msh",
        refinement=None,
    )
else:
    mesh = uw.meshing.SphericalShell(
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
r_uw = mesh.CoordinateSystem.xR[0]
th_uw = mesh.CoordinateSystem.xR[1]
phi_raw = mesh.CoordinateSystem.xR[2]
phi_uw = sp.Piecewise(
    (2 * sp.pi + phi_raw, phi_raw < 0),
    (phi_raw, True),
)
null_mode_expr = sp.Matrix(((0, 1, 1), (-1, 0, 1), (-1, -1, 0))) * mesh.CoordinateSystem.N.T

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
def fill_piecewise(var, fn_above, fn_below):
    radii = uw.function.evaluate(r_uw, var.coords)

    for i, coord in enumerate(var.coords):
        fn = fn_above if radii[i] > r_int else fn_below
        var.data[i] = fn(coord)


# %%
with mesh.access(v_ana, p_ana):
    fill_piecewise(v_ana, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
    fill_piecewise(p_ana, soln_above.pressure_cartesian, soln_below.pressure_cartesian)

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
y_lm_real = (
    sp.sqrt((2 * l + 1) / (4 * sp.pi) * sp.factorial(l - m) / sp.factorial(l + m))
    * sp.cos(m * phi_uw)
    * sp.assoc_legendre(l, m, sp.cos(th_uw))
)

gravity_fn = -1.0 * unit_rvec

if delta_fn:
    rho = sp.exp(-1e5 * ((r_uw - r_int) ** 2)) * y_lm_real
    stokes.add_natural_bc(-rho * unit_rvec, mesh.boundaries.Internal.name)
    stokes.bodyforce = sp.Matrix([0.0, 0.0, 0.0])
else:
    rho = ((r_uw / r_o) ** k) * y_lm_real
    stokes.bodyforce = rho * gravity_fn

with mesh.access(rho_ana):
    rho_ana.data[:] = np.asarray(uw.function.evaluate(rho, rho_ana.coords)).reshape(-1, 1)

# %% [markdown]
# #### Boundary Conditions

# %%
if freeslip:
    if bc_type == "natural":
        v_diff = v_uw.sym - v_ana.sym
        stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Upper.name)
        stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Lower.name)
    elif bc_type == "essential":
        stokes.add_essential_bc(v_ana.sym, mesh.boundaries.Upper.name)
        stokes.add_essential_bc(v_ana.sym, mesh.boundaries.Lower.name)
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")
elif noslip:
    if bc_type == "natural":
        v_diff = v_uw.sym - v_ana.sym
        stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Upper.name)
        stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Lower.name)
    elif bc_type == "essential":
        stokes.add_essential_bc(sp.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.Upper.name)
        stokes.add_essential_bc(sp.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.Lower.name)
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

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
I0 = uw.maths.Integral(mesh, null_mode_expr.dot(v_uw.sym))
norm = I0.evaluate()
I0.fn = null_mode_expr.dot(null_mode_expr)
vnorm = I0.evaluate()

with mesh.access(v_uw):
    dv = uw.function.evaluate(norm * null_mode_expr, v_uw.coords) / vnorm
    v_uw.data[...] -= dv


# %% [markdown]
# ### Errors And L2 Norm

# %%
def fill_error(var_err, var_num, fn_above, fn_below):
    radii = uw.function.evaluate(r_uw, var_err.coords)

    for i, coord in enumerate(var_err.coords):
        fn = fn_above if radii[i] > r_int else fn_below
        var_err.data[i] = var_num.data[i] - fn(coord)


# %%
with mesh.access(v_uw, p_uw, v_err, p_err):
    fill_error(v_err, v_uw, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
    fill_error(p_err, p_uw, soln_above.pressure_cartesian, soln_below.pressure_cartesian)


# %%
def relative_l2_error(mesh, err_expr, ana_expr):
    if isinstance(err_expr, sp.MatrixBase):
        err_expr = err_expr.dot(err_expr)
        ana_expr = ana_expr.dot(ana_expr)
    else:
        err_expr = err_expr * err_expr
        ana_expr = ana_expr * ana_expr

    err_I = uw.maths.Integral(mesh, err_expr)
    ana_I = uw.maths.Integral(mesh, ana_expr)
    return np.sqrt(err_I.evaluate()) / np.sqrt(ana_I.evaluate())


# %%
v_err_l2 = relative_l2_error(mesh, v_err.sym, v_ana.sym)
p_err_l2 = relative_l2_error(mesh, p_err.sym[0], p_ana.sym[0])

if uw.mpi.rank == 0:
    print("Relative velocity L2 error:", v_err_l2)
    print("Relative pressure L2 error:", p_err_l2)

# %% [markdown]
# ### Save Outputs

# %%
if uw.mpi.size == 1 and os.path.isfile(output_dir + "error_norm.h5"):
    os.remove(output_dir + "error_norm.h5")

if uw.mpi.rank == 0:
    with h5py.File(output_dir + "error_norm.h5", "w") as f_h5:
        f_h5.create_dataset("l", data=l)
        f_h5.create_dataset("m", data=m)
        f_h5.create_dataset("k", data=k)
        f_h5.create_dataset("cellsize", data=cellsize)
        f_h5.create_dataset("v_l2_norm", data=v_err_l2)
        f_h5.create_dataset("p_l2_norm", data=p_err_l2)

# %%
mesh.petsc_save_checkpoint(
    index=0,
    meshVars=[v_uw, p_uw, v_ana, p_ana, rho_ana, v_err, p_err],
    outputPath=os.path.relpath(output_dir) + "/output",
)

# %%
# Serial post-processing plots should be generated in a separate script.
