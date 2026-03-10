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
# ##### Case2: Freeslip boundaries and smooth density distribution
# ##### Case3: Noslip boundaries and delta function density perturbation
# ##### Case4: Noslip boundaries and smooth density distribution

# %%
import os
import h5py
import numpy as np
import sympy as sp
import underworld3 as uw
from underworld3.systems import Stokes
import assess

os.environ["UW_TIMING_ENABLE"] = "1"

is_serial = (uw.mpi.size == 1)

# %% [markdown]
# ### Convection Parameters

# %%
params = uw.Params(
    uw_case=uw.Param(
        "case2",
        description="Benchmark case: case1, case2, case3, case4",
    ),
    uw_n=uw.Param(
        2,
        description="Wave number for density perturbation",
    ),
    uw_k=uw.Param(
        3,
        description="Power exponent for smooth density",
    ),
    uw_radius_inner=uw.Param(
        1.22,
        description="Inner radius",
    ),
    uw_radius_internal=uw.Param(
        2.0,
        description="Internal interface radius",
    ),
    uw_radius_outer=uw.Param(
        2.22,
        description="Outer radius",
    ),
    uw_cellsize=uw.Param(
        1.0 / 32.0,
        description="Background mesh cell size",
    ),
    uw_cellsize_internal_boundary_factor=uw.Param(
        2,
        description="Internal-boundary cell-size refinement factor",
    ),
    uw_vdegree=uw.Param(
        2,
        description="Velocity polynomial degree",
    ),
    uw_pdegree=uw.Param(
        1,
        description="Pressure polynomial degree",
    ),
    uw_pcont=uw.Param(
        True,
        description="Pressure continuity flag",
    ),
    uw_stokes_tol=uw.Param(
        1e-10,
        description="Stokes solver tolerance",
    ),
    uw_vel_penalty=uw.Param(
        2.5e8,
        description="Penalty for natural-BC velocity matching",
    ),
    uw_bc_type=uw.Param(
        "natural",
        description="Boundary-condition mode: natural or essential",
    ),
    uw_ana_normal=uw.Param(
        False,
        description="Use analytical radial normal (legacy option)",
    ),
    uw_petsc_normal=uw.Param(
        True,
        description="Use PETSc Gamma normal (legacy option)",
    ),
)

# %% [markdown]
# ### Convection Parameters

# %%
case = params.uw_case
n = int(params.uw_n)
k = int(params.uw_k)

# %% [markdown]
# ### Mesh Parameters

# %%
r_i = float(params.uw_radius_inner)
r_int = float(params.uw_radius_internal)
r_o = float(params.uw_radius_outer)
cellsize = float(params.uw_cellsize)
cellsize_int_bd_fac = int(params.uw_cellsize_internal_boundary_factor)

# %% [markdown]
# ### Boundary Condition Parameters

# %%
# which normals to use
ana_normal = bool(params.uw_ana_normal)  # mesh radial unit vectors
petsc_normal = bool(params.uw_petsc_normal)  # PETSc Gamma

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
    "../../output/annulus/kramer/latest/",
    (
        f"{case}_inv_lc_{int(1/cellsize)}_n_{n}_k_{k}_vdeg_{params.uw_vdegree}_pdeg_{params.uw_pdegree}"
        f"_pcont_{str(params.uw_pcont).lower()}_vel_penalty_{params.uw_vel_penalty:.2g}_stokes_tol_{params.uw_stokes_tol:.2g}"
        f"_ncpus_{uw.mpi.size}/"
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
        qdegree=max(params.uw_pdegree, params.uw_vdegree),
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
v_uw = uw.discretisation.MeshVariable(
    varname="V_u",
    mesh=mesh,
    degree=params.uw_vdegree,
    vtype=uw.VarType.VECTOR,
    varsymbol=r"{V_u}",
)
p_uw = uw.discretisation.MeshVariable(
    varname="P_u",
    mesh=mesh,
    degree=params.uw_pdegree,
    vtype=uw.VarType.SCALAR,
    varsymbol=r"{P_u}",
    continuous=params.uw_pcont,
)

v_ana = uw.discretisation.MeshVariable(
    varname="V_a",
    mesh=mesh,
    degree=params.uw_vdegree,
    vtype=uw.VarType.VECTOR,
    varsymbol=r"{V_a}",
)
p_ana = uw.discretisation.MeshVariable(
    varname="P_a",
    mesh=mesh,
    degree=params.uw_pdegree,
    vtype=uw.VarType.SCALAR,
    varsymbol=r"{P_a}",
    continuous=params.uw_pcont,
)
rho_ana = uw.discretisation.MeshVariable(
    varname="RHO_a",
    mesh=mesh,
    degree=params.uw_pdegree,
    vtype=uw.VarType.SCALAR,
    varsymbol=r"{RHO_a}",
    continuous=True,
)

v_err = uw.discretisation.MeshVariable(
    varname="V_e",
    mesh=mesh,
    degree=params.uw_vdegree,
    vtype=uw.VarType.VECTOR,
    varsymbol=r"{V_e}",
)
p_err = uw.discretisation.MeshVariable(
    varname="P_e",
    mesh=mesh,
    degree=params.uw_pdegree,
    vtype=uw.VarType.SCALAR,
    varsymbol=r"{P_e}",
    continuous=params.uw_pcont,
)


# %% [markdown]
# ### Analytical Field Fill

# %%
def analytical_values(var, r_int, fn_above, fn_below):
    coords = np.asarray(var.coords)
    mask_above = np.hypot(coords[:, 0], coords[:, 1]) > r_int

    ncomp = var.data.shape[1]
    values = np.empty_like(var.data)

    for mask, fn in ((mask_above, fn_above), (~mask_above, fn_below)):
        if np.any(mask):
            values[mask, :] = np.asarray([fn(c) for c in coords[mask]]).reshape(-1, ncomp)

    return values


# %%
v_ana_values = analytical_values(
    v_ana,
    r_int,
    soln_above.velocity_cartesian,
    soln_below.velocity_cartesian,
)
p_ana_values = analytical_values(
    p_ana,
    r_int,
    soln_above.pressure_cartesian,
    soln_below.pressure_cartesian,
)

with uw.synchronised_array_update():
    v_ana.data[...] = v_ana_values
    p_ana.data[...] = p_ana_values

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
# Restore analytical density values into mesh variable
rho_ana_values = np.asarray(uw.function.evaluate(rho, rho_ana.coords)).reshape(-1)
with uw.synchronised_array_update():
    rho_ana.data[:, 0] = rho_ana_values

# %% [markdown]
# #### Boundary Conditions

# %%
if freeslip:
    if ana_normal:
        Gamma = mesh.CoordinateSystem.unit_e_0
    elif petsc_normal:
        Gamma = mesh.Gamma

    if params.uw_bc_type == "natural":
        v_diff = v_uw.sym - v_ana.sym
        stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Upper.name)
        stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Lower.name)
    elif params.uw_bc_type == "essential":
        stokes.add_essential_bc(soln_above.velocity_cartesian, mesh.boundaries.Upper.name)
        stokes.add_essential_bc(soln_below.velocity_cartesian, mesh.boundaries.Lower.name)
elif noslip:
    if params.uw_bc_type == "natural":
        v_diff = v_uw.sym - v_ana.sym
        stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Upper.name)
        stokes.add_natural_bc(params.uw_vel_penalty * v_diff, mesh.boundaries.Lower.name)
    elif params.uw_bc_type == "essential":
        stokes.add_essential_bc(sp.Matrix([0.0, 0.0]), mesh.boundaries.Upper.name)
        stokes.add_essential_bc(sp.Matrix([0.0, 0.0]), mesh.boundaries.Lower.name)

# %% [markdown]
# #### Solver Settings

# %%
stokes.tolerance = params.uw_stokes_tol
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["ksp_monitor_true_residual"] = None
stokes.petsc_options["snes_monitor"] = None
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
uw.timing.reset()
uw.timing.start()
stokes.solve(verbose=False, debug=False)
uw.timing.stop()
uw.timing.print_table(filename=f"{output_dir}/stokes_timing.txt")

if uw.mpi.rank == 0:
    print(stokes.snes.getConvergedReason())
    print(stokes.snes.ksp.getConvergedReason())

# %% [markdown]
# ### Remove Null Mode

# %%
if freeslip:
    I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v_uw.sym))
    norm = I0.evaluate()
    I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
    vnorm = I0.evaluate()

    with mesh.access(v_uw):
        dv = uw.function.evaluate(norm * v_theta_fn_xy, v_uw.coords) / vnorm
        dv = dv.reshape(v_uw.data.shape)
        v_uw.data[...] -= dv


# %% [markdown]
# ### Errors and L2 Norm

# %%
def compute_error(mesh_var, var_num, r_int, fn_above, fn_below):
    ana_values = analytical_values(mesh_var, r_int, fn_above, fn_below)
    return np.asarray(var_num.data) - ana_values

v_err_values = compute_error(v_err, v_uw, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
p_err_values = compute_error(p_err, p_uw, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)

with uw.synchronised_array_update():
    v_err.data[...] = v_err_values
    p_err.data[...] = p_err_values


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
if uw.mpi.rank == 0:
    err_h5 = os.path.join(output_dir, "error_norm.h5")
    if os.path.isfile(err_h5):
        os.remove(err_h5)
    with h5py.File(err_h5, "w") as f_h5:
        f_h5.create_dataset("case", data=np.bytes_(case))
        f_h5.create_dataset("n", data=n)
        f_h5.create_dataset("k", data=k)
        f_h5.create_dataset("cellsize", data=cellsize)
        f_h5.create_dataset("v_l2_norm", data=v_err_l2)
        f_h5.create_dataset("p_l2_norm", data=p_err_l2)

# %%
mesh.petsc_save_checkpoint(
    index=0,
    meshVars=[v_uw, p_uw, v_ana, p_ana, rho_ana, v_err, p_err],
    outputPath=os.path.relpath(output_dir)+'/output',
)

# %%
