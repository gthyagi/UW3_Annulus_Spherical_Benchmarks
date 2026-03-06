# %% [markdown]
# ## Annulus Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark Paper](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-2765/) [ASPECT Results](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/annulus/doc/annulus.html)
#
# ### Authors
# Thyagarajulu Gollapalli ([GitHub](https://github.com/gthyagi)) <br>
# Underworld3 Development Team ([UW3 Repository](https://github.com/underworldcode/underworld3))
#
# ### Analytical solution
#
# This benchmark is based on a manufactured solution in which an analytical solution to the isoviscous incompressible Stokes equations is derived in an annulus geometry. The velocity and pressure fields are as follows:
#
# $$ v_{\theta}(r, \theta) = f(r) \cos(k\theta) $$
#
# $$ v_r(r, \theta) = g(r)k \sin(k\theta) $$
#
# $$ p(r, \theta) = kh(r) \sin(k\theta) + \rho_0g_r(R_2 - r) $$
#
# $$ \rho(r, \theta) = m(r)k \sin(k\theta) + \rho_0 $$
#
# with
#
# $$ f(r) = Ar + \frac{B}{r} $$
#
# $$ g(r) = \frac{A}{2}r + \frac{B}{r}\ln r + \frac{C}{r} $$
#
# $$ h(r) = \frac{2g(r) - f(r)}{r} $$
#
# $$ m(r) = g''(r) - \frac{g'(r)}{r} - \frac{g(r)}{r^2}(k^2 - 1) + \frac{f(r)}{r^2} + \frac{f'(r)}{r} $$
#
# $$ A = -C\frac{2(\ln R_1 - \ln R_2)}{R_2^2 \ln R_1 - R_1^2 \ln R_2} $$
#
# $$ B = -C\frac{R_2^2 - R_1^2}{R_2^2 \ln R_1 - R_1^2 \ln R_2} $$
#
#
# The parameters $A$ and $B$ are chosen so that $ v_r(R_1, \theta) = v_r(R_2, \theta) = 0 $ for all $\theta \in [0, 2\pi]$, i.e. the velocity is tangential to both inner and outer surfaces. The gravity vector is radial inward and of unit length.
#
# The parameter $k$ controls the number of convection cells present in the domain
#
# In the present case, we set $ R_1 = 1, R_2 = 2$ and $C = -1 $.

# %%
import os
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
# ### Mesh Parameters

# %%
cellsize = uw.options.getReal(
    "cellsize",
    default=1 / 64,
)
r_i = uw.options.getReal(
    "radius_inner",
    default=1.0,
)
r_o = uw.options.getReal(
    "radius_outer",
    default=2.0,
)

# %% [markdown]
# ### Convection Cells Parameter

# %%
k = uw.options.getInt(
    "k",
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
vel_penalty = uw.options.getReal(
    "vel_penalty",
    default=2.5e8,
)

analytical = True

# %% [markdown]
# ### Output Directory

# %%
output_dir = os.path.join(
    "../../output/annulus/thieulot/legacy/",
    (
        f"model_inv_lc_{int(1/cellsize)}_k_{k}_vdeg_{vdegree}_pdeg_{pdegree}"
        f"_pcont_{pcont_str}_vel_penalty_{vel_penalty:.2g}_stokes_tol_{stokes_tol:.2g}/"
    ),
)
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)


# %% [markdown]
# ### Analytical Solution Helpers

# %%
def analytic_solution(
    mesh,
    r_i,
    r_o,
    k,
    C=-1,
    rho0=0,
):
    """Return analytical annulus benchmark fields (v, p, rho) as UW expressions."""

    x, y = mesh.CoordinateSystem.X
    r, th = mesh.CoordinateSystem.xR
    unit_rvec = mesh.CoordinateSystem.unit_e_0

    denom = (r_o**2) * sp.log(r_i) - (r_i**2) * sp.log(r_o)

    A = -C * (2 * (sp.log(r_i) - sp.log(r_o)) / denom)
    B = -C * ((r_o**2 - r_i**2) / denom)

    f = A * r + B / r
    g = (A / 2) * r + (B / r) * sp.log(r) + C / r
    h = (2 * g - f) / r

    m = (
        sp.diff(g, r, 2)
        - sp.diff(g, r) / r
        - (g / r**2) * (k**2 - 1)
        + f / r**2
        + sp.diff(f, r) / r
    )

    v_r = g * k * sp.sin(k * th)
    v_th = f * sp.cos(k * th)

    if k == 0:
        v_uw = mesh.CoordinateSystem.rRotN.T * sp.Matrix([0, v_th])
        p_uw = sp.Integer(0)
        rho_uw = sp.Integer(0)
    else:
        v_uw = mesh.CoordinateSystem.rRotN.T * sp.Matrix([v_r, v_th])
        p_uw = k * h * sp.sin(k * th) + rho0 * (r_o - r)
        rho_uw = m * k * sp.sin(k * th) + rho0

    return v_uw, p_uw, rho_uw


# %% [markdown]
# ### Create Mesh

# %%
mesh = uw.meshing.Annulus(
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSize=cellsize,
    qdegree=max(pdegree, vdegree),
    degree=1,
    filename=f"{output_dir}mesh.msh",
)

if is_serial:
    mesh.dm.view()

# %%
x, y = mesh.CoordinateSystem.X
r, th = mesh.CoordinateSystem.xR
unit_rvec = mesh.CoordinateSystem.unit_e_0

# %% [markdown]
# ### Create Mesh Variables

# %%
v_soln = uw.discretisation.MeshVariable(
    r"{V_u}",
    mesh,
    mesh.data.shape[1],
    degree=vdegree,
)
p_soln = uw.discretisation.MeshVariable(
    r"{P_u}",
    mesh,
    1,
    degree=pdegree,
    continuous=pcont,
)

# %%
# Analytical solution and error expressions
v_ana_expr, p_ana_expr, rho_ana_expr = analytic_solution(
    mesh,
    r_i,
    r_o,
    k,
)
v_err_expr = sp.Matrix(v_soln.sym).T - v_ana_expr
p_err_expr = p_soln.sym[0] - p_ana_expr

# %% [markdown]
# ### Stokes

# %% [markdown]
# #### Stokes Setup

# %%
stokes = Stokes(
    mesh,
    velocityField=v_soln,
    pressureField=p_soln,
)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

gravity_fn = -1.0 * unit_rvec
stokes.bodyforce = rho_ana_expr * gravity_fn

# %% [markdown]
# #### Boundary Conditions

# %%
stokes.add_essential_bc(
    v_ana_expr,
    mesh.boundaries.Upper.name,
)
stokes.add_essential_bc(
    v_ana_expr,
    mesh.boundaries.Lower.name,
)

if k == 0:
    stokes.add_condition(
        p_soln.field_id,
        "dirichlet",
        sp.Matrix([0]),
        mesh.boundaries.Lower.name,
        components=(0),
    )
    stokes.add_condition(
        p_soln.field_id,
        "dirichlet",
        sp.Matrix([0]),
        mesh.boundaries.Upper.name,
        components=(0),
    )

# %% [markdown]
# #### Solver Settings

# %%
stokes.tolerance = stokes_tol
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
timing.reset()
timing.start()
stokes.solve(verbose=False, debug=False)
timing.stop()
timing.print_table(display_fraction=0.999, output_file=f"{output_dir}/stokes_timing.txt")

if uw.mpi.rank == 0:
    print(stokes.snes.getConvergedReason())
    print(stokes.snes.ksp.getConvergedReason())


# %% [markdown]
# ### Relative Error Norms

# %%
def relative_l2_error(mesh, err_expr, ana_expr):
    """Relative L2 error for scalar or vector expressions."""

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
if analytical:

    v_err_l2 = relative_l2_error(mesh, v_err_expr, v_ana_expr)
    p_err_l2 = np.inf if k == 0 else relative_l2_error(mesh, p_err_expr, p_ana_expr)

    if uw.mpi.rank == 0:
        print("Relative velocity L2 error:", v_err_l2)

        if k == 0:
            print("Pressure L2 error undefined for k=0.")
        else:
            print("Relative pressure L2 error:", p_err_l2)

# %% [markdown]
# ### Save Outputs

# %%
if uw.mpi.rank == 0:
    err_h5 = os.path.join(output_dir, "error_norm.h5")
    if os.path.isfile(err_h5):
        os.remove(err_h5)
    with h5py.File(err_h5,"w") as f_h5:
        f_h5.create_dataset("k", data=k)
        f_h5.create_dataset("cellsize", data=cellsize)
        f_h5.create_dataset("v_l2_norm", data=v_err_l2)
        f_h5.create_dataset("p_l2_norm", data=p_err_l2)

# %%
# Save solution checkpoint for post-processing script
mesh.petsc_save_checkpoint(
    index=0,
    meshVars=[v_soln, p_soln],
    outputPath=os.path.relpath(output_dir)+'/output',
)

# %%
