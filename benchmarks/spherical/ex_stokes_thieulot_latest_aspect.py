# %% [markdown]
# ## Spherical Benchmark: Thieulot With ASPECT-Style Pressure Workflow
#
# This variant follows the ASPECT-style benchmark workflow more closely:
#
# - prescribe the analytical velocity on the inner and outer spheres
# - include a constant background density in the body force
# - compare against the total pressure
#   `p_total = p_dynamic + rho_0 g (R_2 - r)`
# - fix the pressure gauge by setting the average pressure on the outer surface to zero

# %%
import os

import h5py
import numpy as np
import sympy as sp
import underworld3 as uw
from underworld3.systems import Stokes

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"


# %% [markdown]
# ### Parameters

# %%
params = uw.Params(
    uw_cellsize=uw.Param(
        1.0 / 8.0,
        description="Target spherical-shell mesh cell size",
    ),
    uw_r_i=uw.Param(
        0.5,
        description="Inner spherical-shell radius",
    ),
    uw_r_o=uw.Param(
        1.0,
        description="Outer spherical-shell radius",
    ),
    uw_m=uw.Param(
        -1,
        description="Viscosity exponent in the analytical solution",
    ),
    uw_rho_0=uw.Param(
        1000.0,
        description="Background density used in the ASPECT-style workflow",
    ),
    uw_gravity=uw.Param(
        1.0,
        description="Gravity magnitude",
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
)

if int(params.uw_m) == -4:
    raise ValueError("The Thieulot spherical benchmark is undefined for m = -4.")


# %% [markdown]
# ### Output Directory

# %%
output_dir = os.path.join(
    "../../output/spherical/thieulot/latest_aspect/",
    (
        f"case_inv_lc_{int(1/params.uw_cellsize)}_m_{int(params.uw_m)}"
        f"_rho0_{params.uw_rho_0:.0f}_g_{params.uw_gravity:.1f}"
        f"_vdeg_{int(params.uw_vdegree)}_pdeg_{int(params.uw_pdegree)}"
        f"_pcont_{str(bool(params.uw_pcont)).lower()}"
        f"_stokes_tol_{params.uw_stokes_tol:.2g}_ncpus_{uw.mpi.size}/"
    ),
)
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)


# %% [markdown]
# ### Analytical Solution

# %%
def analytic_solution(
    mesh,
    r_i,
    r_o,
    m,
    rho_0,
    gravity,
):
    """
    Return ASPECT-style analytical velocity, pressure, density, and viscosity.

    The returned pressure is the total pressure:
    `p_total = p_dynamic + rho_0 * gravity * (r_o - r)`.
    """
    r = mesh.CoordinateSystem.xR[0]
    theta = mesh.CoordinateSystem.xR[1]
    phi_raw = mesh.CoordinateSystem.xR[2]
    phi = sp.Piecewise(
        (2 * sp.pi + phi_raw, phi_raw < 0),
        (phi_raw, True),
    )

    gamma = 1.0
    mu_0 = 1.0
    mu_expr = mu_0 * (r ** (m + 1))

    if m == -1:
        alpha = -gamma * (
            (r_o**3 - r_i**3)
            / ((r_o**3) * np.log(r_i) - (r_i**3) * np.log(r_o))
        )
        beta = -3.0 * gamma * (
            (np.log(r_o) - np.log(r_i))
            / ((r_i**3) * np.log(r_o) - (r_o**3) * np.log(r_i))
        )

        f = alpha * (r ** -(m + 3)) + beta * r
        g = (-2.0 / (r**2)) * (alpha * sp.log(r) + (beta / 3.0) * (r**3) + gamma)
        h = (2.0 / r) * mu_0 * g

        rho_dynamic_expr = (
            (alpha / r**4) * (8.0 * sp.log(r) - 6.0)
            + (8.0 * beta) / (3.0 * r)
            + (8.0 * gamma) / r**4
        ) * sp.cos(theta)
    else:
        alpha = gamma * (m + 1) * (
            (r_i**-3 - r_o**-3) / ((r_i ** -(m + 4)) - (r_o ** -(m + 4)))
        )
        beta = -3.0 * gamma * (
            ((r_i ** (m + 1)) - (r_o ** (m + 1)))
            / ((r_i ** (m + 4)) - (r_o ** (m + 4)))
        )

        f = alpha * (r ** -(m + 3)) + beta * r
        g = (-2.0 / (r**2)) * (
            (-alpha / (m + 1)) * r ** (-(m + 1))
            + (beta / 3.0) * (r**3)
            + gamma
        )
        h = ((m + 3) / r) * mu_expr * g

        rho_dynamic_expr = (
            2.0
            * alpha
            * r ** (-(m + 4))
            * ((m + 3) / (m + 1))
            * (m - 1)
            - (2.0 * beta / 3.0) * (m - 1) * (m + 3)
            - m * (m + 5) * (2.0 * gamma / r**3)
        ) * sp.cos(theta)

    p_dynamic_expr = h * sp.cos(theta)
    p_total_expr = p_dynamic_expr + rho_0 * gravity * (r_o - r)
    rho_total_expr = rho_0 - rho_dynamic_expr

    v_r = g * sp.cos(theta)
    v_theta = f * sp.sin(theta)
    v_phi = f * sp.sin(theta)

    v_x = (
        v_r * sp.sin(theta) * sp.cos(phi)
        + v_theta * sp.cos(theta) * sp.cos(phi)
        - v_phi * sp.sin(phi)
    )
    v_y = (
        v_r * sp.sin(theta) * sp.sin(phi)
        + v_theta * sp.cos(theta) * sp.sin(phi)
        + v_phi * sp.cos(phi)
    )
    v_z = v_r * sp.cos(theta) - v_theta * sp.sin(theta)

    v_expr = sp.Matrix([v_x, v_y, v_z])

    return v_expr, p_total_expr, rho_total_expr, mu_expr


# %% [markdown]
# ### Pressure Gauge Helper

# %%
def subtract_surface_pressure_mean(
    mesh,
    pressure_var,
    boundary_name,
):
    """
    Shift pressure so the average pressure on a named boundary is zero.
    """
    p_bd_int = uw.maths.BdIntegral(
        mesh=mesh,
        fn=pressure_var.sym[0],
        boundary=boundary_name,
    ).evaluate()
    bd_measure = uw.maths.BdIntegral(
        mesh=mesh,
        fn=1.0,
        boundary=boundary_name,
    ).evaluate()
    pressure_var.data[:, 0] -= p_bd_int / bd_measure


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


# %% [markdown]
# ### Mesh

# %%
mesh = uw.meshing.SphericalShell(
    radiusInner=params.uw_r_i,
    radiusOuter=params.uw_r_o,
    cellSize=params.uw_cellsize,
    qdegree=max(params.uw_pdegree, params.uw_vdegree),
    degree=1,
    filename=f"{output_dir}mesh.msh",
)

unit_rvec = mesh.CoordinateSystem.unit_e_0


# %% [markdown]
# ### Variables

# %%
v_soln = uw.discretisation.MeshVariable(
    varname="Velocity",
    mesh=mesh,
    degree=params.uw_vdegree,
    vtype=uw.VarType.VECTOR,
    varsymbol=r"V",
)

p_soln = uw.discretisation.MeshVariable(
    varname="Pressure",
    mesh=mesh,
    degree=params.uw_pdegree,
    vtype=uw.VarType.SCALAR,
    varsymbol=r"P",
    continuous=params.uw_pcont,
)


# %% [markdown]
# ### Analytical Fields

# %%
v_ana_expr, p_ana_expr, rho_ana_expr, mu_expr = analytic_solution(
    mesh=mesh,
    r_i=params.uw_r_i,
    r_o=params.uw_r_o,
    m=int(params.uw_m),
    rho_0=params.uw_rho_0,
    gravity=params.uw_gravity,
)

v_err_expr = sp.Matrix(v_soln.sym).T - v_ana_expr
p_err_expr = p_soln.sym[0] - p_ana_expr


# %% [markdown]
# ### Stokes

# %%
stokes = Stokes(mesh, velocityField=v_soln, pressureField=p_soln)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = mu_expr
stokes.saddle_preconditioner = 1.0 / mu_expr

gravity_fn = -float(params.uw_gravity) * unit_rvec
stokes.bodyforce = rho_ana_expr * gravity_fn

stokes.add_essential_bc(
    [v_ana_expr[0], v_ana_expr[1], v_ana_expr[2]],
    mesh.boundaries.Upper.name,
)
stokes.add_essential_bc(
    [v_ana_expr[0], v_ana_expr[1], v_ana_expr[2]],
    mesh.boundaries.Lower.name,
)


# %% [markdown]
# ### Solver Settings

# %%
stokes.tolerance = params.uw_stokes_tol
stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "preonly"
stokes.petsc_options["pc_type"] = "svd"
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor_true_residual"] = None


# %% [markdown]
# ### Solve

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
# ### ASPECT-Style Pressure Normalization

# %%
subtract_surface_pressure_mean(mesh, p_soln, mesh.boundaries.Upper.name)


# %% [markdown]
# ### Error Norms

# %%
v_err_l2 = relative_l2_error(mesh, v_err_expr, v_ana_expr)
p_err_l2 = relative_l2_error(mesh, p_err_expr, p_ana_expr)

uw.pprint("Relative velocity L2 error:", v_err_l2)
uw.pprint("Relative pressure L2 error:", p_err_l2)


# %% [markdown]
# ### Save Outputs

# %%
if uw.mpi.rank == 0:
    err_h5 = os.path.join(output_dir, "error_norm.h5")
    if os.path.isfile(err_h5):
        os.remove(err_h5)
    with h5py.File(err_h5, "w") as f_h5:
        f_h5.create_dataset("m", data=int(params.uw_m))
        f_h5.create_dataset("rho_0", data=float(params.uw_rho_0))
        f_h5.create_dataset("gravity", data=float(params.uw_gravity))
        f_h5.create_dataset("cellsize", data=float(params.uw_cellsize))
        f_h5.create_dataset("v_l2_norm", data=v_err_l2)
        f_h5.create_dataset("p_l2_norm", data=p_err_l2)

mesh.write_timestep(
    "output",
    index=0,
    meshVars=[v_soln, p_soln],
    outputPath=str(output_dir),
)
