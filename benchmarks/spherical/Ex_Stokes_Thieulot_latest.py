# %% [markdown]
# ## Spherical Benchmark: Viscous Incompressible Stokes
#
# #### [Benchmark Paper](https://se.copernicus.org/articles/8/1181/2017/) [ASPECT Results](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/hollow_sphere/doc/hollow_sphere.html)
#
# ### Authors
# Thyagarajulu Gollapalli ([GitHub](https://github.com/gthyagi)) <br>
# Underworld3 Development Team ([UW3 Repository](https://github.com/underworldcode/underworld3))
#
# ### Analytical solution
#
# This benchmark is based on [Thieulot](https://se.copernicus.org/articles/8/1181/2017/) in which an analytical solution to the isoviscous incompressible Stokes equations is derived in a spherical shell geometry. The velocity and pressure fields are as follows:
#
# $$ v_{\theta}(r, \theta) = f(r) \sin(\theta) $$
# $$ v_{\phi}(r, \theta) = f(r) \sin(\theta) $$
# $$ v_r(r, \theta) = g(r) \cos(\theta) $$
# $$ p(r, \theta) = h(r) \cos(\theta) $$
# $$ \mu(r) = \mu_{0}r^{m+1} $$
#
# where $m$ is an integer (positive or negative). Note that $m = -1$ yields a constant viscosity.
#
# $$ f(r) = {\alpha} r^{-(m+3)} + \beta r $$
#
# ##### Case $m = -1$
#
# $$ g(r) = -\frac{2}{r^2} \bigg(\alpha \ln r + \frac{\beta}{3}r^3 + \gamma \bigg) $$
# $$ h(r) = \frac{2}{r} \mu_{0} g(r) $$
# $$ \rho(r, \theta) = \bigg(\frac{\alpha}{r^4} (8\ln r - 6) + \frac{8\beta}{3r} + 8\frac{\gamma}{r^4} \bigg) \cos(\theta)$$
# $$ \alpha = -\gamma \frac{R_2^3 - R_1^3}{R_2^3 \ln R_1 - R_1^3 \ln R_2} $$
# $$ \beta = -3\gamma \frac{\ln R_2 - \ln R_1}{R_1^3 \ln R_2 - R_2^3 \ln R_1} $$
#
# ##### Case $m \neq -1$
#
# $$ g(r) = -\frac{2}{r^2} \bigg(-\frac{\alpha}{m+1} r^{-(m+1)} + \frac{\beta}{3}r^3 + \gamma \bigg) $$
# $$ h(r) = \frac{m+3}{r} \mu(r) g(r) $$
# $$ \rho(r, \theta) = \bigg[2\alpha r^{-(m+4)}\frac{m+3}{m+1}(m-1) - \frac{2\beta}{3}(m-1)(m+3) - m(m+5)\frac{2\gamma}{r^3} \bigg] \cos(\theta) $$
# $$ \alpha = \gamma (m+1) \frac{R_1^{-3} - R_2^{-3}}{R_1^{-(m+4)} - R_2^{-(m+4)}} $$
# $$ \beta = -3\gamma \frac{R_1^{m+1} - R_2^{m+1}}{R_1^{m+4} - R_2^{m+4}} $$
# Note that this imposes that $m \neq -4$.
#
# The radial component of the velocity is zero on the inside $r = R_1$ and outside $r = R_2$ of the domain, thereby ensuring a tangential flow on the boundaries, i.e.
# $$ v_r(R_1, \theta) = v_r(R_2, \theta) = 0 $$
#
# The gravity vector is radial and of unit length. We set $R_1 = 1$ and $R_2 = 2$.
#
# In this work, the following spherical coordinates conventions are used: $r$ is the radial distance, $\theta \in [0,\pi]$ is the polar angle and $\phi \in [0, 2\pi]$ is the azimuthal angle.

# %%
import os
import h5py
import numpy as np
import sympy as sp
import underworld3 as uw
from underworld3.systems import Stokes

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

is_serial = (uw.mpi.size == 1)

# %% [markdown]
# ### Mesh Parameters

# %%
params = uw.Params(
    uw_cellsize=uw.Param(
        1.0 / 8.0,
        description="Target spherical-shell mesh cell size",
    ),
    uw_r_i=uw.Param(
        1.0,
        description="Inner spherical-shell radius",
    ),
    uw_r_o=uw.Param(
        2.0,
        description="Outer spherical-shell radius",
    ),
    uw_m=uw.Param(
        -1,
        description="Viscosity exponent in the analytical solution",
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
        1.0e8,
        description="Penalty for curved-boundary tangential flow",
    ),
    uw_analytical=uw.Param(
        True,
        description="Enable analytical error norms",
    ),
)

if int(params.uw_m) == -4:
    raise ValueError("The Thieulot spherical benchmark is undefined for m = -4.")

# %% [markdown]
# ### Output Directory

# %%
output_dir = os.path.join(
    "../../output/spherical/thieulot/latest/",
    (
        f"case_inv_lc_{int(1/params.uw_cellsize)}_m_{int(params.uw_m)}_vdeg_{int(params.uw_vdegree)}_pdeg_{int(params.uw_pdegree)}"
        f"_pcont_{str(bool(params.uw_pcont)).lower()}_vel_penalty_{params.uw_vel_penalty:.2g}"
        f"_stokes_tol_{params.uw_stokes_tol:.2g}_ncpus_{uw.mpi.size}/"
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
    m,
    gamma=1.0,
    mu_0=1.0,
):
    """Return spherical benchmark fields (v, p, rho, bodyforce-rho, mu) as UW expressions."""

    r = mesh.CoordinateSystem.xR[0]
    theta = mesh.CoordinateSystem.xR[1]
    phi_raw = mesh.CoordinateSystem.xR[2]
    phi = sp.Piecewise(
        (2 * sp.pi + phi_raw, phi_raw < 0),
        (phi_raw, True),
    )

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

        force_term = (
            -(r * sp.diff(f, r, 3))
            - (3.0 * sp.diff(f, r, 2))
            + ((2.0 * sp.diff(f, r) / r) - sp.diff(g, r, 2))
            + 2.0 * ((f + g) / r**2)
        )
        rho_expr = sp.simplify(force_term * sp.cos(theta))
        rho_bodyforce_expr = -rho_expr
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

        force_term = (
            (-r**2) * sp.diff(f, r, 3)
            - ((2 * m) + 5) * r * sp.diff(f, r, 2)
            - (m * (m + 3)) * sp.diff(f, r)
            + (m * (m + 3) + 4) * ((f + g) / r)
            - (m + 1) * sp.diff(g, r)
            - r * sp.diff(g, r, 2)
        )
        rho_expr = sp.simplify((r**m) * force_term * sp.cos(theta))
        rho_bodyforce_expr = rho_expr

    p_expr = h * sp.cos(theta)

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

    return v_expr, p_expr, rho_expr, rho_bodyforce_expr, mu_expr


# %% [markdown]
# ### Create Mesh

# %%
mesh = uw.meshing.SphericalShell(
    radiusInner=params.uw_r_i,
    radiusOuter=params.uw_r_o,
    cellSize=params.uw_cellsize,
    qdegree=max(params.uw_pdegree, params.uw_vdegree),
    degree=1,
    filename=f"{output_dir}mesh.msh",
)

if is_serial:
    mesh.dm.view()

# %%
x, y, z = mesh.CoordinateSystem.X
unit_rvec = mesh.CoordinateSystem.unit_e_0

# %% [markdown]
# ### Create Mesh Variables

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

# %%
v_ana_expr, p_ana_expr, rho_ana_expr, rho_bodyforce_expr, mu_expr = analytic_solution(
    mesh,
    params.uw_r_i,
    params.uw_r_o,
    int(params.uw_m),
)
v_err_expr = sp.Matrix(v_soln.sym).T - v_ana_expr
p_err_expr = p_soln.sym[0] - p_ana_expr

# %% [markdown]
# ### Stokes

# %% [markdown]
# #### Stokes Setup

# %%
stokes = Stokes(mesh, velocityField=v_soln, pressureField=p_soln)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = mu_expr
stokes.saddle_preconditioner = 1.0 / mu_expr

gravity_fn = -1.0 * unit_rvec
stokes.bodyforce = rho_bodyforce_expr * gravity_fn

# %% [markdown]
# #### Boundary Conditions

# %%
stokes.add_natural_bc(params.uw_vel_penalty * v_err_expr, mesh.boundaries.Upper.name)
stokes.add_natural_bc(params.uw_vel_penalty * v_err_expr, mesh.boundaries.Lower.name)

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
# ### Benchmark Calibrations
#
# In a 3-D spherical shell, the velocity null space contains the three rigid-body
# rotation modes
#
# $$ \mathbf{r}_x = (0, -z, y), \qquad \mathbf{r}_y = (z, 0, -x), \qquad \mathbf{r}_z = (-y, x, 0), $$
#
# which are the Cartesian basis fields for
#
# $$ \mathbf{u}_{\mathrm{rot}} = \boldsymbol{\omega} \times \mathbf{x}. $$
#
# We remove them by projecting the numerical velocity onto this basis. With
# $G_{ij} = \int_{\Omega} \mathbf{r}_i \cdot \mathbf{r}_j$ and
# $b_i = \int_{\Omega} \mathbf{r}_i \cdot \mathbf{u}$, we solve
#
# $$ G \mathbf{c} = \mathbf{b}, $$
#
# and subtract
#
# $$ \mathbf{u} \leftarrow \mathbf{u} - \sum_i c_i \mathbf{r}_i. $$

# %%
def subtract_pressure_mean(mesh, pressure_var):
    """
    Subtract the domain-average pressure from the numerical pressure field.

    Parameters
    ----------
    mesh : uw.discretisation.Mesh
        Mesh used to evaluate the pressure and volume integrals.
    pressure_var : uw.discretisation.MeshVariable
        Scalar pressure field to shift to zero mean.
    """
    p_int = uw.maths.Integral(mesh, pressure_var.sym[0]).evaluate()
    volume = uw.maths.Integral(mesh, 1.0).evaluate()
    pressure_var.data[:, 0] -= p_int / volume


def subtract_rigid_rotations(mesh, velocity_var, rotation_modes):
    """
    Remove rigid-body rotation components from the numerical velocity field.

    Parameters
    ----------
    mesh : uw.discretisation.Mesh
        Mesh used to evaluate the projection integrals.
    velocity_var : uw.discretisation.MeshVariable
        Vector velocity field to correct.
    rotation_modes : list[sympy.Matrix]
        Rigid-body rotation null modes, here the Cartesian basis fields for `omega x x`.
    """
    velocity_expr = sp.Matrix(velocity_var.sym).T
    nmodes = len(rotation_modes)

    gram = np.zeros((nmodes, nmodes))
    rhs = np.zeros(nmodes)

    for i, mode_i in enumerate(rotation_modes):
        rhs[i] = uw.maths.Integral(mesh, mode_i.dot(velocity_expr)).evaluate()
        for j, mode_j in enumerate(rotation_modes):
            gram[i, j] = uw.maths.Integral(mesh, mode_i.dot(mode_j)).evaluate()

    coeffs = np.linalg.solve(gram, rhs)

    correction = coeffs[0] * rotation_modes[0]
    for coeff, mode in zip(coeffs[1:], rotation_modes[1:]):
        correction += coeff * mode

    dv = uw.function.evaluate(correction, velocity_var.coords)
    velocity_var.data[...] -= np.asarray(dv).reshape(velocity_var.data.shape)


# %%
rotation_modes = [
    sp.Matrix([0, -z, y]),
    sp.Matrix([z, 0, -x]),
    sp.Matrix([-y, x, 0]),
]
subtract_rigid_rotations(mesh, v_soln, rotation_modes)
subtract_pressure_mean(mesh, p_soln)

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
if params.uw_analytical:
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
        f_h5.create_dataset("cellsize", data=float(params.uw_cellsize))
        f_h5.create_dataset("v_l2_norm", data=v_err_l2)
        f_h5.create_dataset("p_l2_norm", data=p_err_l2)

# %%
mesh.petsc_save_checkpoint(
    index=0,
    meshVars=[v_soln, p_soln],
    outputPath=os.path.relpath(output_dir) + "/output",
)
