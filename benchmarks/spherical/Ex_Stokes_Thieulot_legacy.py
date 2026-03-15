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
# where $m$ is an integer (positive or negative). Note that $m = −1$ yields a constant viscosity.
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
# Note that this imposes that $m \neq −4$.
#
# The radial component of the velocity is nul on the inside $r = R_1$ and outside $r = R_2$ of the domain, thereby insuring a
# tangential flow on the boundaries, i.e.
# $$ v_r(R_1, \theta) = v_r(R_2, \theta) = 0 $$
#
# The gravity vector is radial and of unit length. We set $R_1 = 0.5$ and $R_2 = 1$.
#
# In this work, the following spherical coordinates conventions are used: $r$ is the radial distance, $\theta \in [0,\pi]$ is the polar angle and $\phi \in [0, 2\pi]$ is the azimuthal angle.

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

is_serial = uw.mpi.size == 1

# %% [markdown]
# ### Mesh Parameters

# %%
cellsize = uw.options.getReal(
    "cellsize",
    default=1. / 8,
)
r_i = uw.options.getReal(
    "radius_inner",
    default=0.5,
)
r_o = uw.options.getReal(
    "radius_outer",
    default=1.0,
)

# %% [markdown]
# ### Viscosity Exponent

# %%
m = uw.options.getInt(
    "m",
    default=-1,
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
vel_penalty = uw.options.getReal(
    "vel_penalty",
    default=1e8,
)
stokes_tol = uw.options.getReal(
    "stokes_tol",
    default=1e-10,
)

# %% [markdown]
# ### Output Directory

# %%
output_dir = os.path.join(
    "../../output/spherical/thieulot/legacy/",
    (
        f"case_inv_lc_{int(1/cellsize)}_m_{m}_vdeg_{vdegree}_pdeg_{pdegree}"
        f"_pcont_{pcont_str}_vel_penalty_{vel_penalty:.2g}"
        f"_stokes_tol_{stokes_tol:.2g}_ncpus_{uw.mpi.size}/"
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
    """Return spherical benchmark velocity, pressure, density, and viscosity."""

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
        beta = -3 * gamma * (
            (np.log(r_o) - np.log(r_i))
            / ((r_i**3) * np.log(r_o) - (r_o**3) * np.log(r_i))
        )

        f = alpha * (r ** -(m + 3)) + beta * r
        g = (-2 / (r**2)) * (alpha * sp.ln(r) + (beta / 3) * (r**3) + gamma)
        h = (2 / r) * mu_0 * g

        f_fd = sp.diff(f, r)
        f_sd = sp.diff(f_fd, r)
        f_td = sp.diff(f_sd, r)
        g_fd = sp.diff(g, r)
        g_sd = sp.diff(g_fd, r)

        force_term = (
            -(r * f_td)
            - (3 * f_sd)
            + ((2 * f_fd / r) - g_sd)
            + 2 * ((f + g) / r**2)
        )
        rho_expr = sp.simplify(force_term * sp.cos(theta))
        rho_bodyforce_expr = -rho_expr
    else:
        alpha = gamma * (m + 1) * (
            (r_i**-3 - r_o**-3) / ((r_i ** -(m + 4)) - (r_o ** -(m + 4)))
        )
        beta = -3 * gamma * (
            ((r_i ** (m + 1)) - (r_o ** (m + 1)))
            / ((r_i ** (m + 4)) - (r_o ** (m + 4)))
        )

        f = alpha * (r ** -(m + 3)) + beta * r
        g = (-2 / (r**2)) * (
            (-alpha / (m + 1)) * r ** (-(m + 1))
            + (beta / 3) * (r**3)
            + gamma
        )
        h = ((m + 3) / r) * mu_expr * g

        f_fd = sp.diff(f, r)
        f_sd = sp.diff(f_fd, r)
        f_td = sp.diff(f_sd, r)
        g_fd = sp.diff(g, r)
        g_sd = sp.diff(g_fd, r)

        force_term = (
            (-r**2) * f_td
            - ((2 * m) + 5) * r * f_sd
            - (m * (m + 3)) * f_fd
            + (m * (m + 3) + 4) * ((f + g) / r)
            - (m + 1) * g_fd
            - r * g_sd
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

    v_expr = sp.Matrix([[v_x, v_y, v_z]])

    return v_expr, p_expr, rho_expr, rho_bodyforce_expr, mu_expr


# %% [markdown]
# ### Create Mesh

# %%
mesh = uw.meshing.SphericalShell(
    radiusInner=r_i,
    radiusOuter=r_o,
    cellSize=cellsize,
    qdegree=max(pdegree, vdegree),
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
v_soln = uw.discretisation.MeshVariable("V_u", mesh, mesh.data.shape[1], degree=vdegree)
p_soln = uw.discretisation.MeshVariable("P_u", mesh, 1, degree=pdegree, continuous=pcont)

# %%
# Analytical solution and error expressions
v_ana_expr, p_ana_expr, rho_ana_expr, rho_bodyforce_expr, mu_expr = analytic_solution(
    mesh,
    r_i,
    r_o,
    m,
)
v_err_expr = sp.Matrix(v_soln.sym) - v_ana_expr
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
stokes.add_natural_bc(vel_penalty * v_err_expr, "Upper")
stokes.add_natural_bc(vel_penalty * v_err_expr, "Lower")

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
stokes.petsc_options["fieldsplit_pressure_mg_coarse_pc_type"] = "svd"

# %% [markdown]
# #### Solve Stokes

# %%
timing.reset()
timing.start()
stokes.solve(verbose=False, debug=False)
timing.stop()
timing.print_table(
    group_by="line_routine",
    output_file=f"{output_dir}stokes_solve_time.txt",
    display_fraction=1.00,
)

if uw.mpi.rank == 0:
    print(stokes.snes.getConvergedReason())
    print(stokes.snes.ksp.getConvergedReason())

# %% [markdown]
# ### Rigid-Body Rotation Calibration
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
def subtract_rigid_rotations(
    mesh,
    velocity_var,
    rotation_modes,
):
    """Remove rigid-body rotation components from a 3-D velocity field."""

    velocity_expr = sp.Matrix(velocity_var.sym)
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

    with mesh.access(velocity_var):
        dv = uw.function.evaluate(correction, velocity_var.coords)
        velocity_var.data[...] -= np.asarray(dv).reshape(velocity_var.data.shape)


def subtract_pressure_mean(
    mesh,
    pressure_var,
):
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
    with mesh.access(pressure_var):
        pressure_var.data[:, 0] -= p_int / volume

# %%
rotation_modes = [
    sp.Matrix([[0, -z, y]]),
    sp.Matrix([[z, 0, -x]]),
    sp.Matrix([[-y, x, 0]]),
]
subtract_rigid_rotations(mesh, v_soln, rotation_modes)
subtract_pressure_mean(mesh, p_soln)

# %% [markdown]
# ### Relative Error Norms

# %%
def relative_l2_error(
    mesh,
    err_expr,
    ana_expr,
):
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
v_err_l2 = relative_l2_error(mesh, v_err_expr, v_ana_expr)
p_err_l2 = relative_l2_error(mesh, p_err_expr, p_ana_expr)

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
        f_h5.create_dataset("m", data=m)
        f_h5.create_dataset("cellsize", data=cellsize)
        f_h5.create_dataset("v_l2_norm", data=v_err_l2)
        f_h5.create_dataset("p_l2_norm", data=p_err_l2)

# %%
# Serial post-processing plots are generated by thieulot_field_plots_legacy.py.
mesh.petsc_save_checkpoint(
    index=0,
    meshVars=[v_soln, p_soln],
    outputPath=os.path.relpath(output_dir) + "/output",
)
