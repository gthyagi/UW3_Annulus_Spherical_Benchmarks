import numpy as np
import sympy as sp


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

        # The implemented 3D Cartesian field requires an additional r**m factor
        # in the radial body-force coefficient for m != -1.
        rho_expr = sp.simplify(
            (r**m)
            * (
                2.0 * alpha * r ** (-(m + 4)) * ((m + 3) / (m + 1)) * (m - 1)
                - (2.0 * beta / 3.0) * (m - 1) * (m + 3)
                - m * (m + 5) * (2.0 * gamma / r**3)
            )
            * sp.cos(theta)
        )
        rho_bodyforce_expr = -rho_expr

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


def analytic_radial_stress(
    mesh,
    r_i,
    r_o,
    m,
    gamma=1.0,
    mu_0=1.0,
):
    """Return the analytical radial stress sigma_rr from Thieulot (2017), Sec. 4.6."""

    r = mesh.CoordinateSystem.xR[0]
    theta = mesh.CoordinateSystem.xR[1]
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

    sigma_rr_expr = (mu_expr / r) * (-((m + 7) * g + 4.0 * f) * sp.cos(theta))
    return sp.simplify(sigma_rr_expr)
