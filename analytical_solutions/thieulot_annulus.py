import sympy as sp


def analytic_solution(
    mesh,
    r_i,
    r_o,
    k,
    C=-1,
    rho0=0,
):
    """Return analytical annulus benchmark fields (v, p, rho) as UW expressions."""

    r, th = mesh.CoordinateSystem.xR

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
