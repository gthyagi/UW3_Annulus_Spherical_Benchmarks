import sympy as sp


def _coefficients(m, r_i=sp.Rational(1, 2), r_o=sp.Integer(1), gamma=sp.Integer(1)):
    if m == -1:
        alpha = -gamma * (
            (r_o**3 - r_i**3)
            / ((r_o**3) * sp.log(r_i) - (r_i**3) * sp.log(r_o))
        )
        beta = -sp.Integer(3) * gamma * (
            (sp.log(r_o) - sp.log(r_i))
            / ((r_i**3) * sp.log(r_o) - (r_o**3) * sp.log(r_i))
        )
    else:
        alpha = gamma * (m + 1) * (
            (r_i**-3 - r_o**-3) / ((r_i ** -(m + 4)) - (r_o ** -(m + 4)))
        )
        beta = -sp.Integer(3) * gamma * (
            ((r_i ** (m + 1)) - (r_o ** (m + 1)))
            / ((r_i ** (m + 4)) - (r_o ** (m + 4)))
        )
    return sp.simplify(alpha), sp.simplify(beta)


def _bodyforce_coefficient(m):
    r = sp.symbols("r", positive=True)
    alpha, beta = _coefficients(m)
    gamma = sp.Integer(1)
    mu = r ** (m + 1)
    f = alpha * r ** (-(m + 3)) + beta * r

    if m == -1:
        g = (-sp.Integer(2) / r**2) * (
            alpha * sp.log(r) + (beta / sp.Integer(3)) * r**3 + gamma
        )
        h = (sp.Integer(2) / r) * g
        rho_expected = (
            (alpha / r**4) * (8 * sp.log(r) - 6)
            + 8 * beta / (3 * r)
            + 8 * gamma / r**4
        )
    else:
        m_plus_1 = sp.Integer(m + 1)
        m_plus_3 = sp.Integer(m + 3)
        m_minus_1 = sp.Integer(m - 1)
        m_plus_5 = sp.Integer(m + 5)
        g = (-sp.Integer(2) / r**2) * (
            (-alpha / m_plus_1) * r ** (-(m + 1))
            + (beta / sp.Integer(3)) * r**3
            + gamma
        )
        h = (m_plus_3 / r) * mu * g
        rho_expected = (r**m) * (
            2 * alpha * r ** (-(m + 4)) * (m_plus_3 / m_plus_1) * m_minus_1
            - (sp.Integer(2) * beta / sp.Integer(3)) * m_minus_1 * m_plus_3
            - sp.Integer(m) * m_plus_5 * (2 * gamma / r**3)
        )

    sigma_rr = 2 * mu * sp.diff(g, r) - h
    sigma_rtheta = mu * (sp.diff(f, r) - (f + g) / r)
    sigma_thetatheta = 2 * mu * (f + g) / r - h

    div_sigma_r = (
        (1 / r**2) * sp.diff(r**2 * sigma_rr, r)
        + (2 * sigma_rtheta) / r
        - (sigma_thetatheta + sigma_thetatheta) / r
    )
    bodyforce_coefficient = sp.simplify(-div_sigma_r)

    return sp.simplify(bodyforce_coefficient), sp.simplify(rho_expected)


def test_spherical_thieulot_bodyforce_matches_stress_divergence():
    for m in (-1, 1, 2, 3):
        bodyforce_coefficient, rho_expected = _bodyforce_coefficient(m)
        assert sp.simplify(bodyforce_coefficient - rho_expected) == 0
