#!/usr/bin/env python3
# %% [markdown]
# # Exact annulus mesh with level-based refinement (Notebook-first)
#
# This reproduces the exact annulus mesh used by the authors in the Kramer benchmark paper.
#
# Base setup:
# - `r_inner = 1.22`
# - `r_outer = 2.22`
# - `n_theta_base = 128`
# - `n_radial_base = 16`
# - base triangles = `4096`
#
# Refinement level `L`:
# - `n_theta = n_theta_base * 2^L`
# - `n_radial = n_radial_base * 2^L`
# - triangles = `4096 * 4^L`
#
# Physical groups:
# - `Lower` (tag=1)
# - `Upper` (tag=2)
# - `Elements` (tag=666666)

from __future__ import annotations

import os

import gmsh

# %% [markdown]
# ## Parameters (edit these in notebook)

# %%

r_inner = 1.22
r_outer = 2.22

n_theta_base = 128
n_radial_base = 16
level = 5

msh_version = 4.1
show_gui = True

output_path = f'/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/output/kramer_annulus_mesh/mesh_level_{level}.msh'


# %%
def build_annulus_mesh(
    output_path: str,
    r_inner: float,
    r_outer: float,
    n_theta_base: int,
    n_radial_base: int,
    level: int,
    msh_version: float,
    show_gui: bool,
) -> tuple[int, int, int]:
    if level < 0:
        raise ValueError("level must be >= 0")

    n_theta = n_theta_base * (2**level)
    n_radial = n_radial_base * (2**level)

    if n_theta <= 0 or n_radial <= 0:
        raise ValueError("n_theta and n_radial must be positive.")
    if n_theta % 4 != 0:
        raise ValueError("n_theta must be divisible by 4 for quarter-arc transfinite setup.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n_theta_quarter = n_theta // 4
    theta_nodes = n_theta_quarter + 1
    radial_nodes = n_radial + 1

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", msh_version)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.model.add("AnnulusStructured")

        p_c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)

        p_i0 = gmsh.model.geo.addPoint(r_inner, 0.0, 0.0)
        p_i1 = gmsh.model.geo.addPoint(0.0, r_inner, 0.0)
        p_i2 = gmsh.model.geo.addPoint(-r_inner, 0.0, 0.0)
        p_i3 = gmsh.model.geo.addPoint(0.0, -r_inner, 0.0)

        p_o0 = gmsh.model.geo.addPoint(r_outer, 0.0, 0.0)
        p_o1 = gmsh.model.geo.addPoint(0.0, r_outer, 0.0)
        p_o2 = gmsh.model.geo.addPoint(-r_outer, 0.0, 0.0)
        p_o3 = gmsh.model.geo.addPoint(0.0, -r_outer, 0.0)

        a_i01 = gmsh.model.geo.addCircleArc(p_i0, p_c, p_i1)
        a_i12 = gmsh.model.geo.addCircleArc(p_i1, p_c, p_i2)
        a_i23 = gmsh.model.geo.addCircleArc(p_i2, p_c, p_i3)
        a_i30 = gmsh.model.geo.addCircleArc(p_i3, p_c, p_i0)

        a_o01 = gmsh.model.geo.addCircleArc(p_o0, p_c, p_o1)
        a_o12 = gmsh.model.geo.addCircleArc(p_o1, p_c, p_o2)
        a_o23 = gmsh.model.geo.addCircleArc(p_o2, p_c, p_o3)
        a_o30 = gmsh.model.geo.addCircleArc(p_o3, p_c, p_o0)

        l_0 = gmsh.model.geo.addLine(p_i0, p_o0)
        l_1 = gmsh.model.geo.addLine(p_i1, p_o1)
        l_2 = gmsh.model.geo.addLine(p_i2, p_o2)
        l_3 = gmsh.model.geo.addLine(p_i3, p_o3)

        cl_q1 = gmsh.model.geo.addCurveLoop([l_0, a_o01, -l_1, -a_i01])
        cl_q2 = gmsh.model.geo.addCurveLoop([l_1, a_o12, -l_2, -a_i12])
        cl_q3 = gmsh.model.geo.addCurveLoop([l_2, a_o23, -l_3, -a_i23])
        cl_q4 = gmsh.model.geo.addCurveLoop([l_3, a_o30, -l_0, -a_i30])

        s_q1 = gmsh.model.geo.addPlaneSurface([cl_q1])
        s_q2 = gmsh.model.geo.addPlaneSurface([cl_q2])
        s_q3 = gmsh.model.geo.addPlaneSurface([cl_q3])
        s_q4 = gmsh.model.geo.addPlaneSurface([cl_q4])

        gmsh.model.geo.synchronize()

        for ctag in [a_i01, a_i12, a_i23, a_i30, a_o01, a_o12, a_o23, a_o30]:
            gmsh.model.mesh.setTransfiniteCurve(ctag, theta_nodes)
        for ctag in [l_0, l_1, l_2, l_3]:
            gmsh.model.mesh.setTransfiniteCurve(ctag, radial_nodes)
        for stag in [s_q1, s_q2, s_q3, s_q4]:
            gmsh.model.mesh.setTransfiniteSurface(stag)

        gmsh.model.addPhysicalGroup(1, [a_i01, a_i12, a_i23, a_i30], 1, name="Lower")
        gmsh.model.addPhysicalGroup(1, [a_o01, a_o12, a_o23, a_o30], 2, name="Upper")
        gmsh.model.addPhysicalGroup(2, [s_q1, s_q2, s_q3, s_q4], 666666, name="Elements")

        gmsh.model.mesh.generate(2)

        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
        n_triangles = 0
        for etype, etags in zip(elem_types, elem_tags):
            if etype == 2:
                n_triangles += len(etags)

        expected = n_theta * n_radial * 2
        if n_triangles != expected:
            raise RuntimeError(
                f"Triangle count mismatch: got {n_triangles}, expected {expected} "
                f"(n_theta={n_theta}, n_radial={n_radial})."
            )

        gmsh.write(output_path)
        if show_gui:
            gmsh.fltk.run()
        return n_triangles, n_theta, n_radial
    finally:
        gmsh.finalize()


# %% [markdown]
# ## Run

# %%
n_tri, n_theta, n_radial = build_annulus_mesh(
    output_path=output_path,
    r_inner=r_inner,
    r_outer=r_outer,
    n_theta_base=n_theta_base,
    n_radial_base=n_radial_base,
    level=level,
    msh_version=msh_version,
    show_gui=show_gui,
)
print(f"Wrote mesh: {output_path}")
print(f"Refinement level: {level} | n_theta={n_theta}, n_radial={n_radial}")
print(f"Triangle count: {n_tri} (expected {n_theta * n_radial * 2})")
