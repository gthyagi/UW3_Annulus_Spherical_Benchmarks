#!/usr/bin/env python3

import os
import sys
from fractions import Fraction

import gmsh
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import underworld3 as uw

from underworld3.discretisation import _from_gmsh


uw.parse_cmd_line_options()

params = uw.Params(
    uw_radius_inner=uw.Param(1.22, type=uw.ParamType.FLOAT, description="Inner radius"),
    uw_radius_internal=uw.Param(
        None,
        type=uw.ParamType.STRING,
        description="Internal interface radius. Leave as None for a simple spherical shell.",
    ),
    uw_radius_outer=uw.Param(2.22, type=uw.ParamType.FLOAT, description="Outer radius"),
    uw_cellsize=uw.Param(
        "1/8",
        type=uw.ParamType.STRING,
        description="Target spherical-shell mesh cell size",
    ),
    uw_gadi_mesh_dir=uw.Param(
        "/g/data/m18/tg7098/spherical_mesh_gmsh",
        type=uw.ParamType.STRING,
        description="Directory for spherical .msh and .msh.h5 files",
    ),
    uw_benchmark=uw.Param(
        "kramer",
        type=uw.ParamType.STRING,
        description="kramer or thieulot benchmark",
    ),
)


if MPI.COMM_WORLD.size != 1:
    raise RuntimeError("Run this script in serial only.")

if any(arg in ("--help", "-h", "-help", "-uw_help") for arg in sys.argv[1:]):
    print(params.cli_help())
    raise SystemExit(0)


params.uw_cellsize = float(Fraction(str(params.uw_cellsize).replace(" ", "")))
has_internal_boundary = str(params.uw_radius_internal).strip().lower() not in ("", "none")
inv_cellsize = int(round(1.0 / params.uw_cellsize))

if params.uw_benchmark == "thieulot":
    params.uw_gadi_mesh_dir = os.path.join(params.uw_gadi_mesh_dir, "thieulot")
    filename = (
        f"uw_spherical_shell_ro{params.uw_radius_outer:g}_ri{params.uw_radius_inner:g}"
        f"_inv_cellsize{inv_cellsize}.msh"
    )

elif params.uw_benchmark == "kramer" and not has_internal_boundary:
    params.uw_gadi_mesh_dir = os.path.join(params.uw_gadi_mesh_dir, "kramer")
    filename = (
        f"uw_spherical_shell_ro{params.uw_radius_outer:g}_ri{params.uw_radius_inner:g}"
        f"_inv_cellsize{inv_cellsize}.msh"
    )

elif params.uw_benchmark == "kramer":
    params.uw_gadi_mesh_dir = os.path.join(params.uw_gadi_mesh_dir, "kramer")
    filename = (
        f"uw_spherical_shell_ro{params.uw_radius_outer:g}"
        f"_rint{float(params.uw_radius_internal):g}"
        f"_ri{params.uw_radius_inner:g}"
        f"_inv_cellsize{inv_cellsize}.msh"
    )

else:
    raise ValueError(f"Unknown benchmark: {params.uw_benchmark}")

os.makedirs(params.uw_gadi_mesh_dir, exist_ok=True)

mesh_path = os.path.join(params.uw_gadi_mesh_dir, filename)
mesh_h5_path = f"{mesh_path}.h5"


def create_gmsh_mesh(mesh_path: str) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)

    try:
        if params.uw_benchmark == "thieulot" or (
            params.uw_benchmark == "kramer" and not has_internal_boundary
        ):
            gmsh.model.add("Sphere")

            gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=params.uw_cellsize)

            outer = gmsh.model.occ.addSphere(0, 0, 0, params.uw_radius_outer)
            inner = gmsh.model.occ.addSphere(0, 0, 0, params.uw_radius_inner)

            gmsh.model.occ.cut(
                [(3, outer)],
                [(3, inner)],
                removeObject=True,
                removeTool=True,
            )

            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", params.uw_cellsize)
            gmsh.model.occ.synchronize()

            for dim, tag in gmsh.model.getEntities(2):
                rmax = gmsh.model.get_bounding_box(dim, tag)[-1]

                if np.isclose(rmax, params.uw_radius_inner):
                    gmsh.model.addPhysicalGroup(dim, [tag], 11, name="Lower")
                elif np.isclose(rmax, params.uw_radius_outer):
                    gmsh.model.addPhysicalGroup(dim, [tag], 12, name="Upper")

            volume = gmsh.model.getEntities(3)[0]
            gmsh.model.addPhysicalGroup(volume[0], [volume[1]], 99999)
            gmsh.model.setPhysicalName(volume[0], 99999, "Elements")

        else:
            gmsh.model.add("SphereShell_with_Internal_Surface")

            gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=params.uw_cellsize)

            outer = gmsh.model.occ.addSphere(0, 0, 0, params.uw_radius_outer)
            inner = gmsh.model.occ.addSphere(0, 0, 0, params.uw_radius_inner)

            gmsh.model.occ.cut(
                [(3, outer)],
                [(3, inner)],
                removeObject=True,
                removeTool=True,
            )

            internal = gmsh.model.occ.addSphere(
                0.0,
                0.0,
                0.0,
                float(params.uw_radius_internal),
            )
            inner_copy = gmsh.model.occ.addSphere(0, 0, 0, params.uw_radius_inner)

            gmsh.model.occ.cut(
                [(3, internal)],
                [(3, inner_copy)],
                removeObject=True,
                removeTool=True,
            )

            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", params.uw_cellsize)
            gmsh.model.occ.synchronize()

            gmsh.model.mesh.embed(2, [6], 3, 1)
            gmsh.model.remove_entities([(3, 2)], [(2, 5)])
            gmsh.model.occ.remove([(3, 2)], [(2, 5)])

            for dim, tag in gmsh.model.getEntities(2):
                rmax = gmsh.model.get_bounding_box(dim, tag)[-1]

                if np.isclose(rmax, params.uw_radius_inner):
                    gmsh.model.addPhysicalGroup(dim, [tag], 11, name="Lower")
                elif np.isclose(rmax, float(params.uw_radius_internal)):
                    gmsh.model.addPhysicalGroup(dim, [tag], 12, name="Internal")
                elif np.isclose(rmax, params.uw_radius_outer):
                    gmsh.model.addPhysicalGroup(dim, [tag], 13, name="Upper")

            volume = gmsh.model.getEntities(3)[0]
            gmsh.model.addPhysicalGroup(volume[0], [volume[1]], 99999)
            gmsh.model.setPhysicalName(volume[0], 99999, "Elements")

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(mesh_path)

    finally:
        gmsh.finalize()


if os.path.exists(mesh_path):
    print(f"Using existing mesh: {mesh_path}")
else:
    print(f"Creating mesh: {mesh_path}")
    create_gmsh_mesh(mesh_path)

if os.path.exists(mesh_h5_path):
    print(f"Using existing mesh h5: {mesh_h5_path}")
else:
    print(f"Creating mesh h5 from existing .msh: {mesh_h5_path}")
    _from_gmsh(
        mesh_path,
        comm=PETSc.COMM_SELF,
        markVertices=True,
        useRegions=True,
        useMultipleTags=True,
    )

if MPI.COMM_WORLD.rank == 0:
    print(f"Mesh path:    {mesh_path}")
    print(f"Mesh h5 path: {mesh_h5_path}")