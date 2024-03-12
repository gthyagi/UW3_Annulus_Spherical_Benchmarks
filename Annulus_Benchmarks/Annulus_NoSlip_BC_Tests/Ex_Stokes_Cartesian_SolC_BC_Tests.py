# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Stokes Benchmark SolCx
#


# +
# %%
import petsc4py
from petsc4py import PETSc

import nest_asyncio
nest_asyncio.apply()

# options = PETSc.Options()
# options["help"] = None 

import os
os.environ["UW_TIMING_ENABLE"] = "1"


# +
import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
from underworld3 import timing

import numpy as np
import sympy
from sympy import Piecewise

# +
# %%
n_els = 4
refinement = 3

mesh1 = uw.meshing.UnstructuredSimplexBox(regular=True, minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), 
                                          cellSize=1/n_els, qdegree=3, refinement=refinement)

mesh2 = uw.meshing.UnstructuredSimplexBox(regular=True, minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), 
                                          cellSize=1/n_els, qdegree=3, refinement=refinement)
# -


# coordinate stuff
x1,y1 = mesh1.X
x2,y2 = mesh2.X

# +
# mesh variables
v1 = uw.discretisation.MeshVariable('V1', mesh1, 2, degree=2)
p1 = uw.discretisation.MeshVariable('P1', mesh1, 1, degree=1)

v2 = uw.discretisation.MeshVariable('V2', mesh2, 2, degree=2)
p2 = uw.discretisation.MeshVariable('P2', mesh2, 1, degree=1)

v_err = uw.discretisation.MeshVariable('Ve', mesh1, 2, degree=2)
p_err = uw.discretisation.MeshVariable('Pe', mesh1, 1, degree=1)

# +
stokes1 = uw.systems.Stokes(mesh1, velocityField=v1, pressureField=p1, verbose=True)
stokes1.constitutive_model=uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes1.Unknowns)
stokes1.constitutive_model.Parameters.shear_viscosity_0 = 1

stokes2 = uw.systems.Stokes(mesh2, velocityField=v2, pressureField=p2, verbose=True)
stokes2.constitutive_model=uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes2.Unknowns)
stokes2.constitutive_model.Parameters.shear_viscosity_0 = 1
# -
# constant values
eta_0 = 1.0
x_c = 0.5
f_0 = 1.0


# +
stokes1.penalty = 100.0
stokes1.bodyforce = sympy.Matrix([0, Piecewise((f_0, x1 > x_c), (0.0, True))])

stokes2.penalty = 100.0
stokes2.bodyforce = sympy.Matrix([0, Piecewise((f_0, x2 > x_c), (0.0, True))])

# +
# This is the other way to impose no vertical flow
# stokes.add_natural_bc([0.0, 1e5*v.sym[1]], "Top")              # Top "free slip / penalty"


# +
# free slip.

# stokes1.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes1.add_natural_bc([0.0, 2.5e6*v1.sym[1]], "Top") 
stokes1.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes1.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes1.add_dirichlet_bc((0.0, sympy.oo), "Right")

# stokes2.add_essential_bc((sympy.oo, 0.0), "Top")
# stokes2.add_essential_bc((sympy.oo,0.0), "Bottom")
# stokes2.add_essential_bc((0.0,sympy.oo), "Left")
# stokes2.add_essential_bc((0.0,sympy.oo), "Right")

# stokes2.add_natural_bc([0.0, 1e5*v2.sym[1]], "Top") 
# stokes2.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
# stokes2.add_dirichlet_bc((0.0, sympy.oo), "Left")
# stokes2.add_dirichlet_bc((0.0, sympy.oo), "Right")

Gamma = mesh2.Gamma
stokes2.add_natural_bc(2.5e6 * Gamma.dot(v2.sym) *  Gamma, "Top")
stokes2.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes2.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes2.add_dirichlet_bc((0.0, sympy.oo), "Right")

# stokes2.add_natural_bc(2.5e6 * Gamma.dot(v2.sym) *  Gamma, "Bottom")
# stokes2.add_natural_bc(2.5e6 * Gamma.dot(v2.sym) *  Gamma, "Left")
# stokes2.add_natural_bc(2.5e6 * Gamma.dot(v2.sym) *  Gamma, "Right")
# -


# We may need to adjust the tolerance if $\Delta \eta$ is large

stokes1.tolerance = 1.0e-6
stokes2.tolerance = 1.0e-6

# +
stokes1.petsc_options["snes_monitor"]= None
stokes1.petsc_options["ksp_monitor"] = None

stokes2.petsc_options["snes_monitor"]= None
stokes2.petsc_options["ksp_monitor"] = None


# +
stokes1.petsc_options["snes_type"] = "newtonls"
stokes1.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes1.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes1.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes1.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes1.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes1.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes1.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
stokes1.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# gasm is super-fast ... but mg seems to be bulletproof
# gamg is toughest wrt viscosity

stokes1.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes1.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes1.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")


stokes2.petsc_options["snes_type"] = "newtonls"
stokes2.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes2.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes2.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes2.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes2.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes2.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes2.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
stokes2.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# gasm is super-fast ... but mg seems to be bulletproof
# gamg is toughest wrt viscosity

stokes2.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes2.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes2.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")


# +
# stokes._setup_pointwise_functions(verbose=True)
# stokes._setup_discretisation(verbose=True)
# stokes.dm.ds.view()
# -

# %%
# Solve time
stokes1.solve()
stokes2.solve()

# ### Visualise it !

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v1.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.sqrt(v1.sym.dot(v1.sym)))

    velocity_points = vis.meshVariable_to_pv_cloud(v1)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v1.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvmesh,
                cmap="coolwarm",
                edge_color="Black",
                show_edges=True,
                scalars="Vmag",
                use_transparency=False,
                opacity=1.0,
                )

    arrows = pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=3.0, opacity=1, show_scalar_bar=False)

    pl.show(cpos="xy")


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh2)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v2.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.sqrt(v2.sym.dot(v2.sym)))

    velocity_points = vis.meshVariable_to_pv_cloud(v2)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v2.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvmesh,
                cmap="coolwarm",
                edge_color="Black",
                show_edges=True,
                scalars="Vmag",
                use_transparency=False,
                opacity=1.0,
                )

    arrows = pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=3.0, opacity=1, show_scalar_bar=False)

    pl.show(cpos="xy")
# -

with mesh2.access(v2):
    with mesh1.access(v1, v_err):
        for i, coord in enumerate(v1.coords):
            # print(v1.data[i], v2.data[i])
        #     print(v1.data[i] - v2.data[i])
            v_err.data[i] = v1.data[i] - v2.data[i]
        print(v_err.data[:,0].min(), v_err.data[:,1].min())
        print(v_err.data[:,0].max(), v_err.data[:,1].max())

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_err.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.sqrt(v_err.sym.dot(v_err.sym)))

    velocity_points = vis.meshVariable_to_pv_cloud(v_err)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_err.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvmesh,
                cmap="coolwarm",
                edge_color="Black",
                show_edges=True,
                scalars="Vmag",
                use_transparency=False,
                opacity=1.0,
                )

    arrows = pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=3e5, opacity=1, show_scalar_bar=False)

    pl.show(cpos="xy")
# -


