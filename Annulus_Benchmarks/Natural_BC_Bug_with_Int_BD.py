# ## This is a parallel bug. 
# #### If one of the processor does not contain all boundary labels then the code hangs during solve

import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
from underworld3.systems import Stokes
import sympy
from enum import Enum
import os
import glob

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc
    import underworld3.visualisation as vis
    import gmsh
    import pyvista as pv
    import numpy as np

output_dir = './output/natural_bc_test/'
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)


# +
# Define boundary enumerations
class boundaries(Enum):
    Bottom = 1
    Right = 2
    Top = 3
    Left = 4
    Internal = 5
    All_Boundaries = 1001
    Null_Boundary = 666

class boundary_normals_2D(Enum):
    Bottom = sympy.Matrix([0, 1])
    Top = sympy.Matrix([0, 1])
    Right = sympy.Matrix([1, 0])
    Left = sympy.Matrix([1, 0])
    Internal = sympy.Matrix([0, 1])


# +
# Define the domain limits
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0
yint = 0.67  # Location of internal boundary

simplex = not True
# -

if simplex:
    cellSize = 1/32
else:
    nx = 3 # Number of divisions along x-axis
    ny_total = 4 # Total number of divisions along the entire y-axis

if simplex:
    mesh_fname = f"{output_dir}mesh_ib_simp_res_{int(1/cellSize)}.msh"
    mesh_type = 'simp'
else:
    mesh_fname = f"{output_dir}/mesh_ib_quad_res_{nx}_{ny_total}.msh"
    mesh_type = 'quad'

# creating mesh
if uw.mpi.size == 1 and not os.path.isfile(mesh_fname):
    with open("create_natural_bc_mesh.py") as f:
        code = f.read()
    
    exec(code, globals())

# normal type
# options: 'petsc', 'uw'
norm_type = 'petsc' 

# +
# mesh = uw.meshing.BoxInternalBoundary(minCoords=(0, 0), maxCoords=(1, 1), zintCoord=yint, 
#                                       cellSize=cellSize, simplex=True)

# +
mesh = uw.discretisation.Mesh(mesh_fname, boundaries=boundaries, 
                              boundary_normals=boundary_normals_2D,
                              markVertices=True, useMultipleTags=True, useRegions=True,
                              coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN, )

# try with quad mesh

# +
x,y = mesh.X

mesh.view()
# +
# mesh.print_label_info_parallel()
# -

v_soln = uw.discretisation.MeshVariable(r"u", mesh, 2, degree=2)
p_soln = uw.discretisation.MeshVariable(r"p", mesh, 1, degree=1, continuous=True)
rank_var = uw.discretisation.MeshVariable(r"r", mesh, 1, degree=0, continuous=False)

with mesh.access(rank_var):
    rank_var.data[...] = uw.mpi.rank

# Create Stokes object
stokes = Stokes(mesh, velocityField=v_soln, pressureField=p_soln,) # solver_name="stokes")
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# This first choice is the PETSc normals, re-oriented on the fly
GammaNorm = mesh.Gamma.dot(mesh.CoordinateSystem.unit_e_1) / sympy.sqrt(mesh.Gamma.dot(mesh.Gamma))
Gamma     = GammaNorm * mesh.Gamma 

# +
# bc's
# t_init = sympy.cos(3*x*sympy.pi) * sympy.exp(-1000.0 * ((y - int_ycoord) ** 2)) 
t_init = 1.0

if norm_type=='petsc':
    stokes.add_natural_bc(sympy.Matrix([0.0, -t_init]), "Internal")
if norm_type=='uw':
    stokes.add_natural_bc(-t_init*Gamma, "Internal")

stokes.add_essential_bc(sympy.Matrix([sympy.oo, 0.0]), "Top")
stokes.add_essential_bc(sympy.Matrix([sympy.oo, 0.0]), "Bottom")
stokes.add_essential_bc(sympy.Matrix([0.0,sympy.oo]), "Left")
stokes.add_essential_bc(sympy.Matrix([0.0,sympy.oo]), "Right")

stokes.bodyforce = sympy.Matrix([0, 0])
# +
# print(f'rank: {uw.mpi.rank}, min coord: {mesh.data[:,0].min(), mesh.data[:,1].min()}', flush=True)
# print(f'rank: {uw.mpi.rank}, max coord: {mesh.data[:,0].max(), mesh.data[:,1].max()}', flush=True)
# print(f'rank: {uw.mpi.rank}, coords: {mesh.data}', flush=True)
# -

# stokes.petsc_options["pc_type"] = "lu"
stokes.tolerance = 1e-6
stokes.solve(verbose=False, debug=True,)# debug_name="test")

mesh.dm.view()

stokes.dm.view()

p_stats = p_soln.stats()
if uw.mpi.rank==0:
    # Round off each float in the tuple to 6 decimal places
    rounded_data = tuple(round(num, 6) if isinstance(num, float) else num for num in p_stats)

    # Print the result
    # print(p_stats)
    print(rounded_data)

# +
# stokes.dm.orient()
# -

if uw.mpi.size == 1:
    
    clim=[-1, 1]
    if mesh_type=='simp':
        vmag=1e1
    if mesh_type=='quad':
        vmag=0.5e1
    cmap= cmc.vik.resampled(10)

    # plotting vectors
    vis.plot_vector(mesh, v_soln, 'v', cmap=cmap, clim=clim, window_size=(750, 550),
                    vfreq=1, vmag=vmag, show_arrows=True, save_png=True, show_edges=True,
                    dir_fname=f'{output_dir}v_p_{uw.mpi.size}_n_{norm_type}_{mesh_type}', scalar=p_soln, 
                    scalar_name='p')
    
    # saving colorbar 
    vis.save_colorbar(colormap=cmap, cb_bounds=None, vmin=clim[0], vmax=clim[1], 
                      figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', cb_axis_label='Pressure', 
                      cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', output_path=output_dir, 
                      fname=f'p_{uw.mpi.size}_n_{norm_type}_{mesh_type}')

# +
# # saving h5 and xdmf file
# mesh.petsc_save_checkpoint(index=0, meshVars=[v_soln, p_soln], 
#                            outputPath=os.path.relpath(output_dir)+f'/output_{uw.mpi.size}_cpus_{norm_type}')
# -

mesh.write_timestep(f'output_{uw.mpi.size}_n_{norm_type}_{mesh_type}', meshUpdates=True, 
                    meshVars=[v_soln, p_soln, rank_var], 
                    outputPath=output_dir, index=0,)


def create_merged_debug_txt(filename):
    # Specify the directory containing the text files
    directory = './'
    
    # Get a list of all .txt files in the directory
    txt_files = glob.glob(os.path.join(directory, '*_debug.txt'))
    
    # Sort the files by creation time (latest first)
    txt_files_sorted = sorted(txt_files, key=os.path.getctime, reverse=True)

    if len(txt_files_sorted)>0:
        # Name of the merged output file
        merged_filename = filename
        
        with open(merged_filename, 'w') as outfile:
            for file in txt_files_sorted:
                with open(file, 'r') as infile:
                    outfile.write(infile.read())
                    outfile.write('\n')  # Optionally add a newline between files
                os.remove(file) # remove the file after merging
                
        print(f"Merged {len(txt_files_sorted)} files into {merged_filename}")
    else:
        print('No txt file exists')



if uw.mpi.rank == 0:
    filename = f'merged_debug_{mesh_type}_n_{norm_type}_cpus_{uw.mpi.size}.txt'
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass
    create_merged_debug_txt(filename)

# ### Plot parallel model data

# +
reload = True
cpus = 2

for cpu_no in range(cpus, cpus+1):

    if uw.mpi.size == 1 and reload:
        mesh_2 = uw.discretisation.Mesh(f"{output_dir}output_{cpus}_n_{norm_type}_{mesh_type}.mesh.00000.h5")
        
        v_soln_2 = uw.discretisation.MeshVariable(f"u_{cpus}", mesh_2, 2, degree=2)
        p_soln_2 = uw.discretisation.MeshVariable(f"p_{cpus}", mesh_2, 1, degree=1, continuous=True)
        rank_var_2 = uw.discretisation.MeshVariable(f"r_{cpus}", mesh_2, 1, degree=0, continuous=False)
    
        v_soln_2.read_timestep(data_filename=f'output_{cpus}_n_{norm_type}_{mesh_type}', data_name="u", 
                               index=0, outputPath=output_dir)
        p_soln_2.read_timestep(data_filename=f'output_{cpus}_n_{norm_type}_{mesh_type}', data_name="p", 
                               index=0, outputPath=output_dir)
        rank_var_2.read_timestep(data_filename=f'output_{cpus}_n_{norm_type}_{mesh_type}', data_name="r", 
                                 index=0, outputPath=output_dir)

        clim=[-1, 1]
        if mesh_type=='simp':
            vmag=1e1
        if mesh_type=='quad':
            vmag=0.5e1
        cmap= cmc.vik.resampled(10)
        
        # plotting vectors
        vis.plot_vector(mesh_2, v_soln_2, 'v', cmap=cmap, clim=clim, window_size=(750, 550),
                        vfreq=1, vmag=vmag, show_arrows=True, save_png=True, show_edges=True,
                        dir_fname=f'{output_dir}v_p_{cpus}_n_{norm_type}_{mesh_type}', scalar=p_soln_2, 
                        scalar_name=f'p_{cpus}')
        
        # saving colorbar 
        vis.save_colorbar(colormap=cmap, cb_bounds=None, vmin=clim[0], vmax=clim[1], 
                          figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', 
                          cb_axis_label='Pressure', cb_label_xpos=0.5, cb_label_ypos=-2.05, 
                          fformat='pdf', output_path=output_dir, fname=f'p_{cpus}_n_{norm_type}_{mesh_type}')

    # if uw.mpi.size == 1 and reload:

    #     clim=[0, cpus-1]
    #     cmap= plt.cm.tab10.resampled(cpus)
        
    #     # plotting vectors
    #     vis.plot_vector(mesh_2, v_soln_2, 'v', cmap=cmap, clim=clim, window_size=(750, 550),
    #                     vfreq=1, vmag=vmag, show_arrows=True, save_png=True, show_edges=True,
    #                     dir_fname=f'{output_dir}v_r_{cpus}_n_{norm_type}_{mesh_type}', scalar=rank_var_2, 
    #                     scalar_name=f'r_{cpus}')
        
    #     # saving colorbar 
    #     vis.save_colorbar(colormap=cmap, cb_bounds=None, vmin=clim[0], vmax=clim[1], 
    #                       figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', 
    #                       cb_axis_label='Rank', cb_label_xpos=0.5, cb_label_ypos=-2.05, 
    #                       fformat='pdf', output_path=output_dir, fname=f'r_{cpus}_n_{norm_type}_{mesh_type}')
# -
# file to load
filename = f'merged_debug_{mesh_type}_n_{norm_type}_cpus_{cpu_no}.txt'

if reload and uw.mpi.size == 1 and os.path.isfile(filename):
    import re
    
    all_X = []  # list to store (x, y, z) from lines
    all_N = []  # list to store (nx, ny, nz) from lines
    
    with open(filename, 'r') as f:
        for line in f:
            # Only process lines that start with this specific prefix
            if line.startswith("bdres - equation 18 X / N"):
                # Find the substring in parentheses: "(x, y, z / nx, ny, nz)"
                match = re.search(r"\((.*?)\)", line)
                if not match:
                    continue
                
                content = match.group(1)
                # content is something like "2.96e-01, 6.70e-01, 0.00e+00 / 0e+00, 1e+00, 0.00e+00"
                
                # Split left side (x, y, z) from right side (nx, ny, nz)
                parts = content.split("/")
                if len(parts) < 2:
                    continue
                
                # Left side: e.g. "2.96e-01, 6.70e-01, 0.00e+00"
                left_str = parts[0].strip()
                # Right side: e.g. "0e+00, 1e+00, 0.00e+00"
                right_str = parts[1].strip()
                
                # Split by commas to get separate values
                left_vals = [val.strip() for val in left_str.split(",")]
                right_vals = [val.strip() for val in right_str.split(",")]
                
                # Convert to floats
                if len(left_vals) == 3 and len(right_vals) == 3:
                    x_tuple = tuple(float(x) for x in left_vals)   # (x, y, z)
                    n_tuple = tuple(float(x) for x in right_vals)  # (nx, ny, nz)
                    
                    all_X.append(x_tuple)
                    all_N.append(n_tuple)
    
    # # After the loop, 'all_X' and 'all_N' contain the parsed data
    # print("Extracted X vectors:")
    # for xvec in all_X:
    #     print(xvec)
    
    # print("\nExtracted N vectors:")
    # for nvec in all_N:
    #     print(nvec)

    print(np.array(all_X).shape)
    print(np.array(all_N).shape)

if reload and uw.mpi.size == 1:
    if not reload:
        cpus = 1
        with mesh.access(v_soln, rank_var):
            int_bd = v_soln.coords[np.where(v_soln.coords[:,1]==0.67)]
            rank_data = rank_var.data
    else:
        with mesh_2.access(v_soln_2, rank_var_2):
            int_bd = v_soln_2.coords[np.where(v_soln_2.coords[:,1]==0.67)]
            rank_data = rank_var_2.data

if reload and uw.mpi.size == 1 and os.path.isfile(filename):
    # Convert your int_bd coordinates to PyVista PolyData
    point_cloud = np.column_stack((int_bd, np.zeros(len(int_bd))))

    point_cloud_debug = np.array(all_X)
    poly_debug = pv.PolyData(point_cloud_debug)
    poly_debug["normals"] = np.array(all_N)

    
    glyphs = poly_debug.glyph(orient="normals", scale=False, factor=0.1)
    
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    
    pvmesh.cell_data["rank_var"] = rank_data
    
    pl = pv.Plotter(window_size=(750, 750))

    clim=[0, cpus-1]
    cmap= plt.cm.tab10.resampled(cpus)
    pl.add_mesh(pvmesh, edge_color="k", show_edges=True, scalars="rank_var", 
                cmap=cmap, show_scalar_bar=True, clim=clim)
    
    pl.add_mesh(point_cloud, color="red", point_size=10, render_points_as_spheres=True)

    pl.add_mesh(point_cloud_debug, color="blue", point_size=10, render_points_as_spheres=True)


    pl.add_mesh(glyphs, color="black")

    pl.show(cpos='xy')
    pl.camera.zoom(1.4)

    filename = f'{output_dir}mesh_int_bd_pts_n_{norm_type}_{cpus}_{mesh_type}'
    pl.screenshot(f'{filename}.png', scale=3)


