# ## Spherical Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark paper](https://gmd.copernicus.org/articles/14/1899/2021/) 
#
# *Author: [Thyagarajulu Gollapalli](https://github.com/gthyagi)*

from mpi4py import MPI

# +
import underworld3 as uw
from underworld3.systems import Stokes

import numpy as np
import sympy
import os
import assess
import h5py
# -

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

if uw.mpi.size == 1:
    # to fix trame issue
    import nest_asyncio
    nest_asyncio.apply()
    
    import pyvista as pv
    import underworld3.visualisation as vis
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc
    from scipy import integrate

# +
# mesh options
r_o = 2.22
r_int = 2.0
r_i = 1.22

res_inv = 8 # 4, 8, 16, 32, 64, 128
res = 1/res_inv
res_int_fac = 1/2

# r_o = 2.22
# r_i = 1.22
# r_int = (r_o+r_i)/2

# res_inv = 32 # 8, 16, 32, 64, 128
# res = 1/res_inv
# res_int_fac = 1
# -

# ##### Case1: Freeslip boundaries and delta function density perturbation
# <!--
#     1. Works fine (i.e., bc produce results)
# -->
# ##### Case2: Freeslip boundaries and smooth density distribution
# <!--
#     1. Works fine (i.e., bc produce results)
#     2. Output contains null space (for normals = unit radial vector)
# -->
# ##### Case3: Noslip boundaries and delta function density perturbation
# <!--
#     1. Works fine (i.e., bc produce results)
# -->
# ##### Case4: Noslip boundaries and smooth density distribution 
# <!--
#     1. Works fine (i.e., bc produce results)
# -->

# +
# specify the case 
case = 'case4'

# spherical harmonic fn degree (l) and order (m)
l = 2
m = 1
k = l+1 # power (check the reference paper)

# +
# fem stuff
vdegree  = 2
pdegree = 1
pcont = True
pcont_str = str(pcont).lower()

vel_penalty = 1e8
stokes_tol = 1e-5
vel_penalty_str = str("{:.1e}".format(vel_penalty))
stokes_tol_str = str("{:.1e}".format(stokes_tol))

# +
# boundary condition and density perturbation
freeslip, noslip, delta_fn, smooth = False, False, False, False

if case in ('case1'):
    freeslip, delta_fn = True, True
elif case in ('case2'):
    freeslip, smooth = True, True
elif case in ('case3'):
    noslip, delta_fn = True, True
elif case in ('case4'):
    noslip, smooth = True, True

# +
# output_dir = os.path.join(os.path.join("./output/Latex_Dir/"), f"{case}/")
output_dir = os.path.join(os.path.join("./output/Latex_Dir/"), 
                          f'{case}_l_{l}_m_{m}_k_{k}_res_{res_inv}_vdeg_{vdegree}_pdeg_{pdegree}'\
                          f'_pcont_{pcont_str}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/')

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
# -

# compute analytical solution
analytical = True
visualize = True
timing = True

# ### Analytical Solution

if analytical:
    if freeslip:
        if delta_fn:
            soln_above = assess.SphericalStokesSolutionDeltaFreeSlip(l, m, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
            soln_below = assess.SphericalStokesSolutionDeltaFreeSlip(l, m, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
        elif smooth:
            '''
            For smooth density distribution only single solution exists in the domain. 
            But for sake of code optimization I am creating two solution here.
            '''
            soln_above = assess.SphericalStokesSolutionSmoothFreeSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
            soln_below = assess.SphericalStokesSolutionSmoothFreeSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
    elif noslip:
        if delta_fn:
            soln_above = assess.SphericalStokesSolutionDeltaZeroSlip(l, m, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
            soln_below = assess.SphericalStokesSolutionDeltaZeroSlip(l, m, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
        elif smooth:
            '''
            For smooth density distribution only single solution exists in the domain. 
            But for sake of code optimization I am creating two solution here.
            '''
            soln_above = assess.SphericalStokesSolutionSmoothZeroSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
            soln_below = assess.SphericalStokesSolutionSmoothZeroSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)


# ### Plotting and Analysis functions

def plot_mesh(_mesh, _save_png=False, _dir_fname='', _title='', _show_clip=False):
    # plot mesh
    pvmesh = vis.mesh_to_pv_mesh(_mesh)

    pl = pv.Plotter(window_size=(750, 750))
    if _show_clip:
        clip1_normal = (np.cos(np.deg2rad(135)), np.cos(np.deg2rad(135)), 0.0)
        clip1 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip1_normal, invert=False, crinkle=False)
        pl.add_mesh(clip1, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, show_scalar_bar=False, opacity=1.0,)

        clip2_normal = (np.cos(np.deg2rad(135)), -np.cos(np.deg2rad(135)), 0.0)
        clip2 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip2_normal, invert=False, crinkle=False)
        pl.add_mesh(clip2, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, show_scalar_bar=False, opacity=1.0,)
    else:
        pl.add_mesh(pvmesh, edge_color="Grey", show_edges=True, use_transparency=False, opacity=1.0, )

    pl.show(cpos="yz")

    if len(_title)!=0:
        pl.add_text(_title, font_size=18, position=(950, 2100))
    
    if _save_png:
        pl.camera.zoom(1.4)
        pl.screenshot(_dir_fname, scale=3.5,)


def plot_scalar(_mesh, _scalar, _scalar_name='', _cmap='', _clim='', _save_png=False, _dir_fname='', _title='', _fmt='%10.7f', _show_clip=False):
    # plot scalar quantity from mesh
    pvmesh = vis.mesh_to_pv_mesh(_mesh)
    pvmesh.point_data[_scalar_name] = vis.scalar_fn_to_pv_points(pvmesh, _scalar)

    print(pvmesh.point_data[_scalar_name].min(), pvmesh.point_data[_scalar_name].max())
    
    pl = pv.Plotter(window_size=(750, 750))
    if _show_clip:
        clip1_normal = (np.cos(np.deg2rad(135)), np.cos(np.deg2rad(135)), 0.0)
        clip1 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip1_normal, invert=False, crinkle=False)
        pl.add_mesh(clip1, cmap=_cmap, edge_color="Grey", scalars=_scalar_name, show_edges=False, 
                    use_transparency=False, show_scalar_bar=False, opacity=1.0, clim=_clim)

        clip2_normal = (np.cos(np.deg2rad(135)), -np.cos(np.deg2rad(135)), 0.0)
        clip2 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip2_normal, invert=False, crinkle=False)
        pl.add_mesh(clip2, cmap=_cmap, edge_color="Grey", scalars=_scalar_name, show_edges=False, 
                    use_transparency=False, show_scalar_bar=False, opacity=1.0, clim=_clim)
    else:
        pl.add_mesh(pvmesh, cmap=_cmap, edge_color="Grey", scalars=_scalar_name, show_edges=False, 
                    use_transparency=False, opacity=1.0, clim=_clim, show_scalar_bar=False)
    
    # pl.add_scalar_bar(_scalar_name, vertical=False, title_font_size=25, label_font_size=20, fmt=_fmt, 
    #                   position_x=0.225, position_y=0.01, color='k')
    
    pl.show(cpos="yz")

    if len(_title)!=0:
        pl.add_text(_title, font_size=18, position=(950, 2100))

    if _save_png:
        pl.camera.zoom(1.4)
        pl.screenshot(_dir_fname, scale=3.5,)


def plot_vector(_mesh, _vector, _vector_name='', _cmap='', _clim='', _vmag='', _vfreq='', _save_png=False, _dir_fname='', _title='', _fmt='%10.7f', 
                _show_clip=False, _show_arrows=False):
    # plot vector quantity from mesh
    pvmesh = vis.mesh_to_pv_mesh(_mesh)
    pvmesh.point_data[_vector_name] = vis.vector_fn_to_pv_points(pvmesh, _vector.sym)
    _vector_mag_name = _vector_name+'_mag'
    pvmesh.point_data[_vector_mag_name] = vis.scalar_fn_to_pv_points(pvmesh, 
                                                                     sympy.sqrt(_vector.sym.dot(_vector.sym)))
    
    print(pvmesh.point_data[_vector_mag_name].min(), pvmesh.point_data[_vector_mag_name].max())
    
    velocity_points = vis.meshVariable_to_pv_cloud(_vector)
    velocity_points.point_data[_vector_name] = vis.vector_fn_to_pv_points(velocity_points, _vector.sym)
    
    pl = pv.Plotter(window_size=(750, 750))
    if _show_clip:
        clip1_normal = (np.cos(np.deg2rad(135)), np.cos(np.deg2rad(135)), 0.0)
        clip1 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip1_normal, invert=False, crinkle=False)
        pl.add_mesh(clip1, cmap=_cmap, edge_color="Grey", scalars=_vector_mag_name, show_edges=False, 
                    use_transparency=False, show_scalar_bar=False, opacity=1.0, clim=_clim)

        clip2_normal = (np.cos(np.deg2rad(135)), -np.cos(np.deg2rad(135)), 0.0)
        clip2 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip2_normal, invert=False, crinkle=False)
        pl.add_mesh(clip2, cmap=_cmap, edge_color="Grey", scalars=_vector_mag_name, show_edges=False, 
                    use_transparency=False, show_scalar_bar=False, opacity=1.0, clim=_clim)
    else:
        pl.add_mesh(pvmesh, cmap=_cmap, edge_color="Grey", scalars=_vector_mag_name, show_edges=False, use_transparency=False,
                    opacity=1.0, clim=_clim, show_scalar_bar=False)
               
    # pl.add_scalar_bar(_vector_name, vertical=False, title_font_size=25, label_font_size=20, fmt=_fmt, 
    #                   position_x=0.225, position_y=0.01,)
    if _show_arrows:
        pl.add_arrows(velocity_points.points[::_vfreq], velocity_points.point_data[_vector_name][::_vfreq], mag=_vmag, color='k')

    pl.show(cpos="yz")

    if len(_title)!=0:
        pl.add_text(_title, font_size=18, position=(950, 1075))

    if _save_png:
        pl.camera.zoom(1.4)
        pl.screenshot(_dir_fname, scale=3.5,)


def save_colorbar(_colormap='', _cb_bounds='', _vmin='', _vmax='', _figsize_cb='', _primary_fs=18, _cb_orient='', _cb_axis_label='',
                  _cb_label_xpos='', _cb_label_ypos='', _fformat='', _output_path='', _fname=''):
    # save the colorbar separately
    plt.figure(figsize=_figsize_cb)
    plt.rc('font', size=_primary_fs) # font_size
    if len(_cb_bounds)!=0:
        a = np.array([bounds])
        img = plt.imshow(a, cmap=_colormap, norm=norm)
    else:
        a = np.array([[_vmin,_vmax]])
        img = plt.imshow(a, cmap=_colormap)
        
    plt.gca().set_visible(False)
    if _cb_orient=='vertical':
        cax = plt.axes([0.1, 0.2, 0.06, 1.15])
        cb = plt.colorbar(orientation='vertical', cax=cax)
        cb.ax.set_title(_cb_axis_label, fontsize=_primary_fs, x=_cb_label_xpos, y=_cb_label_ypos, rotation=90) # font_size
        if _fformat=='png':
            plt.savefig(_output_path+_fname+'_cbvert.'+_fformat, dpi=150, bbox_inches='tight')
        elif _fformat=='pdf':
            plt.savefig(_output_path+_fname+"_cbvert."+_fformat, format=_fformat, bbox_inches='tight')
    if _cb_orient=='horizontal':
        cax = plt.axes([0.1, 0.2, 1.15, 0.06])
        cb = plt.colorbar(orientation='horizontal', cax=cax)
        cb.ax.set_title(_cb_axis_label, fontsize=_primary_fs, x=_cb_label_xpos, y=_cb_label_ypos) # font_size
        if _fformat=='png':
            plt.savefig(_output_path+_fname+'_cbhorz.'+_fformat, dpi=150, bbox_inches='tight')
        elif _fformat=='pdf':
            plt.savefig(_output_path+_fname+"_cbhorz."+_fformat, format=_fformat, bbox_inches='tight')


# ### Create Mesh

if timing:
    uw.timing.reset()
    uw.timing.start()

mesh = uw.meshing.SegmentedSphericalShell(radiusInner=r_i, radiusOuter=r_o, cellSize=res, numSegments=5, qdegree=max(pdegree, vdegree))

if timing:
    uw.timing.stop()
    uw.timing.print_table(group_by='line_routine', output_file=f"{output_dir}/mesh_create_time.txt",  display_fraction=1.00)

if uw.mpi.size == 1:
    plot_mesh(mesh, _save_png=True, _dir_fname=output_dir+'mesh.png', _title=case, _show_clip=True)

# +
# mesh variables
v_uw = uw.discretisation.MeshVariable('V_u', mesh, mesh.data.shape[1], degree=vdegree)
p_uw = uw.discretisation.MeshVariable('P_u', mesh, 1, degree=pdegree, continuous=pcont)

if analytical:
    v_ana = uw.discretisation.MeshVariable('V_a', mesh, mesh.data.shape[1], degree=vdegree)
    p_ana = uw.discretisation.MeshVariable('P_a', mesh, 1, degree=pdegree, continuous=pcont)
    rad_ss_ana = uw.discretisation.MeshVariable('RSS_a', mesh, 1, degree=pdegree, continuous=True) # radial stress
    
    v_err = uw.discretisation.MeshVariable('V_e', mesh, mesh.data.shape[1], degree=vdegree)
    p_err = uw.discretisation.MeshVariable('P_e', mesh, 1, degree=pdegree, continuous=pcont)
# -

norm_v = uw.discretisation.MeshVariable("N", mesh, mesh.data.shape[1], degree=pdegree, varsymbol=r"{\hat{n}}")
with mesh.access(norm_v):
    norm_v.data[:,0] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[0], norm_v.coords)
    norm_v.data[:,1] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[1], norm_v.coords)
    norm_v.data[:,2] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[2], norm_v.coords)

# +
# Some useful coordinate stuff
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR[0], mesh.CoordinateSystem.xR[1]
phi_uw =sympy.Piecewise((2*sympy.pi + mesh.CoordinateSystem.xR[2], mesh.CoordinateSystem.xR[2]<0), 
                        (mesh.CoordinateSystem.xR[2], True)
                       )

# # Null space in velocity (constant v_theta) expressed in x,y coordinates
# v_theta_fn_xy = r_uw * mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0,1))
# -

if analytical:
    with mesh.access(v_ana, p_ana, rad_ss_ana):
        
        def get_ana_soln(_var, _r_int, _soln_above, _soln_below):
            # get analytical solution into mesh variables
            r = uw.function.evalf(r_uw, _var.coords)
            for i, coord in enumerate(_var.coords):
                if r[i]>_r_int:
                    _var.data[i] = _soln_above(coord)
                else:
                    _var.data[i] = _soln_below(coord)
                    
        # velocities
        get_ana_soln(v_ana, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)

        # pressure 
        get_ana_soln(p_ana, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)
        
        # density
        get_ana_soln(rad_ss_ana, r_int, soln_above.radial_stress_cartesian, soln_below.radial_stress_cartesian)

# +
# plotting analytical velocities
if case in ('case1'):
    clim, vmag, vfreq = [0., 0.05], 5e0, 75
elif case in ('case2'):
    clim, vmag, vfreq = [0., 0.025], 1e1, 75
elif case in ('case3'):
    clim, vmag, vfreq = [0., 0.01], 2.5e1, 75
elif case in ('case4'):
    clim, vmag, vfreq = [0., 0.00175], 5e2, 75
        
if uw.mpi.size == 1 and analytical and visualize:
    # velocity plot
    plot_vector(mesh, v_ana, _vector_name='v_ana', _cmap=cmc.lapaz.resampled(11), _clim=clim, _vmag=vmag, _vfreq=vfreq,
                _save_png=True, _dir_fname=output_dir+'vel_ana.png', _show_clip=True, _show_arrows=False)
    # saving colobar separately 
    save_colorbar(_colormap=cmc.lapaz.resampled(11), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                  _cb_orient='horizontal', _cb_axis_label='Velocity', _cb_label_xpos=0.5, _cb_label_ypos=-2.05, _fformat='pdf', 
                  _output_path=output_dir, _fname='v_ana')

# +
# plotting analytical pressure
if case in ('case1'):
    clim = [-0.65, 0.65]
elif case in ('case2'):
    clim = [-0.3, 0.3]
elif case in ('case3'):
    clim = [-0.65, 0.65]
elif case in ('case4'):
    clim = [-0.13, 0.13]

if uw.mpi.size == 1 and analytical and visualize:
    # pressure plot
    plot_scalar(mesh, p_ana.sym, 'p_ana', _cmap=cmc.vik.resampled(41), _clim=clim, _save_png=True, _show_clip=True,
                _dir_fname=output_dir+'p_ana.png')
    # saving colobar separately 
    save_colorbar(_colormap=cmc.vik.resampled(41), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                  _cb_orient='horizontal', _cb_axis_label='Pressure', _cb_label_xpos=0.5, _cb_label_ypos=-2.0, _fformat='pdf', 
                  _output_path=output_dir, _fname='p_ana')

# +
# plotting analytical radial stress
if case in ('case1'):
    clim = [-1, 1]
elif case in ('case2'):
    clim = [-0.35, 0.35]
elif case in ('case3'):
    clim = [-0.65, 0.65]
elif case in ('case4'):
    clim = [-0.13, 0.13]
        
if uw.mpi.size == 1 and analytical and visualize:
    # pressure plot
    plot_scalar(mesh, rad_ss_ana.sym, 'Radial Stress', _cmap=cmc.roma.resampled(31), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'rad_stress_ana.png', _show_clip=True)
    # saving colobar separately 
    save_colorbar(_colormap=cmc.roma.resampled(31), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                  _cb_orient='horizontal', _cb_axis_label='Radial Stress', _cb_label_xpos=0.5, _cb_label_ypos=-2.0, _fformat='pdf', 
                  _output_path=output_dir, _fname='rad_stress_ana')
# -

# Create Stokes object
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw, solver_name="stokes")
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# +
# defining rho fn
if delta_fn:
    rho = sympy.cos(n*th_uw) * sympy.exp(-1e5 * ((r_uw - r_int) ** 2))
elif smooth:
    y_lm_real = sympy.sqrt((2*l + 1)/(4*sympy.pi) * sympy.factorial(l - m)/sympy.factorial(l + m)) * sympy.cos(m*phi_uw) * sympy.assoc_legendre(l, m, sympy.cos(th_uw))
    rho = ((r_uw/r_o)**k) * y_lm_real
    

# boundary conditions
v_diff =  v_uw.sym - v_ana.sym
stokes.add_natural_bc(vel_penalty*v_diff, "Upper")
stokes.add_natural_bc(vel_penalty*v_diff, "Lower")
# -

# bodyforce term
if delta_fn:
    stokes.bodyforce = sympy.Matrix([0., 0.])
elif smooth:
    gravity_fn = -1.0 * unit_rvec # gravity
    stokes.bodyforce = rho*gravity_fn 

# +
# visualizing rho
if freeslip:
    if delta_fn:
        clim = [-1, 1]
    elif smooth:
        clim = [-1, 1]
elif noslip:
    if delta_fn:
        clim = [-1, 1]
    elif smooth:
        clim = [-0.38, 0.38]
        
if uw.mpi.size == 1 and visualize:
    # rho plot
    plot_scalar(mesh, rho, 'rho', _cmap=cmc.roma.resampled(31), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'rho_ana.png', _title=case, _show_clip=True)
    # saving colobar separately 
    save_colorbar(_colormap=cmc.roma.resampled(31), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                  _cb_orient='horizontal', _cb_axis_label='Density', _cb_label_xpos=0.5, _cb_label_ypos=-2.0, _fformat='pdf', 
                  _output_path=output_dir, _fname='rho_ana')

# +
# Stokes settings

stokes.tolerance = stokes_tol
stokes.petsc_options["ksp_monitor"] = None

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# mg, multiplicative - very robust ... similar to gamg, additive
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")
# -

if timing:
    uw.timing.reset()
    uw.timing.start()

stokes.solve(verbose=False)

if timing:
    uw.timing.stop()
    uw.timing.print_table(group_by='line_routine', output_file=f"{output_dir}/stokes_solve_time.txt", display_fraction=1.00)

# +
# # Null space evaluation

# I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v_uw.sym))
# norm = I0.evaluate()
# I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
# vnorm = I0.evaluate()
# # print(norm/vnorm, vnorm)

# with mesh.access(v_uw):
#     dv = uw.function.evaluate(norm * v_theta_fn_xy, v_uw.coords) / vnorm
#     v_uw.data[...] -= dv 
# -

# compute error
if analytical:
    with mesh.access(v_uw, p_uw, v_err, p_err):
    
        def get_error(_var_err, _var_uw, _r_int, _soln_above, _soln_below):
            # get error in numerical solution
            r = uw.function.evalf(r_uw, _var_err.coords)
            for i, coord in enumerate(_var_err.coords):
                if r[i]>_r_int:
                    _var_err.data[i] = _var_uw.data[i] - _soln_above(coord)
                else:
                    _var_err.data[i] = _var_uw.data[i] - _soln_below(coord)
                    
        # error in velocities
        get_error(v_err, v_uw, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
        
        # error in pressure 
        get_error(p_err, p_uw, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)

# +
# plotting velocities from uw
if case in ('case1'):
    clim, vmag, vfreq = [0., 0.05], 5e0, 75
elif case in ('case2'):
    clim, vmag, vfreq = [0., 0.025], 1e1, 75
elif case in ('case3'):
    clim, vmag, vfreq = [0., 0.01], 2.5e1, 75
elif case in ('case4'):
    clim, vmag, vfreq = [0., 0.00175], 5e2, 75
    
if uw.mpi.size == 1 and visualize:
    # velocity plot
    plot_vector(mesh, v_uw, _vector_name='v_ana', _cmap=cmc.lapaz.resampled(11), _clim=clim, _vmag=vmag, _vfreq=vfreq,
                _save_png=True, _dir_fname=output_dir+'vel_uw.png', _show_clip=True, _show_arrows=False)

# +
# plotting relative errror in velocities
if case in ('case1'):
    clim, vmag, vfreq = [0., 0.005], 1e2, 75
elif case in ('case2'):
    clim, vmag, vfreq = [0., 7e-4], 1e2, 75
elif case in ('case3'):
    clim, vmag, vfreq = [0., 1e-4], 2e2, 75
elif case in ('case4'):
    clim, vmag, vfreq = [0., 1e-4], 1e5, 75
        
if uw.mpi.size == 1 and analytical and visualize:
    # velocity error plot
    plot_vector(mesh, v_err, _vector_name='v_err(relative)', _cmap=cmc.lapaz.resampled(11), _clim=clim, _vmag=vmag, _vfreq=vfreq,
                _save_png=True, _dir_fname=output_dir+'vel_r_err.png', _show_clip=True, _show_arrows=False)

# +
# plotting magnitude error in percentage
if case in ('case1'):
    clim = [0, 20]
elif case in ('case2'):
    clim = [0, 20]
elif case in ('case3'):
    clim = [0, 5]
elif case in ('case4'):
    clim = [0, 1000]

if uw.mpi.size == 1 and analytical and visualize: 
    # velocity error plot
    vmag_expr = (sympy.sqrt(v_err.sym.dot(v_err.sym))/sympy.sqrt(v_ana.sym.dot(v_ana.sym)))*100
    plot_scalar(mesh, vmag_expr, 'vmag_err(%)', _cmap=cmc.oslo_r.resampled(21), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'vel_p_err.png', _show_clip=True)

# +
# plotting pressure from uw
if case in ('case1'):
    clim = [-0.65, 0.65]
elif case in ('case2'):
    clim = [-0.3, 0.3]
elif case in ('case3'):
    clim = [-0.65, 0.65]
elif case in ('case4'):
    clim = [-0.13, 0.13]
        
if uw.mpi.size == 1 and visualize:
    # pressure plot
    plot_scalar(mesh, p_uw.sym, 'p_uw', _cmap=cmc.vik.resampled(41), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'p_uw.png', _show_clip=True)

# +
# plotting relative error in uw
if case in ('case1'):
    clim = [-0.065, 0.065]
elif case in ('case2'):
    clim = [-0.003, 0.003]
elif case in ('case3'):
    clim = [-0.0065, 0.0065]
elif case in ('case4'):
    clim = [-0.0045, 0.0045]
        
if uw.mpi.size == 1 and analytical and visualize:
    # pressure error plot
    plot_scalar(mesh, p_err.sym, 'p_err(relative)', _cmap=cmc.vik.resampled(41), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'p_r_err.png', _show_clip=True)
# -

# plotting percentage error in uw
if uw.mpi.size == 1 and analytical and visualize:
    # pressure error plot
    plot_scalar(mesh, (p_err.sym[0]/p_ana.sym[0])*100, 'p_err(%)', _cmap=cmc.vik.resampled(41), _clim=[-10, 10], _save_png=True, 
                _dir_fname=output_dir+'p_p_err.png', _show_clip=True)

# computing L2 norm
if analytical:
    with mesh.access(v_err, p_err, p_ana, v_ana):    
        v_err_I = uw.maths.Integral(mesh, v_err.sym.dot(v_err.sym))
        v_ana_I = uw.maths.Integral(mesh, v_ana.sym.dot(v_ana.sym))
        v_err_l2 = np.sqrt(v_err_I.evaluate())/np.sqrt(v_ana_I.evaluate())
        print('Relative error in velocity in the L2 norm: ', v_err_l2)
    
        p_err_I = uw.maths.Integral(mesh, p_err.sym.dot(p_err.sym))
        p_ana_I = uw.maths.Integral(mesh, p_ana.sym.dot(p_ana.sym))
        p_err_l2 = np.sqrt(p_err_I.evaluate())/np.sqrt(p_ana_I.evaluate())
        print('Relative error in pressure in the L2 norm: ', p_err_l2)

# +
# writing l2 norms to h5 file
if uw.mpi.size == 1 and os.path.isfile(output_dir+'error_norm.h5'):
    os.remove(output_dir+'error_norm.h5')
    print('Old file removed')

if uw.mpi.rank == 0:
    print('Creating new h5 file')
    with h5py.File(output_dir+'error_norm.h5', 'w') as f:
        f.create_dataset("k", data=k)
        f.create_dataset("cell_size", data=res)
        f.create_dataset("v_l2_norm", data=v_err_l2)
        f.create_dataset("p_l2_norm", data=p_err_l2)
# -

# saving h5 and xdmf file
mesh.petsc_save_checkpoint(index=0, meshVars=[v_uw, p_uw, v_ana, p_ana, v_err, p_err], outputPath=output_dir+'output')

# +
if uw.mpi.rank == 0:
    print('-------------------------------------------------------------------------------')

# memory stats: needed only on mac
import psutil
process = psutil.Process()
print(f'rank: {uw.mpi.rank}, RAM Used (GB): {process.memory_info().rss/1024 ** 3}')
# -


