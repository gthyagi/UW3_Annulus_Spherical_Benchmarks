# ## Annulus Benchmark: Isoviscous Incompressible Stokes
#
# ### Case: Infinitely thin density anomaly at $r = r'$
# #### [Benchmark paper](https://gmd.copernicus.org/articles/14/1899/2021/) 
#
# *Author: [Thyagarajulu Gollapalli](https://github.com/gthyagi)*

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

res_inv = 32 # 8, 16, 32, 64, 128
res = 1/res_inv
res_int_fac = 1/2

# r_o = 2.22
# r_i = 1.22
# r_int = (r_o+r_i)/2

# res_inv = 32 # 8, 16, 32, 64, 128
# res = 1/res_inv
# res_int_fac = 1
# -

# which normals to use
ana_normal = not True # unit radial vector
petsc_normal = True # gamma function
bc_type = 'nat_bc' # 'nat_bc', 'ess_bc'

# compute analytical solutions
comp_ana = True

# ##### Case1: Freeslip boundaries and delta function density perturbation
#     1. Works fine (i.e., bc produce results)
# ##### Case2: Freeslip boundaries and smooth density distribution
#     1. Works fine (i.e., bc produce results)
#     2. Output contains null space (for normals = unit radial vector)
# ##### Case3: Noslip boundaries and delta function density perturbation
#     1. Works fine (i.e., bc produce results)
# ##### Case4: Noslip boundaries and smooth density distribution 
#     1. Works fine (i.e., bc produce results)

# +
# specify the case 
case = 'case4'

n = 2 # wave number
k = 0 # power (check the reference paper)

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
output_dir = os.path.join(os.path.join("./output/Latex_files/"), f"{case}_ess_bc/")

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
# -

# ### Analytical Solution

if comp_ana:
    if freeslip:
        if delta_fn:
            soln_above = assess.CylindricalStokesSolutionDeltaFreeSlip(n, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
            soln_below = assess.CylindricalStokesSolutionDeltaFreeSlip(n, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
        elif smooth:
            '''
            For smooth density distribution only single solution exists in the domain. 
            But for sake of code optimization I am creating two solution here.
            '''
            soln_above = assess.CylindricalStokesSolutionSmoothFreeSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
            soln_below = assess.CylindricalStokesSolutionSmoothFreeSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
    elif noslip:
        if delta_fn:
            soln_above = assess.CylindricalStokesSolutionDeltaZeroSlip(n, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
            soln_below = assess.CylindricalStokesSolutionDeltaZeroSlip(n, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
        elif smooth:
            '''
            For smooth density distribution only single solution exists in the domain. 
            But for sake of code optimization I am creating two solution here.
            '''
            soln_above = assess.CylindricalStokesSolutionSmoothZeroSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
            soln_below = assess.CylindricalStokesSolutionSmoothZeroSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)


# ### Plotting and Analysis functions

def plot_mesh(_mesh, _save_png=False, _dir_fname='', _title=''):
    # plot mesh
    pvmesh = vis.mesh_to_pv_mesh(_mesh)

    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, edge_color="Grey", show_edges=True, use_transparency=False, opacity=1.0, )

    pl.show(cpos="xy")

    if len(_title)!=0:
        pl.add_text(_title, font_size=18, position=(950, 1075))
    
    if _save_png:
        pl.camera.zoom(1.4)
        pl.screenshot(_dir_fname, scale=3.5,)


def plot_scalar(_mesh, _scalar, _scalar_name='', _cmap='', _clim='', _save_png=False, _dir_fname='', _title='', _fmt='%10.7f' ):
    # plot scalar quantity from mesh
    pvmesh = vis.mesh_to_pv_mesh(_mesh)
    pvmesh.point_data[_scalar_name] = vis.scalar_fn_to_pv_points(pvmesh, _scalar)

    print(pvmesh.point_data[_scalar_name].min(), pvmesh.point_data[_scalar_name].max())
    
    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap=_cmap, edge_color="Grey", scalars=_scalar_name, show_edges=False, 
                use_transparency=False, opacity=1.0, clim=_clim, show_scalar_bar=False)
    
    # pl.add_scalar_bar(_scalar_name, vertical=False, title_font_size=25, label_font_size=20, fmt=_fmt, 
    #                   position_x=0.225, position_y=0.01, color='k')
    
    pl.show(cpos="xy")

    if len(_title)!=0:
        pl.add_text(_title, font_size=18, position=(950, 1075))

    if _save_png:
        pl.camera.zoom(1.4)
        pl.screenshot(_dir_fname, scale=3.5,)


def plot_vector(_mesh, _vector, _vector_name='', _cmap='', _clim='', _vmag='', _vfreq='', _save_png=False, _dir_fname='', _title='', _fmt='%10.7f'):
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
    pl.add_mesh(pvmesh, cmap=_cmap, edge_color="Grey", scalars=_vector_mag_name, show_edges=False, use_transparency=False,
                opacity=0.7, clim=_clim, show_scalar_bar=False)
               
    # pl.add_scalar_bar(_vector_name, vertical=False, title_font_size=25, label_font_size=20, fmt=_fmt, 
    #                   position_x=0.225, position_y=0.01,)
    
    pl.add_arrows(velocity_points.points[::_vfreq], velocity_points.point_data[_vector_name][::_vfreq], mag=_vmag, color='k')

    pl.show(cpos="xy")

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

# mesh
if delta_fn:
    mesh = uw.meshing.AnnulusInternalBoundary(radiusOuter=r_o, 
                                              radiusInternal=r_int, 
                                              radiusInner=r_i, 
                                              cellSize_Inner=res,
                                              cellSize_Internal=res*res_int_fac,
                                              cellSize_Outer=res,)
elif smooth:
    mesh = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=res, 
                              qdegree=2, degree=1)

if uw.mpi.size == 1:
    plot_mesh(mesh, _save_png=True, _dir_fname=output_dir+'mesh.png', _title=case)


# +
# mesh variables
v_uw = uw.discretisation.MeshVariable(r"\mathbf{u}", mesh, 2, degree=2)
p_uw = uw.discretisation.MeshVariable(r"p", mesh, 1, degree=1)

if comp_ana:
    v_ana = uw.discretisation.MeshVariable(r"\mathbf{u_a}", mesh, 2, degree=2)
    p_ana = uw.discretisation.MeshVariable(r"p_a", mesh, 1, degree=1)
    rho_ana = uw.discretisation.MeshVariable(r"rho_a", mesh, 1, degree=1)
    
    v_err = uw.discretisation.MeshVariable(r"\mathbf{u_e}", mesh, 2, degree=2)
    p_err = uw.discretisation.MeshVariable(r"p_e", mesh, 1, degree=1)
# -

norm_v = uw.discretisation.MeshVariable("N", mesh, 2, degree=1, varsymbol=r"{\hat{n}}")
with mesh.access(norm_v):
    norm_v.data[:,0] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[0], norm_v.coords)
    norm_v.data[:,1] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[1], norm_v.coords)

# +
# Some useful coordinate stuff
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR

# Null space in velocity (constant v_theta) expressed in x,y coordinates
v_theta_fn_xy = r_uw * mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0,1))
# -

if comp_ana:
    with mesh.access(v_ana, p_ana, rho_ana):
        
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
        get_ana_soln(rho_ana, r_int, soln_above.radial_stress_cartesian, soln_below.radial_stress_cartesian)

# +
# plotting analytical velocities
if case in ('case1'):
    clim, vmag, vfreq = [0., 0.05], 5e0, 75
elif case in ('case2'):
    clim, vmag, vfreq = [0., 0.0385], 6e0, 75
elif case in ('case3'):
    clim, vmag, vfreq = [0., 0.01], 2.5e1, 75
elif case in ('case4'):
    clim, vmag, vfreq = [0., 0.00925], 3e1, 75
        
if uw.mpi.size == 1 and comp_ana:
    # velocity plot
    plot_vector(mesh, v_ana, _vector_name='v_ana', _cmap=cmc.lapaz.resampled(11), _clim=clim, _vmag=vmag, _vfreq=vfreq,
                _save_png=True, _dir_fname=output_dir+'vel_ana.png')
    # saving colobar separately 
    save_colorbar(_colormap=cmc.lapaz.resampled(11), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                  _cb_orient='horizontal', 
                  _cb_axis_label='Velocity', _cb_label_xpos=0.5, _cb_label_ypos=-2.05, _fformat='pdf', _output_path=output_dir, _fname='v_ana')

# +
# plotting analytical pressure
if case in ('case1'):
    clim = [-0.65, 0.65]
elif case in ('case2'):
    clim = [-0.5, 0.5]
elif case in ('case3'):
    clim = [-0.65, 0.65]
elif case in ('case4'):
    clim = [-0.5, 0.5]

if uw.mpi.size == 1 and comp_ana:
    # pressure plot
    plot_scalar(mesh, p_ana.sym, 'p_ana', _cmap=cmc.vik.resampled(41), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'p_ana.png')
    # saving colobar separately 
    save_colorbar(_colormap=cmc.vik.resampled(41), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, _cb_orient='horizontal', 
                  _cb_axis_label='Pressure', _cb_label_xpos=0.5, _cb_label_ypos=-2.0, _fformat='pdf', _output_path=output_dir, _fname='p_ana')

# +
# plotting analytical density
if case in ('case1'):
    clim = [-1, 1]
elif case in ('case2'):
    clim = [-0.6, 0.6]
elif case in ('case3'):
    clim = [-0.65, 0.65]
elif case in ('case4'):
    clim = [-0.5, 0.5]
        
if uw.mpi.size == 1 and comp_ana:
    # pressure plot
    plot_scalar(mesh, rho_ana.sym, 'Radial Stress', _cmap=cmc.roma.resampled(31), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'rad_stress_ana.png')
    # saving colobar separately 
    save_colorbar(_colormap=cmc.roma.resampled(31), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                  _cb_orient='horizontal', 
                  _cb_axis_label='Radial Stress', _cb_label_xpos=0.5, _cb_label_ypos=-2.0, _fformat='pdf', _output_path=output_dir, _fname='rad_stress_ana')
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
    stokes.add_natural_bc(-rho * unit_rvec, "Internal")
elif smooth:
    rho = ((r_uw/r_o)**k)*sympy.cos(n*th_uw)

# boundary conditions
if freeslip:
    if ana_normal:
        Gamma = mesh.CoordinateSystem.unit_e_0
    elif petsc_normal:
        Gamma = mesh.Gamma
    
    v_diff =  v_uw.sym - v_ana.sym
    stokes.add_natural_bc(2.5e8*v_diff, "Upper")
    stokes.add_natural_bc(2.5e8*v_diff, "Lower")
    
elif noslip:
    if bc_type=='ess_bc':
        stokes.add_essential_bc(sympy.Matrix([0., 0.]), "Upper")
        stokes.add_essential_bc(sympy.Matrix([0., 0.]), "Lower")
    if bc_type=='nat_bc':
        v_diff =  v_uw.sym - v_ana.sym
        stokes.add_natural_bc(2.5e8*v_diff, "Upper")
        stokes.add_natural_bc(2.5e8*v_diff, "Lower")
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
        clim = [-1, 1]
        
if uw.mpi.size == 1:
    # rho plot
    plot_scalar(mesh, rho, 'rho', _cmap=cmc.roma.resampled(31), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'rho_ana.png', _title=case)
    # saving colobar separately 
    save_colorbar(_colormap=cmc.roma.resampled(31), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                  _cb_orient='horizontal', 
                  _cb_axis_label='Density', _cb_label_xpos=0.5, _cb_label_ypos=-2.0, _fformat='pdf', _output_path=output_dir, _fname='rho_ana')

# +
# Stokes settings

stokes.tolerance = 1.0e-10
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

stokes.natural_bcs

for index,bc in enumerate(stokes.natural_bcs):
    print("Setting bc {} ({})".format(index, bc.type))
    print(" - component: {}".format(bc.components))
    print(" - boundary:   {}".format(bc.boundary))
    print(" - fn:         {} ".format(bc.fn_f))

stokes.solve(verbose=True)

# +
# Null space evaluation

I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v_uw.sym))
norm = I0.evaluate()
I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
vnorm = I0.evaluate()

# print(norm/vnorm, vnorm)

with mesh.access(v_uw):
    dv = uw.function.evaluate(norm * v_theta_fn_xy, v_uw.coords) / vnorm
    v_uw.data[...] -= dv 
# -

# compute error
if comp_ana:
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
    clim, vmag, vfreq = [0., 0.0385], 6e0, 75
elif case in ('case3'):
    clim, vmag, vfreq = [0., 0.01], 2.5e1, 75
elif case in ('case4'):
    clim, vmag, vfreq = [0., 0.00925], 3e1, 75
    
if uw.mpi.size == 1:
    # velocity plot
    plot_vector(mesh, v_uw, _vector_name='v_ana', _cmap=cmc.lapaz.resampled(11), _clim=clim, _vmag=vmag, _vfreq=vfreq,
                _save_png=True, _dir_fname=output_dir+'vel_uw.png')

# +
# plotting relative errror in velocities
if case in ('case1'):
    clim, vmag, vfreq = [0., 0.005], 1e2, 75
elif case in ('case2'):
    clim, vmag, vfreq = [0., 7e-4], 1e2, 75
elif case in ('case3'):
    clim, vmag, vfreq = [0., 1e-4], 2e2, 75
elif case in ('case4'):
    clim, vmag, vfreq = [0., 1e-5], 5e4, 75
        
if uw.mpi.size == 1 and comp_ana:
    # velocity error plot
    plot_vector(mesh, v_err, _vector_name='v_err(relative)', _cmap=cmc.lapaz.resampled(11), _clim=clim, _vmag=vmag, _vfreq=vfreq,
                _save_png=True, _dir_fname=output_dir+'vel_r_err.png')
    # saving colobar separately 
    save_colorbar(_colormap=cmc.lapaz.resampled(11), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                  _cb_orient='horizontal', _cb_axis_label='Velocity Relative Error', _cb_label_xpos=0.5, _cb_label_ypos=-2.05, _fformat='pdf', 
                  _output_path=output_dir, _fname='vel_r_err')

# +
# plotting magnitude error in percentage
if case in ('case1'):
    clim = [0, 20]
elif case in ('case2'):
    clim = [0, 20]
elif case in ('case3'):
    clim = [0, 5]
elif case in ('case4'):
    clim = [0, 1]

if comp_ana:   
    vmag_expr = (sympy.sqrt(v_err.sym.dot(v_err.sym))/sympy.sqrt(v_ana.sym.dot(v_ana.sym)))*100
    if uw.mpi.size == 1:
         # velocity error plot
        plot_scalar(mesh, vmag_expr, 'vmag_err(%)', _cmap=cmc.oslo_r.resampled(21), _clim=clim, _save_png=True, 
                    _dir_fname=output_dir+'vel_p_err.png')
        # saving colobar separately 
        save_colorbar(_colormap=cmc.oslo_r.resampled(21), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                      _cb_orient='horizontal', _cb_axis_label='Velocity Percentage Error', _cb_label_xpos=0.5, _cb_label_ypos=-2.0, _fformat='pdf', 
                      _output_path=output_dir, _fname='vel_p_err_ana')

# +
# plotting pressure from uw
if case in ('case1'):
    clim = [-0.65, 0.65]
elif case in ('case2'):
    clim = [-0.5, 0.5]
elif case in ('case3'):
    clim = [-0.65, 0.65]
elif case in ('case4'):
    clim = [-0.5, 0.5]
        
if uw.mpi.size == 1:
    # pressure plot
    plot_scalar(mesh, p_uw.sym, 'p_uw', _cmap=cmc.vik.resampled(41), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'p_uw.png')

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
        
if uw.mpi.size == 1 and comp_ana:
    # pressure error plot
    plot_scalar(mesh, p_err.sym, 'p_err(relative)', _cmap=cmc.vik.resampled(41), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'p_r_err.png')
# -

# plotting percentage error in uw
if uw.mpi.size == 1 and comp_ana:
    # pressure error plot
    plot_scalar(mesh, (p_err.sym[0]/p_ana.sym[0])*100, 'p_err(%)', _cmap=cmc.vik.resampled(41), _clim=[-10, 10], _save_png=True, 
                _dir_fname=output_dir+'p_p_err.png')

# computing L2 norm
if comp_ana:
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
# +
# # saving h5 and xdmf file
# mesh.petsc_save_checkpoint(index=0, meshVars=[v_uw, p_uw, v_ana, p_ana, rho_ana, v_err, p_err], outputPath=output_dir+'output')
# -


# memory stats: needed only on mac
import psutil
process = psutil.Process()
print('RAM Used (GB):', process.memory_info().rss/1024 ** 3) 
# ### From here onwards we compute quantities to compare analytical and numerical solution.
# ### Feel free to comment out this section

# #### Plotting velocity and pressure on lower and outer boundaries

# +
# get indices
lower_indx = uw.discretisation.petsc_discretisation.petsc_dm_find_labeled_points_local(mesh.dm, 'Lower')
upper_indx = uw.discretisation.petsc_discretisation.petsc_dm_find_labeled_points_local(mesh.dm, 'Upper')

# get theta from from x, y
lower_theta = uw.function.evalf(th_uw, mesh.data[lower_indx])
lower_theta[lower_theta<0] += 2 * np.pi
upper_theta = uw.function.evalf(th_uw, mesh.data[upper_indx])
upper_theta[upper_theta<0] += 2 * np.pi
# -

# lower and upper bd velocities
if comp_ana:
    with mesh.access(v_uw, p_uw):
        def get_ana_soln_2(_var, _coords, _r_int, _soln_above, _soln_below):
            # get analytical solution into mesh variables
            r = uw.function.evalf(r_uw, _coords)
            for i, coord in enumerate(_coords):
                if r[i]>_r_int:
                    _var[i] = _soln_above(coord)
                else:
                    _var[i] = _soln_below(coord)
                    
        # pressure arrays
        p_ana_lower = np.zeros((len(lower_indx), 1))
        p_ana_upper = np.zeros((len(upper_indx), 1))
        p_uw_lower = np.zeros((len(lower_indx), 1))
        p_uw_upper = np.zeros((len(upper_indx), 1))

        # pressure analy and uw
        get_ana_soln_2(p_ana_upper, mesh.data[upper_indx], r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)
        get_ana_soln_2(p_ana_lower, mesh.data[lower_indx], r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)
        p_uw_lower[:,0] = uw.function.evalf(p_uw.sym, mesh.data[lower_indx])
        p_uw_upper[:,0] = uw.function.evalf(p_uw.sym, mesh.data[upper_indx]) 

        # velocity arrays
        v_ana_lower = np.zeros_like(mesh.data[lower_indx])
        v_ana_upper = np.zeros_like(mesh.data[upper_indx])
        v_uw_lower = np.zeros_like(mesh.data[lower_indx])
        v_uw_upper = np.zeros_like(mesh.data[upper_indx])
        
        # velocity analy and uw
        get_ana_soln_2(v_ana_upper, mesh.data[upper_indx], r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
        get_ana_soln_2(v_ana_lower, mesh.data[lower_indx], r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
        v_uw_lower = uw.function.evalf(v_uw.sym, mesh.data[lower_indx])
        v_uw_upper = uw.function.evalf(v_uw.sym, mesh.data[upper_indx])

# sort array for plotting
sort_lower = lower_theta.argsort()
sort_upper = upper_theta.argsort()


def plot_stats(_data_list='', _label_list='', _line_style='', _xlabel='', _ylabel='', _xlim='', _ylim='', _mod_xticks=False, 
               _save_pdf='', _output_path='', _fname=''):
    # plot some statiscs 
    fig, ax = plt.subplots()
    for i, data in enumerate(_data_list):
        ax.plot(data[:,0], data[:,1], label=_label_list[i], linestyle=_line_style[i])
    
    ax.set_xlabel(_xlabel)
    ax.set_ylabel(_ylabel)
    ax.grid(linestyle='--')
    ax.legend(loc=(1.01, 0.60), fontsize=14)

    if len(_xlim)!=0:
        ax.set_xlim(_xlim[0], _xlim[1])
        if _mod_xticks:
            ax.set_xticks(np.arange(_xlim[0], _xlim[1]+0.01, np.pi/2))
            labels = ['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
            ax.set_xticklabels(labels)

    if len(_ylim)!=0:
        ax.set_ylim(_ylim[0], _ylim[1])

    if _save_pdf:
        plt.savefig(_output_path+_fname+'.pdf', format='pdf', bbox_inches='tight')


# +
# plotting pressure on lower and upper boundaries
if case in ('case1'):
    ylim = [-0.75, 0.75]
elif case in ('case2'):
    ylim = [-0.65, 0.65]
elif case in ('case3'):
    ylim = [-0.95, 0.95]
elif case in ('case4'):
    ylim = [-0.65, 0.65]
    
data_list = [np.hstack((np.c_[lower_theta[sort_lower]], p_ana_lower[sort_lower])),
             np.hstack((np.c_[upper_theta[sort_upper]], p_ana_upper[sort_upper])), 
             np.hstack((np.c_[lower_theta[sort_lower]], p_uw_lower[sort_lower])), 
             np.hstack((np.c_[upper_theta[sort_upper]], p_uw_upper[sort_upper]))]
label_list = ['k='+str(k)+' (analy.), '+r'$r=R_{1}$',
              'k='+str(k)+' (analy.), '+r'$r=R_{2}$',
              'k='+str(k)+' (UW), '+r'$r=R_{1}$', 
              'k='+str(k)+' (UW), '+r'$r=R_{2}$']
linestyle_list = ['-', '-', '--', '--']

plot_stats(_data_list=data_list, _label_list=label_list, _line_style=linestyle_list, _xlabel=r'$\theta$', _ylabel='Pressure', 
           _xlim=[0, 2*np.pi], _ylim=ylim, _mod_xticks=True, _save_pdf=True, _output_path=output_dir, _fname='p_r_i_o_'+bc_type)


# -

def get_magnitude(_array):
    # compute velocity magnitude
    sqrd_sum = np.zeros((_array.shape[0], 1))
    for i in range(_array.shape[1]):
        sqrd_sum += _array[:, i:i+1]**2 
    return np.sqrt(sqrd_sum)


# compute velocity magnitude
v_ana_lower_mag = get_magnitude(v_ana_lower)
v_ana_upper_mag = get_magnitude(v_ana_upper)
v_uw_lower_mag = get_magnitude(v_uw_lower)
v_uw_upper_mag = get_magnitude(v_uw_upper)

# +
# plotting vel. mag on lower and upper boundaries
if case in ('case1'):
    ylim = [0, 5e-2]
elif case in ('case2'):
    ylim = [0, 4e-2]
elif case in ('case3'):
    ylim = [0, 3.7e-9]
elif case in ('case4'):
    ylim = [0, 2.2e-9]
    
data_list = [np.hstack((np.c_[lower_theta[sort_lower]], v_ana_lower_mag[sort_lower])),
             np.hstack((np.c_[upper_theta[sort_upper]], v_ana_upper_mag[sort_upper])), 
             np.hstack((np.c_[lower_theta[sort_lower]], v_uw_lower_mag[sort_lower])), 
             np.hstack((np.c_[upper_theta[sort_upper]], v_uw_upper_mag[sort_upper]))]
label_list = ['k='+str(k)+' (analy.), '+r'$r=R_{1}$',
              'k='+str(k)+' (analy.), '+r'$r=R_{2}$',
              'k='+str(k)+' (UW), '+r'$r=R_{1}$', 
              'k='+str(k)+' (UW), '+r'$r=R_{2}$']
linestyle_list = ['-', '-', '--', '--']

plot_stats(_data_list=data_list, _label_list=label_list, _line_style=linestyle_list, _xlabel=r'$\theta$', _ylabel='Velocity Magnitude', 
           _xlim=[0, 2*np.pi], _ylim=ylim, _mod_xticks=True, _save_pdf=True, _output_path=output_dir, _fname='vel_r_i_o_'+bc_type)
# -

# uw velocity in (r, theta)
v_uw_r_th = mesh.CoordinateSystem.rRotN*v_uw.sym.T

# radial and theta components of velocity integrated over mesh. Theoretically output should be zero
if comp_ana:   
    v_r_rms_I = uw.maths.Integral(mesh, v_uw_r_th[0])
    print((1/(2*np.pi))*v_r_rms_I.evaluate())

    v_th_rms_I = uw.maths.Integral(mesh, v_uw_r_th[1])
    print((1/(2*np.pi))*v_th_rms_I.evaluate())    

# theta and r arrays
theta_0_2pi = np.linspace(0, 2*np.pi, 1000, endpoint=True)
r_i_o = np.linspace(r_i, r_o-1e-4, 21, endpoint=True)


def get_vel_avg_r(_theta_arr, _r_arr, _vel_comp):
    'Return average velocity'
    vel_avg_arr = np.zeros_like(_r_arr)
    for i, r_val in enumerate(_r_arr):
        x_arr = r_val*np.cos(_theta_arr)
        y_arr = r_val*np.sin(_theta_arr)
        xy_arr = np.stack((x_arr, y_arr), axis=-1)
        vel_xy = uw.function.evaluate(_vel_comp, xy_arr)
        vel_avg_arr[i] = integrate.simpson(vel_xy, x=_theta_arr)/(2*np.pi)
    return vel_avg_arr


# velocity radial component average
vr_avg = get_vel_avg_r(theta_0_2pi, r_i_o, v_uw_r_th[0])

# +
# plotting velocity radial component average
if case in ('case1'):
    ylim = [-1e-6, 1e-6]
elif case in ('case2'):
    ylim = [-2.5e-9, 2.5e-9]
elif case in ('case3'):
    ylim = [-8e-7, 8e-7]
elif case in ('case4'):
    ylim = [-1e-8, 1e-8]
    
data_list = [np.hstack((np.c_[r_i_o], np.c_[np.zeros_like(r_i_o)])),
             np.hstack((np.c_[r_i_o], np.c_[vr_avg]))]
label_list = ['k='+str(k)+' (analy.)',
              'k='+str(k)+' (UW)',]
linestyle_list = ['-', '--']

plot_stats(_data_list=data_list, _label_list=label_list, _line_style=linestyle_list, _xlabel='r', _ylabel=r'$<v_{r}>$', 
           _xlim=[r_i, r_o], _ylim=ylim, _save_pdf=True, _output_path=output_dir, _fname='vel_r_avg_'+bc_type)
# -

# velocity theta component average
vth_avg = get_vel_avg_r(theta_0_2pi, r_i_o, v_uw_r_th[1])

# +
# plotting velocity theta component average
if case in ('case1'):
    ylim = [-5e-7, 5e-7]
elif case in ('case2'):
    ylim = [-2e-8, 2e-8]
elif case in ('case3'):
    ylim = [-5e-7, 5e-7]
elif case in ('case4'):
    ylim = [-2e-8, 2e-8]
    
data_list = [np.hstack((np.c_[r_i_o], np.c_[np.zeros_like(r_i_o)])),
             np.hstack((np.c_[r_i_o], np.c_[vth_avg]))]
label_list = ['k='+str(k)+' (analy.)',
              'k='+str(k)+' (UW)',]
linestyle_list = ['-', '--']

plot_stats(_data_list=data_list, _label_list=label_list, _line_style=linestyle_list, _xlabel='r', _ylabel=r'$<v_{\theta}>$', 
           _xlim=[r_i, r_o], _ylim=ylim, _save_pdf=True, _output_path=output_dir, _fname='vel_th_avg_'+bc_type)
# -


