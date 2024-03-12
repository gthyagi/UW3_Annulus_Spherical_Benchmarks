# ## Annulus Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark ASPECT results](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/annulus/doc/annulus.html)
# #### [Benchmark paper](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-2765/) 
#
# *Author: [Thyagarajulu Gollapalli](https://github.com/gthyagi)*

# ### Analytical solution

# This benchmark is based on a manufactured solution in which an analytical solution to the isoviscous incompressible Stokes equations is derived in an annulus geometry. The velocity and pressure fields are as follows:
#
# $$ v_{\theta}(r, \theta) = f(r) \cos(k\theta) $$
#
# $$ v_r(r, \theta) = g(r)k \sin(k\theta) $$
#
# $$ p(r, \theta) = kh(r) \sin(k\theta) + \rho_0g_r(R_2 - r) $$
#
# $$ \rho(r, \theta) = m(r)k \sin(k\theta) + \rho_0 $$
#
# with
#
# $$ f(r) = Ar + \frac{B}{r} $$
#
# $$ g(r) = \frac{A}{2}r + \frac{B}{r}\ln r + \frac{C}{r} $$
#
# $$ h(r) = \frac{2g(r) - f(r)}{r} $$
#
# $$ m(r) = g''(r) - \frac{g'(r)}{r} - \frac{g(r)}{r^2}(k^2 - 1) + \frac{f(r)}{r^2} + \frac{f'(r)}{r} $$
#
# $$ A = -C\frac{2(\ln R_1 - \ln R_2)}{R_2^2 \ln R_1 - R_1^2 \ln R_2} $$
#
# $$ B = -C\frac{R_2^2 - R_1^2}{R_2^2 \ln R_1 - R_1^2 \ln R_2} $$
#
#
# The parameters $A$ and $B$ are chosen so that $ v_r(R_1, \theta) = v_r(R_2, \theta) = 0 $ for all $\theta \in [0, 2\pi]$, i.e. the velocity is tangential to both inner and outer surfaces. The gravity vector is radial inward and of unit length.
#
# The parameter $k$ controls the number of convection cells present in the domain
#
# In the present case, we set $ R_1 = 1, R_2 = 2$ and $C = -1 $.

# +
import underworld3 as uw
from underworld3.systems import Stokes

import numpy as np
import sympy
import os
import h5py
import sys
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

# +
# radii
r_i = 1
r_o = 2

k = int(sys.argv[1]) # controls the number of convection cells

res_inv = int(sys.argv[2])
res = 1/res_inv

vdegree  = int(sys.argv[3])
pdegree = int(sys.argv[4])
pcont = sys.argv[5].lower()

vel_penalty = float(sys.argv[6])
stokes_tol = float(sys.argv[7])
vel_penalty_str = str("{:.1e}".format(vel_penalty))
stokes_tol_str = str("{:.1e}".format(stokes_tol))
# -

# compute analytical solutions
comp_ana = True
plotting = False
do_timing = True

# +
output_dir = os.path.join(os.path.join("./output/Latex_files/"), 
                          f"model_k_{k}_res_{res_inv}_vdeg_{vdegree}_pdeg_{pdegree}_pcont_{pcont}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/")

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
# -

# ### Analytical solution in sympy

# +
# analytical solution
r = sympy.symbols('r')
theta = sympy.Symbol('theta', real=True)

C=-1
A = -C*(2*(np.log(r_i) - np.log(r_o))/((r_o**2)*np.log(r_i) - (r_i**2)*np.log(r_o)))
B = -C*((r_o**2 - r_i**2)/((r_o**2)*np.log(r_i) - (r_i**2)*np.log(r_o)))
rho_0 = 0

f = sympy.Function('f')(r)
f = A*r + B/r

g = sympy.Function('g')(r)
g = ((A/2)*r) + ((B/r) * sympy.ln(r)) + (C/r)

h = sympy.Function('h')(r)
h = (2*g - f)/r

m = sympy.Function('m')(r)
m = g.diff(r, r) - (g.diff(r)/r) - (g/r**2)*(k**2 - 1) + (f/r**2) + (f.diff(r)/r)

v_r = g*k*sympy.sin(k*theta)
v_theta = f*sympy.cos(k*theta)
p = k*h*sympy.sin(k*theta) + rho_0*(r_o-r)
rho = m*k*sympy.sin(k*theta) + rho_0
v_x = v_r * sympy.cos(theta) - v_theta * sympy.sin(theta) 
v_y = v_r * sympy.sin(theta) + v_theta * sympy.cos(theta)


# -

# ### Plotting and Analysis functions

def plot_mesh(_mesh, _save_png=False, _dir_fname='', _title=''):
    # plot mesh
    pvmesh = vis.mesh_to_pv_mesh(_mesh)

    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, edge_color="Grey", show_edges=True, use_transparency=False, opacity=1.0, )

    pl.show(cpos="xy")

    if len(_title)!=0:
        pl.add_text(_title, font_size=18, position=(1025, 1075))
    
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
        pl.add_text(_title, font_size=18, position=(1025, 1075))

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
        pl.add_text(_title, font_size=18, position=(1025, 1075))

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

if do_timing:
    uw.timing.reset()
    uw.timing.start()

# mesh
mesh = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=res, qdegree=max(pdegree, vdegree), degree=1)

if do_timing:
    uw.timing.stop()
    uw.timing.print_table(group_by='line_routine', output_file=f"{output_dir}/mesh_create_time.txt",  display_fraction=1.00)

# print mesh size in each cpu
if uw.mpi.rank == 0:
    print('-------------------------------------------------------------------------------')
mesh.dm.view()
if uw.mpi.rank == 0:
    print('-------------------------------------------------------------------------------')

# +
# # print mesh size in each cpu
# print(f'rank: {uw.mpi.rank}, mesh size: {mesh.data.shape}')
# -

if uw.mpi.size == 1 and plotting:
    plot_mesh(mesh, _save_png=True, _dir_fname=output_dir+'mesh.png', _title='k='+str(k))

# +
# mesh variables
v_uw = uw.discretisation.MeshVariable(r"\mathbf{u}", mesh, 2, degree=vdegree)
if pcont == 'true':
    p_uw = uw.discretisation.MeshVariable(r"p", mesh, 1, degree=pdegree, continuous=True)
else:
    p_uw = uw.discretisation.MeshVariable(r"p", mesh, 1, degree=pdegree, continuous=False)

if comp_ana:
    v_ana = uw.discretisation.MeshVariable(r"\mathbf{v}", mesh, 2, degree=vdegree)
    v_err = uw.discretisation.MeshVariable(r"\mathbf{u_e}", mesh, 2, degree=vdegree)
    if pcont == 'true':
        p_ana = uw.discretisation.MeshVariable(r"p_a", mesh, 1, degree=pdegree, continuous=True)
        rho_ana = uw.discretisation.MeshVariable(r"rho_a", mesh, 1, degree=pdegree, continuous=True)
        p_err = uw.discretisation.MeshVariable(r"p_e", mesh, 1, degree=pdegree, continuous=True)
    else:
        p_ana = uw.discretisation.MeshVariable(r"p_a", mesh, 1, degree=pdegree, continuous=False)
        rho_ana = uw.discretisation.MeshVariable(r"rho_a", mesh, 1, degree=pdegree, continuous=False)
        p_err = uw.discretisation.MeshVariable(r"p_e", mesh, 1, degree=pdegree, continuous=False)

# +
# # Null space evaluation
# norm_v = uw.discretisation.MeshVariable("N", mesh, 2, degree=1, varsymbol=r"{\hat{n}}")
# with mesh.access(norm_v):
#     norm_v.data[:,0] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[0], norm_v.coords)
#     norm_v.data[:,1] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[1], norm_v.coords)

# +
# Some useful coordinate stuff
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR

# Null space in velocity (constant v_theta) expressed in x,y coordinates
v_theta_fn_xy = r_uw * mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0,1))
# -

# Analytical velocities
if comp_ana:
    with mesh.access(v_ana, p_ana, rho_ana):
        if k==0:
            v_ana_expr = mesh.CoordinateSystem.rRotN.T * sympy.Matrix([0, v_theta.subs({r:r_uw, theta:th_uw})])
            p_ana.data[:,0] = 0
            rho_ana.data[:,0] = 0
        else:
            v_ana_expr = mesh.CoordinateSystem.rRotN.T * sympy.Matrix([v_r.subs({r:r_uw, theta:th_uw}), 
                                                                       v_theta.subs({r:r_uw, theta:th_uw})])
            p_ana.data[:,0] = uw.function.evalf(p.subs({r:r_uw, theta:th_uw}), p_ana.coords)
            rho_ana.data[:,0] = uw.function.evalf(rho.subs({r:r_uw, theta:th_uw}), rho_ana.coords)
            
        v_ana.data[:,0] = uw.function.evalf(v_ana_expr[0], v_ana.coords)
        v_ana.data[:,1] = uw.function.evalf(v_ana_expr[1], v_ana.coords)

# plotting analytical density
if uw.mpi.size == 1 and comp_ana and plotting:
    # density plot
    plot_scalar(mesh, rho_ana.sym, 'rho', _cmap=cmc.roma.resampled(31), _clim=[-67.5, 67.5], _save_png=True, 
                _dir_fname=output_dir+'rho_ana.png', _title='k='+str(k), )
    # saving colobar separately 
    save_colorbar(_colormap=cmc.roma.resampled(31), _cb_bounds='', _vmin=-67.5, _vmax=67.5, _figsize_cb=(5, 5), _primary_fs=18, _cb_orient='horizontal', 
                  _cb_axis_label='Density', _cb_label_xpos=0.5, _cb_label_ypos=-2.0, _fformat='pdf', _output_path=output_dir, _fname='rho_ana')

# plotting analytical velocities
if uw.mpi.size == 1 and comp_ana and plotting:
    # velocity plot
    plot_vector(mesh, v_ana, _vector_name='v_ana', _cmap=cmc.lapaz.resampled(11), _clim=[0., 2.3], _vmag=1e-1, _vfreq=40,
                _save_png=True, _dir_fname=output_dir+'vel_ana.png')
    # saving colobar separately 
    save_colorbar(_colormap=cmc.lapaz.resampled(11), _cb_bounds='', _vmin=0., _vmax=2.3, _figsize_cb=(5, 5), _primary_fs=18, _cb_orient='horizontal', 
                  _cb_axis_label='Velocity', _cb_label_xpos=0.5, _cb_label_ypos=-2.05, _fformat='pdf', _output_path=output_dir, _fname='v_ana')

# plotting analytical pressure
if uw.mpi.size == 1 and comp_ana and plotting:
    # pressure plot
    plot_scalar(mesh, p_ana.sym, 'p_ana', _cmap=cmc.vik.resampled(41), _clim=[-8.5, 8.5], _save_png=True, 
                _dir_fname=output_dir+'p_ana.png')
    # saving colobar separately 
    save_colorbar(_colormap=cmc.vik.resampled(41), _cb_bounds='', _vmin=-8.5, _vmax=8.5, _figsize_cb=(5, 5), _primary_fs=18, _cb_orient='horizontal', 
                  _cb_axis_label='Pressure', _cb_label_xpos=0.5, _cb_label_ypos=-2.0, _fformat='pdf', _output_path=output_dir, _fname='p_ana')

# +
# Create Stokes object
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw, solver_name="stokes")
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# gravity
gravity_fn = -1.0 * unit_rvec

# density
rho_uw = rho.subs({r:r_uw, theta:th_uw})

# bodyforce term
stokes.bodyforce = rho_uw*gravity_fn
# -

# boundary conditions
v_diff =  v_uw.sym - v_ana.sym
stokes.add_natural_bc(vel_penalty*v_diff, "Upper")
stokes.add_natural_bc(vel_penalty*v_diff, "Lower")


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

if do_timing:
    uw.timing.reset()
    uw.timing.start()

stokes.solve()

if do_timing:
    uw.timing.stop()
    uw.timing.print_table(group_by='line_routine', output_file=f"{output_dir}/stokes_solve_time.txt", display_fraction=1.00)

# +
# # Null space evaluation
# I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v_uw.sym))
# norm = I0.evaluate()
# I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
# vnorm = I0.evaluate()

# print(norm/vnorm, vnorm)

# with mesh.access(v_uw):
#     dv = uw.function.evaluate(norm * v_theta_fn_xy, v_uw.coords) / vnorm
#     v_uw.data[...] -= dv 
# -

# compute error
if comp_ana:
    with mesh.access(v_uw, p_uw, v_err, p_err):
        v_err.data[:,0] = v_uw.data[:,0] - v_ana.data[:,0]
        v_err.data[:,1] = v_uw.data[:,1] - v_ana.data[:,1]
        p_err.data[:,0] = p_uw.data[:,0] - p_ana.data[:,0]

# plotting velocities from uw
if uw.mpi.size == 1 and plotting:
    plot_vector(mesh, v_uw, _vector_name='v_uw', _cmap=cmc.lapaz.resampled(11), _clim=[0., 2.3], _vfreq=40, _vmag=1e-1,
                _save_png=True, _dir_fname=output_dir+'vel_uw.png')

# plotting errror in velocities
if uw.mpi.size == 1 and comp_ana and plotting:
    plot_vector(mesh, v_err, _vector_name='v_err', _cmap=cmc.lapaz.resampled(11), _clim=[0., 2.3e-4], _vfreq=20, _vmag=1e3, 
                _save_png=True, _dir_fname=output_dir+'vel_err.png')

if comp_ana and plotting:   
    vmag_expr = (sympy.sqrt(v_err.sym.dot(v_err.sym))/sympy.sqrt(v_ana.sym.dot(v_ana.sym)))*100
    if uw.mpi.size == 1:
        plot_scalar(mesh, vmag_expr, 'vmag_err(%)', _cmap=cmc.oslo_r.resampled(21), _clim=[0, 1], 
                    _save_png=True, _dir_fname=output_dir+'vmag_p_err.png')

# plotting pressure from uw
if uw.mpi.size == 1 and plotting:
    plot_scalar(mesh, p_uw.sym, 'p_uw', _cmap=cmc.vik.resampled(41), _clim=[-8.5, 8.5], 
                _save_png=True, _dir_fname=output_dir+'p_uw.png')

# plotting error in uw
if uw.mpi.size == 1 and comp_ana and plotting:
    plot_scalar(mesh, p_err.sym, 'p_err', _cmap=cmc.vik.resampled(41), _clim=[-0.006, 0.006], 
                _save_png=True, _dir_fname=output_dir+'p_err.png')

# plotting percentage error in uw
if k==0:
    pass
elif uw.mpi.size == 1 and comp_ana and plotting:
    plot_scalar(mesh, (p_err.sym[0]/p_ana.sym[0])*100, 'p_err(%)', _cmap=cmc.vik.resampled(41), _clim=[-100, 100], 
                _save_png=True, _dir_fname=output_dir+'p_p_err.png')

# computing L2 norm
if comp_ana:
    with mesh.access(v_err, p_err, p_ana, v_ana):    
        v_err_I = uw.maths.Integral(mesh, v_err.sym.dot(v_err.sym))
        v_ana_I = uw.maths.Integral(mesh, v_ana.sym.dot(v_ana.sym))
        v_err_l2 = np.sqrt(v_err_I.evaluate())/np.sqrt(v_ana_I.evaluate())
        if uw.mpi.rank == 0:
            print('Relative error in velocity in the L2 norm: ', v_err_l2)
        
        if k==0:
            if uw.mpi.rank == 0:
                print('Integration of analytical solution over the domain is zero. For k=0, it is not possible to compute L2 norm for pressure.')
        else:
            p_err_I = uw.maths.Integral(mesh, p_err.sym.dot(p_err.sym))
            p_ana_I = uw.maths.Integral(mesh, p_ana.sym.dot(p_ana.sym))
            p_err_l2 = np.sqrt(p_err_I.evaluate())/np.sqrt(p_ana_I.evaluate())
            if uw.mpi.rank == 0:
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
        if k==0:
            f.create_dataset("p_l2_norm", data=np.inf)
        else:
            f.create_dataset("p_l2_norm", data=p_err_l2)

# +
# # saving h5 and xdmf file
# mesh.petsc_save_checkpoint(index=0, meshVars=[v_uw, p_uw, v_ana, p_ana, rho_ana, v_err, p_err], outputPath=output_dir+'output')
# -

if uw.mpi.rank == 0:
    print('-------------------------------------------------------------------------------')

# memory stats: needed only on mac
import psutil
process = psutil.Process()
print(f'rank: {uw.mpi.rank}, RAM Used (GB): {process.memory_info().rss/1024 ** 3}')


