# ## Spherical Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark ASPECT results](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/hollow_sphere/doc/hollow_sphere.html)
# #### [Benchmark paper](https://se.copernicus.org/articles/8/1181/2017/) 
#
# *Author: [Thyagarajulu Gollapalli](https://github.com/gthyagi)*

# ### Analytical solution

# This benchmark is based on [Thieulot](https://se.copernicus.org/articles/8/1181/2017/) in which an analytical solution to the isoviscous incompressible Stokes equations is derived in a spherical shell geometry. The velocity and pressure fields are as follows:
#
# $$ v_{\theta}(r, \theta) = f(r) \sin(\theta) $$
# $$ v_{\phi}(r, \theta) = f(r) \sin(\theta) $$
# $$ v_r(r, \theta) = g(r) \cos(\theta) $$
# $$ p(r, \theta) = h(r) \cos(\theta) $$
# $$ \mu(r) = \mu_{0}r^{m+1} $$
#
# where $m$ is an integer (positive or negative). Note that $m = −1$ yields a constant viscosity.
#
# $$ f(r) = {\alpha} r^{-(m+3)} + \beta r $$
#
# ##### Case $m = -1$
#
# $$ g(r) = -\frac{2}{r^2} \bigg(\alpha \ln r + \frac{\beta}{3}r^3 + \gamma \bigg) $$
# $$ h(r) = \frac{2}{r} \mu_{0} g(r) $$
# $$ \rho(r, \theta) = \bigg(\frac{\alpha}{r^4} (8\ln r - 6) + \frac{8\beta}{3r} + 8\frac{\gamma}{r^4} \bigg) \cos(\theta)$$
# $$ \alpha = -\gamma \frac{R_2^3 - R_1^3}{R_2^3 \ln R_1 - R_1^3 \ln R_2} $$
# $$ \beta = -3\gamma \frac{\ln R_2 - \ln R_1}{R_1^3 \ln R_2 - R_2^3 \ln R_1} $$
#
# ##### Case $m \neq -1$
#
# $$ g(r) = -\frac{2}{r^2} \bigg(-\frac{\alpha}{m+1} r^{-(m+1)} + \frac{\beta}{3}r^3 + \gamma \bigg) $$
# $$ h(r) = \frac{m+3}{r} \mu(r) g(r) $$
# $$ \rho(r, \theta) = \bigg[2\alpha r^{-(m+4)}\frac{m+3}{m+1}(m-1) - \frac{2\beta}{3}(m-1)(m+3) - m(m+5)\frac{2\gamma}{r^3} \bigg] \cos(\theta) $$
# $$ \alpha = \gamma (m+1) \frac{R_1^{-3} - R_2^{-3}}{R_1^{-(m+4)} - R_2^{-(m+4)}} $$
# $$ \beta = -3\gamma \frac{R_1^{m+1} - R_2^{m+1}}{R_1^{m+4} - R_2^{m+4}} $$
# Note that this imposes that $m \neq −4$.
#
# The radial component of the velocity is nul on the inside $r = R_1$ and outside $r = R_2$ of the domain, thereby insuring a
# tangential flow on the boundaries, i.e.
# $$ v_r(R_1, \theta) = v_r(R_2, \theta) = 0 $$
#
# The gravity vector is radial and of unit length. We set $R_1 = 0.5$ and $R_2 = 1$.
#
# In this work, the following spherical coordinates conventions are used: $r$ is the radial distance, $\theta \in [0,\pi]$ is the polar angle and $\phi \in [0, 2\pi]$ is the azimuthal angle.

from mpi4py import MPI

# +
import underworld3 as uw
from underworld3.systems import Stokes

import numpy as np
import sympy
import os
import assess
import h5py
from enum import Enum
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
    from scipy import integrate

# +
# mesh options
r_o = 1.0
r_i = 0.5

res = 8 # 4, 8, 16, 32, 64, 128
cellsize = 1/res

# m value
m = -1

# +
# fem stuff
vdegree  = 2
pdegree = 1
pcont = True
pcont_str = str(pcont).lower()

vel_penalty = 2.5e8
stokes_tol = 1e-10
vel_penalty_str = str("{:.1e}".format(vel_penalty))
stokes_tol_str = str("{:.1e}".format(stokes_tol))
# -

# compute analytical solution
analytical = True
visualize = True
timing = True

# ### Analytical solution in sympy

# The Cartesian unit vectors are related to the spherical unit vectors by
# $$ 
# \begin{pmatrix}
# \hat{\mathbf{e}}_x \\
# \hat{\mathbf{e}}_y \\
# \hat{\mathbf{e}}_z \\
# \end{pmatrix}
# =
# \begin{pmatrix}
# \sin(\theta) \cos(\phi) & \cos(\theta) \cos(\phi) & -\sin(\phi) \\
# \sin(\theta) \sin(\phi) & \cos(\theta) \sin(\phi) & \cos(\phi) \\
# \cos(\theta) & -\sin(\theta) & 0 \\
# \end{pmatrix}
# \begin{pmatrix}
# \hat{\mathbf{e}}_r \\
# \hat{\mathbf{e}}_{\theta} \\
# \hat{\mathbf{e}}_{\phi} \\
# \end{pmatrix}
# $$

# +
# analytical solution
r = sympy.symbols('r')
theta = sympy.Symbol('theta', real=True)
phi = sympy.Symbol('phi', real=True)

gamma = 1.0
mu_0 = 1.0
mu = mu_0*(r**(m+1))
rho_0 = 0

if m==-1:
    alpha = -gamma*((r_o**3 - r_i**3)/((r_o**3)*np.log(r_i) - (r_i**3)*np.log(r_o)))
    beta = -3*gamma*((np.log(r_o) - np.log(r_i))/((r_i**3)*np.log(r_o) - (r_o**3)*np.log(r_i)))

    f = sympy.Function('f')(r)
    f = alpha*(r**-(m+3)) + beta*r

    g = sympy.Function('g')(r)
    g = (-2/(r**2))*(alpha*sympy.ln(r) + (beta/3)*(r**3) + gamma)

    h = sympy.Function('h')(r)
    h = (2/r)*mu_0*g

    # rho = (((alpha/(r**4))*(8*sympy.ln(r) - 6)) + ((8*beta)/(3*r)) + ((8*gamma)/(r**4))) * sympy.cos(theta)
    f_fd = sympy.Derivative(f, r, evaluate=True)
    f_sd = sympy.Derivative(f_fd, r, evaluate=True)
    f_td = sympy.Derivative(f_sd, r, evaluate=True)
    g_fd = sympy.Derivative(g, r, evaluate=True)
    g_sd = sympy.Derivative(g_fd, r, evaluate=True)
    F_r = -r*f_td - 3*f_sd + 2*f_fd/r -g_sd + 2*((f+g)/r**2)
    rho_ = F_r * sympy.cos(theta)
    # rho_ = (F_r * sympy.cos(theta)) + rho_0
    rho = rho_.simplify()
else:
    alpha = gamma*(m+1)*((r_i**-3 - r_o**-3)/((r_i**-(m+4)) - (r_o**-(m+4))))
    beta = -3*gamma*((r_i**(m+1)) - (r_o**(m+1)))/((r_i**(m+4)) - (r_o**(m+4)))

    f = sympy.Function('f')(r)
    f = alpha*(r**-(m+3)) + beta*r

    g = sympy.Function('g')(r)
    g = (-2/(r**2))*((-alpha/(m+1))*r**(-(m+1)) + (beta/3)*(r**3) + gamma)

    h = sympy.Function('h')(r)
    h = ((m+3)/r)*mu*g

    # rho = (((2*alpha*(r**(-(m+4))))*((m+3)/(m+1))*(m-1)) - ((2*beta/3)*(m-1)*(m+3)) - (m*(m+5)*(2*gamma/(r**3)))) * sympy.cos(theta)
    # rho = ( ( 2 * alpha * (r**(-(m+4))) * ((m+3)/(m+1))*(m-1) ) - ( 2 * beta * (m-1) * (m+3)/3 ) - ( m * (m+5) * 2*gamma/r**3 ) ) * sympy.cos(theta)
    f_fd = sympy.Derivative(f, r, evaluate=True)
    f_sd = sympy.Derivative(f_fd, r, evaluate=True)
    f_td = sympy.Derivative(f_sd, r, evaluate=True)
    F_r = (-r**2)*f_td - ((2*m)+5)*r*f_sd - ((m*(m+3)) - 2)*f_fd + m*(m+5)*((f+g)/r)
    rho_ = r**m * F_r * sympy.cos(theta)
    # rho_ = (F_r * sympy.cos(theta)) + rho_0
    rho = rho_.simplify()
    
p = h*sympy.cos(theta)
# p = h*sympy.cos(theta) + (rho_0 * 1 * (r_o - r))

v_r = g*sympy.cos(theta)
v_theta = f*sympy.sin(theta)
v_phi = f*sympy.sin(theta)

v_x = v_r*sympy.sin(theta)*sympy.cos(phi) + v_theta*sympy.cos(theta)*sympy.cos(phi) - v_phi*sympy.sin(phi)
v_y = v_r*sympy.sin(theta)*sympy.sin(phi) + v_theta*sympy.cos(theta)*sympy.sin(phi) + v_phi*sympy.cos(phi)
v_z = v_r*sympy.cos(theta) - v_theta*sympy.sin(theta)

# +
# output_dir = os.path.join(os.path.join("./output/Latex_Dir/"), f"{case}/")
output_dir = os.path.join(os.path.join("./output/"), 
                          f'case_m_{m}_res_{res}_vdeg_{vdegree}_pdeg_{pdegree}'\
                          f'_pcont_{pcont_str}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/')

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
# -

if uw.mpi.rank==0:
    # plot f, g, h, viscosity functions
    rad_np = np.linspace(1, 0.5, num=200, endpoint=True)
    f_np = np.zeros_like(rad_np)
    g_np = np.zeros_like(rad_np)
    h_np = np.zeros_like(rad_np)
    mu_np = np.zeros_like(rad_np)
    
    for i, r_val in enumerate(rad_np):
        f_np[i] = f.subs({r:r_val})
        g_np[i] = g.subs({r:r_val})
        h_np[i] = h.subs({r:r_val})
        mu_np[i] = mu.subs({r:r_val})

    fn_list = [f_np, g_np, h_np, mu_np]
    ylim_list = [[-10, 20], [-3, 4], [-10, 10], [1e-2, 1e2]]
    ylabel_list = [r'$f(r)$', r'$g(r)$', r'$h(r)$', 'Viscosity']
    
    # Set global font size
    plt.rcParams.update({'font.size': 14})
    
    # Create a 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Flatten the axs array to simplify iteration
    axs = axs.flatten()
    
    # Plot data on each subplot using a loop
    for i, ax in enumerate(axs):
        ax.plot(rad_np, fn_list[i], color='green', linewidth=1)
        ax.set_xlim(0.5, 1)
        ax.set_ylim(ylim_list[i])
        ax.grid(linewidth=0.7)
        ax.set_xlabel('r')
        ax.set_ylabel(ylabel_list[i])
    
        if i==3:
            # Set the y-axis to be logarithmic
            ax.set_yscale('log')
            
            # Set y axis label tickmark inward
            ax.tick_params(axis='y', direction='in')
    
        # Set the axis grid marks to point inward
        ax.tick_params(axis='both', direction='in', pad=8)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plots
    plt.savefig(output_dir+'analy_fns.pdf', format='pdf', bbox_inches='tight')


# ### Plotting functions

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

mesh = uw.meshing.SphericalShell(radiusInner=r_i, radiusOuter=r_o, cellSize=cellsize, qdegree=max(pdegree, vdegree), 
                                 filename=f'{output_dir}mesh.msh')

if timing:
    uw.timing.stop()
    uw.timing.print_table(group_by='line_routine', output_file=f"{output_dir}mesh_create_time.txt",  display_fraction=1.00)

if uw.mpi.size == 1 and visualize:
    plot_mesh(mesh, _save_png=True, _dir_fname=output_dir+'mesh.png', _title='', _show_clip=True)

# print mesh size in each cpu
if uw.mpi.rank == 0:
    print('-------------------------------------------------------------------------------')
mesh.dm.view()
if uw.mpi.rank == 0:
    print('-------------------------------------------------------------------------------')

# +
# mesh variables
v_uw = uw.discretisation.MeshVariable('V_u', mesh, mesh.data.shape[1], degree=vdegree)
p_uw = uw.discretisation.MeshVariable('P_u', mesh, 1, degree=pdegree, continuous=pcont)

if analytical:
    v_ana = uw.discretisation.MeshVariable('V_a', mesh, mesh.data.shape[1], degree=vdegree)
    p_ana = uw.discretisation.MeshVariable('P_a', mesh, 1, degree=pdegree, continuous=pcont)
    rho_ana = uw.discretisation.MeshVariable('RHO_a', mesh, 1, degree=pdegree, continuous=True)
    
    v_err = uw.discretisation.MeshVariable('V_e', mesh, mesh.data.shape[1], degree=vdegree)
    p_err = uw.discretisation.MeshVariable('P_e', mesh, 1, degree=pdegree, continuous=pcont)

# +
# norm_v = uw.discretisation.MeshVariable("N", mesh, mesh.data.shape[1], degree=pdegree, varsymbol=r"{\hat{n}}")
# with mesh.access(norm_v):
#     norm_v.data[:,0] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[0], norm_v.coords)
#     norm_v.data[:,1] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[1], norm_v.coords)
#     norm_v.data[:,2] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[2], norm_v.coords)
# -

# Some useful coordinate stuff
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR[0], mesh.CoordinateSystem.xR[1]
phi_uw =sympy.Piecewise((2*sympy.pi + mesh.CoordinateSystem.xR[2], mesh.CoordinateSystem.xR[2]<0), 
                        (mesh.CoordinateSystem.xR[2], True)
                       )

if analytical:
    with mesh.access(v_ana, p_ana, rho_ana):
        
        p_ana.data[:,0] = uw.function.evalf(p.subs({r:r_uw, theta:th_uw, phi:phi_uw}), p_ana.coords)
        rho_ana.data[:,0] = uw.function.evalf(rho.subs({r:r_uw, theta:th_uw}), rho_ana.coords)
            
        v_ana.data[:,0] = uw.function.evalf(v_x.subs({r:r_uw, theta:th_uw, phi:phi_uw}), v_ana.coords)
        v_ana.data[:,1] = uw.function.evalf(v_y.subs({r:r_uw, theta:th_uw, phi:phi_uw}), v_ana.coords)
        v_ana.data[:,2] = uw.function.evalf(v_z.subs({r:r_uw, theta:th_uw, phi:phi_uw}), v_ana.coords)

with mesh.access(rho_ana):
    print(rho_ana.data.min(), rho_ana.data.max())

# +
# plotting analytical velocities
if m==-1:
    clim, vmag, vfreq = [0., 5], 5e0, 75
elif m==3:
    clim, vmag, vfreq = [0., 18], 5e0, 75
    
if uw.mpi.size == 1 and analytical and visualize:
    # velocity plot
    plot_vector(mesh, v_ana, _vector_name='v_ana', _cmap=cmc.lapaz.resampled(21), _clim=clim, _vmag=vmag, _vfreq=vfreq,
                _save_png=True, _dir_fname=output_dir+'vel_ana.png', _show_clip=True, _show_arrows=False)
    # saving colobar separately 
    save_colorbar(_colormap=cmc.lapaz.resampled(11), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                  _cb_orient='horizontal', _cb_axis_label='Velocity', _cb_label_xpos=0.5, _cb_label_ypos=-2.05, _fformat='pdf', 
                  _output_path=output_dir, _fname='v_ana')

# +
# plotting analytical pressure
if m==-1:
    clim = [-2.5, 2.5]
    # clim = [0, 25]
elif m==3:
    clim = [-4, 4]
    
if uw.mpi.size == 1 and analytical and visualize:
    # pressure plot
    plot_scalar(mesh, p_ana.sym, 'p_ana', _cmap=cmc.vik.resampled(41), _clim=clim, _save_png=True, _show_clip=True,
                _dir_fname=output_dir+'p_ana.png')
    # saving colobar separately 
    save_colorbar(_colormap=cmc.vik.resampled(41), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                  _cb_orient='horizontal', _cb_axis_label='Pressure', _cb_label_xpos=0.5, _cb_label_ypos=-2.0, _fformat='pdf', 
                  _output_path=output_dir, _fname='p_ana')
# -

# plotting analytical radial stress
if m==-1:
    clim = [-110, 110]
    # clim = [878., 1121.]
elif m==3:
    clim = [-35, 35]
if uw.mpi.size == 1 and analytical and visualize:
    # pressure plot
    plot_scalar(mesh, rho_ana.sym, 'Rho', _cmap=cmc.roma.resampled(31), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'rho_ana.png', _show_clip=True)
    # saving colobar separately 
    save_colorbar(_colormap=cmc.roma.resampled(31), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                  _cb_orient='horizontal', _cb_axis_label='Rho', _cb_label_xpos=0.5, _cb_label_ypos=-2.0, _fformat='pdf', 
                  _output_path=output_dir, _fname='rho_ana')

# Create Stokes object
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw, solver_name="stokes")
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = mu.subs({r:r_uw, theta:th_uw, phi:phi_uw})
stokes.saddle_preconditioner = 1.0/mu.subs({r:r_uw, theta:th_uw, phi:phi_uw})

# +
# gravity
gravity_fn = -1.0 * unit_rvec

# density
rho_uw = rho.subs({r:r_uw, theta:th_uw, phi:phi_uw})

# bodyforce term
stokes.bodyforce = rho_uw*gravity_fn

# +
# boundary conditions
v_diff =  v_uw.sym - v_ana.sym
stokes.add_natural_bc(vel_penalty*v_diff, "Upper")
stokes.add_natural_bc(vel_penalty*v_diff, "Lower")

# stokes.add_dirichlet_bc(v_ana.sym, "Upper")
# stokes.add_dirichlet_bc(v_ana.sym, "Lower")

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
    uw.timing.print_table(group_by='line_routine', output_file=f"{output_dir}stokes_solve_time.txt", display_fraction=1.00)

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
        v_err.data[:,0] = v_uw.data[:,0] - v_ana.data[:,0]
        v_err.data[:,1] = v_uw.data[:,1] - v_ana.data[:,1]
        v_err.data[:,2] = v_uw.data[:,2] - v_ana.data[:,2]
        p_err.data[:,0] = p_uw.data[:,0] - p_ana.data[:,0]

# +
# plotting velocities from uw
if m==-1:
    clim, vmag, vfreq = [0., 5], 5e0, 75
elif m==3:
    clim, vmag, vfreq = [0., 18], 5e0, 75
    
if uw.mpi.size == 1 and visualize:
    # velocity plot
    plot_vector(mesh, v_uw, _vector_name='v_ana', _cmap=cmc.lapaz.resampled(21), _clim=clim, _vmag=vmag, _vfreq=vfreq,
                _save_png=True, _dir_fname=output_dir+'vel_uw.png', _show_clip=True, _show_arrows=False)
# -

# plotting relative errror in velocities
clim, vmag, vfreq = [0., 0.005], 1e2, 75        
if uw.mpi.size == 1 and analytical and visualize:
    # velocity error plot
    plot_vector(mesh, v_err, _vector_name='v_err(relative)', _cmap=cmc.lapaz.resampled(11), _clim=clim, _vmag=vmag, _vfreq=vfreq,
                _save_png=True, _dir_fname=output_dir+'vel_r_err.png', _show_clip=True, _show_arrows=False)

# plotting magnitude error in percentage
clim = [0, 5]
if uw.mpi.size == 1 and analytical and visualize: 
    # velocity error plot
    vmag_expr = (sympy.sqrt(v_err.sym.dot(v_err.sym))/sympy.sqrt(v_ana.sym.dot(v_ana.sym)))*100
    plot_scalar(mesh, vmag_expr, 'vmag_err(%)', _cmap=cmc.oslo_r.resampled(21), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'vel_p_err.png', _show_clip=True)

# +
# plotting pressure from uw
if m==-1:
    # clim = [-2.5, 2.5]
    clim = [-300., -250.]
elif m==3:
    clim = [-4, 4]
        
if uw.mpi.size == 1 and visualize:
    # pressure plot
    plot_scalar(mesh, p_uw.sym, 'p_uw', _cmap=cmc.vik.resampled(41), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'p_uw.png', _show_clip=True)
    # saving colobar separately 
    save_colorbar(_colormap=cmc.vik.resampled(41), _cb_bounds='', _vmin=clim[0], _vmax=clim[1], _figsize_cb=(5, 5), _primary_fs=18, 
                  _cb_orient='horizontal', _cb_axis_label='Pressure', _cb_label_xpos=0.5, _cb_label_ypos=-2.0, _fformat='pdf', 
                  _output_path=output_dir, _fname='p_ana')
# -

# plotting relative error in uw
clim = [-0.065, 0.065]       
if uw.mpi.size == 1 and analytical and visualize:
    # pressure error plot
    plot_scalar(mesh, p_err.sym, 'p_err(relative)', _cmap=cmc.vik.resampled(41), _clim=clim, _save_png=True, 
                _dir_fname=output_dir+'p_r_err.png', _show_clip=True)

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
    
        p_err_I = uw.maths.Integral(mesh, p_err.sym.dot(p_err.sym))
        p_ana_I = uw.maths.Integral(mesh, p_ana.sym.dot(p_ana.sym))
        p_err_l2 = np.sqrt(p_err_I.evaluate())/np.sqrt(p_ana_I.evaluate())

        if uw.mpi.rank == 0:
            print('Relative error in velocity in the L2 norm: ', v_err_l2)
            print('Relative error in pressure in the L2 norm: ', p_err_l2)

# +
# writing l2 norms to h5 file
if uw.mpi.size == 1 and os.path.isfile(output_dir+'error_norm.h5'):
    os.remove(output_dir+'error_norm.h5')
    print('Old file removed')

if uw.mpi.rank == 0:
    print('Creating new h5 file')
    with h5py.File(output_dir+'error_norm.h5', 'w') as f:
        f.create_dataset("m", data=m)
        f.create_dataset("cellsize", data=cellsize)
        f.create_dataset("res", data=res)
        f.create_dataset("v_l2_norm", data=v_err_l2)
        f.create_dataset("p_l2_norm", data=p_err_l2)
# -

# saving h5 and xdmf file
mesh.petsc_save_checkpoint(index=0, meshVars=[v_uw, p_uw, v_ana, p_ana, v_err, p_err, rho_ana], outputPath=os.path.relpath(output_dir)+'/output')



