# ## Plot Error Convergence

import matplotlib.pyplot as plt
import numpy as np
import os
import h5py


def plot_err_convergence(_error_data='', _res_data='', _k_data='', _vdeg_data='', _pdeg_data='', _color_data='', _marker_data='',
                         _err_order='', _err_scaling='', _ax_xlabel='', _ax_ylabel='', _ax_xlim='', _ax_ylim='', _figsize=(6, 10),
                         _save_fig='', _dir_fname=''):
    """
    Plotting Error Convergence
    """
    
    f, ax = plt.subplots(1, 1, figsize=_figsize)

    # axis label
    ax.set_xlabel(_ax_xlabel)
    ax.set_ylabel(_ax_ylabel)

    # axis limits
    ax.set_xlim(_ax_xlim)
    ax.set_ylim(_ax_ylim)

    # setting axis scale to log
    ax.set_xscale('log')
    ax.set_yscale('log')

    # grid on
    ax.grid(linestyle='dotted')

    # ticks marks inward
    ax.tick_params(which='both', direction='in')

    # plot analytical curve for error convergence
    x = np.linspace(2e-3, 5e-1, 50)
    for i, order in enumerate(_err_order):
        
        y = _err_scaling[i]*(x**order)
        
        if order==1:
            _label = r'$\mathcal{O}(h)$'
        elif order==2:
            _label = r'$\mathcal{O}(h^2)$'
        elif order==3:
            _label = r'$\mathcal{O}(h^3)$'
        elif order==4:
            _label = r'$\mathcal{O}(h^4)$'
            
        # ax.plot(x, y, c=_color_data[i], label=_label)
        ax.plot(x, y, c='k', label=_label)

    # plot numerical curves
    for i, k_data in enumerate(_error_data):
        for j, line in enumerate(k_data):
            ax.plot(_res_data[i][j], line, c=_color_data[i], marker=_marker_data[j], markersize=6, mfc='none',
                    label='k='+str(_k_data[i][j])+', '+r'$v_{d}$='+str(_vdeg_data[i][j])+', '+r'$p_{d}$='+str(_pdeg_data[i][j]))
        
    ax.legend(loc=(1.01, 0.0))

    if _save_fig:
        plt.savefig(_dir_fname, format='pdf', bbox_inches='tight')


# output_dir
output_dir = './output/Annulus_Benchmark_Thieulot/benchmark_figs/'
os.makedirs(output_dir, exist_ok=True)

# +
# parameters
k_list = [1, 4, 8] # [1, 4, 8]
res_inv_list = [4, 8, 16, 32, 64] # [4, 8, 16, 32, 64, 128]

# Element pairs for solving Stokes: P1P0, P2P1, P3P2
# stable elements: P2P1 P2P-1
stokes_elements = [[2,-1]] # [[1,0], [2,1], [3,2]]

# continuous pressure
for elements in stokes_elements:
    if elements[1]<=0:
        pcont = 'False'
    else:
        pcont = 'True'

# velocity penalty and stokes tolerance
v_pen_stk_tol_list = [[1e10, 1e-10]] # [[2.5e6, 1e-7], [1e10, 1e-10], [1e10, 1e-10]]
# -

# empty lists to collect data
k_list_plt = []
vdeg_list_plt = []
pdeg_list_plt = []
e_v_list_plt = []
e_p_list_plt = []
cellsize_list_plt = []

if len(v_pen_stk_tol_list)>0:
    for v_pen_stk_tol in v_pen_stk_tol_list:
        vel_penalty = v_pen_stk_tol[0]
        stokes_tol = v_pen_stk_tol[1]
        vel_penalty_str = str("{:.1e}".format(vel_penalty))
        stokes_tol_str = str("{:.1e}".format(stokes_tol))
        
        for elements in stokes_elements:
            vdegree = elements[0]
            pdegree = np.absolute(elements[1])
            if elements[1]<=0:
                pcont = 'False'
            else:
                pcont = 'True'
    
            e_v_k_list = []
            e_p_k_list = []
            cellsize_k_list = []
            k_k_list = []
            vdeg_k_list = []
            pdeg_k_list = []
            for k in k_list:
                k_k_list +=[k]
                vdeg_k_list += [vdegree]
                pdeg_k_list += [pdegree]
        
                e_v_list = []
                e_p_list = []
                cellsize_list = []
                for res_inv in res_inv_list:
                    if res_inv<=16:
                        ncpus=1
                    elif res_inv==32:
                        ncpus=2
                    elif res_inv==64:
                        ncpus=4
                    elif res_inv>64:
                        ncpus=8
                        
                    h5_file = os.path.join(os.path.join("./output/Annulus_Benchmark_Thieulot/"), 
                                           f"model_k_{k}_res_{res_inv}_vdeg_{vdegree}_pdeg_{pdegree}_pcont_{pcont.lower()}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/error_norm.h5")
                    
                    with h5py.File(h5_file, 'r') as f:
                        e_v_list += [np.array(f["v_l2_norm"])]
                        e_p_list += [np.array(f["p_l2_norm"])]
                        cellsize_list += [1/(res_inv)]
                
                e_v_k_list += [e_v_list]
                e_p_k_list += [e_p_list]
                cellsize_k_list += [cellsize_list]
    
            k_list_plt +=[k_k_list]
            vdeg_list_plt += [vdeg_k_list]
            pdeg_list_plt += [pdeg_k_list]
            e_v_list_plt += [e_v_k_list]
            e_p_list_plt += [e_p_k_list]
            cellsize_list_plt += [cellsize_k_list]

# +
# for i, elements in enumerate(stokes_elements):
#     vdegree = elements[0]
#     pdegree = np.absolute(elements[1])
#     if elements[1]<=0:
#         pcont = 'False'
#         vel_penalty = v_pen_stk_tol_list[i][0]
#         stokes_tol = v_pen_stk_tol_list[i][1]
#         vel_penalty_str = str("{:.1e}".format(vel_penalty))
#         stokes_tol_str = str("{:.1e}".format(stokes_tol))
#     else:
#         pcont = 'True'
#         vel_penalty = v_pen_stk_tol_list[i][0]
#         stokes_tol = v_pen_stk_tol_list[i][1]
#         vel_penalty_str = str("{:.1e}".format(vel_penalty))
#         stokes_tol_str = str("{:.1e}".format(stokes_tol))

#     e_v_k_list = []
#     e_p_k_list = []
#     cellsize_k_list = []
#     k_k_list = []
#     vdeg_k_list = []
#     pdeg_k_list = []
#     for k in k_list:
#         k_k_list +=[k]
#         vdeg_k_list += [vdegree]
#         pdeg_k_list += [pdegree]

#         e_v_list = []
#         e_p_list = []
#         cellsize_list = []
#         for res_inv in res_inv_list:
#             if res_inv<=16:
#                 ncpus=1
#             elif res_inv==32:
#                 ncpus=2
#             elif res_inv==64:
#                 ncpus=4
#             elif res_inv>64:
#                 ncpus=8
                
#             h5_file = os.path.join(os.path.join("./output/Annulus_Benchmark_Thieulot/"), 
#                                    f"model_k_{k}_res_{res_inv}_vdeg_{vdegree}_pdeg_{pdegree}_pcont_{pcont.lower()}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/error_norm.h5")
            
#             with h5py.File(h5_file, 'r') as f:
#                 e_v_list += [np.array(f["v_l2_norm"])]
#                 e_p_list += [np.array(f["p_l2_norm"])]
#                 cellsize_list += [1/(res_inv)]
        
#         e_v_k_list += [e_v_list]
#         e_p_k_list += [e_p_list]
#         cellsize_k_list += [cellsize_list]

#     k_list_plt +=[k_k_list]
#     vdeg_list_plt += [vdeg_k_list]
#     pdeg_list_plt += [pdeg_k_list]
#     e_v_list_plt += [e_v_k_list]
#     e_p_list_plt += [e_p_k_list]
#     cellsize_list_plt += [cellsize_k_list]

# +
# err_order = [2, 3, 4]
# err_scaling = [1.4, 1e-1, 10e-2]
err_order = [3]
err_scaling = [1e-1]

ax_xlabel = 'Cell size ($h$)'
ax_ylabel = r'$ |e_{v}|_{2} $'

# ax_xlim = [1e-3, 1e0]
# ax_ylim = [1e-9, 1e1]
ax_xlim = [1e-3, 1e0]
ax_ylim = [1e-9, 1e-1]

color_data = ['mediumorchid', 'forestgreen', 'deepskyblue']
marker_data = ['s', 'o', '^']

# +
# pdeg_list_plt[[-1, -1, -1]]

# +
# # continuous pressure
# for elements in stokes_elements:
#     if elements[1]<0:
# pdeg_list_plt
# -

plot_err_convergence(_error_data=e_v_list_plt, _res_data=cellsize_list_plt, _k_data=k_list_plt, _vdeg_data=vdeg_list_plt, 
                     _pdeg_data=pdeg_list_plt, _color_data=color_data, _marker_data=marker_data,
                     _err_order=err_order, _err_scaling=err_scaling, _ax_xlabel=ax_xlabel, _ax_ylabel=ax_ylabel, 
                     _ax_xlim=ax_xlim, _ax_ylim=ax_ylim, _figsize=(6, 5), _save_fig=True, 
                     _dir_fname=output_dir+f'vel_err_conv_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}.pdf')

# +
# err_order = [1, 2, 3]
# err_scaling = [8.1, 3.5, 1]
err_order = [2]
err_scaling = [3.5]

ax_xlabel = 'Cell size ($h$)'
ax_ylabel = r'$ |e_{p}|_{2} $'

# ax_xlim = [1e-3, 1e0]
# ax_ylim = [1e-6, 1e7]
ax_xlim = [1e-3, 1e0]
ax_ylim = [1e-6, 1e2]
# -

plot_err_convergence(_error_data=e_p_list_plt, _res_data=cellsize_list_plt, _k_data=k_list_plt, _vdeg_data=vdeg_list_plt, 
                     _pdeg_data=pdeg_list_plt, _color_data=color_data, _marker_data=marker_data,
                     _err_order=err_order, _err_scaling=err_scaling, _ax_xlabel=ax_xlabel, _ax_ylabel=ax_ylabel, 
                     _ax_xlim=ax_xlim, _ax_ylim=ax_ylim, _figsize=(6, 5), _save_fig=True, 
                     _dir_fname=output_dir+f'p_err_conv_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}.pdf')


