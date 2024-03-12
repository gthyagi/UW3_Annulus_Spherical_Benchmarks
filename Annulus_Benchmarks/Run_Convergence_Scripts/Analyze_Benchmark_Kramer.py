# ## Plot Error Convergence

import matplotlib.pyplot as plt
import numpy as np
import os
import h5py


def plot_err_convergence(_error_data='', _res_data='', _k_data='', _vdeg_data='', _pdeg_data='', _color_data='', _marker_data='',
                         _err_order='', _err_scaling='', _ax_xlabel='', _ax_ylabel='', _ax_xlim='', _ax_ylim='', _figsize=(6, 10),
                         _save_fig='', _dir_fname='', _plt_title=''):
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
    x = np.linspace(1/128, 1/8, 20, endpoint=True)    
    y = _err_scaling[0]*(x**_err_order[0])
    if _err_order[0]==0.5:
        _label = r'$\mathcal{O}$' + r"$(h^{{0.5}})$" 
    elif _err_order[0]==1.5:
        _label = r'$\mathcal{O}$' + r"$(h^{{1.5}})$"
    elif _err_order[0]==2.0:
        _label = r'$\mathcal{O}$' + r"$(h^{{2.0}})$"
    elif _err_order[0]==3.0:
        _label = r'$\mathcal{O}$' + r"$(h^{{3.0}})$"

    ax.plot(x, y, c='k', label=_label)

    # plot numerical curves
    for i, k_data in enumerate(_error_data):
        for j, line in enumerate(k_data):
            ax.plot(_res_data[i][j], line, c=_color_data[i], marker=_marker_data[j], markersize=6, mfc='none',
                    label='n='+str(_k_data[i][j]))
        
    ax.legend(loc=(0.77, 0.02))

    ax.set_title(_plt_title)

    if _save_fig:
        plt.savefig(_dir_fname, format='pdf', bbox_inches='tight')


# +
# output_dir
output_dir = './output/Annulus_Benchmark_Kramer/benchmark_figs/'
os.makedirs(output_dir, exist_ok=True)

# input_dir
input_dir = './output/Annulus_Benchmark_Kramer/' # "./output/Annulus_Benchmark_Kramer/"
# -

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
# parameters
k_list = [8] # [0, 2, 8]
case_list = ['case4'] # ['case1', 'case2', 'case3', 'case4']

n_list = [2, 8, 32]
res_inv_list = [8, 16, 32, 64, 128]

# Element pairs for solving Stokes: P1P0, P2P1, P3P2, P2P-1
# stable elements: P2P1 
stokes_elements = [[2,1]]

# continuous pressure
for elements in stokes_elements:
    if elements[1]<=0:
        pcont = 'False'
    else:
        pcont = 'True'

# velocity penalty and stokes tolerance
v_pen_stk_tol_list = [[2.5e8, 1e-10]] # [[2.5e6, 1e-10], [1e8, 1e-10], [1e10, 1e-10]]
# -

# empty lists to collect data
n_list_plt = []
vdeg_list_plt = []
pdeg_list_plt = []
e_v_list_plt = []
e_p_list_plt = []
cellsize_list_plt = []

# define case and k 
case = case_list[0]
k = k_list[0]

# making figure title
if case in ('case1', 'case3'):
    plt_title = "Delta-Function"
elif case in ('case2', 'case4'):
    plt_title = "Smooth (k="+str(k)+")"

# plotting
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

        e_v_n_list = []
        e_p_n_list = []
        cellsize_n_list = []
        n_n_list = []
        vdeg_n_list = []
        pdeg_n_list = []
        for n in n_list:
            n_n_list +=[n]
            vdeg_n_list += [vdegree]
            pdeg_n_list += [pdegree]
    
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
                    
                h5_file = os.path.join(os.path.join(input_dir), 
                                       f"{case}_n_{n}_k_{k}_res_{res_inv}_vdeg_{vdegree}_pdeg_{pdegree}_pcont_{pcont.lower()}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/error_norm.h5")
                
                with h5py.File(h5_file, 'r') as f:
                    e_v_list += [np.array(f["v_l2_norm"])]
                    e_p_list += [np.array(f["p_l2_norm"])]
                    cellsize_list += [1/(res_inv)]
            
            e_v_n_list += [e_v_list]
            e_p_n_list += [e_p_list]
            cellsize_n_list += [cellsize_list]

        n_list_plt +=[n_n_list]
        vdeg_list_plt += [vdeg_n_list]
        pdeg_list_plt += [pdeg_n_list]
        e_v_list_plt += [e_v_n_list]
        e_p_list_plt += [e_p_n_list]
        cellsize_list_plt += [cellsize_n_list]

# +
if case=="case1":
    err_order, err_scaling = [1.5], [0.4]
    ax_xlim, ax_ylim = [5e-3, 2e-1], [1e-4, 1e-1]
elif case=="case2" and k==2:
    err_order, err_scaling = [3], [1.5e-1]
    ax_xlim, ax_ylim = [5e-3, 2e-1], [5e-8, 5e-2]
elif case=="case2" and k==8:
    err_order, err_scaling = [3], [3e-1]
    ax_xlim, ax_ylim = [5e-3, 2e-1], [5e-8, 5e-2]
elif case=="case3":
    err_order, err_scaling = [1.5], [0.8]
    ax_xlim, ax_ylim = [5e-3, 2e-1], [1e-4, 2e-1]
elif case=="case4" and k==2:
    err_order, err_scaling = [3], [6e-1]
    ax_xlim, ax_ylim = [5e-3, 2e-1], [5e-8, 5e-2]
elif case=="case4" and k==8:
    err_order, err_scaling = [3], [0.85]
    ax_xlim, ax_ylim = [5e-3, 2e-1], [5e-8, 6e-2]

ax_xlabel = 'Cell size ($h$)'
ax_ylabel = r'$ |e_{v}|_{2} $'

color_data = ['mediumorchid', 'forestgreen', 'deepskyblue']
marker_data = ['o', '*', 's']
# -

plot_err_convergence(_error_data=e_v_list_plt, _res_data=cellsize_list_plt, _k_data=n_list_plt, _vdeg_data=vdeg_list_plt, 
                     _pdeg_data=pdeg_list_plt, _color_data=color_data, _marker_data=marker_data,
                     _err_order=err_order, _err_scaling=err_scaling, _ax_xlabel=ax_xlabel, _ax_ylabel=ax_ylabel, 
                     _ax_xlim=ax_xlim, _ax_ylim=ax_ylim, _figsize=(6, 5), _save_fig=True, _plt_title=plt_title,
                     _dir_fname=output_dir+f'{case}_k_{k}_vel_err_conv_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}.pdf')

# +
if case=="case1":
    err_order, err_scaling = [0.5], [1.05]
    ax_xlim, ax_ylim = [5e-3, 2e-1], [1e-2, 1e0]
elif case=="case2" and k==2:
    err_order, err_scaling = [2], [1.1]
    ax_xlim, ax_ylim = [5e-3, 2e-1], [1e-5, 2e-1]
elif case=="case2" and k==8:
    err_order, err_scaling = [2], [2.9]
    ax_xlim, ax_ylim = [5e-3, 2e-1], [1e-5, 2e-1]
elif case=="case3":
    err_order, err_scaling = [0.5], [0.8]
    ax_xlim, ax_ylim = [5e-3, 2e-1], [1e-2, 1e0]
elif case=="case4" and k==2:
    err_order, err_scaling = [2], [0.65]
    ax_xlim, ax_ylim = [5e-3, 2e-1], [1e-5, 2e-1]
elif case=="case4" and k==8:
    err_order, err_scaling = [2], [2]
    ax_xlim, ax_ylim = [5e-3, 2e-1], [1e-5, 2e-1]

ax_xlabel = 'Cell size ($h$)'
ax_ylabel = r'$ |e_{p}|_{2} $'
# -

plot_err_convergence(_error_data=e_p_list_plt, _res_data=cellsize_list_plt, _k_data=n_list_plt, _vdeg_data=vdeg_list_plt, 
                     _pdeg_data=pdeg_list_plt, _color_data=color_data, _marker_data=marker_data,
                     _err_order=err_order, _err_scaling=err_scaling, _ax_xlabel=ax_xlabel, _ax_ylabel=ax_ylabel, 
                     _ax_xlim=ax_xlim, _ax_ylim=ax_ylim, _figsize=(6, 5), _save_fig=True, _plt_title='',
                     _dir_fname=output_dir+f'{case}_k_{k}_p_err_conv_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}.pdf')

# creating figure labels
font_size = 21
for i, label in enumerate(['Free-Slip', 'Zero-Slip']):
    fig = plt.figure()
    plt.text(0.001, 0.001, label, fontsize=font_size, ) #bbox=dict(facecolor='none', edgecolor='k', boxstyle='round, pad=0.2', alpha=0.2, fc="white"))
    plt.axis('off')
    plt.subplots_adjust(left=0.15, right=0.16, bottom=0.1, top=0.11, wspace=None, hspace=None) # make sure all fig are same size
    plt.savefig(output_dir+'label_'+label+'.pdf', bbox_inches='tight', format='pdf')




