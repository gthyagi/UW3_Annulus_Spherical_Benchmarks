# ## Python script to runs all tests

import numpy as np
import os

# output dir
output_dir = './output/Annulus_Benchmark_Kramer/'
fig_dir = output_dir+'benchmark_figs/'
output_dir_2 = './output/'

# +
# parameters
n_list = [2, 8, 32]
k_list = [0, 2, 8]
res_inv_list = [8, 16, 32, 64, 128]
case_list = ['case1', 'case2', 'case3', 'case4']

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

# +
# 0 - 179
# -

for k in k_list:
    for n in n_list:
        for res_inv in res_inv_list:
            
            if res_inv<=16:
                ncpus=1
            elif res_inv==32:
                ncpus=2
            elif res_inv==64:
                ncpus=4
            elif res_inv>64:
                ncpus=8
                
            for elements in stokes_elements:
                vdegree = elements[0]
                pdegree = np.absolute(elements[1])
                for v_pen_stk_tol in v_pen_stk_tol_list:
                    vel_penalty = v_pen_stk_tol[0]
                    stokes_tol = v_pen_stk_tol[1]
                    vel_penalty_str = str("{:.1e}".format(vel_penalty))
                    stokes_tol_str = str("{:.1e}".format(stokes_tol))

                    for case in case_list:
                    
                        print('-------------------------------------------------------------------------------')
                        print(f"{case}_n_{n}_k_{k}_res_{res_inv}_vdeg_{vdegree}_pdeg_{pdegree}_pcont_{pcont.lower()}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/")
                        print('-------------------------------------------------------------------------------')
        
                        os.system(f'ts mpiexec -n {ncpus} python3 Ex_Stokes_Annulus_Benchmark_Kramer_etal_sh.py {n} {k} {res_inv} {vdegree} {pdegree} {pcont} {vel_penalty} {stokes_tol} {case}')

# +
# start = 144
# end = 180
# while start<end:
#     for k in k_list:
#         for n in n_list:
#             for res_inv in res_inv_list:
                
#                 if res_inv<=16:
#                     ncpus=1
#                 elif res_inv==32:
#                     ncpus=2
#                 elif res_inv==64:
#                     ncpus=4
#                 elif res_inv>64:
#                     ncpus=8
                    
#                 for elements in stokes_elements:
#                     vdegree = elements[0]
#                     pdegree = np.absolute(elements[1])
#                     for v_pen_stk_tol in v_pen_stk_tol_list:
#                         vel_penalty = v_pen_stk_tol[0]
#                         stokes_tol = v_pen_stk_tol[1]
#                         vel_penalty_str = str("{:.1e}".format(vel_penalty))
#                         stokes_tol_str = str("{:.1e}".format(stokes_tol))
    
#                         for case in case_list:
                        
#                             print('-------------------------------------------------------------------------------')
#                             print(f"{case}_n_{n}_k_{k}_res_{res_inv}_vdeg_{vdegree}_pdeg_{pdegree}_pcont_{pcont.lower()}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/")
#                             print('-------------------------------------------------------------------------------')
            
#                             os.system(f'ts -c {start} 2>&1 | tee {output_dir_2}{case}_n_{n}_k_{k}_res_{res_inv}_vdeg_{vdegree}_pdeg_{pdegree}_pcont_{pcont.lower()}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}.txt')

#                             start = start+1
# -


