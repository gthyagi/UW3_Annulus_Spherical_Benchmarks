# ## Python script to runs all tests

import numpy as np
import os

# output dir
output_dir = './output/Annulus_Benchmark_Thieulot/'
fig_dir = output_dir+'benchmark_figs/'
output_dir_2 = './output/'

# +
# parameters
k_list = [0, 1, 2, 4]
res_inv_list = [32] # [4, 8, 16, 32, 64, 128]

# Element pairs for solving Stokes: P1P0, P2P1, P3P2
# stable elements: P2P1 P2P-1
stokes_elements = [[2,1]] # [[1,0], [2,1], [3,2]]

# continuous pressure
for elements in stokes_elements:
    if elements[1]<=0:
        pcont = 'False'
    else:
        pcont = 'True'

# velocity penalty and stokes tolerance
v_pen_stk_tol_list = [[1e8, 1e-10]] # [[2.5e6, 1e-10], [1e8, 1e-10], [1e10, 1e-10]]

# +
# res_inv_list = [4, 8, 16, 32, 64, 128] stokes_elements = [[2,1]] v_pen_stk_tol_list = [[2.5e6, 1e-10], [1e8, 1e-10]] ts:72-107
# res_inv_list = [4, 8, 16, 32, 64] stokes_elements = [[3,2]] v_pen_stk_tol_list = [[1e10, 1e-10]] ts:108-122
# res_inv_list = [4, 8, 16, 32, 64, 128] stokes_elements = [[1,0]] v_pen_stk_tol_list = [[2.5e6, 1e-7]] ts:123-140 (128, 134 140 killed)
# res_inv_list = [4, 8, 16, 32, 64, 128] stokes_elements = [[2,-1]] v_pen_stk_tol_list = [[1e10, 1e-10]] ts:141-158 (146, 152, 158)
# -

for k in k_list:
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
                
                print('-------------------------------------------------------------------------------')
                print(f"model_k_{k}_res_{res_inv}_vdeg_{vdegree}_pdeg_{pdegree}_pcont_{pcont.lower()}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/")
                print('-------------------------------------------------------------------------------')

                os.system(f'ts mpiexec -n {ncpus} python3 Ex_Stokes_Annulus_Benchmark_Thieulot_sh.py {k} {res_inv} {vdegree} {pdegree} {pcont} {vel_penalty} {stokes_tol}')

# +
# start = 141
# end = 159
# while start<end:
#     for k in k_list:
#         for res_inv in res_inv_list:
            
#             if res_inv<=16:
#                 ncpus=1
#             elif res_inv==32:
#                 ncpus=2
#             elif res_inv==64:
#                 ncpus=4
#             elif res_inv>64:
#                 ncpus=8
                
#             for elements in stokes_elements:
#                 vdegree = elements[0]
#                 pdegree = np.absolute(elements[1])
#                 for v_pen_stk_tol in v_pen_stk_tol_list:
#                     vel_penalty = v_pen_stk_tol[0]
#                     stokes_tol = v_pen_stk_tol[1]
#                     vel_penalty_str = str("{:.1e}".format(vel_penalty))
#                     stokes_tol_str = str("{:.1e}".format(stokes_tol))
                    
#                     print('-------------------------------------------------------------------------------')
#                     print(f"model_k_{k}_res_{res_inv}_vdeg_{vdegree}_pdeg_{pdegree}_pcont_{pcont.lower()}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/")
#                     print('-------------------------------------------------------------------------------')
    
#                     os.system(f'ts -c {start} 2>&1 | tee {output_dir_2}model_k_{k}_res_{res_inv}_vdeg_{vdegree}_pdeg_{pdegree}_pcont_{pcont.lower()}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}.txt')

#                     start = start+1
# -


