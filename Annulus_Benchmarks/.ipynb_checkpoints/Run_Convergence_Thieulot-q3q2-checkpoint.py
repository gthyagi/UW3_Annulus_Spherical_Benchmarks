import numpy as np
import os
import glob

# output dir
output_dir = './output/Annulus_Thieulot/'
fig_dir = output_dir+'benchmark_figs/'

# +
# parameters
k_list = [1, 4, 8] # [0, 1, 2, 4, 8]
p_res_list = [2, 3, 4, 5, 6, 7]

# Element pairs for solving Stokes
Q1q0 = np.array([1,0])
Q2q1 = np.array([2,1])
Q3q2 = np.array([3,2])
stokes_elements = [Q3q2]

pcont = 'True'


# -

def run_model(_k_list, _p_res_list, vdegree, pdegree, pcont):
    for k in _k_list:
        for p_res in _p_res_list:
            
            print('----------------------------------------------------------------------')
            print(f"modelT_k_{k}_p_res_{p_res}_vdeg_{vdegree}_pdeg_{pdegree}_pcont_{pcont}")
            print('----------------------------------------------------------------------')
            
            os.system(f'python3 Ex_Stokes_Annulus_Benchmark_Thieulot_sh.py {k} {p_res} {vdegree} {pdegree} {pcont}')


for elements in stokes_elements:
    
    vdegree = elements[0]
    pdegree = elements[1]
    
    run_model(k_list, p_res_list, elements[0], elements[1], pcont)


