# ## Python script to runs all tests

import numpy as np
import os

# +
# parameters
l_m_list = [[2,1], [2,2], [4,2], [4,4], [8,4], [8,8]]
res_list = [8, 16, 32, 64, 128]
case_list = ['case1', 'case3', 'case4']
stokes_elements = [[2,1]] # element pairs for solving Stokes: P1P0, P2P1 (stable), P3P2, P2P-1

# continuous pressure
for elements in stokes_elements:
    if elements[1]<=0:
        pcont = False
    else:
        pcont = True
pcont_str = str(pcont).lower()

# velocity penalty and stokes tolerance
v_pen_stk_tol_list = [[1e8, 1e-10]]
# -

# loop to submit jobs
for l_m in l_m_list:
    l_deg, m_ord = l_m[0], l_m[1]
    for res in res_list:
        for elements in stokes_elements:
            vdegree, pdegree = elements[0], np.absolute(elements[1])
            for v_pen_stk_tol in v_pen_stk_tol_list:
                vel_penalty, stokes_tol = v_pen_stk_tol[0], v_pen_stk_tol[1]
                for case_no in case_list:
                    if res==8:
                        os.system(f'qsub -v res={res},pcont={pcont},l_deg={l_deg},m_ord={m_ord},vdegree={vdegree},pdegree={pdegree},vel_penalty={vel_penalty},stokes_tol={stokes_tol},case_no={case_no} -l ncpus=16,mem=64GB,walltime=06:00:00 jobscript_m18.sh')
                    elif res==16:
                        os.system(f'qsub -v res={res},pcont={pcont},l_deg={l_deg},m_ord={m_ord},vdegree={vdegree},pdegree={pdegree},vel_penalty={vel_penalty},stokes_tol={stokes_tol},case_no={case_no} -l ncpus=32,mem=128GB,walltime=12:00:00 jobscript_m18.sh')
                    elif res==32:
                        os.system(f'qsub -v res={res},pcont={pcont},l_deg={l_deg},m_ord={m_ord},vdegree={vdegree},pdegree={pdegree},vel_penalty={vel_penalty},stokes_tol={stokes_tol},case_no={case_no} -l ncpus=64,mem=256GB,walltime=12:00:00 jobscript_m18.sh')
                    elif res==64:
                        os.system(f'qsub -v res={res},pcont={pcont},l_deg={l_deg},m_ord={m_ord},vdegree={vdegree},pdegree={pdegree},vel_penalty={vel_penalty},stokes_tol={stokes_tol},case_no={case_no} -l ncpus=128,mem=512GB,walltime=12:00:00 jobscript_m18.sh')
                    elif res==128:
                        os.system(f'qsub -v res={res},pcont={pcont},l_deg={l_deg},m_ord={m_ord},vdegree={vdegree},pdegree={pdegree},vel_penalty={vel_penalty},stokes_tol={stokes_tol},case_no={case_no} -l ncpus=256,mem=1024GB,walltime=12:00:00 jobscript_m18.sh')



