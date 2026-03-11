#!/usr/bin/env python3

# %% [markdown]
# # Kramer Latest: CPUs vs L2 Norm
# Simple workflow:
# 1) create output folder
# 2) get list of run directories
# 3) read `error_norm.h5`
# 4) plot `ncpus` vs L2 norms

# %%
import os
import re
import sys

import h5py
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1) User run directory list

# %%
run_dir_names = [
    "case1_inv_lc_32_n_2_k_3_vdeg_2_pdeg_1_pcont_true_vel_penalty_2.5e+08_stokes_tol_1e-10_ncpus_1",
    "case1_inv_lc_32_n_2_k_3_vdeg_2_pdeg_1_pcont_true_vel_penalty_2.5e+08_stokes_tol_1e-10_ncpus_2",
    "case1_inv_lc_32_n_2_k_3_vdeg_2_pdeg_1_pcont_true_vel_penalty_2.5e+08_stokes_tol_1e-10_ncpus_4",
    "case1_inv_lc_32_n_2_k_3_vdeg_2_pdeg_1_pcont_true_vel_penalty_2.5e+08_stokes_tol_1e-10_ncpus_8",
]

print(f"Using {len(run_dir_names)} run directories")
for name in run_dir_names:
    print(name)

# %% [markdown]
# ## 2) Resolve paths

# %%
IN_NOTEBOOK = "ipykernel" in sys.modules
OUTPUT_ROOT = os.path.abspath(f"{os.getcwd()}/../../output/annulus/kramer/latest")
FIG_DIR = f"{OUTPUT_ROOT}/l2norm_plots"
print(f"Output root: {OUTPUT_ROOT}")
print(f"Figure dir : {FIG_DIR}")

# %% [markdown]
# ## 3) Validate run dirs and create output folder

# %%
run_dirs = []
for run_name in run_dir_names:
    run_dir = os.path.join(OUTPUT_ROOT, run_name)
    if os.path.isdir(run_dir):
        run_dirs.append(run_dir)
    else:
        print(f"Skipping missing run directory: {run_dir}")

os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# ## 4) Read `error_norm.h5` from each directory

# %%
ncpus_re = re.compile(r"_ncpus_(\d+)$")
records = []

for run_dir in run_dirs:
    run_name = os.path.basename(run_dir)
    match = ncpus_re.search(run_name)
    if not match:
        continue

    ncpus = int(match.group(1))
    h5_file = os.path.join(run_dir, "error_norm.h5")
    if not os.path.isfile(h5_file):
        continue

    with h5py.File(h5_file, "r") as h5f:
        v_l2 = float(h5f["v_l2_norm"][()])
        p_l2 = float(h5f["p_l2_norm"][()])

    records.append({"ncpus": ncpus, "v_l2": v_l2, "p_l2": p_l2, "h5": h5_file})

records = sorted(records, key=lambda r: r["ncpus"])
for r in records:
    print(
        f"ncpus={r['ncpus']:>2}  v_l2={r['v_l2']:.6e}  "
        f"p_l2={r['p_l2']:.6e}"
    )

# %% [markdown]
# ## 5) Plot CPUs vs L2 norm and save figure

# %%
if not records:
    print("No valid error_norm.h5 files found.")
else:
    x = [r["ncpus"] for r in records]
    y_v = [r["v_l2"] for r in records]
    y_p = [r["p_l2"] for r in records]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_yscale("log")
    ax.plot(x, y_v, marker="o", linewidth=2, label="Velocity L2 norm")
    ax.plot(x, y_p, marker="s", linewidth=2, label="Pressure L2 norm")
    ax.set_xticks(sorted(set(x)))
    ax.set_xlabel("CPUs (MPI ranks)")
    ax.set_ylabel("Relative L2 norm")
    ax.set_title("Kramer Latest: CPUs vs L2 Norm")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    fig_path = os.path.join(FIG_DIR, "kramer_latest_l2_vs_cpus.png")
    fig.savefig(fig_path, dpi=200)
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)

    print(f"Saved figure: {fig_path}")
