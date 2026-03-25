# %% [markdown]
# ## Spherical Thieulot Latest: L2 Norm Plot
#
# Simple workflow:
# 1) give a list of run directory names
# 2) read `error_norm.h5`
# 3) infer the varying parameter from the names
# 4) plot that parameter vs velocity / pressure L2 norms

# %%
import os
import re
import sys

import h5py
import matplotlib.pyplot as plt

# %% [markdown]
# ### User Run Directory List

# %%
run_dir_names = [
    "case_inv_lc_4_m_-1_vdeg_2_pdeg_1_pcont_true_vel_penalty_1e+08_stokes_tol_1e-10_stokes_pen_1_ncpus_8",
    "case_inv_lc_8_m_-1_vdeg_2_pdeg_1_pcont_true_vel_penalty_1e+08_stokes_tol_1e-10_stokes_pen_1_ncpus_8",
    "case_inv_lc_16_m_-1_vdeg_2_pdeg_1_pcont_true_vel_penalty_1e+08_stokes_tol_1e-10_stokes_pen_1_ncpus_8",
    "case_inv_lc_32_m_-1_vdeg_2_pdeg_1_pcont_true_vel_penalty_1e+08_stokes_tol_1e-10_stokes_pen_1_ncpus_8",
]

# %% [markdown]
# ### Output Root

# %%
IN_NOTEBOOK = "ipykernel" in sys.modules
OUTPUT_ROOT = "/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/output/spherical/thieulot/latest"
FIG_DIR = os.path.join(OUTPUT_ROOT, "l2norm_plots")

print(f"Output root: {OUTPUT_ROOT}")
print(f"Figure dir : {FIG_DIR}")
print(f"Using {len(run_dir_names)} run directories")
for name in run_dir_names:
    print(name)

os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# ### Parse Run Names

# %%
run_pattern = re.compile(
    r"case_inv_lc_(?P<inv_lc>\d+)_"
    r"m_(?P<m>-?\d+)_"
    r"vdeg_(?P<vdeg>\d+)_"
    r"pdeg_(?P<pdeg>\d+)_"
    r"pcont_(?P<pcont>true|false)_"
    r"vel_penalty_(?P<vel_penalty>[0-9.eE+\-]+)_"
    r"stokes_tol_(?P<stokes_tol>[0-9.eE+\-]+)_"
    r"(?:stokes_pen_(?P<stokes_pen>[0-9.eE+\-]+)_)?"
    r"ncpus_(?P<ncpus>\d+)$"
)

parsed_runs = []
for run_name in run_dir_names:
    match = run_pattern.fullmatch(run_name)
    if match is None:
        raise ValueError(f"Could not parse run directory name: {run_name}")

    params = match.groupdict()
    params["inv_lc"] = int(params["inv_lc"])
    params["m"] = int(params["m"])
    params["vdeg"] = int(params["vdeg"])
    params["pdeg"] = int(params["pdeg"])
    params["ncpus"] = int(params["ncpus"])
    params["pcont"] = params["pcont"] == "true"
    params["vel_penalty"] = float(params["vel_penalty"])
    params["stokes_tol"] = float(params["stokes_tol"])
    params["stokes_pen"] = None if params["stokes_pen"] is None else float(params["stokes_pen"])

    parsed_runs.append((run_name, params))

# %%
candidate_keys = [
    "inv_lc",
    "m",
    "vdeg",
    "pdeg",
    "pcont",
    "vel_penalty",
    "stokes_tol",
    "stokes_pen",
    "ncpus",
]

varying_keys = []
for key in candidate_keys:
    values = {params[key] for _, params in parsed_runs}
    if len(values) > 1:
        varying_keys.append(key)

if len(varying_keys) != 1:
    raise ValueError(f"Expected exactly one varying parameter, found: {varying_keys}")

varying_key = varying_keys[0]
print(f"Varying parameter: {varying_key}")

# %% [markdown]
# ### Read `error_norm.h5`

# %%
records = []

for run_name, params in parsed_runs:
    run_dir = os.path.join(OUTPUT_ROOT, run_name)
    h5_file = os.path.join(run_dir, "error_norm.h5")

    if not os.path.isfile(h5_file):
        raise FileNotFoundError(f"Missing error_norm.h5: {h5_file}")

    with h5py.File(h5_file, "r") as h5f:
        v_key = "v_l2_norm" if "v_l2_norm" in h5f else "v_l2"
        p_key = "p_l2_norm" if "p_l2_norm" in h5f else "p_l2"

        if v_key not in h5f or p_key not in h5f:
            raise KeyError(f"Missing L2 datasets in: {h5_file}")

        v_l2 = float(h5f[v_key][()])
        p_l2 = float(h5f[p_key][()])

    records.append(
        {
            "run_name": run_name,
            "x": params[varying_key],
            "v_l2": v_l2,
            "p_l2": p_l2,
        }
    )

records.sort(key=lambda rec: rec["x"])

for rec in records:
    print(
        f"{varying_key}={rec['x']}  "
        f"v_l2={rec['v_l2']:.6e}  "
        f"p_l2={rec['p_l2']:.6e}"
    )

# %% [markdown]
# ### Plot

# %%
if not records:
    print("No valid error_norm.h5 files found.")
else:
    x = [rec["x"] for rec in records]
    y_v = [rec["v_l2"] for rec in records]
    y_p = [rec["p_l2"] for rec in records]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(x, y_v, marker="o", linewidth=2)
    axes[1].plot(x, y_p, marker="s", linewidth=2)

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("Velocity L2 norm")
    axes[1].set_ylabel("Pressure L2 norm")
    axes[1].set_xlabel(varying_key)
    axes[0].set_title(f"Spherical Thieulot Latest: {varying_key} vs L2 Norm")

    axes[0].grid(True, which="both", linestyle="--", alpha=0.35)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.35)
    axes[1].set_xticks(x)

    fig.tight_layout()

    fig_path = os.path.join(FIG_DIR, f"thieulot_latest_l2_vs_{varying_key}.png")
    fig.savefig(fig_path, dpi=200)
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)

    print(f"Saved figure: {fig_path}")
