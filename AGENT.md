# AGENT.md

## Purpose

This repository contains Underworld3 benchmark scripts for annulus (2D) and spherical shell (3D) Stokes-flow validation. It is currently in a fresh-start/rebuild state.

## Underworld3 Branch Context

- Legacy branch used for historical compatibility:
  - [jcg-meshvar2meshvar-fix](https://github.com/gthyagi/underworld3/tree/jcg-meshvar2meshvar-fix)
- Latest branch targeted for ongoing runs/migration:
  - [feature/boundary-integrals](https://github.com/gthyagi/underworld3/tree/feature/boundary-integrals)

For parity checks, compare latest-branch benchmark outputs against legacy-branch baseline outputs.

## Local Underworld3 Paths (Debugging Reference)

- Latest UW3 checkout:
  - `/Users/tgol0006/uw_folder/uw3_git_gthyagi_openmpi/worktrees/underworld3-feature-boundary-integrals/`
- Legacy UW3 checkout:
  - `/Users/tgol0006/uw_folder/uw3_git_gthyagi/underworld3`

These are local machine paths and are intended as quick references during debugging.

## Required Startup Reads

Before doing any repository-specific work, read these files in this order and follow them together with this `AGENT.md`:

1. `/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/rules/karpathy_guidelines.md`
2. `/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/rules/personal_agents.md`

Do not start analysis, editing, command execution, or repo-specific recommendations until both files have been read.

## Primary Directories

- `benchmarks/annulus/`: annulus benchmark scripts (current main script: legacy Thieulot benchmark).
- `benchmarks/spherical/`: spherical legacy/latest benchmark scripts and plotting helpers.
- `docs/`: documentation notes and migration guidance.
- `output/`: shared generated output location (if used).

## Benchmark Paper References

Use these PDFs as the primary benchmark references when checking formulas, expected fields, convergence behaviour, and boundary-condition intent.

- Curved-shell benchmark suite:
  - `docs/benchmark_papers/kramer_etal_gmd2021.pdf`
  - Absolute path: `/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/docs/benchmark_papers/kramer_etal_gmd2021.pdf`
  - Scope:
    - analytical Stokes solutions in 2-D cylindrical shells and 3-D spherical shells
    - smooth polynomial forcing and delta-function forcing
    - free-slip and zero-slip cases
  - Practical notes:
    - this is the main reference for Kramer annulus/spherical benchmarks in this repo
    - pressure is only defined up to a constant and should be zero-mean calibrated for comparison
    - free-slip velocity solutions can contain rigid-body rotation null modes and should be projected out before comparing with the analytical solution
    - delta-function forcing can produce suboptimal convergence, especially for continuous-pressure discretisations

- Spherical Thieulot benchmark:
  - `docs/benchmark_papers/cedric_thieulot_se2017.pdf`
  - Absolute path: `/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/docs/benchmark_papers/cedric_thieulot_se2017.pdf`
  - Scope:
    - analytical incompressible Stokes flow in a spherical shell
    - tangential velocity on inner/outer boundaries
    - radial power-law viscosity `mu(r) = mu0 * r^(m+1)`
  - Practical notes:
    - `m = -1` is the constant-viscosity case
    - `m = 3` is the standard variable-viscosity case used in this repo
    - the analytical pressure is zero on both spherical boundaries, so pressure boundary treatment matters
    - use this paper when checking spherical `m=-1` and `m=3` velocity, pressure, density/body-force, and `vrms` expressions

- Annulus Thieulot benchmark:
  - `docs/benchmark_papers/thieulot_puckett_2018.pdf`
  - Absolute path: `/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/docs/benchmark_papers/thieulot_puckett_2018.pdf`
  - Scope:
    - analytical incompressible Stokes flow in a 2-D annulus
    - constant viscosity, inward radial gravity, tangential boundary flow
    - parameter `k` controls the number of convection-cell lobes
  - Practical notes:
    - this is the main reference for annulus Thieulot benchmarks in this repo
    - use it to check pressure, velocity, density, and scalar averages such as `vrms`
    - expected finite-element behaviour is standard: velocity converges one order higher than pressure for stable mixed pairs

## Working Conventions

- Treat this as a script-based benchmark workspace, not a packaged library.
- Keep legacy scripts for reproducibility.
- Add latest-Underworld-compatible updates as separate scripts or minimal, well-documented edits.
- Preserve benchmark output folder conventions used by existing scripts.
- During rebuild, prioritize adding clean canonical scripts before restoring experimental copies.

## Benchmark Priorities

- Annulus:
  - `benchmarks/annulus/legacy/ex_stokes_thieulot_legacy.py`
  - `benchmarks/annulus/ex_stokes_thieulot.py`
  - `benchmarks/annulus/ex_stokes_kramer.py`
- Spherical:
  - `benchmarks/spherical/legacy/ex_stokes_thieulot_legacy.py`
  - `benchmarks/spherical/ex_stokes_thieulot.py`

## Benchmark Troubleshooting Notes

- Spherical Thieulot `m = 3`:
  - this case is more sensitive than `m = -1` to pressure treatment and boundary-condition enforcement
  - if analytical and computed pressures differ strongly, check pressure boundary conditions first, not just pressure mean calibration
  - zero-mean pressure calibration removes only the constant pressure null space; it does not fix an incorrect pressure shape

- Free-slip / null-space handling:
  - for benchmarks following Kramer et al. (2021), remove rigid-body rotation modes from velocity before comparing with the analytical solution
  - also subtract the domain-average pressure before evaluating pressure errors
  - solver-internal null-space removal and final benchmark calibration are related but not identical steps

- Legacy UW3 MPI setup:
  - use the matching legacy OpenMPI and legacy UW3 Python pair from `activate_uw3_legacy.sh`
  - if every MPI process reports rank `0` and size `1`, the wrong `mpirun` / `python` pair is being used and mesh generation can corrupt shared `.msh` files
  - if needed, verify with:

```bash
/Users/tgol0006/manual_install_pkg/openmpi-4.1.6/bin/mpirun -np 4 \
  /Users/tgol0006/manual_install_pkg/petsc_venv_uw3_21125/venv_uw3/bin/python3 \
  -c 'from petsc4py import PETSc; print(PETSc.COMM_WORLD.rank, PETSc.COMM_WORLD.size, flush=True)'
```

## Output and Data Policy

- Generated outputs should go under benchmark-specific `output/` directories.
- Do not commit transient artifacts such as:
  - `.meshes/`
  - `.ipynb_checkpoints/`
  - `__pycache__/`
  - large generated plot/checkpoint files unless explicitly needed.

## Suggested Validation Workflow

1. Run a small baseline matrix (serial and MPI) for key cases.
2. Run updated scripts with the same parameters.
3. Compare timing and error norms from produced output files.
4. Document differences and compatibility notes in commit messages or repo docs.

## Environment Note

Use the repo helper scripts to enter the configured pixi environment:

```bash
./activate_uw3_legacy.sh
# or
./activate_uw3_latest.sh
```
