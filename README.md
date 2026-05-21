# Underworld3 Annulus and Spherical Benchmarks

This repository contains Underworld3 benchmark and post processing scripts for:

- annulus geometries
- spherical-shell geometries


## Underworld3 Environment

The current UW3 checkout is:

```text
/Users/tgol0006/uw_folder/uw3_git_gthyagi_latest/underworld3
```

Use the repository helper to enter the configured pixi environment:

```bash
./activate_uw3.sh
```

An explicit pixi environment name can be supplied when needed:

```bash
./activate_uw3.sh amr-dev
```

The helper can also be sourced to activate the environment in the current
shell:

```bash
source ./activate_uw3.sh
```

## Repository Layout

- `benchmarks/annulus/`
  - Current annulus Stokes benchmark scripts and plotting helpers.
- `benchmarks/spherical/`
  - Current spherical-shell Stokes benchmark scripts and plotting helpers.
- `production_scripts/`
  - Batch and convergence-run helper scripts.
- `docs/`
  - Benchmark notes, article sources, figures, and reference material.
- `output/`
  - Optional shared output area for generated benchmark data.
- `activate_uw3.sh`
  - Helper to activate the current UW3 pixi environment.

## Major Benchmarks Included

### Annulus Benchmarks

- Thieulot--Puckett annulus Stokes benchmark:
  - `benchmarks/annulus/ex_stokes_thieulot.py`
- Kramer annulus Stokes benchmark:
  - `benchmarks/annulus/ex_stokes_kramer.py`

### Spherical Benchmarks

- Thieulot spherical-shell Stokes benchmark:
  - `benchmarks/spherical/ex_stokes_thieulot.py`
- Kramer spherical-shell Stokes benchmark:
  - `benchmarks/spherical/ex_stokes_kramer.py`

## Typical Outputs

Benchmark scripts usually write to case-specific `output/` directories,
including:

- XDMF/HDF5 checkpoint data
- mesh generation and Stokes solve timing files
- error-norm summaries
- diagnostic plots (PNG/PDF)

## Running a Benchmark

From this repository root:

```bash
./activate_uw3.sh
python benchmarks/annulus/ex_stokes_thieulot.py --res 8 --vdegree 2 --pdegree 1
```

## Notes

- Keep generated artifacts untracked, including `.meshes/`,
  `.ipynb_checkpoints/`, `output/`, and `__pycache__/`.
- Prefer updating the current benchmark scripts over reintroducing duplicate
  compatibility copies.
- Document solver, mesh, and metric changes in the relevant benchmark notes or
  figure-script article sources.
