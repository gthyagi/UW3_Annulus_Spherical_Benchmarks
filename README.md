# Underworld3 Annulus and Spherical Benchmarks

This repository is a reset workspace for **Underworld3** benchmark and validation scripts in:

- 2D annulus geometries
- 3D spherical shell geometries

Current status: the repo has been cleaned for a fresh restart, and benchmark content is being rebuilt progressively.

## Underworld3 Branch Context

This benchmark repository has been used with two Underworld3 branches:

- Legacy UW3 branch (historical compatibility):
  - [jcg-meshvar2meshvar-fix](https://github.com/gthyagi/underworld3/tree/jcg-meshvar2meshvar-fix)
- Latest UW3 branch (current development target):
  - [feature/boundary-integrals](https://github.com/gthyagi/underworld3/tree/feature/boundary-integrals)

When comparing results, treat the legacy branch as baseline and validate migrated scripts against it.

## Repository Layout

- `benchmarks/annulus/`
  - Annulus benchmark scripts and annulus-specific temporary mesh/checkpoint folders.
- `benchmarks/spherical/`
  - Spherical benchmark scripts and spherical-specific temporary mesh/checkpoint folders.
- `docs/`
  - Notes and workflow documentation.
- `output/`
  - Top-level generated outputs (optional shared output area).
- `activate_uw3_legacy.sh`
  - Helper to activate the legacy UW3 environment.
- `activate_uw3_latest.sh`
  - Helper to activate the latest UW3 environment.

## Major Benchmarks Included

### Annulus Benchmarks

- **Thieulot annulus benchmark (legacy reference currently present)**
  - `benchmarks/annulus/ex_stokes_thieulot_legacy.py`
  - Manufactured analytical solution benchmark used for solver/error validation.

### Spherical Benchmarks

- Spherical benchmark scripts are being reintroduced in the new structure.
- Existing material currently appears mostly as notebook checkpoints under:
  - `benchmarks/spherical/.ipynb_checkpoints/`

## Typical Outputs

Benchmark scripts usually write to case-specific `output/` directories, including:

- XDMF/HDF5 checkpoint data
- mesh generation and Stokes solve timing files
- error-norm summaries
- diagnostic plots (PNG/PDF)

## Running a Benchmark

From this repository root:

```bash
cd benchmarks/annulus
python ex_stokes_thieulot_legacy.py --res 8 --vdegree 2 --pdegree 1
```

Activate environment helpers:

```bash
./activate_uw3_legacy.sh
# or
./activate_uw3_latest.sh
```

## Notes

- This repo is currently in a rebuild phase after cleanup.
- Keep generated artifacts (`.meshes`, `.ipynb_checkpoints`, `output`, `__pycache__`) untracked.
- Re-add benchmark scripts incrementally and validate against the legacy branch baseline.
