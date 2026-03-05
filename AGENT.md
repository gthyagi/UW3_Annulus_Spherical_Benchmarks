# AGENT.md

## Purpose

This repository contains Underworld3 benchmark scripts for annulus (2D) and spherical shell (3D) Stokes-flow validation. It is currently in a fresh-start/rebuild state.

## Underworld3 Branch Context

- Legacy branch used for historical compatibility:
  - [jcg-meshvar2meshvar-fix](https://github.com/gthyagi/underworld3/tree/jcg-meshvar2meshvar-fix)
- Latest branch targeted for ongoing runs/migration:
  - [feature/boundary-integrals](https://github.com/gthyagi/underworld3/tree/feature/boundary-integrals)

For parity checks, compare latest-branch benchmark outputs against legacy-branch baseline outputs.

## Primary Directories

- `benchmarks/annulus/`: annulus benchmark scripts (current main script: legacy Thieulot benchmark).
- `benchmarks/spherical/`: spherical benchmark area (currently mostly checkpoint/mesh artifacts pending script re-addition).
- `docs/`: documentation notes and migration guidance.
- `output/`: shared generated output location (if used).

## Working Conventions

- Treat this as a script-based benchmark workspace, not a packaged library.
- Keep legacy scripts for reproducibility.
- Add latest-Underworld-compatible updates as separate scripts or minimal, well-documented edits.
- Preserve benchmark output folder conventions used by existing scripts.
- During rebuild, prioritize adding clean canonical scripts before restoring experimental copies.

## Benchmark Priorities

- Annulus:
  - `benchmarks/annulus/Ex_Stokes_Annulus_Benchmark_Thieulot_legacy.py`
  - Next add: latest-compatible Thieulot and Kramer scripts
- Spherical:
  - Next add: latest-compatible Thieulot and Kramer scripts in `benchmarks/spherical/`

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
