# Underworld3 Annulus and Spherical Stokes Benchmarks

![Analytical benchmark fields](docs/benchmarks_banner_figure/combined_density_distribution_figures.jpg)

This repository contains Underworld3 benchmark studies for incompressible
Stokes flow in annulus and spherical-shell geometries. The benchmark suite
focuses on curved-domain with analytical solutions,
including the Thieulot--Puckett and Kramer benchmark families. It includes the
uw3 scripts for benchmarks, production-run helpers, postprocessing scripts, convergence
figures, benchmark articles, and a technical blog-post draft.

The main quantities reported by the benchmark articles are velocity and pressure
error norms, boundary diagnostics, convergence rates, and comparisons across
finite-element pairs and forcing cases.

## Underworld3 Environment

The configured UW3 checkout is:

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

The helper can also be sourced to activate the environment in the current shell if uw3 already installed via pixi:

```bash
source ./activate_uw3.sh
```

## Repository Layout

```text
benchmarks/
  annulus/      Benchmark scripts and field-plot helpers for annulus cases.
  spherical/    Benchmark scripts and field-plot helpers for spherical shells.

production_scripts/
  annulus/      Batch/convergence submission helpers for annulus studies.
  spherical/    Batch/convergence submission helpers for spherical studies.

docs/
  benchmarks_figures_and_articles/
    annulus/    Figure scripts, tables, references, and article PDFs.
    spherical/  Figure scripts, tables, references, and article PDFs.
  benchmarks_banner_figure/
    Combined analytical-field banner figure and paper-section material.
  benchmarks_blog_post/
    Technical blog-post draft for the benchmark suite.
  benchmarks_papers/
    Source papers and reference material.

output/
  Optional local output area for generated benchmark data.
```

## Benchmarks Included

| Geometry | Benchmark | Script | Article |
| --- | --- | --- | --- |
| Annulus | Thieulot--Puckett | `benchmarks/annulus/ex_stokes_thieulot.py` | `docs/benchmarks_figures_and_articles/annulus/thieulot/thieulot_annulus_benchmark_article.pdf` |
| Annulus | Kramer et al. | `benchmarks/annulus/ex_stokes_kramer.py` | `docs/benchmarks_figures_and_articles/annulus/kramer/kramer_annulus_benchmark_article.pdf` |
| Spherical shell | Thieulot | `benchmarks/spherical/ex_stokes_thieulot.py` | `docs/benchmarks_figures_and_articles/spherical/thieulot/thieulot_spherical_benchmark_article.pdf` |
| Spherical shell | Kramer et al. | `benchmarks/spherical/ex_stokes_kramer.py` | `docs/benchmarks_figures_and_articles/spherical/kramer/kramer_spherical_benchmark_article.pdf` |

## Running a Benchmark

From the repository root:

```bash
./activate_uw3.sh
python benchmarks/annulus/ex_stokes_thieulot.py \
  -uw_cellsize "1/32" \
  -uw_k 4 \
  -uw_vdegree 2 \
  -uw_pdegree 1 \
  -uw_pcont true
```

For MPI runs, invoke the same script through `mpirun` or the relevant batch
script:

```bash
mpirun -n 8 python benchmarks/spherical/ex_stokes_thieulot.py \
  -uw_cellsize "1/8" \
  -uw_m -1 \
  -uw_vdegree 2 \
  -uw_pdegree 1 \
  -uw_pcont true
```

Each script supports `--help`:

```bash
python benchmarks/annulus/ex_stokes_thieulot.py --help
```

## Output and Postprocessing

Benchmark runs write case-specific output directories containing some or all of:

- mesh and field HDF5/XDMF output,
- PETSc-reloadable checkpoint data when enabled by the script,
- `benchmark_metrics.h5` error and diagnostic summaries,
- timing information,
- generated PDF/PNG figures.

The article folders under `docs/benchmarks_figures_and_articles/` contain the
plotting scripts and Makefiles used to regenerate figures and tables from those
outputs.

## Documentation Products

- Benchmark articles:
  `docs/benchmarks_figures_and_articles/**/**/*_benchmark_article.pdf`
- Combined analytical-field banner:
  `docs/benchmarks_banner_figure/combined_density_distribution_figures.pdf`
- Blog-post draft:
  `docs/benchmarks_blog_post/blog_annulus_spherical_stokes_benchmarks.md`

## Housekeeping

- Keep generated runtime data out of commits unless it is a curated document or
  figure artifact.
- Do not commit `.meshes/`, `.ipynb_checkpoints/`, `output/`, `__pycache__/`,
  or temporary LaTeX build files unless they are intentionally tracked.
- Prefer updating the current benchmark scripts and article folders rather than
  adding duplicate compatibility copies.
- Record solver, mesh, and metric changes in the relevant benchmark article or
  plotting script.
