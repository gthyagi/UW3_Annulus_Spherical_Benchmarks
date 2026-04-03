# Spherical Thieulot Pressure-Anchor Debugging

## Goal

Investigate why pressure remains inaccurate at the inner spherical boundary, even after the `m = 3` body-force bug was fixed, and determine which current Underworld features are viable for improving the pressure field.

Relevant drivers:

- [benchmarks/spherical/ex_stokes_thieulot.py](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/benchmarks/spherical/ex_stokes_thieulot.py)
- [benchmarks/spherical/ex_stokes_thieulot_bc_normals.py](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/benchmarks/spherical/ex_stokes_thieulot_bc_normals.py)
- [benchmarks/spherical/ex_stokes_thieulot_pressure_anchor.py](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/benchmarks/spherical/ex_stokes_thieulot_pressure_anchor.py)

## Scope

The earlier `m = 3` failure is resolved. The remaining issue is the inner-boundary pressure mismatch.

The working hypothesis tested here was:

1. pressure is highly sensitive near the inner boundary because of strong radial scaling
2. the global constant-pressure nullspace may not control the inner-boundary pressure trace well enough
3. explicit pressure anchoring or alternative gauge choices might reduce the error

## Tests Run

### 1. Normal-vector boundary-condition tests

These were already completed in the separate normal-BC driver.

Cases compared:

- `essential`
- `natural_full`
- `natural_normal_petsc`
- `natural_normal_analytic`
- `natural_normal_projected`

Main result:

- `natural_full` stayed close to `essential`
- `natural_normal_*` did not improve the inner-boundary pressure
- analytical or projected normals were worse than PETSc normals in the normal-only BC

Conclusion:

- boundary-normal accuracy is not the dominant cause of the pressure error in the current benchmark setup

### 2. Pressure-anchor experiments

Implemented in [ex_stokes_thieulot_pressure_anchor.py](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/benchmarks/spherical/ex_stokes_thieulot_pressure_anchor.py).

Pressure modes:

- `none`: PETSc constant-pressure nullspace, no pressure Dirichlet BC
- `lower`: pressure Dirichlet on `Lower`
- `upper`: pressure Dirichlet on `Upper`
- `both`: pressure Dirichlet on both `Lower` and `Upper`

Reporting-only gauges:

- `none`
- `volume_mean`
- `inner_surface_mean`
- `outer_surface_mean`

### 3. Underworld feature inspection

Checked what current UW exposes for pressure control.

Available now:

- named-surface pressure Dirichlet BCs
- PETSc constant-pressure nullspace
- volume and boundary integrals for post-solve diagnostics

Not first-class:

- boundary mean pressure constraints
- global pressure mean constraints as solver constraints
- point pressure pins

## Results

### Serial direct solve, `np = 1`, `1/8`, `m = -1`

Summary file:

- [pressure_anchor_stage m=-1 summary](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/output/spherical/thieulot/pressure_anchor_stage/inv_lc_8_m_-1_vdeg_2_pdeg_1_pcont_true_tol_1e-05_np_1/summary.txt)

Key result:

- `none`, `lower`, `upper`, and `both` were effectively identical
- gauge shifts changed `p_inner_l2` only in the fourth decimal place

Representative numbers:

| mode | gauge | `v_vol_l2` | `p_vol_l2` | `p_inner_l2` |
|---|---:|---:|---:|---:|
| `none` | `volume_mean` | `2.29948e-2` | `8.70782e-1` | `2.00023` |
| `lower` | `volume_mean` | `2.29948e-2` | `8.70782e-1` | `2.00023` |
| `upper` | `volume_mean` | `2.29948e-2` | `8.70782e-1` | `2.00023` |
| `both` | `volume_mean` | `2.29948e-2` | `8.70782e-1` | `2.00023` |

### Serial direct solve, `np = 1`, `1/8`, `m = 3`

Summary file:

- [pressure_anchor_stage m=3 summary](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/output/spherical/thieulot/pressure_anchor_stage/inv_lc_8_m_3_vdeg_2_pdeg_1_pcont_true_tol_1e-05_np_1/summary.txt)

Key result:

- same pattern as `m = -1`
- pressure anchors did not materially change the solved pressure
- gauge shifts changed `p_inner_l2` only slightly

Representative numbers:

| mode | gauge | `v_vol_l2` | `p_vol_l2` | `p_inner_l2` |
|---|---:|---:|---:|---:|
| `none` | `volume_mean` | `7.17376e-2` | `8.93876e-1` | `1.83758` |
| `lower` | `volume_mean` | `7.17376e-2` | `8.93876e-1` | `1.83759` |
| `upper` | `volume_mean` | `7.17376e-2` | `8.93876e-1` | `1.83759` |
| `both` | `volume_mean` | `7.17376e-2` | `8.93876e-1` | `1.83759` |

### MPI iterative solve, `np = 2`, `1/8`, `m = -1`

Summary file:

- [pressure_anchor_mpi2 m=-1 summary](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/output/spherical/thieulot/pressure_anchor_mpi2/inv_lc_8_m_-1_vdeg_2_pdeg_1_pcont_true_tol_1e-05_np_2/summary.txt)

Key result:

- `none` converged and reproduced the known good low-error solve
- `both` diverged in the linear solve with `DIVERGED_PC_FAILED`

Representative numbers:

| mode | gauge | `v_vol_l2` | `p_vol_l2` | `p_inner_l2` | status |
|---|---:|---:|---:|---:|---|
| `none` | `volume_mean` | `5.90108e-3` | `2.61469e-1` | `11.7999` | converged |
| `both` | `volume_mean` | `8.20630e-1` | `1.0` | `1.0` | diverged |

This means the pressure-anchored solve path is not robust in the current parallel fieldsplit/preconditioner setup.

### MPI iterative solve, `np = 2`, `1/8`, `m = 3`

Summary file:

- [pressure_anchor_mpi2_clean m=3 summary](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/output/spherical/thieulot/pressure_anchor_mpi2_clean/inv_lc_8_m_3_vdeg_2_pdeg_1_pcont_true_tol_1e-05_np_2/summary.txt)

Key result:

- `none` converged cleanly
- a mixed `none,both` run previously stalled on the anchored case and had to be killed

Representative `none` numbers:

| mode | gauge | `v_vol_l2` | `p_vol_l2` | `p_inner_l2` |
|---|---:|---:|---:|---:|
| `none` | `volume_mean` | `3.99436e-2` | `3.59465e-1` | `20.5627` |

## What Worked

- fixing the earlier `m = 3` body-force bug
- keeping `bc_type = essential` or `natural_full` for the actual benchmark comparison
- using PETSc pressure nullspace with post-solve diagnostics
- using `BdIntegral`-based diagnostics to inspect inner and outer pressure means separately

## What Did Not Work

- replacing PETSc normals with analytical normals in a normal-only BC
- projected normals in the same normal-only BC
- using normal-only weak BCs as a way to improve pressure
- relying on a different constant gauge to fix the inner-boundary pressure mismatch
- using hard pressure anchors as a general fix
  - in serial/direct solves they barely changed the solution
  - in parallel/iterative solves they destabilized the solve path

## Root Cause

The remaining inner-boundary pressure issue is not primarily a normal-vector problem and not primarily a constant pressure-gauge problem.

The evidence points to a boundary-local pressure-trace sensitivity near the inner spherical surface, where pressure has strong radial dependence and small algebraic errors are amplified. The global constant-pressure nullspace removes only the volume-wide additive mode; it does not provide a surgical control on the inner-boundary pressure trace. Hard pressure Dirichlet anchors change the algebraic problem and can alter or destabilize the iterative Stokes solve, but they do not represent a clean fix for the actual inner-boundary pressure shape.

In short:

- the pressure error is localized and boundary-sensitive
- the current nullspace/gauge treatment is too coarse to target it directly
- the available hard-anchor workaround is not robust in the iterative parallel path

## Practical Recommendations Using Current UW Features

### Best current approach for benchmark-quality solves

- keep `bc_type = essential`
- keep `petsc_use_pressure_nullspace = True` when there is no physical pressure BC
- do not use normal-only BCs as a pressure fix
- do not treat `p_bc = true` as a general solution improvement unless the physical model really has a pressure BC

### Best current approach for debugging

- compare inner and outer pressure diagnostics separately using `BdIntegral`
- monitor:
  - volume pressure mean
  - inner-surface pressure mean
  - outer-surface pressure mean
  - inner/outer boundary relative pressure errors
- use reduced serial or small-MPI cases to separate algebraic issues from formulation issues

### Cautious use of pressure anchoring

- named-surface pressure Dirichlet BCs are available now
- but they should only be used if the physics genuinely prescribes pressure there
- for this benchmark, they are better treated as a diagnostic perturbation than as a fix

## Worthwhile UW Improvements

These would address the problem much more directly than current hard Dirichlet anchors.

1. First-class boundary-mean pressure constraints
   - example: enforce `∫_Lower p dS = 0`
   - this is the most surgical control for the present problem

2. First-class point pressure pins
   - example: fix pressure at a single inner-boundary point
   - useful for gauge control without overconstraining the whole boundary

3. Better support for pressure constraints in parallel fieldsplit/preconditioned Stokes solves
   - current hard pressure anchors can destabilize the parallel solve path

4. Better boundary pressure / traction recovery utilities on curved spherical boundaries
   - especially for applications where tractions and surface/basal stresses matter

5. More explicit nullspace / gauge-control utilities in Python
   - boundary-local gauge tools are missing at present

## Implications for Large-Scale Applications

This matters for global mantle convection, spherical-cap models, and global subduction models because those models also depend on accurate pressure or traction interpretation near curved shell boundaries.

Even when the velocity field looks acceptable:

- boundary-local pressure errors can remain significant
- global zero-mean pressure is not enough to control the boundary trace
- hard pressure anchors may change solver behavior rather than actually fixing the pressure field

For larger production models, the safest current practice is:

- preserve a physically consistent velocity BC treatment
- use PETSc nullspace when pressure is gauge-free
- diagnose boundary pressure separately from global pressure
- treat hard pressure BCs as model changes, not as benchmark-neutral fixes
