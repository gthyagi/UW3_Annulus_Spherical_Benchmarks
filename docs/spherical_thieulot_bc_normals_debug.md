# Spherical Thieulot BC-Normal Debug

## Goal

Test whether switching the normal definition in a natural normal-only boundary
condition improves pressure on the inner spherical boundary, using the
`essential` solution as the reference baseline.

## Driver

Standalone script:

`benchmarks/sphere/ex_stokes_thieulot_bc_normals.py`

The main benchmark driver
`benchmarks/sphere/ex_stokes_thieulot.py`
was not modified for this experiment.

For each parameter set, the script runs:

- `essential`
- `natural_full`
- `natural_normal_petsc`
- `natural_normal_analytic`
- `natural_normal_projected` when requested

and records:

- global pressure L2 vs analytical pressure
- inner / outer boundary pressure L2 vs analytical pressure
- inner / outer boundary pressure L2 vs the essential baseline
- inner / outer tangential velocity RMS

The key normal-only term is:

`penalty * ((n . v_num) - (n . v_ana)) * n`

## Steps Taken

1. Added a dedicated standalone driver for boundary-normal experiments.
2. Verified the driver on `m = -1` first.
3. Repeated the same matrix for `m = 3`.
4. Added a projected-normal variant as a third normal definition.
5. Ran targeted `m = 3` PETSc-normal retests with:
   - higher penalty
   - tighter Stokes tolerance

## Main Comparison Settings

- `uw_cellsize = 1/8`
- `Q2/Q1`
- continuous pressure
- `uw_p_bc = False`
- default pressure gauge: volume mean
- serial solve with the standalone driver
- baseline natural penalty: `1e8`
- baseline Stokes tolerance: `1e-5`

## Results

### `m = -1`

```text
case                    p_vol_l2(ana)   p_inner_l2(ana)   p_inner_l2(base)   inner_tan_rms
essential               0.870782        2.00023           -                  4.1907
natural_full            0.870904        1.99414           0.023663           4.19116
natural_normal_petsc    0.903829        2.38951           2.23654            3.99403
natural_normal_analytic 1.27216         15.2982           12.985             0.144261
natural_normal_projected
                        1.27379         15.6381           13.2424            0.213833
```

### `m = 3`

```text
case                    p_vol_l2(ana)   p_inner_l2(ana)   p_inner_l2(base)   inner_tan_rms
essential               0.893876        1.83758           -                  15.3523
natural_full            0.894121        1.83568           0.00880887         15.3557
natural_normal_petsc    0.910169        3.06533           0.805627           14.3317
natural_normal_analytic 0.895693        10.4955           3.60611            0.659744
natural_normal_projected
                        0.89858         11.1348           3.87889            1.42643
```

## Tuning Checks

Targeted `m = 3` PETSc-normal retests:

### Higher penalty

- `uw_vel_penalty = 1e10`
- `uw_stokes_tol = 1e-5`

```text
natural_normal_petsc: p_inner_l2(base) = 0.805639
baseline PETSc-normal: p_inner_l2(base) = 0.805627
```

### Tighter tolerance

- `uw_vel_penalty = 1e8`
- `uw_stokes_tol = 1e-8`

```text
natural_normal_petsc: p_inner_l2(base) = 0.805627
baseline PETSc-normal: p_inner_l2(base) = 0.805627
```

Neither change materially improved the PETSc-normal result.

## Positive Results

- The standalone driver isolates the experiment cleanly from the production
  benchmark path.
- `natural_full` matches the essential baseline very closely in both `m = -1`
  and `m = 3`.
- All three normal-only variants were run successfully: PETSc, analytical, and
  projected.

## Negative Results

- Normal-only weak BCs do not improve inner-boundary pressure relative to the
  essential baseline.
- Analytical normals are substantially worse than PETSc normals for this
  benchmark objective.
- Projected normals track the analytical-normal behavior more than the
  PETSc-normal behavior.
- Increasing the penalty or tightening the solve tolerance does not recover the
  essential baseline for the PETSc-normal normal-only case.

## Key Observation

The important distinction is not just `PETSc normal` versus `analytical
normal`. The bigger change is `full-vector benchmark BC` versus `normal-only
BC`.

`natural_full` still tries to match the full analytical boundary velocity, so it
stays very close to `essential`.

`natural_normal_*` changes the boundary-condition class. It only penalizes the
normal component, so it behaves like a free-slip style condition rather than
the original benchmark BC.

That is visible in the inner-boundary tangential RMS:

- `natural_full` stays almost identical to `essential`
- `natural_normal_analytic` suppresses tangential motion very strongly
- `natural_normal_projected` behaves similarly
- `natural_normal_petsc` leaks more tangential motion because the facet normals
  are imperfect

That leakage makes PETSc normals look less bad relative to the essential
baseline, but it is not evidence that PETSc normals are a more accurate version
of the benchmark BC.

## Conclusion

Replacing PETSc facet normals with analytical or projected normals in a
natural normal-only BC does **not** reduce inner-boundary pressure error for
this spherical Thieulot benchmark.

If the goal is to match the benchmark / essential solution, the best choices
remain:

- `essential`
- `natural_full`

If the goal is a genuine free-slip style experiment, `natural_normal_*` is a
valid separate test, but it should be treated as a different boundary-condition
problem rather than as a pressure fix for the benchmark.
