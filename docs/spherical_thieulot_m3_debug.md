# Spherical Thieulot `m=3` Pressure Debug

## Summary

The remaining `m != -1` pressure bug was in the spherical body-force expression, not in the pressure gauge, quadrature, or mixed FE order.

For the implemented 3D Cartesian velocity/pressure field, the required radial body-force coefficient is:

- `m = -1`: the existing closed-form `rho(r,theta)` branch
- `m != -1`: `r**m` times the closed-form coefficient that had been used in the paper-style branch

The correct spherical solve path is therefore:

- keep the `r**m` factor in the `m != -1` `rho_expr`
- keep `rho_bodyforce_expr = -rho_expr`

## Steps Taken

1. Verified the analytical helper formulas in the spherical benchmark script and created an isolated debug driver:
   [ex_stokes_thieulot_debug.py](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/benchmarks/spherical/ex_stokes_thieulot_debug.py)
2. Validated the setup on cheap serial `m=-1` runs.
3. Reproduced the bad `m=3` pressure behavior in the debug driver.
4. Checked gauge, pressure BC, quadrature, polynomial order, and pressure continuity variations.
5. Derived the radial body-force coefficient independently from the spherical stress-divergence identity.
6. Confirmed that the derived coefficient matches:
   - the `m=-1` branch as already implemented
   - the `m != -1` branch only when the extra `r**m` factor is present
7. Patched the main spherical benchmark and the analytical rho plotting helper.
8. Added a pure-SymPy regression test for the forcing formula.

## Variations Explored

- `m = -1`, `1`, `2`, `3`
- `uw_cellsize = 1/8`, `1/16`
- `uw_p_bc = False`, `True`
- `uw_pcont = True`, `False`
- `uw_qdegree = 6`, `8`
- `uw_vdegree/uw_pdegree = 2/1`, `3/2`
- body-force sign flip in the debug driver

## Positive Results

- Restoring the `r**m` factor in the `m != -1` forcing while keeping the negative body-force sign fixes the main issue.
- Cheap serial debug-driver results improved substantially:
  - `m=3`, `1/8`, before correction: `v_l2 ~ 7.01e-2`, `p_l2 ~ 3.09`
  - `m=3`, `1/8`, after correction: `v_l2 ~ 3.99e-2`, `p_l2 ~ 3.59e-1`
  - `m=1`, `1/8`, after correction: `v_l2 ~ 1.67e-2`, `p_l2 ~ 2.48e-1`
- The main spherical script matches the corrected debug driver at `m=3`, `1/8`:
  - `v_l2 ~ 3.99e-2`
  - `p_l2 ~ 3.59e-1`
- The new regression test passes:
  - [test_spherical_thieulot_bodyforce.py](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/tests/test_spherical_thieulot_bodyforce.py)

## Negative Results

- Removing the `r**m` factor was a false lead. It makes the `m != -1` forcing disagree with the implemented 3D field.
- Flipping the body-force sign to `+rho * gravity` is worse for both `m=-1` and `m=3`.
- Increasing quadrature (`qdeg 6`, `8`) did not materially change the bad pre-fix `m=3` results.
- Raising FE order from `P2/P1` to `P3/P2` did not materially change the bad pre-fix `m=3` results.
- `uw_p_bc=True` did not fix the pressure mismatch; it generally made the `m=3` run worse.
- `uw_pcont=False` is not a useful diagnostic with the current solver settings because the DG-pressure solve path becomes very slow / stalls.

## Key Observations

- The pressure mismatch grows with `m` in the pre-fix branch:
  `m=-1` is reasonable, `m=1` is worse, `m=2` worse again, `m=3` worst.
- This monotonic trend pointed to a general `m != -1` forcing error rather than an `m=3`-specific typo.
- The independent stress-divergence derivation showed that the implemented spherical field requires the `r**m` factor in the `m != -1` radial body-force coefficient.

## Files Changed

- [ex_stokes_thieulot.py](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/benchmarks/spherical/ex_stokes_thieulot.py)
- [ex_stokes_thieulot_debug.py](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/benchmarks/spherical/ex_stokes_thieulot_debug.py)
- [thieulot_field_plots.py](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/benchmarks/spherical/thieulot_field_plots.py)
- [test_spherical_thieulot_bodyforce.py](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/tests/test_spherical_thieulot_bodyforce.py)
