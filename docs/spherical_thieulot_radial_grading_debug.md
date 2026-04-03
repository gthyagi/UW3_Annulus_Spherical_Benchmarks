# Spherical Thieulot Radial Grading

Driver:
- [ex_stokes_thieulot_radial_grading.py](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/benchmarks/sphere/ex_stokes_thieulot_radial_grading.py)

Setup:
- radial mesh deformation only; benchmark fields and PDE are unchanged
- `grading_ratio = h_outer / h_inner`
- `grading_ratio = 4` gives `inner_scale ≈ 0.462`

## Short Summary

- `1/8`, serial/direct:
  - `m = -1`: moderate inner grading helped; best tested `grading_ratio ≈ 4`
  - `m = 3`: inner grading did not help
- `1/8`, `np = 8`:
  - `m = 3`: inner grading helped strongly; `grading_ratio = 4` was the best tested compromise
- `1/16`, `np = 8`, after the UW `BdIntegral` fix:
  - full inner/outer boundary diagnostics now complete for graded meshes
  - grading improves inner-boundary pressure but worsens outer-boundary pressure

## Latest `1/16`, `np = 8` Results

Using the `volume_mean` gauge:

| case | grading ratio | inner scale | velocity L2 | pressure volume L2 | pressure inner L2 | pressure outer L2 |
|---|---:|---:|---:|---:|---:|---:|
| `m=-1` | `1` | `1.000` | `7.71041e-4` | `5.96441e-2` | `14.4094` | `3.88626` |
| `m=-1` | `4` | `0.462` | `4.65460e-4` | `2.41538e-2` | `5.51350` | `13.1031` |
| `m=3` | `1` | `1.000` | `5.57220e-3` | `4.05191e-2` | `16.4700` | `9.00753` |
| `m=3` | `4` | `0.462` | `1.29199e-3` | `2.45953e-2` | `4.56341` | `13.0849` |

Interpretation:
- `m=-1`: inner pressure `14.41 -> 5.51`, outer pressure `3.89 -> 13.10`
- `m=3`: inner pressure `16.47 -> 4.56`, outer pressure `9.01 -> 13.08`
- in both cases, grading helps the inner shell and global pressure norm, but pushes error toward the outer shell

## Files

Recent full-diagnostic summaries:
- [m=-1 grade 1](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/output/sphere/thieulot/radial_grading_16_mpi8_full_bugfix/inv_lc_16_m_-1_grade_1_inner_scale_1_vdeg_2_pdeg_1_pcont_true_tol_1e-05_np_8/summary.txt)
- [m=-1 grade 4](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/output/sphere/thieulot/radial_grading_16_mpi8_full_bugfix/inv_lc_16_m_-1_grade_4_inner_scale_0.462_vdeg_2_pdeg_1_pcont_true_tol_1e-05_np_8/summary.txt)
- [m=3 grade 1](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/output/sphere/thieulot/radial_grading_16_mpi8_full_bugfix/inv_lc_16_m_3_grade_1_inner_scale_1_vdeg_2_pdeg_1_pcont_true_tol_1e-05_np_8/summary.txt)
- [m=3 grade 4](/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/output/sphere/thieulot/radial_grading_16_mpi8_full_bugfix/inv_lc_16_m_3_grade_4_inner_scale_0.462_vdeg_2_pdeg_1_pcont_true_tol_1e-05_np_8/summary.txt)
