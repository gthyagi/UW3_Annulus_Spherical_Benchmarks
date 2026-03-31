# Kramer Benchmark Reference L2 Values

This note collects approximate `L2` error references from the benchmark paper:

- Kramer, Davies, and Wilson (2021), "Analytical solutions for mantle flow in cylindrical and spherical shells"
- Paper: [GMD 14, 1899-1926 (2021)](https://gmd.copernicus.org/articles/14/1899/2021/)
- Figure 3: [2-D cylindrical convergence](https://gmd.copernicus.org/articles/14/1899/2021/gmd-14-1899-2021-f03-web.png)
- Figure 4: [3-D spherical convergence](https://gmd.copernicus.org/articles/14/1899/2021/gmd-14-1899-2021-f04-web.png)
- Figure 5: [2-D cylindrical smooth cases without isoparametric geometry](https://gmd.copernicus.org/articles/14/1899/2021/gmd-14-1899-2021-f05-web.png)

These values are approximate figure readings, not author-supplied tables. Use them as a reference check, not as exact regression targets.

## Refinement Level vs `uw_cellsize`

The paper does not define refinement level by a Gmsh-style target edge length. It defines a mesh family and then doubles the mesh resolution at each level.

From the paper text:

- cylindrical level 1: `128 x 16 x 2 = 4096` triangles
- spherical level 1: `1280 x 16 x 3 = 61440` tetrahedra
- each next level doubles resolution in all directions

For easy comparison with this repo:

| Paper level step | Equivalent trend in this repo |
|---|---|
| `level -> level+1` | `uw_cellsize -> uw_cellsize / 2` |

So the relative mapping is straightforward: each paper refinement level corresponds to halving `uw_cellsize` in our runs.

However, there is **no exact one-to-one absolute mapping** between paper level and our `uw_cellsize = 1/8, 1/16, ...` because:

- the paper uses a specific structured/extruded mesh family
- this repo uses Gmsh target edge length on an unstructured mesh
- `uw_cellsize` is only a target length, not an exact radial-layer count

Two practical anchor points are useful:

### By radial spacing through shell thickness

The shell thickness is `R_+ - R_- = 1`. The paper's level-1 meshes use `16` radial layers, so the level-1 radial spacing is approximately:

```text
dr ≈ 1 / 16
```

By that measure, paper level 1 is closest to:

- `uw_cellsize ≈ 1/16`

and then approximately:

| Paper level | Approx. comparable `uw_cellsize` |
|---:|---:|
| 1 | `1/16` |
| 2 | `1/32` |
| 3 | `1/64` |
| 4 | `1/128` |
| 5 | `1/256` |

### By total cell count

The match is looser. For example:

- paper spherical level 1: `61440` tetrahedra
- this repo spherical `uw_cellsize = 1/8` example: about `102952` tetrahedra
- paper cylindrical level 1: `4096` triangles
- this repo annulus `uw_cellsize = 1/32` example: about `26394` triangles

So by total element count, our meshes are generally not aligned exactly with the paper levels.

## Recommended Comparison Rule

For practical benchmark comparison:

- compare **convergence trend** by halving `uw_cellsize`
- use the paper's `O(h^p)` slope as the main reference
- use the figure-read `L2` values as approximate magnitude checks
- treat `1/16, 1/32, 1/64, ...` as the best approximate mapping to paper levels `1, 2, 3, ...` when you want a simple level-to-cellsize table

## Convergence Orders

For the continuous `P2-P1` benchmark family shown in Figs. 3 and 4:

| Geometry | Case family | Velocity | Pressure |
|---|---|---:|---:|
| Cylindrical shell | Smooth forcing | `O(h^3)` | `O(h^2)` |
| Cylindrical shell | Delta-function forcing | `O(h^1.5)` | `O(h^0.5)` |
| Spherical shell | Smooth forcing | `O(h^3)` | `O(h^2)` |
| Spherical shell | Delta-function forcing | `O(h^1.5)` | `O(h^0.5)` |

Additional note from Fig. 5:

| Geometry | Case family | Velocity | Pressure |
|---|---|---:|---:|
| Cylindrical shell, smooth `k=2`, linear geometry only | Free-slip / zero-slip | `O(h^2)` | `O(h^2)` |

Additional note from the paper text for the discontinuous-pressure cylindrical delta cases (Fig. 6):

| Geometry | Element pair | Case family | Velocity | Pressure |
|---|---|---|---:|---:|
| Cylindrical shell | `P2_bubble-P1DG` | Delta-function forcing | `O(h^3)` | `O(h^2)` |

## 3-D Spherical Reference Series

Figure 4 contains multiple `(l,m)` series. The tables below use the repo-default-like series `l=2, m=1`. For the smooth cases, the paper uses `k=l+1`, so this corresponds to `k=3`.

### Case 1: Free-Slip, Delta Function

Approximate `l=2, m=1` values from Fig. 4 panels (a) and (c):

| Level | Velocity L2 | Pressure L2 |
|---:|---:|---:|
| 1 | `~1.3e-2` | `~3.6e-1` |
| 2 | `~4.5e-3` | `~2.7e-1` |
| 3 | `~1.5e-3` | `~1.9e-1` |
| 4 | `~5.5e-4` | `~1.3e-1` |

### Case 2: Free-Slip, Smooth

Approximate `l=2, m=1, k=3` values from Fig. 4 panels (b) and (d):

| Level | Velocity L2 | Pressure L2 |
|---:|---:|---:|
| 1 | `~1.3e-3` | `~3.2e-2` |
| 2 | `~2.2e-4` | `~7.5e-3` |
| 3 | `~2.8e-5` | `~2.1e-3` |
| 4 | `~3.2e-6` | `~5.0e-4` |

### Case 3: Zero-Slip, Delta Function

Approximate `l=2, m=1` values from Fig. 4 panels (e) and (g):

| Level | Velocity L2 | Pressure L2 |
|---:|---:|---:|
| 1 | `~4.6e-2` | `~3.2e-1` |
| 2 | `~1.4e-2` | `~2.2e-1` |
| 3 | `~5.0e-3` | `~1.5e-1` |
| 4 | `~1.8e-3` | `~1.1e-1` |

### Case 4: Zero-Slip, Smooth

Approximate `l=2, m=1, k=3` values from Fig. 4 panels (f) and (h):

| Level | Velocity L2 | Pressure L2 |
|---:|---:|---:|
| 1 | `~6.5e-3` | `~2.1e-2` |
| 2 | `~7.5e-4` | `~5.7e-3` |
| 3 | `~9.0e-5` | `~1.4e-3` |
| 4 | `~1.0e-5` | `~3.5e-4` |

## 2-D Cylindrical Reference Series

Figure 3 contains multiple wave-number series. The tables below use the default-like `n=2` series. For the smooth cases, note that the paper plots `k=2` and `k=8`; the repo default `k=3` is not plotted exactly, so treat the `k=2` values below as the nearest smooth reference.

### Case 1: Free-Slip, Delta Function

Approximate `n=2` values from Fig. 3 panels (a) and (d):

| Level | Velocity L2 | Pressure L2 |
|---:|---:|---:|
| 1 | `~1.7e-2` | `~2.0e-1` |
| 2 | `~6.0e-3` | `~1.5e-1` |
| 3 | `~2.1e-3` | `~1.0e-1` |
| 4 | `~7.4e-4` | `~6.7e-2` |
| 5 | `~2.6e-4` | `~4.8e-2` |

### Case 2: Free-Slip, Smooth (`k=2` reference)

Approximate `n=2` values from Fig. 3 panels (b) and (e):

| Level | Velocity L2 | Pressure L2 |
|---:|---:|---:|
| 1 | `~1.6e-4` | `~1.7e-3` |
| 2 | `~2.0e-5` | `~4.1e-4` |
| 3 | `~2.4e-6` | `~1.1e-4` |
| 4 | `~3.0e-7` | `~2.8e-5` |
| 5 | `~4.0e-8` | `~7.0e-6` |

### Case 3: Zero-Slip, Delta Function

Approximate `n=2` values from Fig. 3 panels (g) and (j):

| Level | Velocity L2 | Pressure L2 |
|---:|---:|---:|
| 1 | `~2.7e-2` | `~2.1e-1` |
| 2 | `~9.5e-3` | `~1.3e-1` |
| 3 | `~4.3e-3` | `~8.5e-2` |
| 4 | `~2.0e-3` | `~5.8e-2` |
| 5 | `~9.5e-4` | `~4.0e-2` |

### Case 4: Zero-Slip, Smooth (`k=2` reference)

Approximate `n=2` values from Fig. 3 panels (h) and (k):

| Level | Velocity L2 | Pressure L2 |
|---:|---:|---:|
| 1 | `~8.0e-4` | `~1.5e-3` |
| 2 | `~8.0e-5` | `~3.0e-4` |
| 3 | `~1.0e-5` | `~8.5e-5` |
| 4 | `~1.3e-6` | `~2.3e-5` |
| 5 | `~2.0e-7` | `~6.5e-6` |

## Cylindrical Smooth Cases Without Isoparametric Geometry

Figure 5 is useful when checking linear-geometry meshes in the annulus. It shows that smooth `k=2` cylindrical cases drop from the optimal `O(h^3)` velocity convergence in Fig. 3 to `O(h^2)` when the curved domain is not represented isoparametrically.

Approximate `n=2` values from Fig. 5:

| Case | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
|---|---:|---:|---:|---:|---:|
| Free-slip velocity | `~1.2e-3` | `~3.0e-4` | `~8.0e-5` | `~1.8e-5` | `~4.5e-6` |
| Free-slip pressure | `~1.8e-3` | `~4.2e-4` | `~1.1e-4` | `~2.8e-5` | `~7.0e-6` |
| Zero-slip velocity | `~2.2e-5` | `~5.5e-6` | `~1.4e-6` | `~3.3e-7` | `~9.0e-8` |
| Zero-slip pressure | `~7.2e-4` | `~1.8e-4` | `~4.4e-5` | `~1.1e-5` | `~2.8e-6` |

## Practical Use

- For spherical runs in this repo, the most relevant direct paper reference is Fig. 4 with `l=2, m=1`; smooth cases use `k=3`.
- For annulus runs in this repo, the delta-function `n=2` curves in Fig. 3 are direct references.
- For annulus smooth runs, the repo default `k=3` is not shown exactly in the paper. Use the `k=2` and `k=8` curves in Fig. 3 as lower/high-order smooth references and Fig. 5 when diagnosing the effect of linear geometry.
