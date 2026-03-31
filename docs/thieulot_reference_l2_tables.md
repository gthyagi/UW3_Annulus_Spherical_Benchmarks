# Thieulot Spherical Benchmark Reference L2 Tables

This file stores reference `L2` error values extracted from published figures for
future comparison with Underworld runs.

These are figure-extracted values, not values copied from a numeric table.
They should therefore be treated as approximate reference data.

## Sources

- ASPECT documentation benchmark page:
  [The hollow sphere benchmark](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/hollow_sphere/doc/hollow_sphere.html)
- ASPECT documentation error figure:
  [errors_hollowsphere.svg](https://aspect-documentation.readthedocs.io/en/latest/_images/errors_hollowsphere.svg)
- Thieulot benchmark paper:
  [Thieulot (2017)](https://se.copernicus.org/articles/8/1181/2017/)
- Thieulot paper PDF:
  [se-8-1181-2017.pdf](https://se.copernicus.org/articles/8/1181/2017/se-8-1181-2017.pdf)

## Important Notes

- The ASPECT documentation page only gives the `gamma = -1` case. It does not
  provide an `m = +3` curve.
- The ASPECT documentation figure uses `h` as the radial extent of the
  elements.
- The Thieulot paper uses `<h>` as the average element resolution.
- The extracted points lie very close to `1/4`, `1/8`, `1/16`, `1/32`, and
  `1/64`, so those rounded labels are used below for practical comparison.

## ASPECT Docs: `gamma = -1`

| Resolution | Velocity L2 | Pressure L2 |
|---|---:|---:|
| `1/4` | `5.44e-02` | `7.25e-01` |
| `1/8` | `7.64e-03` | `1.25e-01` |
| `1/16` | `9.91e-04` | `2.91e-02` |
| `1/32` | `1.26e-04` | `7.18e-03` |
| `1/64` | `1.57e-05` | `1.78e-03` |

## Thieulot Paper: ASPECT, `m = -1`, `Q2Q1`

| Resolution | Velocity L2 | Pressure L2 |
|---|---:|---:|
| `1/4` | `5.44e-02` | `7.25e-01` |
| `1/8` | `7.65e-03` | `1.25e-01` |
| `1/16` | `9.91e-04` | `2.91e-02` |
| `1/32` | `1.25e-04` | `7.17e-03` |
| `1/64` | `1.58e-05` | `1.78e-03` |

## Thieulot Paper: ASPECT, `m = +3`, `Q2Q1`

| Resolution | Velocity L2 | Pressure L2 |
|---|---:|---:|
| `1/4` | `9.09e-01` | `4.64e+00` |
| `1/8` | `1.53e-01` | `4.63e-01` |
| `1/16` | `2.03e-02` | `4.11e-02` |
| `1/32` | `2.55e-03` | `5.95e-03` |
| `1/64` | `3.19e-04` | `1.37e-03` |
