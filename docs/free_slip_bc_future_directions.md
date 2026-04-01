# Free-Slip BC Future Directions

This note records the main implementation directions for improving free-slip boundary conditions in Underworld on annulus and spherical meshes.

## Current Issue

The current benchmark scripts impose free-slip through a pure penalty term on the normal velocity:

$$
\int_{\Gamma} \alpha \, (u \cdot n)(v \cdot n)\, dS
$$

with `alpha = uw_vel_penalty`.

This creates a tradeoff:

- small `alpha`: weak enforcement of `u . n = 0`, boundary leakage, larger velocity / pressure error
- large `alpha`: stiffer linear system, slower iterative solve, stronger sensitivity to preconditioner quality
- fixed `alpha` across mesh refinements: the BC error can pollute the convergence curve and saturate `L2` norms

This is exactly the pattern seen in the Kramer free-slip runs:

- lower penalty can solve much faster
- but if it is too low, the solution quality degrades
- a single fixed penalty is not a robust long-term choice for convergence studies

## Near-Term Practical Rule

If the current penalty method is kept for now, the penalty should not be fixed across mesh sizes.

Use a mesh-scaled penalty:

$$
\alpha \sim C \mu / h
$$

where:

- `mu` is viscosity
- `h` is the local mesh size near the boundary
- `C` is a moderate dimensionless tuning constant

For the benchmark scripts with `mu = 1`, this reduces to:

$$
\alpha \sim C / h
$$

This is better than using the same `uw_vel_penalty` at `1/8`, `1/16`, and `1/32`.

## Better Long-Term Direction 1: Nitsche Free-Slip BC

A proper Nitsche slip method avoids relying on penalty alone.

The idea is to impose `u . n = 0` weakly using:

- a consistency term from the Stokes traction
- a symmetric adjoint term
- a stabilization term scaled like `beta * mu / h`

The normal-component structure is roughly:

$$
-\int_{\Gamma} (\sigma(u,p)n \cdot n)(v \cdot n)\, dS
-\int_{\Gamma} (\sigma(v,q)n \cdot n)(u \cdot n)\, dS
+\int_{\Gamma} \frac{\beta \mu}{h}(u \cdot n)(v \cdot n)\, dS
$$

Why this is better:

- the weak form is consistent with the continuous Stokes equations
- the stabilization term is no longer doing all the enforcement by itself
- the solution is much less sensitive to the exact choice of penalty coefficient
- mesh-convergence behavior is usually much cleaner than with a pure penalty method

Why it is attractive for Underworld:

- it fits the existing weak-form / natural-BC style better than an exact-constraint method
- it should be possible to expose as a dedicated Stokes boundary-condition helper
- it is likely the most realistic solver-level improvement in the short term

Possible UW-level API direction:

```python
stokes.add_slip_bc(
    boundary=mesh.boundaries.Upper.name,
    normal_fn=mesh.CoordinateSystem.unit_e_0,
    method="nitsche",
    beta=10.0,
)
```

Open implementation details:

- exact weak-form choice for the Stokes operator currently used in UW
- how to define / pass boundary `h`
- whether to support PETSc facet normals, analytical normals, or projected normals, or all three

## Better Long-Term Direction 2: True Normal-Constraint Free-Slip

The more robust end-state is to enforce `u . n = 0` as an exact discrete constraint rather than by penalty.

Three possible directions:

### 1. Boundary Lagrange Multiplier

Add a boundary unknown `lambda` and enforce:

$$
\int_{\Gamma} \lambda (v \cdot n)\, dS
+\int_{\Gamma} \eta (u \cdot n)\, dS
$$

Pros:

- exact normal-velocity enforcement
- no penalty tuning
- no penalty-driven conditioning blow-up

Cons:

- larger saddle-point system
- new boundary multiplier space required
- more demanding preconditioning work

### 2. Geometry-Aware Essential Constraint

At boundary nodes, rotate the local basis into normal and tangential components and constrain the normal component directly.

Pros:

- exact enforcement
- no extra multiplier field

Cons:

- UW currently applies essential BCs component-wise in Cartesian coordinates
- curved boundaries need local geometry-aware constraints
- this is deeper infrastructure work

### 3. Tangential Boundary Subspace

Construct the boundary velocity space so the boundary DOFs are tangential by design.

Pros:

- mathematically clean
- exact constraint in the discrete space

Cons:

- most invasive implementation option
- likely a larger mesh / FE infrastructure change

## Why This Matters For Pressure

Pressure is often more sensitive than velocity to small normal-velocity leakage at curved boundaries, especially near the inner boundary of annulus and spherical shells.

If `u . n` is only approximately zero, pressure can absorb that error. This is one reason pressure may show stronger sensitivity than velocity in free-slip shell benchmarks.

An exact or more consistent free-slip treatment should therefore help:

- inner-boundary pressure quality
- global pressure convergence
- robustness across mesh refinements
- solver behavior for mantle convection and global subduction models

## Curved Geometry Still Matters

Even with an improved free-slip formulation, geometry accuracy still matters.

On linear-facet annulus / spherical meshes:

- normals are piecewise flat
- curvature is only approximated
- traction and pressure near boundaries can still be affected

So boundary-condition improvements and curved-element support are complementary, not competing, directions.

## Recommended Implementation Order

1. Keep the current penalty method, but scale `uw_vel_penalty` like `C / h`.
2. Add a dedicated Nitsche free-slip option in UW.
3. Add a true geometry-aware normal-constraint free-slip feature.
4. Improve curved mesh geometry support alongside the above.

## Benchmarking Checklist For Future Work

When testing any new free-slip implementation, measure:

- velocity `L2`
- pressure `L2`
- inner-boundary pressure error
- outer-boundary pressure error
- boundary normal leakage, `||u . n||`
- sensitivity to mesh refinement
- sensitivity to solver tolerance
- sensitivity to MPI partitioning

For comparison, always keep one baseline run with the current penalty implementation.
