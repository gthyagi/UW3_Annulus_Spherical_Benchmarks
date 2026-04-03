# Pressure Normalization And Velocity Nullspaces In The Shell Benchmarks

This note is organized in the order you asked for:

1. what pressure normalization and velocity nullspaces are, including the theory and why they matter
2. how each benchmark paper deals with them
3. how the current Underworld benchmark scripts implement them, with code snippets

The benchmark papers discussed here are:

- Thieulot and Puckett (2018): annulus benchmark
- Thieulot (2017): spherical-shell benchmark
- Kramer, Davies, and Wilson (2021): cylindrical and spherical shell benchmarks

The Underworld scripts discussed here are:

- `benchmarks/annulus/ex_stokes_thieulot.py`
- `benchmarks/annulus/ex_stokes_kramer.py`
- `benchmarks/spherical/ex_stokes_thieulot.py`
- `benchmarks/spherical/ex_stokes_kramer.py`

## 1. What Are Pressure Normalization And Velocity Nullspaces?

### 1.1 Pressure normalization

For incompressible Stokes flow, the governing equations are

$$
-\nabla \cdot \left(2 \mu \, \varepsilon(\mathbf{u})\right) + \nabla p = \mathbf{f},
\qquad
\nabla \cdot \mathbf{u} = 0,
$$

with

$$
\varepsilon(\mathbf{u}) = \frac{1}{2}\left(\nabla \mathbf{u} + \nabla \mathbf{u}^{T}\right).
$$

The key point is that the equations only depend on the gradient of pressure. If $p$ is a solution, then

$$
\tilde{p} = p + c
$$

is also a solution for any constant $c$, because

$$
\nabla (p + c) = \nabla p.
$$

So pressure is only defined up to an additive constant. This is the pressure gauge freedom.

To make pressure unique, one extra scalar condition must be imposed. Common choices are:

- pointwise or Dirichlet pinning:

$$
p = p_0
$$

- zero domain-mean pressure:

$$
\frac{1}{|\Omega|}\int_{\Omega} p \, dV = 0
$$

- prescribed domain-mean pressure:

$$
\frac{1}{|\Omega|}\int_{\Omega} p \, dV = p_{\mathrm{ref}}
$$

- zero boundary-mean pressure on a boundary $\Gamma$:

$$
\frac{1}{|\Gamma|}\int_{\Gamma} p \, dS = 0
$$

This is what people usually mean by pressure normalization or pressure gauge fixing.

### 1.2 Why pressure normalization matters

Pressure normalization matters for three reasons.

First, without a pressure gauge the discrete Stokes system is singular or rank-deficient in the constant-pressure direction.

Second, if you compare a numerical pressure field to an analytical pressure field without putting them in the same gauge, the pressure error can be contaminated by an arbitrary constant offset.

Third, when you run benchmarks and compare convergence rates, you want the reported error to measure the actual discretization error, not a meaningless pressure shift.

### 1.3 Velocity nullspace

In some shell problems, the velocity itself can also be non-unique. The main extra null modes are rigid-body rotations:

$$
\mathbf{u}_{\mathrm{rb}} = \boldsymbol{\omega} \times \mathbf{x}.
$$

These modes are special because:

- they are divergence free
- their symmetric strain is zero:

$$
\varepsilon(\mathbf{u}_{\mathrm{rb}}) = 0
$$

- so they create no viscous stress
- on circular and spherical boundaries they are tangent to the boundary because

$$
(\boldsymbol{\omega} \times \mathbf{x}) \cdot \mathbf{x} = 0
$$

That means rigid rotations can survive free-slip or tangential-velocity boundary conditions without changing the Stokes balance.

### 1.4 Velocity nullspaces in 2D and 3D shell geometry

In a 2D annulus, the rigid-rotation nullspace is one-dimensional:

$$
\mathbf{r} = r \, \mathbf{e}_{\theta}.
$$

Equivalently, in Cartesian coordinates,

$$
\mathbf{r} = (-y, x) = r \, \hat{\theta}.
$$

In a 3D spherical shell, the rigid-rotation nullspace is three-dimensional, with basis

$$
\mathbf{r}_x = (0,-z,y), \qquad
\mathbf{r}_y = (z,0,-x), \qquad
\mathbf{r}_z = (-y,x,0).
$$

Any rigid rotation is a linear combination of these three fields.

### 1.5 Why velocity nullspaces matter

If a rigid rotation is left in the numerical solution, then:

- the velocity is not unique
- iterative solvers can drift in the null direction
- velocity error norms can look worse than they should
- benchmark comparisons become ambiguous because the analytical solution typically has zero angular momentum, while the numerical solution may contain an arbitrary rigid rotation

So, just as pressure needs a gauge, the velocity may need nullspace removal when the benchmark geometry and boundary conditions permit rigid rotations.

### 1.6 The projection formulas used to remove velocity nullspaces

If the numerical velocity $\mathbf{u}_h$ contains one null mode $\mathbf{r}$, remove it by an $L^2$ projection:

$$
c = \frac{\int_{\Omega} \mathbf{r} \cdot \mathbf{u}_h \, dV}
         {\int_{\Omega} \mathbf{r} \cdot \mathbf{r} \, dV},
\qquad
\mathbf{u}_h \leftarrow \mathbf{u}_h - c \mathbf{r}.
$$

If there are several modes $\{\mathbf{r}_i\}$, solve the Gram system

$$
G_{ij} = \int_{\Omega} \mathbf{r}_i \cdot \mathbf{r}_j \, dV,
\qquad
b_i = \int_{\Omega} \mathbf{r}_i \cdot \mathbf{u}_h \, dV,
\qquad
G \mathbf{c} = \mathbf{b},
$$

then subtract

$$
\mathbf{u}_h \leftarrow \mathbf{u}_h - \sum_i c_i \mathbf{r}_i.
$$

This is the mathematically clean way to remove only the nullspace part of the velocity.

## 2. How Each Benchmark Paper Deals With Them

## 2.1 Thieulot And Puckett (2018): Annulus Benchmark

This paper derives an exact incompressible Stokes solution in a 2D annulus with tangential flow on the inner and outer boundaries.

### Pressure normalization in the paper

The paper fixes the pressure constant explicitly in the axisymmetric case $k=0$. In the derivation, the authors write

$$
p(r,\theta)|_{k=0} = l(r) = \rho_0 g_r (R_2 - r),
$$

and state that this comes from imposing

$$
p(r,\theta) = 0 \quad \text{at } r = R_2.
$$

So for the $k=0$ case, the pressure gauge is fixed by prescribing the pressure at the outer boundary.

For $k>0$, the analytical pressure is the non-axisymmetric part

$$
p(r,\theta) = k h(r)\sin(k\theta) + \rho_0 g_r (R_2-r).
$$

In the numerical section, the authors use $\rho_0 = 0$, so the pressure is purely sinusoidal in $\theta$. Its angular mean is then zero automatically. The paper does not spend time discussing a separate post-processing pressure normalization for those non-axisymmetric cases.

### Velocity nullspace in the paper

The paper does not treat rigid rotation as a separate numerical issue. The reason is that the benchmark prescribes the analytical velocity on both annulus boundaries. That anchors the tangential motion, so a free rigid-body rotation is not left available in the benchmark formulation used in the paper.

### Practical reading of the paper

For this benchmark paper:

- pressure gauge is fixed explicitly for the $k=0$ derivation by setting pressure at the outer boundary
- for $k>0$ with $\rho_0 = 0$, the analytical pressure already has zero angular mean
- no separate rotational nullspace removal is emphasized because the boundary velocity is prescribed

## 2.2 Thieulot (2017): Spherical-Shell Benchmark

This paper derives an exact spherical-shell Stokes solution with tangential velocity on the inner and outer boundaries and a radial viscosity profile.

### Pressure normalization in the paper

This paper is the clearest of the three about pressure gauge freedom. In the numerical setup section, the author says:

- pressure is determined only up to a constant
- the codes can use either a surface normalization or a volume normalization
- for this particular analytical pressure field, both of those averages are already zero

So the paper’s point is:

$$
\text{surface-average pressure} = 0,
\qquad
\text{volume-average pressure} = 0
$$

for the analytical benchmark field, which means the particular pressure gauge chosen by the code does not affect the comparison.

The paper also remarks that the analytical pressure is zero at both $r=R_1$ and $r=R_2$, and interprets it as an overpressure relative to a background lithostatic pressure.

### Velocity nullspace in the paper

The same section says that the boundary conditions preclude a pure rotational mode of numerical origin. The logic is simple: the analytical velocity is imposed on both shell boundaries, so the rigid-body rotation freedom is already suppressed by the boundary data.

So in the paper’s benchmark design:

- pressure gauge freedom exists in principle, but does not matter because the analytical field already satisfies the standard zero-average gauges
- rigid rotation is not expected to survive because the boundary velocities are prescribed

## 2.3 Kramer, Davies, And Wilson (2021): Cylindrical And Spherical Shell Benchmarks

This paper is the most explicit about both issues.

### Pressure normalization in the paper

In Section 3.4, the paper states that the pressure is defined only up to an arbitrary constant and that the analytical pressure has zero mean. Therefore, for comparison they subtract the volume-averaged numerical pressure:

$$
p \leftarrow p - \frac{\int_{\Omega_h} p}{\int_{\Omega_h} 1}.
$$

This is exactly a zero-mean pressure normalization.

### Velocity nullspace in the paper

The paper then says that for free-slip cases in 2D, an arbitrary rigid rotation

$$
(-y,x) = r \, \hat{\theta}
$$

may be added to the velocity solution, so they project it out:

$$
\mathbf{u} \leftarrow \mathbf{u}
- \frac{\int_{\Omega_h} r \hat{\theta}\cdot \mathbf{u}}
        {\int_{\Omega_h} r^2}
  r \hat{\theta}.
$$

In 3D they subtract the three rotational rigid-body modes in the same way.

### A very important extra point from the paper

Kramer et al. also distinguish between two different things:

1. the constant-pressure gauge freedom
2. the discontinuity of the analytical pressure in the delta-function cases

Those are not the same issue.

The first issue is fixed by subtracting a constant mean pressure.

The second issue is an approximation issue: if the analytical pressure is discontinuous across an internal interface, then a continuous pressure space cannot represent it optimally. That is why the paper shows reduced convergence for the delta-function cases with continuous pressure and recovers optimal convergence with discontinuous pressure.

So this paper treats pressure normalization and velocity nullspace removal as essential benchmark post-processing steps, not optional details.

### One more solver detail from the paper

The paper also explains that the linear system itself has these zero modes. During the Krylov solve, they remove the algebraic null modes with an $\ell^2$ projection each iteration, but they still apply the final $L^2$ projections after the solve so the computed field matches the analytical benchmark field in the physically correct gauge.

## 3. How The Current Underworld Benchmark Scripts Implement Them

This section describes the current scripts in this repository, not the papers.

In a few places the Underworld implementation matches the papers exactly, and in a few places it differs. I call those differences out explicitly.

## 3.1 `benchmarks/annulus/ex_stokes_thieulot.py`

### Pressure normalization in Underworld

This script has two branches.

For `params.uw_k == 0`, it adds explicit pressure Dirichlet conditions on both annulus boundaries:

```python
if params.uw_k == 0:
    stokes.add_condition(
        p_soln.field_id,
        "dirichlet",
        sp.Matrix([0]),
        mesh.boundaries.Lower.name,
        components=(0,),
    )
    stokes.add_condition(
        p_soln.field_id,
        "dirichlet",
        sp.Matrix([0]),
        mesh.boundaries.Upper.name,
        components=(0,),
    )
```

For `params.uw_k != 0`, it does not pin pressure before the solve. Instead it subtracts the domain-average pressure afterwards:

```python
def subtract_pressure_mean(mesh, pressure_var):
    p_int = uw.maths.Integral(mesh, pressure_var.sym[0]).evaluate()
    volume = uw.maths.Integral(mesh, 1.0).evaluate()
    pressure_var.data[:, 0] -= p_int / volume

if params.uw_k != 0:
    subtract_pressure_mean(mesh, p_soln)
```

### Velocity nullspace removal in Underworld

The annulus rigid rotation is constructed as

```python
v_theta_fn_xy = r * mesh.CoordinateSystem.rRotN.T * sp.Matrix((0, 1))
```

This is the 2D rigid-rotation field

$$
\mathbf{r} = (-y, x) = r \, \hat{\theta}.
$$

and then projected out:

```python
def subtract_rigid_rotation(mesh, velocity_var, rotation_mode):
    mode_int = uw.maths.Integral(mesh, rotation_mode.dot(velocity_var.sym)).evaluate()
    mode_norm = uw.maths.Integral(mesh, rotation_mode.dot(rotation_mode)).evaluate()
    dv = uw.function.evaluate((mode_int / mode_norm) * rotation_mode, velocity_var.coords)
    velocity_var.data[...] -= np.asarray(dv).reshape(velocity_var.data.shape)

if params.uw_k != 0:
    subtract_pressure_mean(mesh, p_soln)
    if params.uw_bc_type == "natural":
        subtract_rigid_rotation(mesh, v_soln, v_theta_fn_xy)
```

### Relation to the paper

This is consistent with the annulus paper in spirit, but not identical in form.

- The paper explicitly fixes pressure for the $k=0$ derivation. The UW script does the same by direct pressure pinning.
- For $k>0$, the script uses a zero-mean gauge after the solve.
- The script removes rigid rotation only for the natural-BC path, which is a practical numerical correction in Underworld rather than something emphasized in the paper.

## 3.2 `benchmarks/annulus/ex_stokes_kramer.py`

### Pressure normalization in Underworld

This script always subtracts the domain-average pressure after the solve:

```python
def subtract_pressure_mean(mesh, pressure_var):
    p_int = uw.maths.Integral(mesh, pressure_var.sym[0]).evaluate()
    volume = uw.maths.Integral(mesh, 1.0).evaluate()
    p_mean = p_int / volume
    pressure_var.data[:, 0] -= p_mean

subtract_pressure_mean(mesh, p_uw)
```

This matches the gauge choice described in Kramer et al.

### Velocity nullspace removal in Underworld

The 2D rigid-rotation mode is built as

```python
v_theta_fn_xy = r_uw * mesh.CoordinateSystem.rRotN.T * sp.Matrix((0, 1))
```

Again, this is just

$$
\mathbf{r} = (-y, x) = r \, \hat{\theta}.
$$

and then removed for free-slip cases:

```python
def subtract_rigid_rotation(mesh, velocity_var, rotation_mode):
    mode_int = uw.maths.Integral(mesh, rotation_mode.dot(velocity_var.sym)).evaluate()
    mode_norm = uw.maths.Integral(mesh, rotation_mode.dot(rotation_mode)).evaluate()
    coeff = mode_int / mode_norm
    dv = uw.function.evaluate(coeff * rotation_mode, velocity_var.coords)
    velocity_var.data[...] -= dv.reshape(velocity_var.data.shape)

if freeslip:
    subtract_rigid_rotation(mesh, v_uw, v_theta_fn_xy)
```

### Relation to the paper

This is close to the Kramer paper:

- pressure is shifted to zero mean
- the annulus free-slip rigid rotation is projected out

The main difference is implementation style, not mathematics.

## 3.3 `benchmarks/spherical/ex_stokes_thieulot.py`

### Pressure normalization in Underworld

This script currently applies three successive pressure shifts after the solve.

First it subtracts the domain-average pressure:

```python
def subtract_pressure_mean(mesh, pressure_var):
    p_int_local = float(uw.maths.Integral(mesh, pressure_var.sym[0]).evaluate())
    volume_local = float(uw.maths.Integral(mesh, 1.0).evaluate())

    p_int = float(uw.mpi.comm.allreduce(p_int_local))
    volume = float(uw.mpi.comm.allreduce(volume_local))
    pressure_var.data[:, 0] -= p_int / volume
```

Then it shifts pressure to remove the mean on a named boundary:

```python
def subtract_surface_pressure_mean(mesh, pressure_var, boundary_name):
    p_bd_int = uw.maths.BdIntegral(mesh=mesh, fn=pressure_var.sym[0], boundary=boundary_name).evaluate()
    bd_measure = uw.maths.BdIntegral(mesh=mesh, fn=1.0, boundary=boundary_name).evaluate()
    pressure_var.data[:, 0] -= p_bd_int / bd_measure

subtract_pressure_mean(mesh, p_soln)
subtract_surface_pressure_mean(mesh, p_soln, mesh.boundaries.Upper.name)
subtract_surface_pressure_mean(mesh, p_soln, mesh.boundaries.Lower.name)
```

There is also an alternate gauge helper that is currently commented out:

```python
def enforce_pressure_reference(mesh, pressure_var, pressure_reference):
    target_mean = float(pressure_reference)
    p_local = np.asarray(pressure_var.data[:, 0], dtype=np.float64)
    local_sum = float(p_local.sum())
    local_count = int(p_local.size)

    global_sum = uw.mpi.comm.allreduce(local_sum)
    global_count = uw.mpi.comm.allreduce(local_count)

    current_mean = global_sum / max(global_count, 1)
    shift = current_mean - target_mean
    pressure_var.data[:, 0] -= shift
```

### Velocity nullspace removal in Underworld

The script defines the full 3D rigid-rotation basis:

```python
rotation_modes = [
    sp.Matrix([0, -z, y]),
    sp.Matrix([z, 0, -x]),
    sp.Matrix([-y, x, 0]),
]
```

and implements the correct Gram-matrix projector:

```python
def subtract_rigid_rotations(mesh, velocity_var, rotation_modes):
    velocity_expr = sp.Matrix(velocity_var.sym).T
    nmodes = len(rotation_modes)

    gram = np.zeros((nmodes, nmodes))
    rhs = np.zeros(nmodes)

    for i, mode_i in enumerate(rotation_modes):
        rhs[i] = uw.maths.Integral(mesh, mode_i.dot(velocity_expr)).evaluate()
        for j, mode_j in enumerate(rotation_modes):
            gram[i, j] = uw.maths.Integral(mesh, mode_i.dot(mode_j)).evaluate()

    coeffs = np.linalg.solve(gram, rhs)
```

But the actual call is commented out:

```python
# subtract_rigid_rotations(mesh, v_soln, rotation_modes)
```

### Relation to the paper

This is where the current UW script differs most from the paper.

In the paper:

- either surface or volume pressure normalization is acceptable because the analytical field satisfies both
- the prescribed boundary velocities prevent a rigid-rotation mode

In the current UW script:

- pressure is shifted several times in succession
- the full rigid-rotation remover exists, but is not currently activated

So the script contains the right machinery, but its active behavior is not exactly the same as the benchmark discussion in the paper.

## 3.4 `benchmarks/spherical/ex_stokes_kramer.py`

### Pressure normalization in Underworld

The current script does not explicitly normalize pressure after the solve and does not add an explicit pressure Dirichlet condition.

So, at present, there is no visible counterpart of Kramer’s

$$
p \leftarrow p - \frac{\int_{\Omega_h} p}{\int_{\Omega_h} 1}
$$

step in this Underworld script.

### Velocity nullspace removal in Underworld

The script defines one explicit rigid-rotation combination:

```python
null_mode_expr = (
    sp.Matrix(((0, 1, 1), (-1, 0, 1), (-1, -1, 0)))
    * mesh.CoordinateSystem.N.T
)
```

and projects it out:

```python
I0 = uw.maths.Integral(mesh, null_mode_expr.dot(v_uw.sym))
norm = I0.evaluate()
I0.fn = null_mode_expr.dot(null_mode_expr)
vnorm = I0.evaluate()

dv = uw.function.evaluate(norm * null_mode_expr, v_uw.coords) / vnorm
v_uw.data[...] -= np.asarray(dv).reshape(v_uw.data.shape)
```

The matrix used above is antisymmetric, so this is indeed one rigid-body rotation of the form $\boldsymbol{\omega} \times \mathbf{x}$.

### Relation to the paper

Here the difference from Kramer et al. is important.

The paper says:

- subtract the volume-averaged pressure
- in 3D free-slip cases subtract the full three-dimensional rotational subspace

The current Underworld script does:

- no explicit pressure normalization
- removal of only one rigid-rotation combination, not all three

So this script captures part of the paper’s nullspace handling, but not the full paper prescription.

## 4. Short Comparison Table

| Benchmark | Pressure handling in paper | Velocity nullspace handling in paper | Current UW implementation |
| --- | --- | --- | --- |
| Annulus Thieulot | $k=0$ gauge fixed by imposing pressure at outer boundary; non-axisymmetric pressure otherwise used directly | not emphasized separately because boundary velocity is prescribed | `k=0`: pressure Dirichlet pinning; `k>0`: subtract domain mean; remove 2D rigid rotation for natural BC |
| Spherical Thieulot | pressure determined up to constant, but both surface and volume averages are already zero for the analytical field | boundary conditions preclude pure rotation | subtract volume mean and two boundary means; 3D rotation remover exists but is commented out |
| Kramer shell benchmarks | subtract volume-average pressure | remove 2D rotation in cylindrical free-slip, remove three 3D rigid rotations in spherical free-slip | annulus script matches closely; spherical script currently removes only one rotation and does not explicitly shift pressure |

## 5. Main Takeaway

The main mathematical ideas are simple:

- pressure must be put in a chosen gauge because Stokes pressure is only defined up to a constant
- free-slip shell problems can admit rigid-body rotations, so the velocity may also need nullspace removal

The main benchmark-specific takeaway is:

- the annulus Thieulot paper mostly avoids nullspace trouble by prescribing boundary velocity
- the spherical Thieulot paper says pressure gauge choice does not matter for that exact solution and that prescribed boundary velocity removes rotation
- the Kramer paper treats both pressure normalization and velocity-nullspace projection as explicit, necessary benchmark steps

The main Underworld takeaway is:

- `annulus/ex_stokes_kramer.py` is the closest match to the Kramer paper’s gauge/nullspace handling
- `annulus/ex_stokes_thieulot.py` is reasonable and practical, with an extra numerical rigid-rotation cleanup for natural BCs
- `spherical/ex_stokes_thieulot.py` and `spherical/ex_stokes_kramer.py` currently differ in important ways from the most direct paper formulations
