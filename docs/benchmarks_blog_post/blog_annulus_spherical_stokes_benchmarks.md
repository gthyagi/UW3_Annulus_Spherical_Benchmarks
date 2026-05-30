# Benchmarking Stokes Flow in Underworld3 Using Annulus and Spherical-Shell Geometries

This post introduces a suite of annulus and spherical-shell Stokes benchmarks reproduced using Underworld3.<sup>[1](#ref-1)</sup> These benchmark problems have previously been implemented in several numerical codes, but here we bring them together within a single framework to highlight both the strengths and practical challenges of curved-domain finite-element modelling. The post is aimed at researchers working on geodynamics numerical modelling. Rather than focusing on heavy mathematical derivations or low-level implementation details, the goal is to provide an intuitive and practical guide to what each benchmark is designed to test, what numerical behaviour should be expected, and how to interpret the results. By the end, readers should have a clear understanding of the purpose of these benchmarks and the key ideas behind verifying Stokes flow in curved geometries.

## What Is Benchmarking and Why Is It Important?

Benchmarking is the process of testing a numerical method or software implementation against problems with known analytical solutions or well-established reference results. In computational geodynamics, benchmarks are essential because they help verify that a code correctly solves the governing equations before it is applied to complex Earth-science problems where the true solution is unknown. A good benchmark does more than produce a visually reasonable result; it tests numerical convergence behaviour, boundary-condition implementation, mesh geometry, and solver robustness under controlled conditions. Curved-domain benchmarks such as annulus and spherical-shell Stokes problems are particularly important because they expose numerical challenges that do not appear in simple Cartesian geometries. Successfully reproducing benchmark results therefore builds confidence that a numerical framework can model real-world problems with reliable and physically meaningful approximations.

## What Are the Stokes Equations?

The Stokes equations describe the slow, viscous flow of fluids in situations where inertial forces are negligible compared with viscous forces. In geodynamics, this approximation is widely used because rocks in the Earth's mantle deform extremely slowly over geological timescales and behave like highly viscous fluids. The incompressible Stokes equations are written as

$$
\begin{aligned}
-\nabla \cdot \left(2 \eta \dot{\varepsilon}(\mathbf{u})\right)
+ \nabla p &= \rho \mathbf{g}, \\
\nabla \cdot \mathbf{u} &= 0 .
\end{aligned}
$$

Here $\mathbf{u}$ is velocity, $p$ is pressure, $\eta$ is viscosity, $\rho$ is density, and $\mathbf{g}$ is gravity. The first equation represents conservation of momentum, balancing viscous stresses, pressure gradients, and body forces. The second equation enforces mass conservation through incompressibility. Solving these equations allows us to model mantle convection, lithospheric deformation, subduction, and many other large-scale Earth processes. Although the equations appear compact, solving them accurately in curved geometries with complex boundary conditions and variable material properties is computationally challenging, which is why benchmark problems are important.

## Why Do We Need Curved Geometries?

Curved geometries are important in geodynamics because the Earth itself is curved. Many large-scale Earth processes, such as mantle convection, subduction, plume dynamics, and lithospheric deformation, occur within spherical or shell-like domains rather than simple rectangular boxes. While Cartesian geometries are useful for developing intuition and testing numerical methods, they cannot fully represent radial gravity, curved boundaries, or global-scale flow patterns. Curved-domain models also introduce additional numerical challenges, including geometric approximation errors, coordinate transformations, and the accurate implementation of free-slip or zero-slip boundary conditions on non-planar surfaces. As a result, annulus and spherical-shell benchmarks provide a more realistic and demanding test of Stokes solvers.

## Benchmark Suite

In this work, we reproduce four widely used Stokes benchmark suites in curved geometries using Underworld3.<sup>[1](#ref-1)</sup> The first is the Thieulot--Puckett annulus benchmark, which provides a smooth analytical solution in an annulus and is mainly used to test optimal finite-element convergence behaviour.<sup>[2](#ref-2)</sup> The second is the Kramer annulus benchmark, which extends the problem to include both smooth volumetric forcing and singular delta-function forcing on an internal interface, together with free-slip and zero-slip boundary conditions.<sup>[3](#ref-3)</sup> We then consider the spherical-shell counterparts of these problems: the Thieulot spherical benchmark, which tests smooth Stokes flow in spherical geometry with both constant and radially varying viscosity,<sup>[4](#ref-4)</sup> and the Kramer spherical benchmark, which again introduces internal interface forcing and reduced solution regularity.<sup>[3](#ref-3)</sup> Together, these four benchmark suites test curved geometries, pressure treatment, mesh approximation, boundary-condition implementation, smooth and singular forcing, and convergence behaviour in both two- and three-dimensional Stokes flow.

<figure>
  <img src="../benchmarks_banner_figure/combined_density_distribution_figures.jpg" alt="Combined analytical fields for the annulus and spherical-shell Stokes benchmarks" style="width:100%; height:auto;">
  <figcaption>Analytical benchmark fields used in the annulus and spherical-shell Stokes benchmark suite. The panels collect the Thieulot--Puckett annulus, Kramer annulus, Thieulot spherical-shell, and Kramer spherical-shell cases.</figcaption>
</figure>

## What Do We Measure? Error Quantification

The benchmark comparisons use $L_2$-norm errors because they give a single quantitative measure of the difference between the numerical solution and the analytical solution over the whole domain. For a computed field $q_h$ and analytical field $q^*$, the absolute volume error is

$$
E_{L_2}(q)
=
\left(
\int_{\Omega} |q_h-q^*|^2\,\mathrm{d}\Omega
\right)^{1/2}.
$$

The corresponding relative volume error is

$$
E_{L_2}^{\mathrm{rel}}(q)
=
\frac{E_{L_2}(q)}{\|q^*\|_{L_2}}
=
\left(
\frac{\int_{\Omega} |q_h-q^*|^2\,\mathrm{d}\Omega}
     {\int_{\Omega} |q^*|^2\,\mathrm{d}\Omega}
\right)^{1/2}.
$$

The relative form is useful when comparing velocity and pressure errors across different benchmark cases because it normalises the error by the size of the analytical solution. For pressure, the numerical and analytical fields are first compared in the same pressure gauge, since incompressible Stokes pressure is determined only up to an additive constant.

Boundary pressure errors are measured separately on the inner and outer surfaces. For a boundary $\Gamma \in \{\Gamma_{\mathrm{inner}},\Gamma_{\mathrm{outer}}\}$, the absolute pressure-trace error is

$$
E_{L_2,\Gamma}(p)
=
\left(
\int_{\Gamma} |p_h-p^*|^2\,\mathrm{d}\Gamma
\right)^{1/2}.
$$

Where the analytical boundary pressure has a nonzero $L_2$ norm, the relative boundary pressure error is

$$
E_{L_2,\Gamma}^{\mathrm{rel}}(p)
=
\left(
\frac{\int_{\Gamma} |p_h-p^*|^2\,\mathrm{d}\Gamma}
     {\int_{\Gamma} |p^*|^2\,\mathrm{d}\Gamma}
\right)^{1/2}.
$$

Convergence is measured by comparing errors across successively refined meshes:

$$
\mathrm{rate}
=
\frac{
\log\left(E_{h_1}/E_{h_2}\right)
}{
\log\left(h_1/h_2\right)
}.
$$

Here $h$ is the characteristic cell size. If the mesh is uniformly refined so that $h_2=h_1/2$, this becomes

$$
\mathrm{rate}
=
\log_2\left(\frac{E_{h_1}}{E_{h_2}}\right).
$$

The expected behaviour is simple: as $h$ decreases, the error should decrease. On a log-log convergence plot, a method with $E \sim C h^r$ appears approximately as a straight line with slope $r$. For smooth Stokes solutions, stable mixed finite-element pairs have well-defined optimal convergence expectations; for example, Taylor--Hood $P_2\times P_1$ commonly gives third-order velocity and second-order pressure convergence in the volume $L_2$ norm when the geometry and solution are sufficiently smooth.<sup>[5](#ref-5)</sup> Singular forcing cases are different: the solution is less regular near the internal interface, so reduced convergence rates are expected even when the solver is implemented correctly.

## What Do We Observe? Results of Benchmarks in Underworld3

### Volumetric Convergence

The benchmark results show that Underworld3 reproduces the expected convergence behaviour for both annulus and spherical-shell Stokes problems. For smooth analytical solutions, such as the Thieulot annulus and spherical benchmarks, the Taylor--Hood $P_2 \times P_1$ discretisation achieves close to the theoretically expected convergence rates, with approximately third-order velocity convergence and second-order pressure convergence in the volume $L_2$ norm. Higher-order element pairs further improve accuracy, while lower-order discretisations show the expected reduction in convergence order.

For the Kramer benchmarks with smooth forcing, the velocity convergence is closer to second order because the curved geometry is represented using linear meshes, consistent with the original benchmark studies.<sup>[3](#ref-3)</sup> In the delta-function forcing cases, the convergence rates reduce significantly because the internal singular interface lowers the regularity of the analytical solution. Overall, the volumetric results demonstrate that Underworld3 captures both optimal convergence behaviour for smooth problems and the expected degradation in accuracy for singular forcing cases.

<figure>
  <img src="figures/figure_5_thieulot_annulus_convergence.jpg" alt="Velocity and pressure convergence for the Thieulot--Puckett annulus benchmark" style="width:100%; height:auto;">
  <figcaption>Velocity and pressure convergence for the Thieulot--Puckett annulus benchmark.</figcaption>
</figure>

<figure>
  <img src="figures/figure_3_kramer_annulus_convergence.jpg" alt="Velocity and pressure convergence for the Kramer annulus benchmark" style="width:100%; height:auto;">
  <figcaption>Velocity and pressure convergence for the Kramer annulus benchmark.</figcaption>
</figure>

<figure>
  <img src="figures/figures_4_5_thieulot_convergence.jpg" alt="Velocity and pressure convergence for the Thieulot spherical-shell benchmark" style="width:100%; height:auto;">
  <figcaption>Velocity and pressure convergence for the Thieulot spherical-shell benchmark.</figcaption>
</figure>

<figure>
  <img src="figures/figure_4_kramer_spherical_convergence.jpg" alt="Velocity and pressure convergence for the Kramer spherical-shell benchmark" style="width:100%; height:auto;">
  <figcaption>Velocity and pressure convergence for the Kramer spherical-shell benchmark.</figcaption>
</figure>

### Boundary Pressure Convergence

Boundary diagnostics provide a more local and often more sensitive measure of solver accuracy in curved geometries. The volume pressure norm measures an error over the full domain, but boundary pressure errors isolate the pressure trace on the surfaces where boundary conditions, radial normal stresses, and traction-related diagnostics are evaluated. This is useful because geometric approximation errors and boundary quadrature errors can be more visible on curved inner and outer boundaries than in a volume-averaged metric.

For the Thieulot--Puckett annulus benchmark, the $P_2\times P_1$ boundary pressure errors decrease close to the expected $O(h^2)$ trend. The inner and outer boundaries do not have identical error constants, which is acceptable: the two curves have different radii, different geometric representation errors, and different boundary integration paths.

<figure>
  <img src="figures/figure_p2p1_boundary_pressure_convergence.jpg" alt="Boundary pressure convergence for the Thieulot--Puckett annulus benchmark" style="width:100%; height:auto;">
  <figcaption>Absolute boundary pressure error convergence for the Thieulot--Puckett annulus benchmark using the $P_2\times P_1$ discretisation. Solid curves denote the inner boundary and dashed curves denote the outer boundary.</figcaption>
</figure>

For the Kramer annulus benchmark, the boundary pressure errors converge close to second order for the smooth cases. The delta-function cases also show stronger boundary pressure convergence than the corresponding volume pressure norm because the reduced regularity is localized at the internal interface rather than on the annulus boundaries.

<figure>
  <img src="figures/figure_boundary_pressure_convergence.jpg" alt="Boundary pressure convergence for the Kramer annulus benchmark" style="width:100%; height:auto;">
  <figcaption>Relative boundary pressure error convergence for the Kramer annulus benchmark. Solid curves denote the inner boundary and dashed curves denote the outer boundary.</figcaption>
</figure>

The Thieulot spherical-shell benchmark gives the corresponding smooth three-dimensional boundary test. The boundary-pressure panel shows that the absolute pressure-trace errors for both $m=-1$ and $m=3$ decrease systematically with refinement. These errors are shown together with radial normal-stress errors because the stress diagnostic combines pressure and velocity-gradient errors at the same curved spherical boundaries.

<figure>
  <img src="figures/boundary_metric_convergence.jpg" alt="Boundary pressure and radial normal-stress convergence for the Thieulot spherical-shell benchmark" style="width:100%; height:auto;">
  <figcaption>Boundary radial normal-stress and absolute boundary pressure error convergence for the Thieulot spherical-shell benchmark. The right panel shows $E_{L_2,\Gamma}(p)$ on the inner and outer spherical boundaries.</figcaption>
</figure>

For the Kramer spherical-shell benchmark, the current boundary figure reports boundary velocity and radial normal-stress convergence for the free-slip delta-function case. A separate boundary pressure-trace convergence plot is not part of the current spherical Kramer article outputs. The radial normal-stress diagnostic is included here because it is pressure-sensitive: $\sigma_{rr}$ contains the pressure contribution and therefore tests pressure recovery together with velocity-gradient recovery on the spherical boundaries.

<figure>
  <img src="figures/figure_7_kramer_boundary_convergence.jpg" alt="Boundary velocity and radial normal-stress convergence for the Kramer spherical-shell benchmark" style="width:100%; height:auto;">
  <figcaption>Boundary velocity and radial normal-stress convergence for the Kramer spherical-shell benchmark. This is the available pressure-sensitive boundary diagnostic for the spherical Kramer case.</figcaption>
</figure>

Overall, the boundary diagnostics complement the volume $L_2$ pressure errors. Smooth benchmark cases recover approximately second-order pressure-trace convergence for the $P_2\times P_1$ pair, while singular forcing primarily affects the volume pressure norm through reduced regularity at the internal interface. Persistent differences between inner- and outer-boundary error constants should therefore be interpreted as geometric and boundary-evaluation effects rather than as a failure of pressure normalisation.

The full details are in the article PDFs:

- [Thieulot--Puckett annulus benchmark](../benchmarks_figures_and_articles/annulus/thieulot/thieulot_annulus_benchmark_article.pdf)
- [Kramer annulus benchmark](../benchmarks_figures_and_articles/annulus/kramer/kramer_annulus_benchmark_article.pdf)
- [Thieulot spherical-shell benchmark](../benchmarks_figures_and_articles/spherical/thieulot/thieulot_spherical_benchmark_article.pdf)
- [Kramer spherical-shell benchmark](../benchmarks_figures_and_articles/spherical/kramer/kramer_spherical_benchmark_article.pdf)

Those articles contain the analytical fields, discretisation choices, convergence tables, boundary diagnostics, and full reference details.

## References

1. <span id="ref-1"></span>Moresi, L., Mansour, J., Giordani, J., Knepley, M., Knight, B., Graciosa, J. C., Gollapalli, T., Lu, N., and Beucher, R.: Underworld3: Mathematically Self-Describing Modelling in Python for Desktop, HPC and Cloud, *Journal of Open Source Software*, 10, 7831, [https://doi.org/10.21105/joss.07831](https://doi.org/10.21105/joss.07831), 2025.
2. <span id="ref-2"></span>Thieulot, C. and Puckett, E. G.: Incompressible Stokes flow in an annulus: An analytical solution and numerical benchmark, preprint submitted to *Computers & Geosciences*, [https://www.math.ucdavis.edu/~egp/PUBLICATIONS/JOURNAL_ARTICLES/SUBMITTED/CAPT-EGP-2018.pdf](https://www.math.ucdavis.edu/~egp/PUBLICATIONS/JOURNAL_ARTICLES/SUBMITTED/CAPT-EGP-2018.pdf), 2018.
3. <span id="ref-3"></span>Kramer, S. C., Davies, D. R., and Wilson, C. R.: Analytical solutions for mantle flow in cylindrical and spherical shells, *Geoscientific Model Development*, 14, 1899--1919, [https://doi.org/10.5194/gmd-14-1899-2021](https://doi.org/10.5194/gmd-14-1899-2021), 2021.
4. <span id="ref-4"></span>Thieulot, C.: Analytical solution for viscous incompressible Stokes flow in a spherical shell, *Solid Earth*, 8, 1181--1191, [https://doi.org/10.5194/se-8-1181-2017](https://doi.org/10.5194/se-8-1181-2017), 2017.
5. <span id="ref-5"></span>Boffi, D., Brezzi, F., and Fortin, M.: *Mixed Finite Element Methods and Applications*, Springer Series in Computational Mathematics, Springer, [https://doi.org/10.1007/978-3-642-36519-5](https://doi.org/10.1007/978-3-642-36519-5), 2013.
