# Underworld3 Curved-Domain Stokes Benchmarks

The accuracy of a finite-element Stokes solution in a curved domain depends on both the approximation of the velocity and pressure fields and the numerical representation of the boundaries. Geometric approximation and boundary-condition enforcement can therefore affect the observed convergence, even when the finite-element pair has well-established approximation properties. Analytical benchmarks in annulus and spherical-shell geometries provide a controlled setting for measuring these combined errors and assessing how they decrease under mesh refinement. This assessment complements Cartesian benchmarks by examining the solver in the curved geometries for which it is intended.

This technical note evaluates Underworld3 (UW3)<sup><a href="#ref-1">1</a></sup> using four analytical Stokes benchmark families. Accuracy is quantified using volume $L_2$-norm errors in velocity and pressure, supplemented by boundary $L_2$-norm errors in pressure and radial normal stress where available. Smooth solutions are used to assess the convergence orders associated with the chosen finite-element spaces, whereas delta-function interface forcing tests the lower rates expected when solution regularity is reduced. The volume and boundary measures provide complementary assessments of numerical accuracy, allowing convergence within the domain and on its inner and outer boundaries to be examined separately. The detailed benchmark articles provide the analytical derivations and complete convergence tables.

## Benchmark Problems and Numerical Objectives

The four benchmark families are selected to examine the effects of geometry, forcing regularity, viscosity variation, and boundary conditions. The Thieulot solutions provide smooth reference fields with prescribed boundary velocities, while the Kramer solutions compare smooth and interface-localised forcing under free-slip and zero-slip conditions.

On the domain $\Omega$, all benchmark cases satisfy the steady incompressible Stokes equations

$$
\begin{aligned}
-\nabla \cdot \boldsymbol{\sigma} &= \rho\mathbf{g}, \\
\nabla \cdot \mathbf{u} &= 0.
\end{aligned}
$$

The Cauchy stress and strain-rate tensors are defined by

$$
\begin{aligned}
\boldsymbol{\sigma}
&= -p\mathbf{I}+2\eta\boldsymbol{\varepsilon}(\mathbf{u}), \\
\boldsymbol{\varepsilon}(\mathbf{u})
&= \frac{1}{2}\left(\nabla\mathbf{u}+(\nabla\mathbf{u})^{\mathsf{T}}\right).
\end{aligned}
$$

Here $\mathbf{u}$ is velocity, $p$ is pressure, $\eta$ is viscosity, $\rho$ is density, $\mathbf{g}$ is gravitational acceleration, and $\mathbf{I}$ is the identity tensor.

For smooth forcing, the body force $\rho\mathbf{g}$ is distributed throughout the domain. In the Kramer delta-function cases, the force is concentrated on an internal circular or spherical interface rather than distributed through a layer of finite thickness. The momentum equation is then understood in a weak sense, and the resulting reduction in solution regularity changes the expected convergence rates.<sup><a href="#ref-3">3</a></sup>

The boundary conditions also distinguish the families. The Thieulot cases prescribe the analytical velocity on both inner and outer boundaries, with zero normal flow and nonzero tangential motion. The Kramer cases impose either free slip, meaning zero normal velocity and zero tangential traction, or zero slip, meaning that all velocity components vanish. Free slip therefore leaves tangential velocity unconstrained rather than prescribing its analytical value.<sup><a href="#ref-3">3</a></sup>

Table 1 summarises these distinctions and the principal comparisons made in this study. Detailed parameter values are given with the corresponding results and in the benchmark articles.

Table 1. Benchmark families, forcing, boundary conditions, and numerical objectives.

| Benchmark | Forcing and viscosity | Boundary condition | Principal comparison |
|---|---|---|---|
| Thieulot–Puckett annulus<sup><a href="#ref-2">2</a></sup> | Smooth; constant viscosity | Prescribed analytical velocity | Effect of element pair on volume and boundary-pressure convergence |
| Kramer annulus<sup><a href="#ref-3">3</a></sup> | Smooth or delta function; constant viscosity | Free slip or zero slip | Effect of forcing and boundary conditions on volume and boundary-pressure errors |
| Thieulot spherical shell<sup><a href="#ref-4">4</a></sup> | Smooth; constant or radially varying viscosity | Prescribed analytical velocity | Effect of viscosity variation on velocity, pressure, and radial normal-stress errors |
| Kramer spherical shell<sup><a href="#ref-3">3</a></sup> | Smooth or delta function; constant viscosity | Free slip or zero slip | Volume convergence for both forcing types; boundary velocity and radial normal-stress errors |

Figure 1 illustrates the spatial structure of the analytical fields and the distinction between volumetric and interface-localised forcing.

<div align="center">

<img src="../benchmarks_banner_figure/combined_density_distribution_figures.jpg" alt="Thieulot density, velocity magnitude, and pressure fields alongside Kramer smooth and interface-localised density forcing in annulus and spherical-shell geometries" width="75%">

Figure 1. Representative analytical fields for the four benchmark families. The Thieulot panels show density, velocity magnitude, and pressure; the spherical-shell panel compares constant viscosity (first row) with radially varying viscosity (second row). The Kramer panels show smooth volumetric density forcing in their first rows and interface-localised forcing in their second rows. The distributions illustrate the angular structure that the mesh must resolve and the localisation of the singular forcing. Parameter labels follow each benchmark's notation: $k$ denotes the azimuthal harmonic in the Thieulot–Puckett annulus; in the Kramer panels, $n$ or $(l,m)$ specifies angular structure, while $k$ controls the radial profile of smooth forcing.

</div>

## Spatial Discretisation and Solver Configuration

### Mesh Geometry and Finite-Element Spaces

UW3 uses Gmsh to generate unstructured triangular meshes for the annulus and tetrahedral meshes for the spherical shell.<sup><a href="#ref-6">6</a></sup> The benchmark meshes use linear geometric representation: circular boundaries are approximated by straight edges and spherical boundaries by planar triangular faces. This geometric approximation is distinct from the polynomial degree used to interpolate velocity and pressure. A quadratic velocity space on these meshes does not make the boundary representation quadratic.

The convergence plots use the Gmsh characteristic cell size $h$, a target local element size rather than a measured mesh statistic. Each value of $h$ defines an independently generated mesh, not a nested subdivision of the preceding mesh. Actual element sizes vary with geometry and mesh-quality constraints; nominal halving of $h$ therefore describes the refinement sequence without implying that every element diameter is halved.

The Kramer benchmarks and the Thieulot spherical-shell benchmark use the $P_2\times P_1$ Taylor–Hood pair, with continuous quadratic velocity and continuous linear pressure. The Thieulot–Puckett annulus additionally compares $P_1\times P_0$, $P_1\times P_1$, $P_2\times P_0$, $P_2\times P_1^{\mathrm{disc}}$, and $P_3\times P_2$. Velocity is continuous in all these pairs. Pressure is discontinuous for the cellwise-constant $P_0$ spaces and for $P_1^{\mathrm{disc}}$, labelled $P_{-1}$ in the figures; the remaining pressure spaces are continuous. Here the subscript denotes polynomial degree, except for the shorthand $P_{-1}$, which denotes discontinuous linear pressure.

### Boundary Enforcement and Pressure Reference

The Thieulot convergence submissions prescribe analytical boundary velocities through essential conditions, and the Kramer zero-slip cases likewise impose zero velocity through essential conditions. The documented Kramer free-slip submissions select Nitsche enforcement of impermeability, with radial unit normals and parameter $\gamma=10$. This specifies the numerical enforcement of the free-slip condition defined above, rather than prescribing the analytical tangential velocity. Later tests of other free-slip implementations should be distinguished from these convergence configurations.

The constant-pressure mode of the Stokes formulation is accounted for through PETSc's nullspace machinery during the linear solve.<sup><a href="#ref-7">7</a></sup> This treatment of the singular system is distinct from fixing the pressure reference for error evaluation. After solution, the numerical pressure is shifted to the analytical zero-mean gauge. The corresponding zero-mean representatives are defined by

$$
\begin{aligned}
p_h^{\circ}
&= p_h-\frac{1}{|\Omega|}\int_{\Omega}p_h\,\mathrm{d}\Omega, \\
p_*^{\circ}
&= p_*-\frac{1}{|\Omega|}\int_{\Omega}p_*\,\mathrm{d}\Omega.
\end{aligned}
$$

Here $|\Omega|$ denotes the domain area in the annulus and volume in the spherical shell. Each field has one domain-wide pressure shift, which is retained when evaluating errors on both boundaries; the inner and outer boundary means are not removed separately. The Kramer free-slip scripts also subtract rigid-rotation components from the computed velocity before comparison with the analytical solution. This velocity adjustment is not applied to the prescribed-velocity Thieulot cases.

### Linear Solver and Tolerances

UW3 assembles the Stokes system from symbolic expressions and passes the resulting saddle-point problem to PETSc.<sup><a href="#ref-7">7</a></sup> The documented benchmark configuration uses linear-solve mode (`snes_type=ksponly`), since viscosity, forcing, and boundary data are prescribed. Except for $P_1\times P_0$, the scripts configure flexible GMRES (`fgmres`) with velocity–pressure field-split preconditioning and multigrid subsolves. The $P_1\times P_0$ path uses direct LU factorisation in serial, or GMRES with additive-Schwarz preconditioning and local LU subsolves under MPI.

The parameter `uw_stokes_tol` is assigned to `ksp_rtol`, which controls the relative reduction of PETSc's selected linear residual norm; `ksp_atol` is set to zero. This residual criterion is not an $L_2$-norm error bound on velocity or pressure.<sup><a href="#ref-7">7</a></sup> The documented convergence submissions request $10^{-9}$ for the annulus benchmarks and the Thieulot spherical shell, and $10^{-8}$ for the Kramer spherical shell. The spherical Thieulot plotting script also explicitly selects runs labelled with a $10^{-9}$ tolerance. These are documented sweep settings, not confirmation of the options used by every archived run; reproducing a plotted series requires its recorded command-line options and software revision, including any overrides.

The benchmark implementations are available in the [annulus Thieulot](../../benchmarks/annulus/ex_stokes_thieulot.py), [annulus Kramer](../../benchmarks/annulus/ex_stokes_kramer.py), [spherical Thieulot](../../benchmarks/spherical/ex_stokes_thieulot.py), and [spherical Kramer](../../benchmarks/spherical/ex_stokes_kramer.py) scripts. The associated [convergence submissions](../../production_scripts) specify the mesh sequences and requested solver settings.

## Error Measures and Convergence Rates

Numerical accuracy is assessed by comparing the computed and analytical fields over the domain and separately on its inner and outer boundaries. The definitions below retain the integration symbols $\Omega$ and $\Gamma$ used in the benchmark articles. In the scripts, these integrals are evaluated by numerical quadrature on the computational domain $\Omega_h$ and its boundary $\Gamma_h$, with the analytical fields evaluated at the same physical quadrature points. The reported errors therefore reflect the numerical solution on the represented geometry; they are not exact integrals over the circular or spherical boundaries and do not isolate geometric error from the other error sources.

### Volume Errors

For a numerical field $q_h$ and analytical reference $q_*$, the absolute volume $L_2$-norm error is

$$
E_{L_2}(q) = \left(
\int_{\Omega}\lVert q_h-q_*\rVert^2\,\mathrm{d}\Omega
\right)^{1/2},
$$

and the relative error, when the analytical norm is nonzero, is

$$
\begin{aligned}
E_{L_2}^{\mathrm{rel}}(q)
&= \frac{E_{L_2}(q)}{\lVert q_*\rVert_{L_2}} \\
&= \left(
\frac{\int_{\Omega}\lVert q_h-q_*\rVert^2\,\mathrm{d}\Omega}
     {\int_{\Omega}\lVert q_*\rVert^2\,\mathrm{d}\Omega}
\right)^{1/2}.
\end{aligned}
$$

The pointwise norm denotes absolute value for a scalar and the Euclidean norm for a vector. For pressure, $q_h$ and $q_*$ are the gauge-normalised fields defined above. The absolute error retains the field amplitude and integration-domain scaling, whereas the relative error is dimensionless and measures error against the magnitude of the analytical field. The volume convergence figures report relative errors in velocity and pressure.

### Boundary Errors

Boundary errors are evaluated independently on $\Gamma_{\mathrm{inner}}$ and $\Gamma_{\mathrm{outer}}$. For either boundary $\Gamma$ and a scalar or vector quantity $q$, the absolute error is

$$
E_{L_2,\Gamma}(q) = \left(
\int_{\Gamma}\lVert q_h-q_*\rVert^2\,\mathrm{d}\Gamma
\right)^{1/2}.
$$

When the analytical boundary norm is nonzero, the corresponding relative error is

$$
E_{L_2,\Gamma}^{\mathrm{rel}}(q) = \left(
\frac{\int_{\Gamma}\lVert q_h-q_*\rVert^2\,\mathrm{d}\Gamma}
     {\int_{\Gamma}\lVert q_*\rVert^2\,\mathrm{d}\Gamma}
\right)^{1/2}.
$$

These definitions apply to pressure $p$, velocity $\mathbf{u}$, and radial normal stress $\sigma_{rr}$, according to the quantity reported in each figure. Pressure and the pressure contribution to normal stress retain the domain-wide gauge defined above; no independent boundary-mean subtraction is performed. Normal-stress error includes velocity-gradient contributions as well as pressure error and is therefore distinct from pressure-trace error.

If the analytical boundary norm vanishes, the relative error is undefined and the absolute measure must be used, as for velocity on a zero-slip boundary. A zero signed mean pressure does not imply a zero pressure $L_2$ norm, because the norm integrates the squared field rather than the signed field.

The absolute boundary norm also depends on the boundary measure $|\Gamma|$, a length in the annulus and an area in the spherical shell. A supplementary root-mean-square measure removes this measure-dependent scaling:

$$
E_{\mathrm{RMS},\Gamma}(q)
= \frac{E_{L_2,\Gamma}(q)}{\sqrt{|\Gamma|}}.
$$

This is the root-mean-square error on the boundary, not the arithmetic mean of the error magnitude. It retains the units of $q$ and does not remove geometric approximation errors or normalise by the analytical field amplitude. It is introduced here to aid interpretation of differences between inner and outer error magnitudes; the existing figures show the stated absolute or relative $L_2$-norm errors, not RMS errors.

### Convergence Rates

For a fixed benchmark case, element pair, and error measure, let $E_h>0$ denote the error at target cell size $h$. The pairwise rate between successive resolutions $h_1>h_2$ is

$$
r_{\mathrm{pair}} = \frac{\log(E_{h_1}/E_{h_2})}
     {\log(h_1/h_2)}.
$$

For the nominal halving used here, this reduces to

$$
r_{\mathrm{pair}}=\log_2\left(\frac{E_h}{E_{h/2}}\right).
$$

Pairwise rates show how convergence changes along the refinement sequence. They are used in the detailed refinement tables and for the finest-mesh Thieulot spherical-shell rates quoted below. A single summary rate over several resolutions can instead be obtained by an unweighted least-squares fit of $\log E_h$ against $\log h$, including an intercept. For the selected data points,

$$
\begin{aligned}
x_i &= \log h_i, \qquad y_i=\log E_{h_i}, \\
r_{\mathrm{fit}}
&= \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}
        {\sum_i (x_i-\bar{x})^2},
\end{aligned}
$$

where $\bar{x}$ and $\bar{y}$ are the means over the fitted data. This slope summarises the selected refinement range and is not, in general, the arithmetic mean of the pairwise rates. The Thieulot–Puckett annulus summary table in the detailed article uses this fit separately for velocity and pressure, for each element pair and each $k=1,4,8$; it does not average rates across the harmonic cases.

The saved annulus tables cover $h=1/8$ to $1/512$ for $P_1\times P_0$, $P_2\times P_0$, and $P_2\times P_1$; $1/8$ to $1/256$ for $P_1\times P_1$; and $1/8$ to $1/128$ for $P_2\times P_{-1}$ and $P_3\times P_2$. The $P_3\times P_2$ sequence stops at $h=1/128$ because the higher computational resource requirements prevented runs at $h=1/256$; no results at that resolution were omitted from the analysis. The summary rates use all finite, positive errors in each reported sequence, rather than a subset selected automatically as an asymptotic regime.

On log–log axes, $E_h\simeq Ch^r$ has slope $r$; halving $h$ reduces the error by approximately $2^r$ within that regime. Interpreting either a pairwise or fitted slope requires examining the selected range: coarse-mesh behaviour and fine-mesh error saturation can both obscure the asymptotic order. Reference lines labelled $O(h^r)$ indicate comparison slopes, not fitted rates for the numerical data.

For a sufficiently smooth Stokes solution, a stable $P_2\times P_1$ Taylor–Hood discretisation can attain third-order velocity and second-order pressure convergence in the volume $L_2$ norm under the usual regularity assumptions, provided that geometry, quadrature, boundary treatment, and solver error do not limit the rate.<sup><a href="#ref-5">5</a></sup> Delta-function interface forcing reduces solution regularity and lowers the expected asymptotic rates. These volume-error estimates do not by themselves establish the convergence orders of boundary traces or radial normal stress; those quantities are assessed separately in the boundary results.

## Volume-Error Convergence

Figures 2–5 compare the relative volume errors $E_{L_2}^{\mathrm{rel}}(\mathbf{u})$ and $E_{L_2}^{\mathrm{rel}}(p)$ as the target cell size decreases. The results are considered separately for each benchmark family because the solution regularity, boundary conditions, and tested finite-element spaces differ.

### Thieulot–Puckett Annulus

The smooth annulus solution exhibits a clear convergence hierarchy among the tested element pairs (Figure 2). For each azimuthal harmonic $k=1,4,8$, the $P_2\times P_1$ pair gives whole-sequence fitted rates of approximately 3.0 for velocity and 2.0 for pressure. The $P_3\times P_2$ pair follows approximately fourth-order velocity and third-order pressure trends over most of the retained refinements. Its $k=1$ velocity rate decreases on the finest retained mesh, reducing the whole-sequence velocity fit to 3.7, compared with 4.0 for $k=4$ and $k=8$. This is a departure from the preceding fourth-order trend, rather than evidence that the error has reached a resolution-independent plateau.

The other element pairs show different velocity and pressure behaviour. The $P_2\times P_0$ pair gives fitted rates of approximately 2.0 and 1.0, respectively. Both $P_1\times P_1$ and $P_2\times P_{-1}$ reduce velocity error at roughly second order, but pressure converges more slowly, particularly for $P_2\times P_{-1}$. In contrast, the $P_1\times P_0$ errors approach nonzero levels rather than displaying sustained convergence. These observations characterise the individual benchmark calculations; convergence for one analytical solution does not establish uniform inf–sup stability or a general convergence guarantee for an element pair.<sup><a href="#ref-5">5</a></sup>

<div align="center">

<img src="figures/figure_5_thieulot_annulus_convergence.jpg" alt="Velocity and pressure convergence for the Thieulot–Puckett annulus benchmark" width="75%">

Figure 2. Relative volume $L_2$-norm errors in velocity and pressure for the Thieulot–Puckett annulus benchmark. Curves compare the tested mixed finite-element pairs for $k=1,4,8$; reference lines indicate comparison slopes.

</div>

### Kramer Annulus

The free-slip and zero-slip cases show the same qualitative distinction between smooth and interface-localised forcing (Figure 3). For smooth forcing, pressure errors approach second-order convergence, while velocity errors are approximately second order for the lower azimuthal wavenumbers $n=2$ and $n=8$. The $n=32$ cases have larger errors and more pronounced coarse-mesh departures from these trends, consistent with the finer angular structure that must be resolved.

The principal explanation for the reduced smooth velocity order is the use of linear rather than quadratic/isoparametric geometry. Kramer et al. demonstrate this effect in their Figure 5: repeating the smooth annulus calculations on linear meshes reduces velocity convergence from third to second order, while pressure convergence remains second order.<sup><a href="#ref-3">3</a></sup> The UW3 results agree with this comparison. Quadratic velocity interpolation alone does not remove the error introduced by the straight-sided approximation of the annulus boundaries.

For delta-function forcing, the refined-mesh trends approach $O(h^{1.5})$ for velocity and $O(h^{0.5})$ for pressure under both boundary conditions. These lower rates are consistent with the reduced solution regularity associated with the internal forcing interface.<sup><a href="#ref-3">3</a></sup> They should therefore be distinguished from the reduced velocity order in the smooth cases, where the same regularity limitation does not apply.

<div align="center">

<img src="figures/figure_3_kramer_annulus_convergence.jpg" alt="Velocity and pressure convergence for the Kramer annulus benchmark" width="75%">

Figure 3. Relative volume $L_2$-norm errors for the $P_2\times P_1$ Kramer annulus benchmark. The first two rows show free-slip cases and the last two show zero-slip cases, with velocity followed by pressure in each pair. Columns show delta-function forcing and smooth forcing with $k=2$ and $k=8$. Symbols denote azimuthal wavenumbers $n=2,8,32$.

</div>

### Thieulot Spherical Shell

The spherical-shell results compare constant and radially varying viscosity using the $P_2\times P_1$ pair (Figure 4). In this benchmark, viscosity varies as $\eta(r)\propto r^{m+1}$, so $m=-1$ gives constant viscosity and $m=3$ gives a radial power-law profile. Both cases exhibit decreasing velocity and pressure errors throughout the reported refinement sequence.

Between $h=1/64$ and $1/128$, the pairwise velocity rates are 3.03 for $m=-1$ and 3.11 for $m=3$; the pressure rate is 2.20 in both cases. These values are consistent with velocity convergence near third order and pressure convergence approaching second order over the tested meshes. The variable-viscosity case has less uniform coarse-mesh pressure rates, but comparable fine-mesh orders are obtained for both viscosity profiles. The quoted rates describe the final refinement interval, not a fit over the complete sequence.

<div align="center">

<img src="figures/figures_4_5_thieulot_convergence.jpg" alt="Velocity and pressure convergence for the Thieulot spherical-shell benchmark" width="75%">

Figure 4. Relative volume $L_2$-norm errors in velocity and pressure for the $P_2\times P_1$ Thieulot spherical-shell benchmark. The $m=-1$ case has constant viscosity, whereas $m=3$ has radially varying viscosity.

</div>

### Kramer Spherical Shell

The spherical Kramer results show a similar separation between the smooth and singular forcing families (Figure 5). For smooth forcing, pressure converges close to second order under both free-slip and zero-slip conditions. Velocity rates are predominantly between 1.9 and 2.1, with larger coarse-mesh departures for some higher spherical harmonics. Thus, as in the annulus, the smooth velocity results remain closer to second than third order over the reported resolutions.

We interpret the second-order smooth velocity convergence primarily as a consequence of using linear rather than quadratic/isoparametric geometry. The planar boundary faces in the present meshes do not provide the higher-order geometric representation used by Kramer et al., who obtain third-order smooth velocity convergence in the spherical shell.<sup><a href="#ref-3">3</a></sup> The UW3 spherical results are consistent with the geometric limitation demonstrated explicitly in their annulus comparison.

The delta-function cases instead approach $O(h^{1.5})$ velocity and $O(h^{0.5})$ pressure convergence on the finest meshes, consistent with the regularity-limited behaviour of the analytical solutions.<sup><a href="#ref-3">3</a></sup> Their lower rates are therefore interpreted in terms of interface regularity, in contrast to the primarily geometric limitation of the smooth cases.

<div align="center">

<img src="figures/figure_4_kramer_spherical_convergence.jpg" alt="Velocity and pressure convergence for the Kramer spherical-shell benchmark" width="75%">

Figure 5. Relative volume $L_2$-norm errors for the $P_2\times P_1$ Kramer spherical-shell benchmark. The first two rows show free-slip cases and the last two show zero-slip cases, with velocity followed by pressure in each pair. Columns separate delta-function and smooth forcing. Colours denote spherical-harmonic degree and order $(l,m)$; all smooth cases use $k=l+1$.

</div>

## Boundary Pressure and Normal-Stress Convergence

Volume-error convergence alone does not establish boundary accuracy. Errors near a boundary may contribute relatively little to a domain integral, while boundary pressure and stress remain quantities of interest in their own right. The pressure trace measures the boundary restriction of the pressure field. Radial normal stress additionally involves velocity derivatives:

$$
\begin{aligned}
\sigma_{rr}
&= \mathbf{e}_r\cdot\boldsymbol{\sigma}\mathbf{e}_r \\
&= -p+2\eta\,\mathbf{e}_r\cdot
\boldsymbol{\varepsilon}(\mathbf{u})\mathbf{e}_r.
\end{aligned}
$$

Here $\mathbf{e}_r$ is the radial unit vector. Errors in $\sigma_{rr}$ combine pressure and velocity-gradient contributions, which cannot be separated from the stress norm alone. The reported stress and boundary-velocity errors use the same absolute or relative $L_2$ definitions with the corresponding field substituted for pressure.

### Annulus Pressure Traces

For the Thieulot–Puckett annulus, the $P_2\times P_1$ boundary pressure error converges at approximately second order. Differences between the inner and outer error constants may reflect boundary length, analytical pressure variation, local mesh geometry, and quadrature. The available curves do not separate these contributions. Their similar convergence rates indicate comparable orders of boundary-pressure accuracy, without requiring equal absolute error magnitudes.

<div align="center">

<img src="figures/figure_p2p1_boundary_pressure_convergence.jpg" alt="Boundary pressure convergence for the Thieulot–Puckett annulus benchmark" width="75%">

Figure 6. Absolute pressure-trace errors for the $P_2\times P_1$ Thieulot–Puckett annulus benchmark. Solid curves denote the inner boundary and dashed curves the outer boundary.

</div>

The smooth Kramer annulus cases also approach second-order boundary pressure convergence. For delta-function forcing, the boundary pressure trace can converge faster than the corresponding volume pressure norm. This behaviour is consistent with localisation of the singular forcing on an internal interface, although elliptic coupling, boundary geometry, quadrature, and pressure-trace approximation also contribute. The available results demonstrate the difference in convergence behaviour but do not isolate a single controlling mechanism.

<div align="center">

<img src="figures/figure_boundary_pressure_convergence.jpg" alt="Boundary pressure convergence for the Kramer annulus benchmark" width="75%">

Figure 7. Relative inner- and outer-boundary pressure errors for the Kramer annulus benchmark. Rows separate free-slip and zero-slip cases; columns separate delta-function and smooth forcing.

</div>

### Spherical-Shell Boundary Diagnostics

For the Thieulot spherical shell, pressure-trace errors decrease systematically on both boundaries for $m=-1$ and $m=3$. The $\sigma_{rr}$ metric is more demanding because it combines pressure error with velocity-derivative error. Inner and outer values also differ through their surface areas, analytical amplitudes, local facets, and quadrature weights. The results quantify this asymmetry but do not identify a dominant contribution.

<div align="center">

<img src="figures/boundary_metric_convergence.jpg" alt="Boundary pressure and radial normal-stress convergence for the Thieulot spherical-shell benchmark" width="75%">

Figure 8. Relative radial normal-stress errors and absolute pressure-trace errors for the $P_2\times P_1$ Thieulot spherical-shell benchmark. Solid and dashed curves denote the inner and outer boundaries.

</div>

The Kramer spherical-shell boundary results report velocity and radial normal stress for the free-slip delta-function case; a separate pressure-trace norm is not included. The $\sigma_{rr}$ metric remains sensitive to pressure but also includes velocity-gradient error and must therefore be interpreted as a coupled stress diagnostic.

<div align="center">

<img src="figures/figure_7_kramer_boundary_convergence.jpg" alt="Boundary velocity and radial normal-stress convergence for the Kramer spherical-shell benchmark" width="75%">

Figure 9. Boundary velocity and radial normal-stress convergence for the free-slip, delta-function Kramer spherical-shell benchmark. This is a pressure-sensitive stress diagnostic, not a pressure-trace error.

</div>

## Summary

The volume $L_2$-norm errors show that UW3 reproduces the expected convergence hierarchy for the smooth Thieulot benchmarks over the reported refinement sequences. The $P_2\times P_1$ pair achieves approximately third-order velocity and second-order pressure convergence in both geometries. In the annulus, the $P_3\times P_2$ pair follows approximately fourth- and third-order trends over most retained refinements, whereas $P_2\times P_0$ gives second- and first-order convergence. These results demonstrate the different levels of velocity and pressure accuracy obtained with the tested finite-element spaces.

The Kramer benchmarks distinguish two principal limitations on convergence. With delta-function forcing, the refined-mesh trends approach $O(h^{1.5})$ for velocity and $O(h^{0.5})$ for pressure, consistent with the reduced regularity at the forcing interface. With smooth forcing, the absence of quadratic/isoparametric geometry is the principal explanation for the predominantly second-order velocity convergence in both the annulus and spherical shell; pressure retains its expected second-order trend. The boundary results complement these volume measures by resolving accuracy on the inner and outer surfaces separately. Pressure-trace errors assess pressure directly, while radial normal-stress errors assess the combined pressure and velocity-gradient contributions to boundary stress.

## Limitations

The conclusions are restricted to the reported benchmark parameters, mesh sequences, finite-element spaces, boundary treatments, and solver settings. Rates measured over a finite refinement range do not necessarily represent the asymptotic order, particularly when coarse meshes remain under-resolved or the rate changes on the finest meshes. Similarly, convergence for a particular analytical solution does not establish uniform inf–sup stability or a general accuracy guarantee for an element pair.

The geometric explanation for the smooth Kramer velocity rates is supported by the published annulus comparison; its application to the UW3 spherical results is an inference, not a separate controlled geometry comparison. Boundary enforcement, quadrature, and algebraic-solver errors are treated as secondary contributions whose magnitudes have not been isolated here. A UW3 comparison with quadratic/isoparametric meshes would test the expected recovery of third-order smooth velocity convergence. Linear geometry does not impose a universal second-order ceiling for every benchmark, as the higher rates obtained for the Thieulot solutions demonstrate.

Boundary coverage is also incomplete. For the Kramer spherical shell, the reported boundary results concern velocity and radial normal stress in the free-slip delta-function case. A separate pressure-trace norm is needed to assess boundary pressure directly, and additional cases are required to extend that comparison to smooth forcing and zero-slip conditions. More generally, unequal inner- and outer-boundary errors are measured, but the present results do not quantify the contribution of each proposed mechanism to that difference.

## Detailed Benchmark Articles

The complete analytical definitions, convergence tables, and discussion are provided in the benchmark articles:

- [Thieulot–Puckett annulus benchmark](https://github.com/gthyagi/UW3_Annulus_Spherical_Benchmarks/blob/main/docs/benchmarks_figures_and_articles/annulus/thieulot/thieulot_annulus_benchmark_article.pdf)
- [Kramer annulus benchmark](https://github.com/gthyagi/UW3_Annulus_Spherical_Benchmarks/blob/main/docs/benchmarks_figures_and_articles/annulus/kramer/kramer_annulus_benchmark_article.pdf)
- [Thieulot spherical-shell benchmark](https://github.com/gthyagi/UW3_Annulus_Spherical_Benchmarks/blob/main/docs/benchmarks_figures_and_articles/spherical/thieulot/thieulot_spherical_benchmark_article.pdf)
- [Kramer spherical-shell benchmark](https://github.com/gthyagi/UW3_Annulus_Spherical_Benchmarks/blob/main/docs/benchmarks_figures_and_articles/spherical/kramer/kramer_spherical_benchmark_article.pdf)

## Code and Data Availability

The benchmark scripts, analytical solutions, and convergence-submission settings are available in the [benchmark repository at revision 30bcb9d](https://github.com/gthyagi/UW3_Annulus_Spherical_Benchmarks/tree/30bcb9de214aace1fb8d33c5f5798bd637f5c4ec). The repository also contains the [plotting scripts and tabulated errors](https://github.com/gthyagi/UW3_Annulus_Spherical_Benchmarks/tree/30bcb9de214aace1fb8d33c5f5798bd637f5c4ec/docs/benchmarks_figures_and_articles), alongside the detailed benchmark articles. This fixed snapshot provides a reference for the code and documented configurations; the simulations themselves span earlier revisions. The UW3 source is maintained in the [Underworld3 repository](https://github.com/underworldcode/underworld3).

The archived `benchmark_metrics.h5` files record the benchmark-repository commit and command line used to generate the metrics. For the spherical benchmarks, these identify checkpoint-based post-processing rather than the original solve. The files do not record the UW3 or PETSc build revisions, so they cannot establish the exact solver build used for each simulation. Raw HDF5 metrics, meshes, and solution checkpoints are not included in the public repository; the tabulated errors provide the publicly available numerical results.

## References

1. <span id="ref-1"></span>Moresi, L., Mansour, J., Giordani, J., Knepley, M., Knight, B., Graciosa, J. C., Gollapalli, T., Lu, N., and Beucher, R.: Underworld3: Mathematically Self-Describing Modelling in Python for Desktop, HPC and Cloud, *Journal of Open Source Software*, 10, 7831, [https://doi.org/10.21105/joss.07831](https://doi.org/10.21105/joss.07831), 2025.
2. <span id="ref-2"></span>Thieulot, C. and Puckett, E. G.: Incompressible Stokes flow in an annulus: An analytical solution and numerical benchmark, preprint submitted to *Computers & Geosciences*, [https://www.math.ucdavis.edu/~egp/PUBLICATIONS/JOURNAL_ARTICLES/SUBMITTED/CAPT-EGP-2018.pdf](https://www.math.ucdavis.edu/~egp/PUBLICATIONS/JOURNAL_ARTICLES/SUBMITTED/CAPT-EGP-2018.pdf), 2018.
3. <span id="ref-3"></span>Kramer, S. C., Davies, D. R., and Wilson, C. R.: Analytical solutions for mantle flow in cylindrical and spherical shells, *Geoscientific Model Development*, 14, 1899–1919, [https://doi.org/10.5194/gmd-14-1899-2021](https://doi.org/10.5194/gmd-14-1899-2021), 2021.
4. <span id="ref-4"></span>Thieulot, C.: Analytical solution for viscous incompressible Stokes flow in a spherical shell, *Solid Earth*, 8, 1181–1191, [https://doi.org/10.5194/se-8-1181-2017](https://doi.org/10.5194/se-8-1181-2017), 2017.
5. <span id="ref-5"></span>Boffi, D., Brezzi, F., and Fortin, M.: *Mixed Finite Element Methods and Applications*, Springer Series in Computational Mathematics, Springer, [https://doi.org/10.1007/978-3-642-36519-5](https://doi.org/10.1007/978-3-642-36519-5), 2013.
6. <span id="ref-6"></span>Geuzaine, C. and Remacle, J.-F.: Gmsh: A 3-D finite element mesh generator with built-in pre- and post-processing facilities, *International Journal for Numerical Methods in Engineering*, 79, 1309–1331, [https://doi.org/10.1002/nme.2579](https://doi.org/10.1002/nme.2579), 2009.
7. <span id="ref-7"></span>Balay, S., Abhyankar, S., Adams, M. F., Benson, S., Brown, J., Brune, P., Buschelman, K., Constantinescu, E. M., Dalcin, L., Dener, A., Eijkhout, V., Faibussowitsch, J., Gropp, W. D., Hapla, V., Isaac, T., Jolivet, P., Karpeev, D., Kaushik, D., Knepley, M. G., and others: PETSc/TAO Users Manual, ANL-21/39 Rev. 3.21, Argonne National Laboratory, [https://doi.org/10.2172/2337606](https://doi.org/10.2172/2337606), 2024.
