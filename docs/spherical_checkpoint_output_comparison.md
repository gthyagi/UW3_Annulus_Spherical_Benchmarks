# Spherical Benchmark Output Modes: `write_timestep()` vs `write_checkpoint()`

## Goal

Document the difference between the two UW3 output paths used during the
spherical benchmark restart / postprocessing work:

- `mesh.write_timestep(...)`
- `mesh.write_checkpoint(...)`

This note focuses on what each method writes, what can be reloaded reliably,
and the practical tradeoffs for the spherical Thieulot benchmark workflow.

## Short Summary

- `write_timestep()` is the visualisation / remap-oriented output path.
- `write_checkpoint()` is the restart-oriented output path.
- They are related in UW3, but they are not equivalent.
- `write_timestep()` writes enough information for XDMF visualisation and
  geometric remap reloads.
- `write_checkpoint()` writes extra PETSc DM / section metadata needed for
  exact field restart.

## Relationship in Current UW3

- `petsc_save_checkpoint()` is effectively a wrapper around
  `write_timestep()`.
- `write_checkpoint()` is a separate write path.

So:

- `write_timestep()` and `petsc_save_checkpoint()` are closely related
- `write_checkpoint()` is the distinct restart-format writer

## What `write_timestep()` Writes

For a spherical Stokes solve with mesh, velocity, and pressure:

- one mesh HDF5 file
- one HDF5 file per mesh variable
- optional XDMF file
- ParaView compatibility groups such as `/vertex_fields`

Typical file set:

- `output.mesh.00000.h5`
- `output.mesh.Velocity.00000.h5`
- `output.mesh.Pressure.00000.h5`
- `output.mesh.00000.xdmf`

Typical variable-file structure:

- `/fields/<name>`
- `/fields/coordinates`
- `/vertex_fields/...`

This is sufficient for:

- visualisation
- coordinate-based field reconstruction with `read_timestep(...)`

This does **not** include the full PETSc subDM section/order metadata needed
for exact FE vector reconstruction.

## What `write_checkpoint()` Writes

For the same solve, `write_checkpoint()` writes:

- one mesh HDF5 file
- one restart/checkpoint HDF5 file per mesh variable by default in the current
  benchmark workflow

Typical file set:

- `checkout.mesh.00000.h5`
- `checkout.Velocity.00000.h5`
- `checkout.Pressure.00000.h5`

If combined output is requested with `separate_variable_files=False`, the
variable data can instead be written into one combined checkpoint file.

The restart file includes PETSc restart metadata under paths like:

- `/topologies/uw_mesh/dms/uw_mesh/order`
- `/topologies/uw_mesh/dms/uw_mesh/section/...`
- `/topologies/uw_mesh/dms/Velocity/order`
- `/topologies/uw_mesh/dms/Velocity/section/atlasDof`
- `/topologies/uw_mesh/dms/Velocity/section/atlasOff`
- `/topologies/uw_mesh/dms/Velocity/vecs/Velocity/Velocity`
- `/topologies/uw_mesh/dms/Pressure/order`
- `/topologies/uw_mesh/dms/Pressure/section/atlasDof`
- `/topologies/uw_mesh/dms/Pressure/section/atlasOff`
- `/topologies/uw_mesh/dms/Pressure/vecs/Pressure/Pressure`

This is the metadata missing from `write_timestep()` for exact restart.

## Why `checkout.mesh.00000.h5` and `output.mesh.00000.h5` Are Different

These two files are produced by different output paths and should not be
treated as interchangeable restart files.

`checkout.mesh.00000.h5` is produced by:

```python
mesh.write_checkpoint(
    "checkout",
    outputPath=str(output_dir),
    meshVars=[v_soln, p_soln],
)
```

This is the PETSc DMPlex checkpoint / restart path. It stores the mesh in
PETSc's restart-oriented DMPlex HDF5 layout and is intended to be used with
checkpoint variable files such as:

- `checkout.Velocity.00000.h5`
- `checkout.Pressure.00000.h5`

`output.mesh.00000.h5` is produced by:

```python
mesh.write_timestep(
    "output",
    index=0,
    meshVars=[v_soln, p_soln],
    outputPath=str(output_dir),
)
```

This is the timestep / visualisation path. It stores geometry, topology,
labels, and separate variable files intended for XDMF visualisation and
coordinate-based reload via `read_timestep(...)`.

The key distinction is:

- `checkout.mesh.00000.h5` belongs to the exact PETSc checkpoint/restart
  workflow
- `output.mesh.00000.h5` belongs to the visualisation/remap workflow

So a clean comparison must avoid mixing mesh-reload effects with
field-reload effects. To isolate the field reload only, both field reload paths
must be tested on the same mesh/partition.

For example:

```text
same checkout.mesh.00000.h5
  + checkpoint variable reload from checkout.Velocity.00000.h5
  + timestep/KDTree reload from output.mesh.Velocity.00000.h5
```

This isolates the difference between `read_checkpoint(...)` and
`read_timestep(...)`. If instead one run uses `checkout.mesh.00000.h5` and
another uses `output.mesh.00000.h5`, derivative-based quantities such as
`stokes.tau`, `tau_rr`, and `sigma_rr` may differ because the mesh/reload path
is also changing.

## XDMF Difference

- `write_timestep()` can generate XDMF output
- `write_checkpoint()` does **not** generate XDMF output

So if both restart and visualisation are needed, the current clean approach is
to write both formats.

## Small Mac Test

Test setup:

- spherical Thieulot benchmark
- `8` MPI ranks
- `uw_cellsize = 1/8`
- temporary copied driver used only for testing

### File Sizes

`write_timestep()` outputs:

- `output.mesh.00000.h5`: `1.9M`
- `output.mesh.Velocity.00000.h5`: `871K`
- `output.mesh.Pressure.00000.h5`: `161K`
- total: about `2.9M`

`write_checkpoint()` outputs:

- `restart.mesh.0.h5`: `1.9M`
- `restart.checkpoint.00000.h5`: `3.5M`
- total: about `5.4M`

So in this test:

- checkpoint output was about `1.9x` larger overall
- the extra size came from the restart metadata in
  `restart.checkpoint.00000.h5`

Also, in this test:

- `restart.mesh.0.h5`
- `output.mesh.00000.h5`

were byte-identical, so the meaningful extra restart information lived in the
checkpoint file, not the mesh file.

## Same-Mesh Reload Diagnostic

A later focused diagnostic compared both reload methods on the same mesh and
MPI partition:

- mesh: `checkout.mesh.00000.h5`
- checkpoint field reload:
  - `v_soln.read_checkpoint("checkout.Velocity.00000.h5", data_name="Velocity")`
  - `p_soln.read_checkpoint("checkout.Pressure.00000.h5", data_name="Pressure")`
- timestep field reload:
  - `v_soln.read_timestep("output", "Velocity", 0, outputPath=...)`
  - `p_soln.read_timestep("output", "Pressure", 0, outputPath=...)`

Test setup:

- spherical Thieulot benchmark
- `8` MPI ranks
- `uw_cellsize = 1/8`
- both reloads evaluated on the same `checkout.mesh.00000.h5` mesh

Direct field data comparison:

| field | max abs diff | L2 abs diff | relative L2 diff |
|---|---:|---:|---:|
| `v_soln.data` | `2.17e-19` | `4.50e-19` | `1.67e-21` |
| `p_soln.data` | `0.0` | `0.0` | `0.0` |

Boundary integral numerator comparison before square root / normalization:

| boundary | checkpoint sigma numerator | timestep sigma numerator | sigma diff numerator | tau diff numerator |
|---|---:|---:|---:|---:|
| lower | `33.62487735638558` | `33.62487735638558` | `0.0` | `0.0` |
| upper | `0.23697408816503557` | `0.23697408816503557` | `0.0` | `0.0` |

Boundary DOF sample comparison:

| quantity | boundary | max abs diff |
|---|---|---:|
| `tau_rr` | lower | `7.11e-15` |
| `tau_rr` | upper | `1.78e-15` |
| `sigma_rr` | lower | `3.55e-15` |
| `sigma_rr` | upper | `1.78e-15` |

This shows that, on the same mesh/partition, checkpoint reload and
timestep/KDTree reload produce the same velocity, pressure, `tau_rr`, and
`sigma_rr` to roundoff for the small `1/8` case.

Therefore, a non-roundoff sigma difference observed between separate full
reload runs is not caused by different `v_soln.data` or `p_soln.data` values.
It is caused by comparing outputs that also changed the mesh/reload path:

- `checkout.mesh.00000.h5` PETSc checkpoint mesh
- `output.mesh.00000.h5` timestep/visualisation mesh

The checkpoint path remains the correct production path for exact restart and
large MPI runs because it avoids the memory-heavy KDTree reconstruction used by
`read_timestep(...)`.

### Reload Behavior

There are two different reload mechanisms involved here.

### `read_timestep(...)` reload

`read_timestep(...)` does **not** perform an exact PETSc restart.

Instead it:

1. opens the variable HDF5 file with `h5py`
2. reads the saved field values from `/fields/<name>`
3. reads the saved coordinates from `/fields/coordinates`
4. builds a KDTree on the saved coordinates
5. reconstructs values on the current mesh variable by coordinate-based
   matching / interpolation

So `read_timestep(...)` works because `write_timestep()` stores:

- field values
- field coordinates

It does **not** need PETSc section/order metadata.

In the spherical Thieulot test, this coordinate/KDTree-based reconstruction
reproduced the expected metrics.

Example baseline metrics:

```text
v_l2_norm               = 0.005901070392565955
p_l2_norm               = 0.2614689522200766
sigma_rr_l2_norm_lower  = 0.1900342889700841
```

### Exact FE vector restart attempt

An exact restart-style reload is a different mechanism.

That path would ideally:

1. rebuild the distributed mesh / subDM layout
2. restore PETSc section/order metadata
3. load the saved FE vector into the distributed PETSc global vector
4. scatter that global vector to the local FE storage

That requires metadata such as:

- subDM order
- section atlasDof
- section atlasOff

Direct PETSc vector load from the `write_timestep()` `/fields/...` datasets
did **not** reproduce the correct fields in this test.

Example incorrect direct-load metrics:

```text
v_l2_norm               = 1.3251666143181036
p_l2_norm               = 1.0475947618599912
sigma_rr_l2_norm_lower  = 3.650470590001755
```

That shows:

- `write_timestep()` files are sufficient for UW3's `read_timestep(...)`
  coordinate/KDTree reconstruction path
- `write_timestep()` files do not, by themselves, provide a reliable exact FE
  vector restart path in this workflow

### PETSc metadata reload test

A focused Mac test was also run against `write_checkpoint()` output using
PETSc's intended restart path:

1. load `restart.mesh.0.h5`
2. open `restart.checkpoint.00000.h5`
3. call `DMPlex.sectionLoad(...)`
4. call `DMPlex.globalVectorLoad(...)`

The first failure was a DM object-name mismatch:

```text
Object (dataset) "order" not stored in group /topologies/plex/dms/Velocity
```

That was fixed in the temporary loader by setting the runtime mesh DM name to
`uw_mesh`, matching the checkpoint file.

After that, the reload still failed inside `DMPlex.sectionLoad(...)`:

```text
Nonconforming object sizes
SF roots 6421 < pEnd 47112
```

The same class of failure occurred with:

- `mesh.sf`
- `mesh.sf0`
- `mesh.dm.getDefaultSF()`

So the checkpoint file contains useful PETSc metadata, but the current public
UW3/PETSc-level reload sequence is not yet cleanly usable from the benchmark
script.

The benchmark should therefore **not** add a fragile manual PETSc loader. The
right next fix is in UW3: provide a supported checkpoint reload helper that
loads mesh variables using the same topology/load SF and section layout expected
by `write_checkpoint()`.

Required UW3-side API shape:

```python
mesh = uw.discretisation.Mesh("restart.mesh.0.h5", ...)
velocity = uw.discretisation.MeshVariable(...)
pressure = uw.discretisation.MeshVariable(...)

velocity.load_from_checkpoint(
    "restart.checkpoint.00000.h5",
    data_name="Velocity",
)
pressure.load_from_checkpoint(
    "restart.checkpoint.00000.h5",
    data_name="Pressure",
)
```

Internally, that helper should own the exact PETSc sequence:

- use the correct topology SF from the mesh-file load
- restore the variable subDM section/order metadata
- load the PETSc global vector
- update the variable local vector
- avoid KDTree reconstruction entirely

## Advantages and Disadvantages

## `write_timestep()`

### Advantages

- Produces the standard benchmark output files already used by the repo
- Generates XDMF for ParaView / visualisation
- Works with `read_timestep(...)`
- Flexible when reloading onto a different mesh or ordering because the reload
  path uses coordinates and remap logic

### Disadvantages

- Reload path is expensive because `read_timestep(...)` uses `h5py` plus KDTree
  remap logic
- At large MPI size this can be very memory-heavy
- Does not contain the PETSc section/order metadata needed for exact direct FE
  vector restart
- Direct PETSc vector load from `/fields/...` is not reliable for these files

## `write_checkpoint()`

### Advantages

- Writes the PETSc DM / subDM restart metadata needed for exact restart
- Stores vectors together with section/order information
- Better aligned with exact restart / postprocessing from distributed FE data
- Avoids relying on geometric remap as the fundamental reload mechanism

### Disadvantages

- Does not create XDMF
- Produces an extra restart file, so total output is larger
- Current UW3 source clearly writes the right metadata, but the clean
  user-facing restore helper is not obvious in the latest codebase
- So it is the right format for restart, but the exact current reload API still
  needs to be pinned down carefully

## Practical Recommendation for These Spherical Benchmarks

If the goal is:

- visualisation only, or flexible remap-based reload:
  - use `write_timestep()`

- exact restart / exact FE field reconstruction:
  - use `write_checkpoint()`

If the goal is both:

- exact restart
- and standard visualisation outputs

then the safest current workflow is to write both:

- `write_timestep(...)`
- `write_checkpoint(...)`

instead of trying to make one substitute for the other.

## Proposed Improved Restart+Viz Layout

For large spherical benchmark runs, a cleaner long-term design would be to
separate:

- mesh topology
- shared restart metadata
- per-variable restart data
- visualisation payload

A practical target layout would be:

- `restart.mesh.0.h5`
  - mesh topology / labels / mesh metadata
- `restart.layout.00000.h5`
  - shared mesh DM restart metadata
- `restart.Velocity.checkpoint.00000.h5`
  - Velocity section/order metadata
  - Velocity PETSc vector
  - optional `/vertex_fields/...` for visualisation
- `restart.Pressure.checkpoint.00000.h5`
  - Pressure section/order metadata
  - Pressure PETSc vector
  - optional `/vertex_fields/...` for visualisation

In this design:

- exact restart uses PETSc vectors plus section/order metadata
- visualisation uses `/vertex_fields/...`
- `/fields/...` would not be required

## Why This Layout Is Better

### 1. Avoids one giant checkpoint file

The current `write_checkpoint()` design places all restart variables into a
single checkpoint file.

At very large scale, such as `1/128` spherical runs on `1000+` ranks, that can
create a large shared-file I/O hotspot during both write and reload.

Splitting variable restart data into separate files reduces that concentration.

### 2. Avoids duplicating shared mesh restart metadata

The mesh-level PETSc restart metadata should only be written once.

If the full mesh restart metadata were repeated in every variable file, the
format would become unnecessarily redundant.

A separate shared layout file keeps the design cleaner and smaller.

### 3. Keeps exact restart separate from visualisation

Restart and visualisation have different requirements:

- restart needs PETSc vector + section/order metadata
- visualisation needs projected array data that ParaView/XDMF can consume

Separating these concerns makes the file format easier to reason about.

### 4. Better fit for discontinuous variables

For discontinuous fields, coordinate-based reconstruction is even less reliable,
because values may be element-local and not uniquely determined by spatial
location alone.

A PETSc metadata-based restart path is therefore more robust for both:

- continuous variables
- discontinuous variables

### 5. Keeps visualisation lightweight and explicit

If visualisation is wanted, the per-variable files can carry:

- `/vertex_fields/...`

without needing to also carry:

- `/fields/...`

This keeps the visualisation payload focused on viewing rather than pretending
to be a restart format.

## Main Advantage in One Sentence

The proposed split layout gives:

- exact PETSc-based restart
- lighter and more modular per-variable files
- no duplication of shared mesh restart metadata
- optional visualisation support

without relying on the expensive KDTree-based `read_timestep(...)` path.

## Bottom Line

`write_timestep()` and `write_checkpoint()` are related, but they serve
different purposes.

- `write_timestep()` is the visualisation/remap output path
- `write_checkpoint()` is the restart metadata path

For the spherical benchmark reload problem, the missing piece in
`write_timestep()` is the PETSc restart metadata stored by
`write_checkpoint()`.
