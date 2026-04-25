# PETSc Timing Output: Text vs CSV at Large MPI Scale

## Summary

At large MPI counts, `uw.timing.print_table()` can behave very differently
depending on the output format:

```python
uw.timing.print_table(filename="integrals_timing.txt")
```

uses PETSc's default human-readable ASCII/text log viewer, while:

```python
uw.timing.print_table(filename="integrals_timing.csv", format="csv")
```

uses PETSc's CSV log viewer.

In the spherical benchmark large-rank runs, the metric computation completed, but
the job stalled when writing the default `.txt` timing table. The same run
completed when the timing output was written as `.csv`.

## Observed Behavior

The failing `.txt` timing file reached the end of the PETSc performance summary:

```text
Average time to get PetscTime(): ...
Average time for MPI_Barrier(): ...
```

but did not reach:

```text
Average time for zero size MPI_Send(): ...
```

This means the benchmark metrics had already completed. The stall happened
inside PETSc's text timing report finalization, not inside the Underworld3
metric calculations.

## PETSc Version

The relevant local PETSc checkout reports:

```text
PETSc version: 3.25.0
Local vendored PETSc commit: d3616053e302ec9a00f9e1eb098278d7544f73aa
```

This was also the version shown in the Gadi timing output:

```text
Using PETSc Release Version 3.25.0
```

## Source-Level Cause

The PETSc source file is:

```text
src/sys/logging/handler/impls/default/logdefault.c
```

PETSc dispatches the timing output based on viewer format:

```c
if (format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_ASCII_INFO) {
  PetscLogHandlerView_Default_Info(handler, viewer);
} else if (format == PETSC_VIEWER_ASCII_CSV) {
  PetscLogHandlerView_Default_CSV(handler, viewer);
}
```

Reference:

https://gitlab.com/petsc/petsc/-/blob/v3.25.0/src/sys/logging/handler/impls/default/logdefault.c#L1854-1866

The CSV path writes synchronized event rows and returns:

https://gitlab.com/petsc/petsc/-/blob/v3.25.0/src/sys/logging/handler/impls/default/logdefault.c#L1100-1166

The default text path does more work. After printing the full performance
summary, it runs extra MPI timing checks:

https://gitlab.com/petsc/petsc/-/blob/v3.25.0/src/sys/logging/handler/impls/default/logdefault.c#L1770-1814

That text-only block includes:

- repeated `MPI_Barrier(comm)` calls
- `PetscCommDuplicate(comm, &newcomm, &tag)`
- a zero-size rank-to-rank `MPI_Send` / `MPI_Recv` ring test
- printing `Average time for zero size MPI_Send()`

Because the failed file printed `Average time for MPI_Barrier()` but not
`Average time for zero size MPI_Send()`, the likely stall point is after the
barrier timing and before the zero-size send timing print. That points to the
communicator duplicate / zero-size ring-send section, not benchmark metric
evaluation.

## Recommended Benchmark Practice

For large MPI-count production benchmark runs, use CSV timing output:

```python
uw.timing.print_table(
    filename=os.path.join(output_dir, "integrals_timing.csv"),
    format="csv",
)
```

Avoid default `.txt` timing output for large-rank postprocessing stages:

```python
uw.timing.print_table(filename=os.path.join(output_dir, "integrals_timing.txt"))
```

CSV avoids the text-only MPI microbenchmark/final-report path and has completed
successfully at large MPI scale.

## Practical Fix

For benchmark scripts:

- Use `format="csv"` for large-rank timing output.
- Prefer `.csv` timing files for postprocessing and metric stages.
- Treat `.txt` timing tables as a small-to-moderate scale diagnostic format.

For a possible Underworld3-side improvement:

- make CSV the recommended or default timing format for large MPI runs, or
- avoid PETSc's default text log viewer for benchmark postprocessing stages, or
- expose a safe timing-output option that does not run PETSc's text-only MPI
  communication microbenchmarks.

## Related Issue

GitHub issue:

https://github.com/underworldcode/underworld3/issues/134
