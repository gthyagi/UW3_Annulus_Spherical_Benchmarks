#!/usr/bin/env bash
# Render the technical note with native MathML and embedded figures.
set -euo pipefail

note_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
note_source="${note_dir}/blog_annulus_spherical_stokes_benchmarks.md"
note_output="${note_source%.md}.html"

if ! command -v pandoc >/dev/null 2>&1; then
    printf 'Rendering requires pandoc on PATH.\n' >&2
    exit 1
fi

pandoc "${note_source}" \
    --from=markdown+tex_math_dollars+raw_html+pipe_tables \
    --to=html5 \
    --standalone \
    --mathml \
    --embed-resources \
    --fail-if-warnings \
    --resource-path="${note_dir}" \
    --css="${note_dir}/technical_note.css" \
    --metadata pagetitle="Underworld3 Curved-Domain Stokes Benchmarks" \
    --output="${note_output}"

printf 'Rendered technical note: %s\n' "${note_output}"
