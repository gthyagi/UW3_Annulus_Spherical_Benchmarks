#!/usr/bin/env python3
"""Copy convergence plots used by stokes_annulus_spherical_benchmarks.tex."""

from pathlib import Path
from shutil import copy2


PLOTS = (
    Path("annulus/thieulot/figure_5_thieulot_annulus_convergence.pdf"),
    Path("annulus/kramer/figure_3_kramer_annulus_convergence.pdf"),
    Path("spherical/thieulot/figures_4_5_thieulot_convergence.pdf"),
    Path("spherical/kramer/figure_4_kramer_spherical_convergence.pdf"),
)


def main() -> None:
    combined_dir = Path(__file__).resolve().parent
    figure_scripts_dir = combined_dir.parent
    output_dir = combined_dir / "Figures"
    output_dir.mkdir(exist_ok=True)

    for relative_path in PLOTS:
        source = figure_scripts_dir / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Missing convergence plot: {source}")

        destination = output_dir / source.name
        copy2(source, destination)
        print(f"Copied {source} -> {destination}")


if __name__ == "__main__":
    main()
