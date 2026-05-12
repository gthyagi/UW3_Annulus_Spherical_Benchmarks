#!/usr/bin/env python3
"""Create a compact LSQ-rate table for Kramer annulus boundary pressure."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_TEX = OUTPUT_DIR / "table_boundary_pressure_convergence_rates.tex"
DETAILED_TABLE_SCRIPT = OUTPUT_DIR / "make_table_boundary_pressure_convergence.py"


def load_detailed_table_module():
    spec = importlib.util.spec_from_file_location(
        "make_table_boundary_pressure_convergence",
        DETAILED_TABLE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {DETAILED_TABLE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def lsq_rate(h_values: tuple[float, ...], y_values: np.ndarray) -> float:
    h = np.asarray(h_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(h) & np.isfinite(y) & (h > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.nan
    return float(np.polyfit(np.log(h[mask]), np.log(y[mask]), deg=1)[0])


def latex_rate(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.2f}"


def make_latex() -> str:
    table_data = load_detailed_table_module()
    records = table_data.read_metrics()
    table_data.validate_records(records)

    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        "",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{multirow}",
        r"\usepackage{newtxtext,newtxmath}",
        "",
        r"\pagestyle{empty}",
        "",
        r"\begin{document}",
        "",
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Least-squares convergence rates for the relative pressure error on the inner and outer boundaries, \(E_{L_2,\Gamma}^{\mathrm{rel}}(p)\).}",
        r"\label{tab:kramer_annulus_boundary_pressure_lsq_rates}",
        r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Case} & \multicolumn{2}{c}{$n=2$} & \multicolumn{2}{c}{$n=8$} & \multicolumn{2}{c}{$n=32$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & inner & outer & inner & outer & inner & outer \\",
        r"\midrule",
    ]

    for case, k, title in table_data.TABLE_SPECS:
        row = [title]
        for n in table_data.N_VALUES:
            for metric_name, _ in table_data.BOUNDARIES:
                values = table_data.values_for_series(
                    records,
                    case=case,
                    k=k,
                    n=n,
                    metric_name=metric_name,
                )
                row.append(latex_rate(lsq_rate(table_data.H_VALUES, values)))
        lines.append(" & ".join(row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_TEX.write_text(make_latex(), encoding="utf-8")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
