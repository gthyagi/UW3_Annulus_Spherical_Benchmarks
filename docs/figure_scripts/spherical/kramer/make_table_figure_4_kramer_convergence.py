#!/usr/bin/env python3
"""Create a LaTeX convergence table for Kramer spherical Figure 4 metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

METRICS_ROOT = Path("/Volumes/seagate4_1/output/spherical/kramer/latest")
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_TEX = OUTPUT_DIR / "table_figure_4_kramer_convergence.tex"

H_VALUES = (1 / 8, 1 / 16, 1 / 32, 1 / 64)
H_LABELS = ("1/8", "1/16", "1/32", "1/64")
CASE_TITLES = {
    "case1": "Case 1: free-slip, delta-function forcing",
    "case2": "Case 2: free-slip, smooth forcing",
    "case3": "Case 3: zero-slip, delta-function forcing",
    "case4": "Case 4: zero-slip, smooth forcing",
}
CASE_ORDER = ("case1", "case2", "case3", "case4")

DIR_PATTERN = re.compile(
    r"(?P<case>case\d+)_"
    r"inv_lc_(?P<inv_lc>\d+)_"
    r"l_(?P<l>\d+)_"
    r"m_(?P<m>\d+)_"
    r"k_(?P<k>\d+)_"
)


@dataclass(frozen=True)
class MetricRecord:
    case: str
    h: float
    l: int
    m: int
    k: int
    v_l2_norm: float
    p_l2_norm: float


def read_scalar(h5f: h5py.File, name: str) -> float:
    return float(np.asarray(h5f[name][()]))


def read_metrics() -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for metric_file in sorted(METRICS_ROOT.glob("case*_inv_lc_*/benchmark_metrics.h5")):
        match = DIR_PATTERN.search(metric_file.parent.name)
        if match is None:
            continue

        case = match.group("case")
        if case not in CASE_TITLES:
            continue

        with h5py.File(metric_file, "r") as h5f:
            records.append(
                MetricRecord(
                    case=case,
                    h=read_scalar(h5f, "cellsize"),
                    l=int(read_scalar(h5f, "l")),
                    m=int(read_scalar(h5f, "m")),
                    k=int(read_scalar(h5f, "k")),
                    v_l2_norm=read_scalar(h5f, "v_l2_norm"),
                    p_l2_norm=read_scalar(h5f, "p_l2_norm"),
                )
            )

    if not records:
        raise FileNotFoundError(f"No benchmark_metrics.h5 files found in {METRICS_ROOT}")

    return records


def latex_num(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return rf"\num{{{value:.6e}}}"


def latex_rate(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.2f}"


def pairwise_rates(h_values: tuple[float, ...], y_values: np.ndarray) -> list[float]:
    """Return convergence rates between successive h values."""

    rates: list[float] = [np.nan]
    for idx in range(1, len(h_values)):
        y_prev = y_values[idx - 1]
        y_curr = y_values[idx]
        if not np.isfinite(y_prev) or not np.isfinite(y_curr) or y_prev <= 0.0 or y_curr <= 0.0:
            rates.append(np.nan)
            continue
        rates.append(float(np.log(y_curr / y_prev) / np.log(h_values[idx] / h_values[idx - 1])))
    return rates


def values_for_group(
    records: list[MetricRecord],
    case: str,
    group: tuple[int, int, int],
    metric_name: str,
) -> np.ndarray:
    l_val, m_val, k_val = group
    values_by_h = {
        record.h: getattr(record, metric_name)
        for record in records
        if record.case == case and record.l == l_val and record.m == m_val and record.k == k_val
    }
    return np.array([values_by_h.get(h, np.nan) for h in H_VALUES], dtype=float)


def case_groups(records: list[MetricRecord], case: str) -> list[tuple[int, int, int]]:
    return sorted({(record.l, record.m, record.k) for record in records if record.case == case})


def table_for_case(records: list[MetricRecord], case: str) -> str:
    groups = case_groups(records, case)
    column_spec = "c" + "cccc" * len(groups)
    header_cmidrules = []
    for group_idx in range(len(groups)):
        first_col = 2 + 4 * group_idx
        last_col = first_col + 3
        header_cmidrules.append(rf"\cmidrule(lr){{{first_col}-{last_col}}}")

    group_header = [r"\multirow{2}{*}{$h$}"]
    sub_header = [""]
    for l_val, m_val, k_val in groups:
        group_header.append(rf"\multicolumn{{4}}{{c}}{{$l={l_val},\,m={m_val},\,k={k_val}$}}")
        sub_header.extend(
            [
                r"$|e_v|_2$",
                "rate",
                r"$|e_p|_2$",
                "rate",
            ]
        )

    lines = [
        r"\begin{table}[p]",
        r"\centering",
        rf"\caption{{{CASE_TITLES[case]}.}}",
        r"\tiny",
        r"\setlength{\tabcolsep}{1.7pt}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(group_header) + r" \\",
        "\n".join(header_cmidrules),
        " & ".join(sub_header) + r" \\",
        r"\midrule",
    ]

    group_values = []
    for group in groups:
        velocity = values_for_group(records, case, group, "v_l2_norm")
        pressure = values_for_group(records, case, group, "p_l2_norm")
        group_values.append(
            (
                velocity,
                pairwise_rates(H_VALUES, velocity),
                pressure,
                pairwise_rates(H_VALUES, pressure),
            )
        )

    for h_idx, h_label in enumerate(H_LABELS):
        row = [rf"${h_label}$"]
        for velocity, velocity_rates, pressure, pressure_rates in group_values:
            row.extend(
                [
                    latex_num(velocity[h_idx]),
                    latex_rate(velocity_rates[h_idx]),
                    latex_num(pressure[h_idx]),
                    latex_rate(pressure_rates[h_idx]),
                ]
            )
        lines.append(" & ".join(row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def make_latex(records: list[MetricRecord]) -> str:
    tables = "\n".join(table_for_case(records, case) for case in CASE_ORDER)
    return rf"""\documentclass[11pt,a4paper,landscape]{{article}}

\usepackage[margin=0.45in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{multirow}}
\usepackage{{graphicx}}
\usepackage{{siunitx}}
\usepackage{{newtxtext,newtxmath}}

\pagestyle{{empty}}

\sisetup{{
  scientific-notation = true,
  round-mode = places,
  round-precision = 3,
}}

\begin{{document}}

{tables}
\end{{document}}
"""


def main() -> None:
    records = read_metrics()
    OUTPUT_TEX.write_text(make_latex(records), encoding="utf-8")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
