#!/usr/bin/env python3
"""Create a LaTeX convergence table for Kramer spherical Figure 7 metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

METRICS_ROOT = Path("/Volumes/seagate4_1/output/spherical/kramer/latest")
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_TEX = OUTPUT_DIR / "table_figure_7_kramer_boundary_convergence.tex"

H_VALUES = (1 / 8, 1 / 16, 1 / 32, 1 / 64)
H_LABELS = ("1/8", "1/16", "1/32", "1/64")
GROUPS_PER_TABLE = 3
DIR_PATTERN = re.compile(
    r"case1_"
    r"inv_lc_(?P<inv_lc>\d+)_"
    r"l_(?P<l>\d+)_"
    r"m_(?P<m>\d+)_"
    r"k_(?P<k>\d+)_"
)


@dataclass(frozen=True)
class MetricRecord:
    h: float
    l: int
    m: int
    k: int
    v_l2_norm_upper: float
    v_l2_norm_lower: float
    sigma_rr_l2_norm_upper: float
    sigma_rr_l2_norm_lower: float


def read_scalar(h5f: h5py.File, name: str) -> float:
    return float(np.asarray(h5f[name][()]))


def read_metrics() -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for metric_file in sorted(METRICS_ROOT.glob("case1_inv_lc_*/benchmark_metrics.h5")):
        match = DIR_PATTERN.search(metric_file.parent.name)
        if match is None:
            continue

        with h5py.File(metric_file, "r") as h5f:
            records.append(
                MetricRecord(
                    h=read_scalar(h5f, "cellsize"),
                    l=int(read_scalar(h5f, "l")),
                    m=int(read_scalar(h5f, "m")),
                    k=int(read_scalar(h5f, "k")),
                    v_l2_norm_upper=read_scalar(h5f, "v_l2_norm_upper"),
                    v_l2_norm_lower=read_scalar(h5f, "v_l2_norm_lower"),
                    sigma_rr_l2_norm_upper=read_scalar(h5f, "sigma_rr_l2_norm_upper"),
                    sigma_rr_l2_norm_lower=read_scalar(h5f, "sigma_rr_l2_norm_lower"),
                )
            )

    if not records:
        raise FileNotFoundError(f"No case1 benchmark_metrics.h5 files found in {METRICS_ROOT}")

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
    rates: list[float] = [np.nan]
    for idx in range(1, len(h_values)):
        y_prev = y_values[idx - 1]
        y_curr = y_values[idx]
        if not np.isfinite(y_prev) or not np.isfinite(y_curr) or y_prev <= 0.0 or y_curr <= 0.0:
            rates.append(np.nan)
            continue
        rates.append(float(np.log(y_curr / y_prev) / np.log(h_values[idx] / h_values[idx - 1])))
    return rates


def groups(records: list[MetricRecord]) -> list[tuple[int, int, int]]:
    return sorted({(record.l, record.m, record.k) for record in records})


def chunked_groups(metric_groups: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    return [
        metric_groups[index : index + GROUPS_PER_TABLE]
        for index in range(0, len(metric_groups), GROUPS_PER_TABLE)
    ]


def values_for_group(records: list[MetricRecord], group: tuple[int, int, int], metric_name: str) -> np.ndarray:
    l_val, m_val, k_val = group
    values_by_h = {
        record.h: getattr(record, metric_name)
        for record in records
        if record.l == l_val and record.m == m_val and record.k == k_val
    }
    return np.array([values_by_h.get(h, np.nan) for h in H_VALUES], dtype=float)


def table_for_metric_group(
    records: list[MetricRecord],
    caption: str,
    metric_groups: list[tuple[int, int, int]],
    table_idx: int,
    table_count: int,
    upper_metric: str,
    lower_metric: str,
    upper_label: str,
    lower_label: str,
) -> str:
    column_spec = "c" + "cccc" * len(metric_groups)
    caption_text = caption.rstrip(".")
    header_cmidrules = []
    for group_idx in range(len(metric_groups)):
        first_col = 2 + 4 * group_idx
        last_col = first_col + 3
        header_cmidrules.append(rf"\cmidrule(lr){{{first_col}-{last_col}}}")

    group_header = [r"\multirow{2}{*}{$h$}"]
    sub_header = [""]
    for l_val, m_val, k_val in metric_groups:
        group_header.append(rf"\multicolumn{{4}}{{c}}{{$l={l_val},\,m={m_val},\,k={k_val}$}}")
        sub_header.extend(
            [
                rf"${upper_label}$",
                "rate",
                rf"${lower_label}$",
                "rate",
            ]
        )

    lines = [
        r"\begin{table}[p]",
        r"\centering",
        rf"\caption{{{caption_text} ({table_idx} of {table_count}).}}",
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
    for group in metric_groups:
        upper_values = values_for_group(records, group, upper_metric)
        lower_values = values_for_group(records, group, lower_metric)
        group_values.append(
            (
                upper_values,
                pairwise_rates(H_VALUES, upper_values),
                lower_values,
                pairwise_rates(H_VALUES, lower_values),
            )
        )

    for h_idx, h_label in enumerate(H_LABELS):
        row = [rf"${h_label}$"]
        for upper_values, upper_rates, lower_values, lower_rates in group_values:
            row.extend(
                [
                    latex_num(upper_values[h_idx]),
                    latex_rate(upper_rates[h_idx]),
                    latex_num(lower_values[h_idx]),
                    latex_rate(lower_rates[h_idx]),
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


def tables_for_metric(
    records: list[MetricRecord],
    caption: str,
    upper_metric: str,
    lower_metric: str,
    upper_label: str,
    lower_label: str,
) -> str:
    group_chunks = chunked_groups(groups(records))
    table_count = len(group_chunks)
    return "\n".join(
        table_for_metric_group(
            records=records,
            caption=caption,
            metric_groups=metric_groups,
            table_idx=table_idx,
            table_count=table_count,
            upper_metric=upper_metric,
            lower_metric=lower_metric,
            upper_label=upper_label,
            lower_label=lower_label,
        )
        for table_idx, metric_groups in enumerate(group_chunks, start=1)
    )


def make_latex(records: list[MetricRecord]) -> str:
    velocity_table = tables_for_metric(
        records,
        "Case 1 boundary velocity convergence.",
        "v_l2_norm_upper",
        "v_l2_norm_lower",
        r"E_{L_2,\Gamma_{\mathrm{outer}}}^{\mathrm{rel}}(\mathbf{u})",
        r"E_{L_2,\Gamma_{\mathrm{inner}}}^{\mathrm{rel}}(\mathbf{u})",
    )
    stress_table = tables_for_metric(
        records,
        r"Case 1 boundary radial-stress convergence.",
        "sigma_rr_l2_norm_upper",
        "sigma_rr_l2_norm_lower",
        r"E_{L_2,\Gamma_{\mathrm{outer}}}^{\mathrm{rel}}(\sigma_{rr})",
        r"E_{L_2,\Gamma_{\mathrm{inner}}}^{\mathrm{rel}}(\sigma_{rr})",
    )
    return rf"""\documentclass[11pt,a4paper]{{article}}

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

{velocity_table}
{stress_table}
\end{{document}}
"""


def main() -> None:
    records = read_metrics()
    OUTPUT_TEX.write_text(make_latex(records), encoding="utf-8")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
