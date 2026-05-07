#!/usr/bin/env python3
"""Create a LaTeX convergence table for Kramer annulus Figure 3 metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


METRICS_ROOT = Path("/Volumes/seagate4_1/output/annulus/kramer/latest")
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_TEX = OUTPUT_DIR / "table_figure_3_kramer_annulus_convergence.tex"

H_VALUES = tuple(1.0 / inv_lc for inv_lc in (8, 16, 32, 64, 128, 256))
H_LABELS = ("1/8", "1/16", "1/32", "1/64", "1/128", "1/256")
N_VALUES = (2, 8, 32)

TABLE_SPECS = (
    ("case1", None, "Free-slip, delta-function forcing"),
    ("case2", 2, r"Free-slip, smooth forcing ($k=2$)"),
    ("case2", 8, r"Free-slip, smooth forcing ($k=8$)"),
    ("case3", None, "Zero-slip, delta-function forcing"),
    ("case4", 2, r"Zero-slip, smooth forcing ($k=2$)"),
    ("case4", 8, r"Zero-slip, smooth forcing ($k=8$)"),
)

DIR_PATTERN = re.compile(
    r"(?P<case>case\d+)_"
    r"inv_lc_(?P<inv_lc>\d+)_"
    r"n_(?P<n>\d+)_"
    r"(?:k_(?P<k>\d+)_)?"
)


@dataclass(frozen=True)
class MetricRecord:
    case: str
    h: float
    n: int
    k: int | None
    v_l2_norm: float
    p_l2_norm: float


def read_scalar(h5f: h5py.File, name: str) -> float:
    value = h5f[name][()]
    if isinstance(value, bytes):
        return value.decode()
    return float(np.asarray(value))


def read_metrics() -> list[MetricRecord]:
    records: list[MetricRecord] = []

    for metric_file in sorted(METRICS_ROOT.glob("case*_inv_lc_*/benchmark_metrics.h5")):
        match = DIR_PATTERN.search(metric_file.parent.name)
        if match is None:
            continue

        case = match.group("case")
        if case not in {"case1", "case2", "case3", "case4"}:
            continue

        with h5py.File(metric_file, "r") as h5f:
            h = read_scalar(h5f, "cellsize")
            if h not in H_VALUES:
                continue

            records.append(
                MetricRecord(
                    case=case,
                    h=h,
                    n=int(read_scalar(h5f, "n")),
                    k=None if match.group("k") is None else int(read_scalar(h5f, "k")),
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


def pairwise_rates(y_values: np.ndarray) -> list[float]:
    rates: list[float] = [np.nan]
    for idx in range(1, len(H_VALUES)):
        y_prev = y_values[idx - 1]
        y_curr = y_values[idx]
        if not np.isfinite(y_prev) or not np.isfinite(y_curr) or y_prev <= 0.0 or y_curr <= 0.0:
            rates.append(np.nan)
            continue
        rates.append(float(np.log(y_curr / y_prev) / np.log(H_VALUES[idx] / H_VALUES[idx - 1])))
    return rates


def values_for_series(
    records: list[MetricRecord],
    *,
    case: str,
    k: int | None,
    n: int,
    metric_name: str,
) -> np.ndarray:
    values_by_h = {
        record.h: getattr(record, metric_name)
        for record in records
        if record.case == case and record.k == k and record.n == n
    }
    return np.array([values_by_h.get(h, np.nan) for h in H_VALUES], dtype=float)


def table_for_spec(records: list[MetricRecord], case: str, k: int | None, title: str) -> str:
    column_spec = "c" + "cccc" * len(N_VALUES)
    header_cmidrules = []
    for group_idx in range(len(N_VALUES)):
        first_col = 2 + 4 * group_idx
        last_col = first_col + 3
        header_cmidrules.append(rf"\cmidrule(lr){{{first_col}-{last_col}}}")

    group_header = [r"\multirow{2}{*}{$h$}"]
    sub_header = [""]
    for n in N_VALUES:
        group_header.append(rf"\multicolumn{{4}}{{c}}{{$n={n}$}}")
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
        rf"\caption{{{title}.}}",
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

    series_values = []
    for n in N_VALUES:
        velocity = values_for_series(records, case=case, k=k, n=n, metric_name="v_l2_norm")
        pressure = values_for_series(records, case=case, k=k, n=n, metric_name="p_l2_norm")
        series_values.append(
            (
                velocity,
                pairwise_rates(velocity),
                pressure,
                pairwise_rates(pressure),
            )
        )

    for h_idx, h_label in enumerate(H_LABELS):
        row = [rf"${h_label}$"]
        for velocity, velocity_rates, pressure, pressure_rates in series_values:
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
    tables = "\n".join(
        table_for_spec(records, case=case, k=k, title=title)
        for case, k, title in TABLE_SPECS
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

{tables}
\end{{document}}
"""


def main() -> None:
    records = read_metrics()
    OUTPUT_TEX.write_text(make_latex(records), encoding="utf-8")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
