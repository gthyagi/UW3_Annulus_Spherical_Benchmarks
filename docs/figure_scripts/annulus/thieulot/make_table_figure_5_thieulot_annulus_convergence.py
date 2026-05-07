#!/usr/bin/env python3
"""Create LaTeX tables for annulus Thieulot L2 convergence metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

METRICS_ROOT = Path("/Volumes/seagate4_1/output/annulus/thieulot/latest")
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_TEX = OUTPUT_DIR / "table_figure_5_thieulot_annulus_convergence.tex"

ELEMENTS = (
    (1, 0, False, r"$P_1\times P_0$"),
    (1, 1, True, r"$P_1\times P_1$"),
    (2, 1, True, r"$P_2\times P_1$"),
    (3, 2, True, r"$P_3\times P_2$"),
)
K_VALUES = (1, 4, 8)
P3P2_EXCLUDED_H = 1 / 256

DIR_PATTERN = re.compile(
    r"model_inv_lc_(?P<inv_lc>\d+)_"
    r"k_(?P<k>\d+)_"
    r"vdeg_(?P<vdeg>\d+)_"
    r"pdeg_(?P<pdeg>\d+)_"
    r"pcont_(?P<pcont>true|false)_"
)


@dataclass(frozen=True)
class MetricRecord:
    h: float
    k: int
    vdegree: int
    pdegree: int
    pcont: bool
    v_l2_norm: float
    p_l2_norm: float


def read_scalar(h5f: h5py.File, name: str) -> float:
    value = h5f[name][()]
    if isinstance(value, bytes):
        return float(value.decode())
    return float(np.asarray(value))


def read_records() -> list[MetricRecord]:
    valid_elements = {(vdegree, pdegree, pcont) for vdegree, pdegree, pcont, _ in ELEMENTS}
    records: list[MetricRecord] = []

    for metrics_file in sorted(METRICS_ROOT.glob("model_inv_lc_*/benchmark_metrics.h5")):
        match = DIR_PATTERN.search(metrics_file.parent.name)
        if match is None:
            continue

        vdegree = int(match.group("vdeg"))
        pdegree = int(match.group("pdeg"))
        pcont = match.group("pcont") == "true"
        k = int(match.group("k"))
        if (vdegree, pdegree, pcont) not in valid_elements or k not in K_VALUES:
            continue

        with h5py.File(metrics_file, "r") as h5f:
            h = read_scalar(h5f, "cellsize")
            if (
                vdegree == 3
                and pdegree == 2
                and pcont
                and np.isclose(h, P3P2_EXCLUDED_H)
            ):
                continue

            records.append(
                MetricRecord(
                    h=h,
                    k=k,
                    vdegree=vdegree,
                    pdegree=pdegree,
                    pcont=pcont,
                    v_l2_norm=read_scalar(h5f, "v_l2_norm"),
                    p_l2_norm=read_scalar(h5f, "p_l2_norm"),
                )
            )

    return records


def values_for_series(
    records: list[MetricRecord],
    vdegree: int,
    pdegree: int,
    pcont: bool,
    k: int,
) -> list[MetricRecord]:
    return sorted(
        [
            record
            for record in records
            if (
                record.vdegree == vdegree
                and record.pdegree == pdegree
                and record.pcont == pcont
                and record.k == k
            )
        ],
        key=lambda record: record.h,
    )


def rates(h: list[float], values: list[float]) -> list[float | None]:
    result: list[float | None] = [None]
    for idx in range(1, len(values)):
        if values[idx - 1] <= 0.0 or values[idx] <= 0.0:
            result.append(None)
            continue
        result.append(np.log(values[idx] / values[idx - 1]) / np.log(h[idx] / h[idx - 1]))
    return result


def fmt_value(value: float) -> str:
    return f"{value:.3e}"


def fmt_h(value: float) -> str:
    inv_h = int(round(1.0 / value))
    if np.isclose(value, 1.0 / inv_h):
        return rf"$1/{inv_h}$"
    return f"${value:g}$"


def fmt_rate(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.2f}"


def table_block(records: list[MetricRecord], element: tuple[int, int, bool, str]) -> str:
    vdegree, pdegree, pcont, label = element
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        rf"\caption{{Annulus Thieulot convergence for {label}.}}",
        rf"\label{{tab:annulus_thieulot_v{vdegree}_p{pdegree}_{'cont' if pcont else 'disc'}}}",
        r"\small",
        r"\begin{tabular}{",
        r"r rr rr rr rr rr rr",
        r"}",
        r"\toprule",
        r"& \multicolumn{4}{c}{$k=1$} & \multicolumn{4}{c}{$k=4$} & \multicolumn{4}{c}{$k=8$} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
        r"$h$ & $|e_v|_2$ & rate & $|e_p|_2$ & rate & $|e_v|_2$ & rate & $|e_p|_2$ & rate & $|e_v|_2$ & rate & $|e_p|_2$ & rate \\",
        r"\midrule",
    ]

    by_k = {k: values_for_series(records, vdegree, pdegree, pcont, k) for k in K_VALUES}
    h_values = sorted({record.h for series in by_k.values() for record in series}, reverse=True)

    rates_by_k = {}
    for k, series in by_k.items():
        display_series = sorted(series, key=lambda record: record.h, reverse=True)
        h = [record.h for record in display_series]
        v = [record.v_l2_norm for record in display_series]
        p = [record.p_l2_norm for record in display_series]
        rates_by_k[k] = {
            record.h: (v_rate, p_rate)
            for record, v_rate, p_rate in zip(display_series, rates(h, v), rates(h, p))
        }

    records_by_k_h = {
        k: {record.h: record for record in series} for k, series in by_k.items()
    }

    for h in h_values:
        row = [fmt_h(h)]
        for k in K_VALUES:
            record = records_by_k_h[k].get(h)
            if record is None:
                row.extend(["--", "--", "--", "--"])
                continue

            v_rate, p_rate = rates_by_k[k][h]
            row.extend(
                [
                    fmt_value(record.v_l2_norm),
                    fmt_rate(v_rate),
                    fmt_value(record.p_l2_norm),
                    fmt_rate(p_rate),
                ]
            )
        lines.append(" & ".join(row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    records = read_records()
    if not records:
        raise FileNotFoundError(f"No matching benchmark_metrics.h5 files found in {METRICS_ROOT}")

    body = "\n\n".join(table_block(records, element) for element in ELEMENTS)
    OUTPUT_TEX.write_text(
        "\n".join(
            [
                r"\documentclass[10pt,a4paper]{article}",
                r"\usepackage[margin=0.55in,landscape]{geometry}",
                r"\usepackage{booktabs}",
                r"\usepackage{amsmath}",
                r"\usepackage{newtxtext,newtxmath}",
                r"\pagestyle{empty}",
                r"\begin{document}",
                body,
                r"\end{document}",
                "",
            ]
        )
    )
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
