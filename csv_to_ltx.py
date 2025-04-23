#!/usr/bin/env python3
"""
csv2latex_multi.py —— Convert CSV with arbitrary number of columns
                       to a LaTeX table (booktabs + vertical rules).

2025-04-23
"""
import argparse
import pandas as pd
from pathlib import Path

TEMPLATE = r"""
\begin{{table}}[htbp]
  \centering
  \caption{{{caption}}}
  \label{{{label}}}
  \begin{{tabular}}{{{colspec}}}
    \toprule
    {header}\\
    \midrule
{rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""

LATEX_SPECIALS = {
    "&":  r"\&", "%":  r"\%", "$": r"\$", "#": r"\#",
    "_":  r"\_", "{":  r"\{", "}": r"\}",
    "~":  r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}

def escape(s: str) -> str:
    for ch, rep in LATEX_SPECIALS.items():
        s = s.replace(ch, rep)
    return s

def build_colspec(n_cols: int, pwidths):
    spec = []
    for i in range(n_cols):
        if i < len(pwidths) and pwidths[i] > 0:
            spec.append(f"p{{{pwidths[i]}cm}}")
        else:
            spec.append("c")
    # 竖线隔开
    return "|".join(spec)

def df_to_latex(df: pd.DataFrame, caption, label, pwidths):
    n_cols = df.shape[1]
    colspec = build_colspec(n_cols, pwidths)

    # 表头
    header_cells = [f"\\textbf{{{escape(str(col))}}}" for col in df.columns]
    header = " & ".join(header_cells)

    # 数据行
    body_lines = []
    for row in df.itertuples(index=False):
        cells = [escape(str(x)) for x in row]
        body_lines.append("    " + " & ".join(cells) + r"\\")
    rows = "\n".join(body_lines)

    return TEMPLATE.format(caption=escape(caption),
                           label=escape(label),
                           colspec=colspec,
                           header=header,
                           rows=rows)

def main():
    ap = argparse.ArgumentParser(description="CSV → LaTeX table (any #cols)")
    ap.add_argument("csv", help="Input CSV file")
    ap.add_argument("--caption", default="Table caption")
    ap.add_argument("--label",   default="tab:label")
    ap.add_argument("--pwidths", nargs="*", type=float, default=[],
                    metavar="W", help="Column widths in cm; 0 = centred 'c'.")
    ap.add_argument("--no-header", action="store_true",
                    help="Treat first row as data, generate generic headers")
    args = ap.parse_args()

    # 读取 CSV
    if args.no_header:
        df = pd.read_csv(args.csv, header=None)
        df.columns = [f"Col {i+1}" for i in range(df.shape[1])]
    else:
        df = pd.read_csv(args.csv)

    tex = df_to_latex(df, args.caption, args.label, args.pwidths)
    print(tex)

if __name__ == "__main__":
    main()
