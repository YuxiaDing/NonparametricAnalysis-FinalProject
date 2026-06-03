from pathlib import Path

import pandas as pd


def save_table(df, name, caption, label, float_format="%.4f", placement="htbp", small=False):
    """Save a table as CSV and LaTeX booktabs."""
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    latex = dataframe_to_tabular(df, float_format=float_format)
    wrapped = (
        f"\\begin{{table}}[{placement}]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        + ("\\small\n" if small else "")
        + latex
        + "\\end{table}\n"
    )
    path.with_suffix(".tex").write_text(wrapped, encoding="utf-8")


def dataframe_to_tabular(df, float_format="%.4f"):
    """Convert a small DataFrame to a booktabs tabular without pandas Styler."""
    columns = list(df.columns)
    align = "l" + "c" * (len(columns) - 1)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(escape_latex(str(c)) for c in columns) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        cells = [format_cell(row[c], float_format) for c in columns]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}\n")
    return "\n".join(lines)


def format_cell(value, float_format):
    if isinstance(value, float):
        return float_format % value
    if pd.isna(value):
        return ""
    return escape_latex(str(value))


def escape_latex(text):
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
