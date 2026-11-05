"""Convert QF_Attack_Report.md to LaTeX (QF_Attack_Report.tex).

Faithful structural conversion: headings, paragraphs, bold, inline code,
tables (booktabs), figures and references. Table values are copied verbatim
from the Markdown, which is itself checked against the result CSVs.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "QF_Attack_Report.md"
OUT = ROOT / "QF_Attack_Report.tex"

# Fixed math fragments, longest first; each maps to the final LaTeX (with $...$).
MATH = [
    ("cos(I \u00b7 \u03c4(x), I \u00b7 \u03c4(x\u2032))", r"$\cos(I \cdot \tau(x),\, I \cdot \tau(x'))$"),
    ("cos(\u03c4(x), \u03c4(x\u2032))", r"$\cos(\tau(x), \tau(x'))$"),
    ("|\u03a3_i sign(d_i,j)| > \u03b5n", r"$|\sum_i \mathrm{sign}(d_{i,j})| > \varepsilon n$"),
    ("62^5 = 916,132,832", r"$62^{5} = 916{,}132{,}832$"),
    ("62^5", r"$62^{5}$"),
    ("d_i,j", r"$d_{i,j}$"),
    ("d_i", r"$d_i$"),
    ("n = 10", r"$n = 10$"),
    ("\u03b5 = 0.9", r"$\varepsilon = 0.9$"),
    ("\u03c4(x\u2032)", r"$\tau(x')$"),
    ("\u03c4(x)", r"$\tau(x)$"),
    ("x\u2032", r"$x'$"),
]
# Stray single characters, applied after escaping.
UNICODE = [("\u03c4", r"$\tau$"), ("\u03b5", r"$\varepsilon$"), ("\u03a3", r"$\Sigma$"),
           ("\u2032", r"$'$"), ("\u00d7", r"$\times$"), ("\u00b1", r"$\pm$"),
           ("\u2212", r"$-$"), ("\u2299", r"$\odot$"), ("\u00b7", r"\textperiodcentered{}")]


def escape_text(s):
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def escape_tt(s):
    out = []
    for ch in s:
        out.append({"\\": r"\textbackslash{}", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
                    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "|": r"\textbar{}", "<": r"\textless{}", ">": r"\textgreater{}",
                    "·": r"\textperiodcentered{}"}.get(ch, ch))
    return "".join(out)


def inline(text):
    codes = []
    text = re.sub(r"`([^`]*)`", lambda m: codes.append(m.group(1)) or f"ZZCODE{len(codes)-1}ZZ", text)
    maths = []
    for pattern, latex in MATH:
        while pattern in text:
            maths.append(latex)
            text = text.replace(pattern, f"ZZMATH{len(maths)-1}ZZ", 1)
    text = re.sub(r'"([^"]*)"', r"``\1''", text)
    text = escape_text(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![\w/])(https?://[^\s)]+)", r"\\url{\1}", text)
    for a, b in UNICODE:
        text = text.replace(a, b)
    for i, latex in enumerate(maths):
        text = text.replace(f"ZZMATH{i}ZZ", latex)
    for i, code in enumerate(codes):
        text = text.replace(f"ZZCODE{i}ZZ", r"\texttt{" + escape_tt(code) + "}")
    return text


def split_row(line):
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.strip().replace("\\|", "|") for c in cells]


def emit_table(caption, rows):
    ncol = len(rows[0])
    spec = "l" * ncol
    body = [r"\begin{table}[htbp]", r"\centering", r"\small", rf"\caption{{{inline(caption)}}}",
            r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    body.append(" & ".join(r"\textbf{" + inline(c) + "}" for c in rows[0]) + r" \\")
    body.append(r"\midrule")
    for row in rows[1:]:
        body.append(" & ".join(inline(c) for c in row) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(body)


def emit_figure(caption, path):
    return "\n".join([r"\begin{figure}[htbp]", r"\centering",
                      rf"\includegraphics[width=\linewidth]{{\detokenize{{{path}}}}}",
                      rf"\caption{{{inline(caption)}}}", r"\end{figure}"])


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
\title{Query-Free Adversarial Attacks on Stable Diffusion:\\ A Small-Scale Reproduction}
\author{S M Raihanul Alam\\ Department of Computer Science, University of Iowa}
\date{November 2026}
\begin{document}
\maketitle
"""


def main():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    body = []
    i = next(k for k, l in enumerate(lines) if l.startswith("## "))  # skip title and byline
    pending_caption = None
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
        elif line.startswith("## "):
            title = line[3:]
            if title == "Abstract":
                i += 1
                paras = []
                while i < len(lines) and not lines[i].startswith("## "):
                    if lines[i].strip():
                        paras.append(inline(lines[i].strip()))
                    i += 1
                body.append(r"\begin{abstract}" + "\n" + "\n\n".join(paras) + "\n" + r"\end{abstract}")
            elif title in ("References", "Appendix: Software and models") or title.startswith("Appendix"):
                body.append(r"\section*{" + inline(title) + "}")
                i += 1
            else:
                body.append(r"\section{" + inline(re.sub(r"^\d+\.\s*", "", title)) + "}")
                i += 1
        elif line.startswith("### "):
            body.append(r"\subsection{" + inline(re.sub(r"^\d+\.\d+\s*", "", line[4:])) + "}")
            i += 1
        elif line.startswith("!["):
            caption, path = re.match(r"!\[(.+)\]\((.+)\)", line).groups()
            body.append(emit_figure(re.sub(r"^Figure \d+\.\s*", "", caption), path))
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.fullmatch(r"\|(\s*-+\s*\|)+", lines[i].strip()):
                    rows.append(split_row(lines[i]))
                i += 1
            body.append(emit_table(pending_caption or "", rows))
            pending_caption = None
        elif re.match(r"Table \d+\.\s", line):
            pending_caption = re.sub(r"^Table \d+\.\s*", "", line)
            i += 1
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(r"\item " + inline(lines[i][2:]))
                i += 1
            body.append(r"\begin{itemize}" + "\n" + "\n".join(items) + "\n" + r"\end{itemize}")
        elif re.match(r"\[\d+\]\s", line):
            items = []
            while i < len(lines):
                if re.match(r"\[\d+\]\s", lines[i].strip()):
                    items.append(r"\item " + inline(re.sub(r"^\[\d+\]\s*", "", lines[i].strip())))
                elif lines[i].strip():
                    break
                i += 1
            body.append(r"\begin{enumerate}[label={[\arabic*]},leftmargin=*]" + "\n"
                        + "\n".join(items) + "\n" + r"\end{enumerate}")
        else:
            body.append(inline(line))
            i += 1

    OUT.write_text(PREAMBLE + "\n" + "\n\n".join(body) + "\n\n" + r"\end{document}" + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
