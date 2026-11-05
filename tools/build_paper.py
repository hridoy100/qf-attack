"""Render QF_Attack_Report.md to output/pdf/QF_Attack_Report.pdf."""
from pathlib import Path
import html
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "QF_Attack_Report.md"
OUT = ROOT / "output/pdf/QF_Attack_Report.pdf"

# Font candidates in order of preference: macOS Times New Roman, then common Linux
# equivalents, then the DejaVu fonts bundled with matplotlib (always available).
FONT_CANDIDATES = {
    "Serif": ["Times New Roman.ttf", "LiberationSerif-Regular.ttf", "DejaVuSerif.ttf"],
    "Serif-Bold": ["Times New Roman Bold.ttf", "LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf"],
    "Serif-Italic": ["Times New Roman Italic.ttf", "LiberationSerif-Italic.ttf", "DejaVuSerif-Italic.ttf"],
    "Serif-BoldItalic": ["Times New Roman Bold Italic.ttf", "LiberationSerif-BoldItalic.ttf",
                         "DejaVuSerif-BoldItalic.ttf"],
    "Mono": ["Courier New.ttf", "LiberationMono-Regular.ttf", "DejaVuSansMono.ttf"],
    "Symbol": ["Arial Unicode.ttf", "DejaVuSans.ttf"],
}


def font_dirs():
    import matplotlib
    dirs = [Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
            Path.home() / "Library/Fonts", Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
            Path.home() / ".fonts", Path.home() / ".local/share/fonts"]
    dirs.append(Path(matplotlib.get_data_path()) / "fonts/ttf")
    return [d for d in dirs if d.is_dir()]


def find_font(candidates, dirs):
    for file in candidates:
        for d in dirs:
            hits = [d / file] if (d / file).is_file() else sorted(d.rglob(file))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"None of these fonts were found: {candidates}")


_dirs = font_dirs()
for name, candidates in FONT_CANDIDATES.items():
    pdfmetrics.registerFont(TTFont(name, str(find_font(candidates, _dirs))))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold",
                              italic="Serif-Italic", boldItalic="Serif-BoldItalic")


def style(name, **kw):
    base = dict(fontName="Serif", fontSize=11, leading=14.5)
    base.update(kw)
    return ParagraphStyle(name, **base)


TITLE = style("Title", fontName="Serif-Bold", fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=8)
BYLINE = style("Byline", alignment=TA_CENTER, spaceAfter=16)
H2 = style("H2", fontName="Serif-Bold", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=5, keepWithNext=1)
H3 = style("H3", fontName="Serif-Bold", spaceBefore=8, spaceAfter=3, keepWithNext=1)
BODY = style("Body", alignment=TA_JUSTIFY, spaceAfter=7)
REF = style("Ref", fontSize=10, leading=13, leftIndent=18, firstLineIndent=-18, spaceAfter=4)
CAPTION = style("Caption", fontSize=10, leading=12.5, spaceBefore=4, spaceAfter=4, keepWithNext=1)
FIG_CAPTION = style("FigCaption", fontSize=10, leading=12.5, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10)
CELL = style("Cell", fontSize=9.5, leading=11.5)
HEAD = style("Head", fontName="Serif-Bold", fontSize=9.5, leading=11.5)


def inline(text):
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            code = part[1:-1].replace("\\|", "|")
            out.append(f'<font face="Mono" size="9">{html.escape(code, quote=False)}</font>')
        else:
            piece = html.escape(part.replace("\\|", "|"), quote=False)
            piece = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", piece)
            piece = re.sub(r"d_i,j", "<i>d</i><sub>i,j</sub>", piece)
            piece = re.sub(r"d_i\b", "<i>d</i><sub>i</sub>", piece)
            piece = piece.replace("⊙", '<font face="Symbol">⊙</font>')
            piece = piece.replace("Σ_i", "Σ<sub>i</sub>").replace("62^5", "62<super>5</super>")
            out.append(piece)
    return "".join(out)


def column_widths(rows, available=A4[0] - 144):
    """Size columns to their widest cell; headers may wrap between words."""
    def width(cell, font="Serif", size=9.5):
        if cell.startswith("`"):
            return pdfmetrics.stringWidth(cell.strip("`").replace("\\|", "|"), "Mono", 9)
        return pdfmetrics.stringWidth(cell, font, size)
    widths = []
    for i in range(len(rows[0])):
        body = max(width(row[i]) for row in rows[1:])
        head = max(width(word, "Serif-Bold") for word in rows[0][i].split())
        widths.append(max(body, head) + 9)
    total = sum(widths)
    return widths if total <= available else [w * available / total for w in widths]


def build_table(rows):
    data = [[Paragraph(inline(c), HEAD) for c in rows[0]]]
    data += [[Paragraph(inline(c), CELL) for c in row] for row in rows[1:]]
    table = Table(data, colWidths=column_widths(rows), repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def split_row(line):
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.strip() for c in cells]


def build_story(lines):
    story, i, caption = [], 0, None
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]).replace(": ", ":<br/>", 1), TITLE))
            i += 1
            while not lines[i].strip():
                i += 1
            story.append(Paragraph(inline(lines[i]), BYLINE))
            i += 1
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), H2))
            i += 1
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), H3))
            i += 1
        elif line.startswith("!["):
            alt, path = re.match(r"!\[(.+)\]\((.+)\)", line).groups()
            width, height = ImageReader(str(ROOT / path)).getSize()
            w = 451 if "embedding_analysis" in path else 285
            story.append(KeepTogether([Image(str(ROOT / path), width=w, height=w * height / width),
                                       Paragraph(inline(alt), FIG_CAPTION)]))
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.fullmatch(r"\|(-+\|)+", lines[i].strip()):
                    rows.append(split_row(lines[i]))
                i += 1
            if caption:
                story.append(Paragraph(inline(caption), CAPTION))
            story += [build_table(rows), Spacer(1, 9)]
            caption = None
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(ListItem(Paragraph(inline(lines[i][2:]), style("Item", alignment=TA_JUSTIFY, spaceAfter=3))))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=14, bulletFontSize=9))
            story.append(Spacer(1, 6))
        elif re.match(r"Table \d+\. ", line):
            caption = line
            i += 1
        elif line.startswith("["):
            story.append(Paragraph(inline(line), REF))
            i += 1
        else:
            story.append(Paragraph(inline(line), BODY))
            i += 1
    return story


def page_number(canvas, doc):
    canvas.setFont("Serif", 10)
    canvas.drawCentredString(A4[0] / 2, 30, str(doc.page))


def main():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    title = lines[0].lstrip("# ")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=72, rightMargin=72,
                            topMargin=68, bottomMargin=62, title=title)
    doc.build(build_story(lines), onFirstPage=page_number, onLaterPages=page_number)
    print(OUT)


if __name__ == "__main__":
    main()
