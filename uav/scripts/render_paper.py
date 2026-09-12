"""
6.0 -- Render docs/SA_DRL_DMOEA_UAV_CURRENT.md to .docx and .pdf.

No pandoc/LaTeX toolchain available in this environment, so this is a
minimal, dependency-light Markdown -> DOCX (python-docx) and
Markdown -> PDF (reportlab) renderer. It handles exactly the subset of
Markdown used in that one file: headings (#.....######), tables
(| ... |), bold (**x**), inline code (`x`), bullet lists (- x),
horizontal rules (---), and $$...$$ display-math blocks (rendered as
monospace text, not typeset LaTeX -- there is no LaTeX engine here).

This is NOT a general Markdown renderer. Re-run after every edit to
the .md source; the .md file remains the single canonical source.
"""
import os
import re

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, PageBreak, HRFlowable)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

_HERE = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(_HERE, "..", "docs", "SA_DRL_DMOEA_UAV_CURRENT.md")
DOCX_PATH = os.path.join(_HERE, "..", "docs", "SA_DRL_DMOEA_UAV_CURRENT.docx")
PDF_PATH = os.path.join(_HERE, "..", "docs", "SA_DRL_DMOEA_UAV_CURRENT.pdf")

# Vietnamese diacritics + math symbols (subscripts, in) need a broad
# Unicode TTF -- ReportLab's built-in Helvetica/Courier are Latin-1 only
# and plain Arial.ttf is missing subscript digits / set-membership signs.
_FONT_DIR = "/System/Library/Fonts/Supplemental"
_AU = "/Library/Fonts/Arial Unicode.ttf"
pdfmetrics.registerFont(TTFont("VNSans", _AU))
pdfmetrics.registerFont(TTFont("VNSans-Bold", _AU))       # AU has no separate bold file
pdfmetrics.registerFont(TTFont("VNSans-Italic", os.path.join(_FONT_DIR, "Arial Italic.ttf")))
pdfmetrics.registerFont(TTFont("VNMono", os.path.join(_FONT_DIR, "Courier New.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402
registerFontFamily("VNSans", normal="VNSans", bold="VNSans-Bold",
                   italic="VNSans-Italic", boldItalic="VNSans-Bold")


def parse_blocks(text):
    """Split the markdown into a list of (kind, payload) blocks."""
    lines = text.split("\n")
    blocks = []
    i = 0
    para_buf = []

    def flush_para():
        if para_buf:
            blocks.append(("para", " ".join(para_buf).strip()))
            para_buf.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "":
            flush_para()
            i += 1
            continue

        if stripped.startswith("$$") and stripped.count("$$") < 2:
            flush_para()
            math_lines = [stripped.lstrip("$")]
            i += 1
            while i < len(lines) and "$$" not in lines[i]:
                math_lines.append(lines[i])
                i += 1
            if i < len(lines):
                math_lines.append(lines[i].replace("$$", ""))
                i += 1
            blocks.append(("math", "\n".join(l for l in math_lines if l.strip())))
            continue

        if stripped.startswith("$$") and stripped.count("$$") == 2:
            flush_para()
            blocks.append(("math", stripped.strip("$")))
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_para()
            blocks.append(("h%d" % len(m.group(1)), m.group(2).strip()))
            i += 1
            continue

        if stripped == "---":
            flush_para()
            blocks.append(("hr", None))
            i += 1
            continue

        if stripped.startswith("|"):
            flush_para()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = []
            for tl in table_lines:
                if re.match(r"^\|[\s:|-]+\|$", tl):
                    continue
                cells = [c.strip() for c in tl.strip("|").split("|")]
                rows.append(cells)
            blocks.append(("table", rows))
            continue

        if re.match(r"^[-*]\s+", stripped):
            flush_para()
            bullet_lines = []
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                bullet_lines.append(re.sub(r"^[-*]\s+", "", lines[i].strip()))
                i += 1
            blocks.append(("list", bullet_lines))
            continue

        if re.match(r"^\d+\.\s+", stripped) and "```" not in stripped:
            flush_para()
            num_lines = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                num_lines.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            blocks.append(("numlist", num_lines))
            continue

        if stripped.startswith("```"):
            flush_para()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", "\n".join(code_lines)))
            continue

        if stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
            flush_para()
            blocks.append(("italic", stripped.strip("*")))
            i += 1
            continue

        para_buf.append(stripped)
        i += 1

    flush_para()
    return blocks


def strip_inline_md(s):
    """Remove markdown emphasis markers for plain-text contexts (not used
    where we keep **bold** runs, e.g. DOCX paragraphs)."""
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"`(.*?)`", r"\1", s)
    return s


# ---------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------
def add_runs_with_bold(paragraph, text):
    parts = re.split(r"(\*\*.*?\*\*|`.*?`)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = paragraph.add_run(part[1:-1])
            r.font.name = "Courier New"
        else:
            paragraph.add_run(part)


def render_docx(blocks):
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    for kind, payload in blocks:
        if re.match(r"^h[1-6]$", kind):
            level = int(kind[1])
            doc.add_heading(strip_inline_md(payload), level=min(level, 4))
        elif kind == "para":
            p = doc.add_paragraph()
            add_runs_with_bold(p, payload)
        elif kind == "list":
            for item in payload:
                p = doc.add_paragraph(style="List Bullet")
                add_runs_with_bold(p, item)
        elif kind == "numlist":
            for item in payload:
                p = doc.add_paragraph(style="List Number")
                add_runs_with_bold(p, item)
        elif kind == "hr":
            doc.add_paragraph("_" * 60)
        elif kind == "code":
            p = doc.add_paragraph()
            r = p.add_run(payload)
            r.font.name = "Courier New"
            r.font.size = Pt(9)
        elif kind == "math":
            p = doc.add_paragraph()
            r = p.add_run(payload.strip())
            r.font.name = "Courier New"
            r.font.size = Pt(10)
            r.italic = True
        elif kind == "italic":
            p = doc.add_paragraph()
            r = p.add_run(strip_inline_md(payload))
            r.italic = True
        elif kind == "table":
            rows = payload
            if not rows:
                continue
            ncols = max(len(r) for r in rows)
            table = doc.add_table(rows=len(rows), cols=ncols)
            table.style = "Light Grid Accent 1"
            for ri, row in enumerate(rows):
                for ci in range(ncols):
                    cell_text = row[ci] if ci < len(row) else ""
                    cell = table.cell(ri, ci)
                    cell.text = ""
                    p = cell.paragraphs[0]
                    add_runs_with_bold(p, strip_inline_md_keep_bold(cell_text))
                    if ri == 0:
                        for run in p.runs:
                            run.bold = True
            doc.add_paragraph()
        elif kind == "para" and payload == "":
            pass

    doc.save(DOCX_PATH)
    print(f"wrote {DOCX_PATH}")


def strip_inline_md_keep_bold(s):
    # keep ** markers (handled by add_runs_with_bold) but drop backticks
    return s


# ---------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------
def render_pdf(blocks):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="VNSans",
                          fontSize=9.5, leading=13, spaceAfter=6,
                          alignment=TA_LEFT)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="VNSans-Bold",
                        fontSize=16, spaceBefore=14, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="VNSans-Bold",
                        fontSize=13, spaceBefore=12, spaceAfter=6)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontName="VNSans-Bold",
                        fontSize=11.5, spaceBefore=10, spaceAfter=5)
    h4 = ParagraphStyle("h4", parent=styles["Heading4"], fontName="VNSans-Bold",
                        fontSize=10.5, spaceBefore=8, spaceAfter=4)
    mono = ParagraphStyle("mono", parent=styles["Code"], fontName="VNMono",
                          fontSize=8.5, leading=11, spaceAfter=6,
                          backColor=colors.whitesmoke)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=14,
                            bulletIndent=4)
    tablecell = ParagraphStyle("tablecell", parent=body, fontName="VNSans",
                               fontSize=7.8, leading=10)

    def inline(text):
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"`(.*?)`", r"<font face='VNMono'>\1</font>", text)
        return text

    story = []
    for kind, payload in blocks:
        if kind == "h1":
            story.append(Paragraph(inline(payload), h1))
        elif kind == "h2":
            story.append(Paragraph(inline(payload), h2))
        elif kind == "h3":
            story.append(Paragraph(inline(payload), h3))
        elif kind in ("h4", "h5", "h6"):
            story.append(Paragraph(inline(payload), h4))
        elif kind == "para":
            story.append(Paragraph(inline(payload), body))
        elif kind == "list":
            for item in payload:
                story.append(Paragraph("&bull; " + inline(item), bullet))
        elif kind == "numlist":
            for n, item in enumerate(payload, 1):
                story.append(Paragraph(f"{n}. " + inline(item), bullet))
        elif kind == "hr":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", color=colors.grey))
            story.append(Spacer(1, 8))
        elif kind == "code":
            story.append(Paragraph(payload.replace("\n", "<br/>"), mono))
        elif kind == "math":
            story.append(Paragraph(payload.strip().replace("\n", "<br/>"), mono))
        elif kind == "italic":
            story.append(Paragraph("<i>" + inline(payload) + "</i>", body))
        elif kind == "table":
            rows = payload
            if not rows:
                continue
            ncols = max(len(r) for r in rows)
            data = []
            for row in rows:
                cells = list(row) + [""] * (ncols - len(row))
                data.append([Paragraph(inline(c), tablecell) for c in cells])
            avail_width = A4[0] - 3.0 * cm
            col_w = avail_width / ncols
            t = Table(data, colWidths=[col_w] * ncols, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe5f1")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.8),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t)
            story.append(Spacer(1, 8))

    doc = SimpleDocTemplate(PDF_PATH, pagesize=A4,
                            leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="SA-DRL-DMOEA UAV -- current status")
    doc.build(story)
    print(f"wrote {PDF_PATH}")


def main():
    text = open(MD_PATH, encoding="utf-8").read()
    blocks = parse_blocks(text)
    render_docx(blocks)
    render_pdf(blocks)


if __name__ == "__main__":
    main()
