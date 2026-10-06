"""
P0R -- Render docs/PR_DRL_DMOEA_UAV_CURRENT.md to PDF.

Adapted from scripts/render_paper.py (SA-DRL-DMOEA renderer).
Handles the same Markdown subset: headings, tables, bold (**x**),
inline code, bullet lists, numbered lists, horizontal rules (---),
$$...$$ display-math blocks (monospace, not LaTeX), code blocks.

The .md file remains the single canonical source.
Do NOT edit the generated PDF directly.
"""
import os
import re
import hashlib

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, HRFlowable)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

_HERE = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(_HERE, "..", "docs", "PR_DRL_DMOEA_UAV_CURRENT.md")
PDF_PATH = os.path.join(_HERE, "..", "paper",
                        "PR_DRL_DMOEA_UAV_New_Design_2026-09-20_O1.pdf")

_FONT_DIR = "/System/Library/Fonts/Supplemental"
_AU = "/Library/Fonts/Arial Unicode.ttf"
pdfmetrics.registerFont(TTFont("VNSans", _AU))
pdfmetrics.registerFont(TTFont("VNSans-Bold", _AU))
pdfmetrics.registerFont(TTFont("VNSans-Italic",
                                os.path.join(_FONT_DIR, "Arial Italic.ttf")))
pdfmetrics.registerFont(TTFont("VNMono",
                                os.path.join(_FONT_DIR, "Courier New.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("VNSans", normal="VNSans", bold="VNSans-Bold",
                   italic="VNSans-Italic", boldItalic="VNSans-Bold")


def parse_blocks(text):
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

    doc = SimpleDocTemplate(
        PDF_PATH, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title="PR-DRL-DMOEA UAV -- O1 adopted 2026-09-20"
    )
    doc.build(story)
    print(f"wrote {PDF_PATH}")
    sha = hashlib.sha256(open(PDF_PATH, "rb").read()).hexdigest()
    print(f"SHA256: {sha}")
    return sha


def main():
    text = open(MD_PATH, encoding="utf-8").read()
    blocks = parse_blocks(text)
    sha = render_pdf(blocks)
    return sha


if __name__ == "__main__":
    main()
