# -*- coding: utf-8 -*-
"""Wspólne style i komponenty PDF (reportlab) z obsługą polskich znaków (DejaVu)."""
import os
from functools import partial
from xml.sax.saxutils import escape as _esc

import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, Preformatted, KeepTogether,
                                PageBreak, ListFlowable, ListItem)

# ---------- Fonts (Polish-capable) ----------
_F = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(_F, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(_F, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Italic", os.path.join(_F, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-BoldItalic", os.path.join(_F, "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", os.path.join(_F, "DejaVuSansMono.ttf")))
registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                   italic="DejaVu-Italic", boldItalic="DejaVu-BoldItalic")

# ---------- Colors ----------
NAVY  = HexColor("#0E2238")
BLUE  = HexColor("#2563EB")
GOLD  = HexColor("#B7791F")   # ciemny bursztyn — czytelny na białym
AMBER = HexColor("#FBBF24")
INK   = HexColor("#1A2A3D")
MUTED = HexColor("#5C6B80")
LIGHT = HexColor("#F2F6FB")
LINE  = HexColor("#D8E3F0")
GREEN = HexColor("#16A34A")
CODEBG = HexColor("#F4F6F9")

MARGIN = 2 * cm
CONTENT_W = A4[0] - 2 * MARGIN

# ---------- Styles ----------
body = ParagraphStyle("body", fontName="DejaVu", fontSize=10.3, leading=15.2,
                      textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7)
lead = ParagraphStyle("lead", parent=body, fontSize=11.5, leading=17,
                      textColor=INK, alignment=TA_LEFT, spaceAfter=10)
h1 = ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=16, leading=20,
                    textColor=NAVY, spaceBefore=18, spaceAfter=4)
h2 = ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=12.5, leading=16,
                    textColor=BLUE, spaceBefore=12, spaceAfter=3)
h3 = ParagraphStyle("h3", fontName="DejaVu-Bold", fontSize=11, leading=15,
                    textColor=INK, spaceBefore=8, spaceAfter=2)
cap = ParagraphStyle("cap", fontName="DejaVu-Italic", fontSize=9, leading=12,
                     textColor=MUTED, alignment=TA_CENTER, spaceAfter=10, spaceBefore=3)
bullet = ParagraphStyle("bullet", parent=body, alignment=TA_LEFT, spaceAfter=4)
code = ParagraphStyle("code", fontName="DejaVuMono", fontSize=8.2, leading=11.5,
                      textColor=INK)
cell = ParagraphStyle("cell", fontName="DejaVu", fontSize=9.3, leading=12.5, textColor=INK)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DejaVu-Bold")
cellh = ParagraphStyle("cellh", fontName="DejaVu-Bold", fontSize=9.6, leading=12.5,
                       textColor=HexColor("#FFFFFF"))


# ---------- Flowable helpers ----------
def P(text, style=body):
    return Paragraph(text, style)

def H1(text): return Paragraph(text, h1)
def H2(text): return Paragraph(text, h2)
def H3(text): return Paragraph(text, h3)
def CAP(text): return Paragraph(text, cap)
def SP(h=6): return Spacer(1, h)

def BULLETS(items, style=bullet, gap=4):
    its = []
    for it in items:
        its.append(ListItem(Paragraph(it, style), leftIndent=14, value="•",
                            bulletColor=BLUE, spaceAfter=gap))
    return ListFlowable(its, bulletType="bullet", start="•", leftIndent=12,
                        bulletFontName="DejaVu", bulletColor=BLUE)

def CODE(text, width=CONTENT_W):
    pre = Preformatted(_esc(text), code)
    t = Table([[pre]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODEBG),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t

def IMG(path, width):
    ir = ImageReader(path); iw, ih = ir.getSize()
    return Image(path, width=width, height=width * ih / iw)

def TABLE(rows, col_w, header=True, align_left_cols=None):
    """rows: list of list of strings (header row first). Renders with paragraphs."""
    align_left_cols = align_left_cols or []
    data = []
    for r, row in enumerate(rows):
        line = []
        for c, val in enumerate(row):
            if r == 0 and header:
                line.append(Paragraph(val, cellh))
            else:
                st = cellb if c == 0 and not header else cell
                line.append(Paragraph(val, st))
        data.append(line)
    t = Table(data, colWidths=col_w, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
        ("LINEAFTER", (0, 0), (-2, -1), 0.5, LINE),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                  ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
                  ("TOPPADDING", (0, 0), (-1, 0), 7)]
        style += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), LIGHT])]
    t.setStyle(TableStyle(style))
    return t


def esc(s):
    return _esc(s)


# ---------- Document scaffold ----------
def _footer(canvas, doc, doc_title):
    canvas.saveState()
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 1.2 * cm, doc_title)
    canvas.drawRightString(A4[0] - MARGIN, 1.2 * cm, "Strona %d" % doc.page)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 1.5 * cm, A4[0] - MARGIN, 1.5 * cm)
    canvas.restoreState()


def title_page(doc_type, subtitle):
    """Zwraca listę flowables tworzących stronę tytułową."""
    fl = [SP(70)]
    fl.append(Paragraph(doc_type, ParagraphStyle("tt", fontName="DejaVu-Bold",
              fontSize=13, textColor=GOLD, alignment=TA_CENTER, spaceAfter=22)))
    fl.append(Paragraph("AirSense Weather AI", ParagraphStyle("tbig", fontName="DejaVu-Bold",
              fontSize=30, textColor=NAVY, alignment=TA_CENTER, leading=36, spaceAfter=10)))
    fl.append(Paragraph(subtitle, ParagraphStyle("tsub", fontName="DejaVu", fontSize=13,
              textColor=MUTED, alignment=TA_CENTER, leading=18, spaceAfter=40)))
    fl.append(Table([[""]], colWidths=[6 * cm], rowHeights=[2],
                    style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), AMBER)])))
    fl.append(SP(40))
    meta = [
        ["Projekt", "AirSense Weather AI — system predykcji jakości powietrza i pogody"],
        ["Grupa", "Grupa 5"],
        ["Autorzy", "Jakub Moszyński, Damian Tomczyk, Nikol Olszewska, Łukasz Duss"],
        ["Technologie", "Python, Streamlit, TensorFlow/Keras (LSTM), scikit-learn, Plotly"],
        ["Źródła danych", "GIOŚ API v1, Open-Meteo API"],
        ["Data", "3 czerwca 2026"],
    ]
    rows = [[Paragraph("<b>%s</b>" % k, cell), Paragraph(v, cell)] for k, v in meta]
    t = Table(rows, colWidths=[3.4 * cm, CONTENT_W - 3.4 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
    ]))
    fl.append(t)
    fl.append(PageBreak())
    return fl


def build(path, doc_title, story):
    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title=doc_title, author="Grupa 5")
    cb = partial(_footer, doc_title=doc_title)
    doc.build(story, onFirstPage=cb, onLaterPages=cb)
    print("Zapisano:", path)
