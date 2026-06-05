# -*- coding: utf-8 -*-
import os, glob, sys
import pypdfium2 as pdfium

HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "render_pdf")
os.makedirs(out, exist_ok=True)
for f in glob.glob(os.path.join(out, "*.png")):
    os.remove(f)

pdfs = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "*.pdf")))
for path in pdfs:
    name = os.path.splitext(os.path.basename(path))[0]
    pdf = pdfium.PdfDocument(path)
    for i in range(len(pdf)):
        bmp = pdf[i].render(scale=1.9)
        bmp.to_pil().save(os.path.join(out, f"{name}_p{i+1:02d}.png"))
    print(f"{name}: {len(pdf)} stron")
    pdf.close()
print("OUT:", out)
