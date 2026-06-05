# -*- coding: utf-8 -*-
"""Renderuje slajdy PPTX do PNG przez PowerPoint COM (do wizualnego QA)."""
import os, sys, glob, win32com.client

HERE = os.path.dirname(os.path.abspath(__file__))
pptx = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "Prezentacja_AirSense_Weather_AI.pptx")
out = os.path.join(HERE, "render")
os.makedirs(out, exist_ok=True)
for f in glob.glob(os.path.join(out, "*.PNG")) + glob.glob(os.path.join(out, "*.png")):
    os.remove(f)

ppt = win32com.client.Dispatch("PowerPoint.Application")
ppt.Visible = True
pres = ppt.Presentations.Open(pptx, ReadOnly=True, WithWindow=False)
try:
    pres.Export(out, "PNG", 1600, 900)
finally:
    pres.Close()
    ppt.Quit()
pngs = sorted(glob.glob(os.path.join(out, "*.PNG")) + glob.glob(os.path.join(out, "*.png")))
print("PNG:", len(pngs))
for p in pngs:
    print(p)
