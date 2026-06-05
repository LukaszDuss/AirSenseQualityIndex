# -*- coding: utf-8 -*-
"""Stabilny wykres błędu względnego (nMAE %) per cecha z metrics_real.json (bez ponownego treningu)."""
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data_fetch as dfm

HERE = os.path.dirname(os.path.abspath(__file__)); ASSETS = os.path.join(HERE, "assets")
BG="#0E2238"; BLUE="#3B82F6"; AMBER="#FBBF24"; GREEN="#22C55E"; RED="#FF4B4B"; TEXT="#E6EDF5"; MUTED="#9FB3C8"; GRID="#24405F"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":TEXT,"axes.labelcolor":TEXT,
    "xtick.color":MUTED,"ytick.color":MUTED,"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG})

with open(os.path.join(HERE, "metrics_real.json"), encoding="utf-8") as fh:
    M = json.load(fh)
df = dfm.fetch_real_climate_data(550, 50.3124, 18.7711)
means = df.abs().mean()

CAP = 60.0  # NO ma znikomą średnią -> błąd względny ~200%; przycinamy słupek dla czytelności
rows = [(p["cecha"], 100.0 * p["mae"] / means[p["cecha"]]) for p in M["per_feature"]]
labels = [r[0] for r in rows]; nmae = [r[1] for r in rows]
drawn = [min(v, CAP) for v in nmae]
colors = [GREEN if v < 15 else (AMBER if v < 30 else RED) for v in nmae]

fig, ax = plt.subplots(figsize=(10.2, 5.2), dpi=170)
yy = np.arange(len(labels))[::-1]
ax.barh(yy, drawn, color=colors, height=0.62)
for y_, v, d in zip(yy, nmae, drawn):
    ax.text(min(d + 1.0, CAP - 1), y_, f"{v:.0f}%", va="center", color=TEXT, fontweight="bold", fontsize=10)
ax.set_yticks(yy); ax.set_yticklabels(labels)
ax.set_xlabel("Średni błąd względny  nMAE = MAE / średnia  [%]  (mniej = lepiej; >60% przycięte)")
ax.set_xlim(0, CAP * 1.08)
ax.grid(True, axis="x", color=GRID, linewidth=0.7, alpha=0.6)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.set_title(f"Błąd prognozy per cecha — {M['stacja']} (dane rzeczywiste, +1 h)",
             color=TEXT, fontsize=14, fontweight="bold", pad=12)
fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "metrics_per_feature.png")); plt.close(fig)
print("nMAE%:", {l: round(v) for l, v in rows})
