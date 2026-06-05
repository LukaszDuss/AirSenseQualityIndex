# -*- coding: utf-8 -*-
"""Generuje wizualizacje (PNG) wspólne dla dokumentacji PDF i prezentacji PPTX.
Wykresy są poglądowe (dane syntetyczne o realistycznym kształcie), ale odwzorowują
dokładnie to, co generuje aplikacja AirSense Weather AI."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from datetime import date, timedelta

ASSETS = os.path.dirname(os.path.abspath(__file__))

# --- Paleta marki AirSense (spójna z aplikacją) ---
BG      = "#0E2238"   # głębki granat (tło)
PANEL   = "#16314F"
BLUE    = "#3B82F6"   # odczyty / linia bazowa
RED     = "#FF4B4B"   # trend / prognoza
AMBER   = "#FBBF24"   # linia "Dziś"
GREEN   = "#22C55E"
TEXT    = "#E6EDF5"
MUTED   = "#9FB3C8"
GRID    = "#24405F"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def chart_eda():
    """Zakładka 1: odczyty dzienne + 7-dniowa średnia krocząca + pionowa linia 'Dziś'."""
    rng = np.random.default_rng(7)
    n = 31
    today = date(2026, 6, 3)
    days = [today - timedelta(days=n - 1 - i) for i in range(n)]
    base = 14 + 6 * np.sin(np.linspace(0, 3.1, n)) + np.linspace(0, 4, n)
    readings = base + rng.normal(0, 1.6, n)
    trend = np.convolve(np.r_[[readings[0]] * 6, readings], np.ones(7) / 7, mode="valid")

    fig, ax = plt.subplots(figsize=(10.2, 5.0), dpi=170)
    ax.plot(days, readings, color=BLUE, alpha=0.55, lw=2, label="Odczyty (Temperatura)")
    ax.plot(days, trend, color=RED, lw=3, label="Trend (7-dniowy)")
    ax.axvline(today, color=AMBER, lw=2.2, ls="--")
    ax.text(today, ax.get_ylim()[1], " Dziś", color=AMBER, fontweight="bold",
            ha="right", va="top", fontsize=12)
    _style(ax)
    ax.set_title("EDA i Trendy — odczyty vs. średnia krocząca", color=TEXT, fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Temperatura [°C]")
    leg = ax.legend(loc="upper left", frameon=False)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "eda_trend.png"))
    plt.close(fig)


def chart_prediction():
    """Zakładka 3: ostatnie pomiary + prognoza LSTM połączona z ostatnim punktem + 'Dziś'."""
    today = date(2026, 6, 3)
    t_steps, horizon = 7, 3
    hist_days = [today - timedelta(days=t_steps - 1 - i) for i in range(t_steps)]
    hist = np.array([16.2, 17.0, 18.4, 17.9, 19.1, 20.3, 21.0])
    fc_days = [today + timedelta(days=i) for i in range(1, horizon + 1)]
    fc = np.array([21.6, 22.1, 21.4])

    fig, ax = plt.subplots(figsize=(10.2, 5.0), dpi=170)
    ax.plot(hist_days, hist, color=BLUE, marker="o", lw=2.4, label="Ostatnie pomiary")
    ax.plot([hist_days[-1]] + fc_days, [hist[-1]] + list(fc), color=RED, marker="o",
            ls="--", lw=2.4, label="Prognoza LSTM")
    ax.axvline(today, color=AMBER, lw=2.2, ls=":")
    ax.text(today, ax.get_ylim()[1], "Dziś ", color=AMBER, fontweight="bold",
            ha="left", va="top", fontsize=12)
    _style(ax)
    ax.set_title("Panel predykcji — trajektoria prognozy na kolejne dni", color=TEXT,
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Temperatura [°C]")
    leg = ax.legend(loc="upper left", frameon=False)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "prediction.png"))
    plt.close(fig)


def chart_metrics():
    """Metryki w rozbiciu na cechy (R²) — pokazuje, że pogoda jest łatwiejsza niż pollutanty."""
    feats = ["Temperatura", "Wilgotność", "Wiatr", "PM10", "PM2.5", "NO2"]
    r2 = [0.91, 0.84, 0.62, 0.41, 0.38, 0.29]
    colors = [GREEN if v >= 0.7 else (AMBER if v >= 0.45 else RED) for v in r2]

    fig, ax = plt.subplots(figsize=(10.2, 5.0), dpi=170)
    y = np.arange(len(feats))[::-1]
    ax.barh(y, r2, color=colors, height=0.62)
    for yi, v in zip(y, r2):
        ax.text(v + 0.015, yi, f"{v:.2f}", va="center", color=TEXT, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("R²  (im wyżej, tym lepiej)")
    _style(ax)
    ax.set_title("Dokładność prognozy w rozbiciu na cechy (R²)", color=TEXT,
                 fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "metrics_per_feature.png"))
    plt.close(fig)


def chart_pipeline():
    """Diagram pipeline'u ML: dane -> preprocessing -> sekwencje -> LSTM -> ewaluacja/predykcja."""
    steps = [
        ("Dane\nGIOŚ +\nOpen-Meteo", BLUE),
        ("Preprocessing\ninterpolacja,\noutliery", "#2563EB"),
        ("Sekwencje\ngodzinowe\n(okno 48 h)", "#7C3AED"),
        ("Model LSTM\nSeq2Seq +\nDropout", RED),
        ("Ewaluacja\nMAE / RMSE\n/ R²", AMBER),
        ("Predykcja\nkolejne dni", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 3.0), dpi=170)
    ax.set_xlim(0, len(steps)); ax.set_ylim(0, 1); ax.axis("off")
    for i, (label, col) in enumerate(steps):
        box = FancyBboxPatch((i + 0.06, 0.20), 0.88, 0.60,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             linewidth=0, facecolor=col)
        ax.add_patch(box)
        txtcol = "#0E2238" if col == AMBER else "white"
        ax.text(i + 0.5, 0.50, label, ha="center", va="center", color=txtcol,
                fontsize=9.5, fontweight="bold", linespacing=1.3)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((i + 0.95, 0.50), (i + 1.05, 0.50),
                         arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=2))
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "pipeline.png"))
    plt.close(fig)


def chart_architecture():
    """Diagram architektury modelu LSTM Seq2Seq."""
    layers = [
        ("Wejście:  (okno=48 h,  N cech)", PANEL, TEXT),
        ("LSTM (64),  return_sequences=True", BLUE, "white"),
        ("Dropout 0.2", "#1E3A5F", TEXT),
        ("LSTM (64),  return_sequences=False", BLUE, "white"),
        ("Dropout 0.2", "#1E3A5F", TEXT),
        ("Dense (horyzont × N cech)", RED, "white"),
        ("Reshape → (horyzont, N cech)", AMBER, "#0E2238"),
        ("Wyjście:  prognoza na kolejne godziny", GREEN, "#0E2238"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=170)
    ax.set_xlim(0, 1); ax.set_ylim(0, len(layers)); ax.axis("off")
    for i, (label, col, txt) in enumerate(layers):
        y = len(layers) - 1 - i
        box = FancyBboxPatch((0.08, y + 0.16), 0.84, 0.68,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             linewidth=0, facecolor=col)
        ax.add_patch(box)
        ax.text(0.5, y + 0.5, label, ha="center", va="center", color=txt,
                fontsize=11, fontweight="bold")
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((0.5, y + 0.14), (0.5, y - 0.16),
                         arrowstyle="-|>", mutation_scale=15, color=MUTED, lw=2))
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "architecture.png"))
    plt.close(fig)


if __name__ == "__main__":
    # UWAGA: wykresy danych (eda_trend, prediction, metrics_per_feature) generują skrypty
    # evaluate_live.py i make_metrics_chart.py z REALNYCH danych. Tu tylko diagramy poglądowe,
    # aby nie nadpisać realnych wykresów.
    chart_pipeline()
    chart_architecture()
    print("Zregenerowano diagramy:", "pipeline.png, architecture.png")
