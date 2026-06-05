# -*- coding: utf-8 -*-
"""Generuje diagramy PNG do dokumentacji WYMAGANE PLIKI (AirSense 2026)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ASSETS = os.path.dirname(os.path.abspath(__file__))
BG = "#202124"
PANEL = "#3c4043"
BLUE = "#8ab4f8"
RED = "#ff5050"
AMBER = "#fbbc04"
GREEN = "#81c995"
TEXT = "#e8eaed"
MUTED = "#9aa0a6"
GRID = "#5f6368"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
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
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def chart_pipeline():
    steps = [
        ("GIOŚ API\n+ Open-Meteo", BLUE),
        ("raw JSON\n(przyrost)", "#5f6368"),
        ("processed\nimputacja+AQI", "#7c4dff"),
        ("sekwencje\nokno+horyzont", AMBER),
        ("LSTM\n→ ASQI", RED),
        ("ewaluacja\n+ rejestr", GREEN),
        ("klient\n/admin UI", "#34a853"),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 2.8), dpi=170)
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (label, col) in enumerate(steps):
        box = FancyBboxPatch(
            (i + 0.05, 0.18), 0.9, 0.64,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=0, facecolor=col,
        )
        ax.add_patch(box)
        tc = "#202124" if col in (AMBER, GREEN) else "white"
        ax.text(i + 0.5, 0.5, label, ha="center", va="center", color=tc,
                fontsize=9, fontweight="bold", linespacing=1.25)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch(
                (i + 0.96, 0.5), (i + 1.04, 0.5),
                arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.8,
            ))
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "pipeline.png"))
    plt.close(fig)


def chart_architecture():
    layers = [
        ("Wejście: (time_steps × n_features)", PANEL, TEXT),
        ("LSTM (units), return_sequences=True [opcjonalnie 2. warstwa]", BLUE, "white"),
        ("Dropout", "#2d2e31", TEXT),
        ("LSTM (units) lub Dense", BLUE, "white"),
        ("Dense(forecast_horizon) → wektor ASQI", RED, "white"),
        ("Dekodowanie MinMax + clip 0–100", AMBER, "#202124"),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=170)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(layers))
    ax.axis("off")
    for i, (label, col, txt) in enumerate(layers):
        y = len(layers) - 1 - i
        box = FancyBboxPatch(
            (0.08, y + 0.14), 0.84, 0.72,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=0, facecolor=col,
        )
        ax.add_patch(box)
        ax.text(0.5, y + 0.5, label, ha="center", va="center", color=txt,
                fontsize=10, fontweight="bold")
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch(
                (0.5, y + 0.12), (0.5, y - 0.18),
                arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=2,
            ))
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "architecture.png"))
    plt.close(fig)


def chart_sequences():
    """Ilustracja: dlaczego odejmujemy okno + horyzont od liczby godzin."""
    n = 14
    ts, fh = 4, 3
    fig, ax = plt.subplots(figsize=(10, 2.2), dpi=170)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 3)
    ax.axis("off")
    colors = [PANEL] * n
    for i in range(ts):
        colors[i] = BLUE
    for i in range(ts, ts + fh):
        colors[i] = RED
    for j in range(n - ts - fh + 1):
        ax.bar(range(n), [0.6] * n, bottom=2.2 - j * 0.35, color=colors, width=0.85, alpha=0.35 if j else 0.9)
    ax.bar(range(ts), [0.6] * ts, bottom=2.2, color=BLUE, width=0.85, label="okno wejściowe")
    ax.bar(range(ts, ts + fh), [0.6] * fh, bottom=2.2, color=RED, width=0.85, label="horyzont (etykieta)")
    ax.text(n / 2 - 0.5, 2.85, f"Przykład: {n} h, okno={ts} h, horyzont={fh} h → {n - ts - fh + 1} sekwencji",
            ha="center", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "sequences.png"))
    plt.close(fig)


def chart_training_time():
    """Wpływ parametrów na szacowany czas (model heurystyczny)."""
    epochs = 60
    lstm_units = [32, 64, 128, 256]
    sec = []
    for u in lstm_units:
        n_train = 400
        per_epoch = 0.035 * (n_train / 100) * (u / 64) ** 1.3 * (504 / 48) ** 0.9 * (12 / 8) ** 0.35
        eff = max(5, epochs * 0.72)
        sec.append((per_epoch * eff + 6) / 60)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=170)
    ax.bar([str(u) for u in lstm_units], sec, color=[GREEN, BLUE, AMBER, RED], width=0.55)
    for i, v in enumerate(sec):
        ax.text(i, v + 0.3, f"{v:.1f} min", ha="center", fontweight="bold")
    ax.set_xlabel("Neurony LSTM (units)")
    ax.set_ylabel("Szac. czas treningu [min]")
    ax.set_title("Heurystyczny szacunek czasu (bez kalibracji historycznej)", fontweight="bold", pad=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "training_time.png"))
    plt.close(fig)


def chart_indices():
    """Porównanie EuropeanIndex vs GiosAQI na syntetycznych sub-indeksach."""
    subs = np.array([25, 40, 55, 30, 20])
    names = ["PM2.5", "PM10", "NO₂", "O₃", "SO₂"]
    eu = subs.max()
    gios = 0.7 * subs.max() + 0.3 * subs.mean()
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=170)
    y = np.arange(len(names))
    ax.barh(y, subs, color=BLUE, height=0.5, label="Sub-indeksy")
    ax.axvline(eu, color=RED, lw=2.5, ls="--", label=f"EuropeanIndex (max) = {eu:.0f}")
    ax.axvline(gios, color=AMBER, lw=2.5, ls=":", label=f"GiosAQI (0.7·max+0.3·śr.) = {gios:.0f}")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Indeks 0–100")
    ax.set_title("Agregacja wskaźników ze stężeń", fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "indices.png"))
    plt.close(fig)


def chart_client_windows():
    """Okna 24 h ASQI — ciągłość między dniami (nie kalendarz)."""
    rng = np.random.default_rng(42)
    h = 72
    t = np.arange(h)
    v = 35 + 15 * np.sin(t / 8) + rng.normal(0, 2, h)
    v = np.clip(v, 0, 100)
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=170)
    ax.fill_between(t, v, alpha=0.25, color=GREEN)
    ax.plot(t, v, color=GREEN, lw=2)
    for d in range(3):
        s, e = d * 24, (d + 1) * 24 - 1
        ax.axvspan(s, e, alpha=0.08, color=AMBER if d == 1 else PANEL)
        ax.text(s + 12, 52, f"Dzień {d + 1}\n24 h", ha="center", fontsize=9, color=MUTED)
    ax.set_xlabel("Godzina prognozy LSTM (kolejno)")
    ax.set_ylabel("ASQI")
    ax.set_title("Widok klienta: sekwencyjne okna 24 h (nie północ–północ)", fontweight="bold", pad=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "client_windows.png"))
    plt.close(fig)


if __name__ == "__main__":
    chart_pipeline()
    chart_architecture()
    chart_sequences()
    chart_training_time()
    chart_indices()
    chart_client_windows()
    print("Wygenerowano PNG w", ASSETS)
