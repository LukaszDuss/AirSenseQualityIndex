# -*- coding: utf-8 -*-
"""Headless: pobiera realne dane stacji, trenuje ten sam model co aplikacja,
liczy REALNE metryki i regeneruje wykresy z prawdziwych danych.
Zapisuje metrics_real.json oraz nadpisuje wykresy w assets/."""
import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date, timedelta
import data_fetch as dfm

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
STATION_NAME, SID, LAT, LON = "Zabrze", 550, 50.3124, 18.7711
TIME_STEPS, HORIZON, UNITS, EPOCHS = 48, 1, 64, 120   # godzinowo: okno 48 h -> nowcasting +1 h (early stopping)

# ---- paleta (jak w make_charts.py) ----
BG="#0E2238"; PANEL="#16314F"; BLUE="#3B82F6"; RED="#FF4B4B"; AMBER="#FBBF24"
GREEN="#22C55E"; TEXT="#E6EDF5"; MUTED="#9FB3C8"; GRID="#24405F"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"text.color":TEXT,
    "axes.labelcolor":TEXT,"xtick.color":MUTED,"ytick.color":MUTED,"axes.edgecolor":GRID,
    "figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG})
def _style(ax):
    ax.grid(True,color=GRID,linewidth=0.7,alpha=0.6)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)

# ---- dane ----
print(f"Pobieram dane: {STATION_NAME} ...")
df = dfm.fetch_real_climate_data(SID, LAT, LON)
feats = df.columns.tolist(); NF = len(feats)
print(f"Cechy ({NF}): {feats}")
print(f"Zakres dat: {df.index[0]} -> {df.index[-1]} ({len(df)} dni)")

scaler = MinMaxScaler(); scaled = scaler.fit_transform(df)
X, y = dfm.prepare_sequences(scaled, TIME_STEPS, HORIZON)
total = len(X); tr = int(total*0.8); va = int(total*0.1)
Xtr, ytr = X[:tr], y[:tr]
Xva, yva = X[tr:tr+va], y[tr:tr+va]
Xte, yte = X[tr+va:], y[tr+va:]
print(f"Sekwencje: {total} (trening {len(Xtr)} / walid. {len(Xva)} / test {len(Xte)})")

model = Sequential([
    LSTM(UNITS, return_sequences=True, input_shape=(TIME_STEPS, NF)),
    Dropout(0.2),
    LSTM(UNITS, return_sequences=False),
    Dropout(0.2),
    Dense(HORIZON*NF),
    Reshape((HORIZON, NF)),
])
model.compile(optimizer="adam", loss="mse")
es = tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)
model.fit(Xtr, ytr, batch_size=16, epochs=EPOCHS, validation_data=(Xva, yva), verbose=0, callbacks=[es])

yp = model.predict(Xte, verbose=0)
yte_real = scaler.inverse_transform(yte.reshape(-1, NF))
yp_real  = scaler.inverse_transform(yp.reshape(-1, NF))

mae = float(mean_absolute_error(yte_real, yp_real))
mse = float(mean_squared_error(yte_real, yp_real))
rmse = float(np.sqrt(mse))
r2 = float(r2_score(yte_real, yp_real))

per = []
for i, f in enumerate(feats):
    yt, ypi = yte_real[:, i], yp_real[:, i]
    r2f = float(r2_score(yt, ypi)) if np.ptp(yt) > 0 else None
    per.append({"cecha": f,
                "mae": float(mean_absolute_error(yt, ypi)),
                "rmse": float(np.sqrt(mean_squared_error(yt, ypi))),
                "r2": r2f})

result = {"stacja": STATION_NAME, "data": str(date.today()), "cechy": feats,
          "n_dni": len(df), "n_test_sekw": len(Xte),
          "overall": {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2},
          "per_feature": per}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_real.json"), "w", encoding="utf-8") as fh:
    json.dump(result, fh, ensure_ascii=False, indent=2)

print("\n=== REALNE METRYKI (zbiór testowy, jednostki rzeczywiste) ===")
print(f"MAE={mae:.3f}  MSE={mse:.3f}  RMSE={rmse:.3f}  R2(śr.)={r2:.3f}")
for p in per:
    r2s = "n/d" if p["r2"] is None else f"{p['r2']:.3f}"
    print(f"  {p['cecha']:>12}: MAE={p['mae']:.3f}  RMSE={p['rmse']:.3f}  R2={r2s}")

# ================= WYKRESY Z REALNYCH DANYCH =================
last_ts = df.index[-1]
midnight = pd.Timestamp(date.today())

# 1) EDA — temperatura godzinowo (widoczny cykl dobowy) + 24-godz. średnia krocząca
feat_eda = "Temperatura"
trend = df[feat_eda].rolling(24, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(10.2, 5.0), dpi=170)
ax.plot(df.index, df[feat_eda], color=BLUE, alpha=0.55, lw=1.3, label=f"Odczyty godzinowe ({feat_eda})")
ax.plot(df.index, trend, color=RED, lw=3, label="Średnia krocząca 24 h")
ax.axvline(midnight, color=AMBER, lw=2.2, ls="--")
ax.text(midnight, ax.get_ylim()[1], " Dziś", color=AMBER, fontweight="bold", ha="right", va="top")
_style(ax); ax.set_title(f"EDA i Trendy — {STATION_NAME} (dane godzinowe, rzeczywiste)", color=TEXT, fontsize=15, fontweight="bold", pad=12)
ax.set_ylabel("Temperatura [°C]")
leg = ax.legend(loc="upper left", frameon=False); [t.set_color(TEXT) for t in leg.get_texts()]
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "eda_trend.png")); plt.close(fig)

# 2) Predykcja vs rzeczywistość na zbiorze testowym (+1 h) — temperatura
fi = feats.index(feat_eda)
times_all = df.index[TIME_STEPS:TIME_STEPS + total]          # znacznik czasu celu (horyzont = 1 h)
test_times = list(times_all[tr + va: tr + va + len(Xte)])
fig, ax = plt.subplots(figsize=(10.2, 5.0), dpi=170)
ax.plot(test_times, yte_real[:, fi], color=BLUE, lw=2.4, label="Rzeczywiste")
ax.plot(test_times, yp_real[:, fi], color=RED, lw=2.2, ls="--", label="Prognoza modelu (+1 h)")
_style(ax); ax.set_title(f"Predykcja vs rzeczywistość — {feat_eda} ({STATION_NAME}, zbiór testowy +1 h)", color=TEXT, fontsize=14.5, fontweight="bold", pad=12)
ax.set_ylabel("Temperatura [°C]")
leg = ax.legend(loc="upper left", frameon=False); [t.set_color(TEXT) for t in leg.get_texts()]
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "prediction.png")); plt.close(fig)

# 3) Metryki per cecha — realne R² (przycięte do -1 na wykresie, wartość anotowana)
labels = [p["cecha"] for p in per]
r2v = [(-1.0 if (p["r2"] is None) else max(p["r2"], -1.0)) for p in per]
true = [p["r2"] for p in per]
colors = [GREEN if (t is not None and t>=0.6) else (AMBER if (t is not None and t>=0.2) else RED) for t in true]
fig, ax = plt.subplots(figsize=(10.2, 5.2), dpi=170)
yy = np.arange(len(labels))[::-1]
ax.barh(yy, r2v, color=colors, height=0.6)
for y_, t, drawn in zip(yy, true, r2v):
    txt = "n/d" if t is None else f"{t:.2f}"
    ax.text(min(drawn+0.03, 0.97), y_, txt, va="center", color=TEXT, fontweight="bold", fontsize=10)
ax.set_yticks(yy); ax.set_yticklabels(labels)
ax.set_xlim(-1.05, 1.0); ax.axvline(0, color=MUTED, lw=1)
ax.set_xlabel("R²  (przycięte do -1; im wyżej, tym lepiej)")
_style(ax); ax.set_title(f"Dokładność per cecha — {STATION_NAME} (dane rzeczywiste)", color=TEXT, fontsize=14, fontweight="bold", pad=12)
fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "metrics_per_feature.png")); plt.close(fig)

print("\nWykresy zaktualizowane z realnych danych. Metryki -> metrics_real.json")
