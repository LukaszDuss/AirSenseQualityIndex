# -*- coding: utf-8 -*-
"""Wskaźniki jakości powietrza AirSense.

Dwa wskaźniki na wspólnej skali ciągłej 0–100 (patrz METODYKA_AirSenseQuality.md):
  * EuropeanIndex   — wariant EAQI: decyduje NAJGORSZY parametr (max sub-indeksów),
  * GiosAQI (GIOŚ AQI) — kompozyt ze stężeń pomiarowych stacji GIOŚ: 0.7·max + 0.3·średnia
                      sub-indeksów. Kolumna w warstwie processed (pomiar, nie prognoza).
  * AirSenseQualityIndex (ASQI) — ta sama formuła co GiosAQI, ale nazwa zarezerwowana
                      wyłącznie dla prognozy LSTM (nie zapisujemy jej w processed).

Ten sam wzór można policzyć ze stężeń prognozowanych przez Open-Meteo — wtedy zapis
w kolumnie OpenMeteoCompositeIndex.

Sub-indeks per zanieczyszczenie liczony jest przez interpolację liniową stężenia [µg/m³]
w pasmach europejskich (EAQI) — bez konwersji jednostek, spójnie z danymi GIOŚ.
Moduł niezależny od Streamlita (testowalny headless).
"""
import numpy as np
import pandas as pd

# Wagi autorskiego kompozytu (parametry projektowe — łatwe do zmiany).
W_MAX = 0.7
W_MEAN = 0.3

# Punkty indeksu odpowiadające 7 granicom 6 pasm EAQI.
_INDEX_ANCHORS = np.linspace(0, 100, 7)  # [0, 16.7, 33.3, 50, 66.7, 83.3, 100]

# Pasma stężeń [µg/m³] dla 5 zanieczyszczeń objętych EAQI. Klucze = kody GIOŚ po
# usunięciu kropek (np. "PM2.5" -> "PM25").
INDEX_BREAKPOINTS = {
    "PM25": [0, 10, 20, 25, 50, 75, 800],
    "PM10": [0, 20, 40, 50, 100, 150, 1200],
    "NO2":  [0, 40, 90, 120, 230, 340, 1000],
    "O3":   [0, 50, 100, 130, 240, 380, 800],
    "SO2":  [0, 100, 200, 350, 500, 750, 1250],
}

# Klasy EAQI (1–6) na skali 0–100.
CLASS_BOUNDS = _INDEX_ANCHORS  # granice klas pokrywają się z kotwicami pasm
CLASS_LABELS = {
    1: "Bardzo dobry",
    2: "Dobry",
    3: "Umiarkowany",
    4: "Dostateczny",
    5: "Zły",
    6: "Bardzo zły",
}
CLASS_COLORS = {
    1: "#50f0e6", 2: "#50ccaa", 3: "#f0e641",
    4: "#ff5050", 5: "#960032", 6: "#7d2181",
}

GIOS_AQI_COL = "GiosAQI"
AQI_COL = "AirSenseQualityIndex"
LEGACY_AQI_COL = "AirSenseQuality"
OM_COMPOSITE_COL = "OpenMeteoCompositeIndex"
EU_COL = "EuropeanIndex"
ASQI_SHORT = "ASQI"
ASQI_LABEL = "AirSenseQualityIndex (ASQI)"
GIOS_AQI_LABEL = "GIOŚ AQI"
OM_COMPOSITE_LABEL = "Indeks z prognozy stężeń (Open-Meteo, godz.)"


def index_pollutant_columns(columns):
    """Spośród kolumn ramki wybiera te, które są zanieczyszczeniami objętymi EAQI."""
    return [c for c in columns if c in INDEX_BREAKPOINTS]


def pollutant_subindex(param_code, concentration):
    """Sub-indeks 0–100 dla pojedynczego zanieczyszczenia (interpolacja w pasmach EAQI)."""
    if param_code not in INDEX_BREAKPOINTS:
        return np.nan
    bps = INDEX_BREAKPOINTS[param_code]
    return float(np.interp(concentration, bps, _INDEX_ANCHORS))


def _subindex_array(param_code, values):
    """Wektorowy sub-indeks dla serii stężeń danego zanieczyszczenia."""
    bps = INDEX_BREAKPOINTS[param_code]
    arr = np.asarray(values, dtype=float)
    return np.interp(arr, bps, _INDEX_ANCHORS)


def european_class(value):
    """Mapuje wartość 0–100 na klasę EAQI 1–6."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    # granice: (0,16.7,33.3,50,66.7,83.3,100] -> klasy 1..6
    for k in range(1, 6):
        if value <= CLASS_BOUNDS[k]:
            return k
    return 6


def class_label(klasa):
    if klasa is None or (isinstance(klasa, float) and np.isnan(klasa)):
        return "—"
    return CLASS_LABELS.get(int(klasa), "—")


def class_color(klasa):
    if klasa is None or (isinstance(klasa, float) and np.isnan(klasa)):
        return "#888888"
    return CLASS_COLORS.get(int(klasa), "#888888")


def compute_subindices(df):
    """Zwraca DataFrame sub-indeksów (0–100) dla zanieczyszczeń obecnych w df."""
    cols = index_pollutant_columns(df.columns)
    sub = pd.DataFrame(index=df.index)
    for c in cols:
        sub[c] = _subindex_array(c, df[c].values)
    return sub


def compute_indices(df):
    """Dolicza GiosAQI, EuropeanIndex i EuropeanClass do kopii df.

    Wymaga co najmniej jednego zanieczyszczenia objętego EAQI. Gdy ich brak,
    wskaźniki będą NaN (aplikacja powinna to obsłużyć komunikatem).
    """
    out = df.copy()
    sub = compute_subindices(df)
    if sub.shape[1] == 0:
        out[GIOS_AQI_COL] = np.nan
        out[EU_COL] = np.nan
        out["EuropeanClass"] = np.nan
        return out

    sub_max = sub.max(axis=1)
    sub_mean = sub.mean(axis=1)
    out[EU_COL] = sub_max
    out[GIOS_AQI_COL] = W_MAX * sub_max + W_MEAN * sub_mean
    out["EuropeanClass"] = out[EU_COL].apply(european_class)
    return out


def has_index_pollutants(columns):
    """Czy w danych jest cokolwiek, z czego można policzyć indeks."""
    return len(index_pollutant_columns(columns)) > 0


def rename_legacy_index_columns(df):
    """Migracja starych nazw w processed → GiosAQI (pomiar GIOŚ, nie ASQI/LSTM)."""
    if df is None or df.empty:
        return df
    renames = {}
    if LEGACY_AQI_COL in df.columns and GIOS_AQI_COL not in df.columns:
        renames[LEGACY_AQI_COL] = GIOS_AQI_COL
    if AQI_COL in df.columns and GIOS_AQI_COL not in df.columns:
        renames[AQI_COL] = GIOS_AQI_COL
    if renames:
        df = df.rename(columns=renames)
    if AQI_COL in df.columns and GIOS_AQI_COL in df.columns:
        df = df.drop(columns=[AQI_COL])
    return df


def rename_legacy_aqi_column(df):
    """Alias wstecznej zgodności — deleguje do rename_legacy_index_columns."""
    return rename_legacy_index_columns(df)
