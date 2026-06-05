# -*- coding: utf-8 -*-
"""Cechy czasowe i finalizacja ramki pod LSTM (kalendarz, braki)."""
import numpy as np
import pandas as pd

import air_quality

CALENDAR_FEATURE_COLS = (
    "Godzina_sin", "Godzina_cos",
    "DzienTyg_sin", "DzienTyg_cos",
    "Sezon_grzewczy",
)

# Miesiące sezonu grzewczego (PL): październik–kwiecień
_HEATING_MONTHS = {10, 11, 12, 1, 2, 3, 4}


def add_calendar_features(df):
    """Dodaje cechy cykliczne z indeksu datetime (bez NaN)."""
    if df.empty:
        return df
    out = df.copy()
    idx = pd.to_datetime(out.index)
    hour = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek
    out["Godzina_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["Godzina_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["DzienTyg_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["DzienTyg_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["Sezon_grzewczy"] = idx.month.isin(_HEATING_MONTHS).astype(float)
    return out


def finalize_features_for_model(df):
    """Imputacja pogody/opadów; wiersze bez kompletnych zanieczyszczeń EAQI są odrzucane.

    Gdy brakuje całego sensora (kolumna), znika po dropna(axis=1, how='all').
    Gdy brakuje pojedynczych godzin — interpolacja; wiersz pada tylko jeśli nadal NaN
    w kolumnach zanieczyszczeń obecnych w ramce.
    """
    if df.empty:
        return df
    out = df.copy()
    poll_cols = [c for c in out.columns if c in air_quality.INDEX_BREAKPOINTS]
    other = [c for c in out.columns if c not in poll_cols]
    for c in other:
        if out[c].isna().any():
            med = out[c].median()
            fill = med if pd.notna(med) else 0.0
            out[c] = out[c].fillna(fill)
    if poll_cols:
        out = out.dropna(subset=poll_cols, how="any")
    return out
