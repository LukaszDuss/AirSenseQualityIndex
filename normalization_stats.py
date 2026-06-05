# -*- coding: utf-8 -*-
"""Statystyki normalizacji: braki, imputacja, MinMax (warstwa processed)."""
import numpy as np
import pandas as pd

import data_store


def raw_vs_processed_frames(store, df_proc):
    """(df_raw, df_proc_aligned) na wspólnej osi czasu processed."""
    df_raw = data_store.raw_to_dataframe(store)
    if df_proc.empty:
        return df_raw, df_proc
    if df_raw.empty:
        return df_raw, df_proc
    df_raw = df_raw.reindex(df_proc.index)
    return df_raw, df_proc


def column_stats(df_raw, df_proc):
    """Tabela per kolumna: realne, braki, uzupełnione (imputowane)."""
    rows = []
    cols = sorted(set(df_raw.columns) | set(df_proc.columns))
    for c in cols:
        raw_col = df_raw[c] if c in df_raw.columns else pd.Series(dtype=float)
        proc_col = df_proc[c] if c in df_proc.columns else pd.Series(dtype=float)
        n = len(df_proc) if not df_proc.empty else len(raw_col)
        real = int(raw_col.notna().sum()) if not raw_col.empty else 0
        missing = int(raw_col.isna().sum()) if not raw_col.empty else n
        imputed = 0
        if not raw_col.empty and not proc_col.empty:
            aligned = raw_col.reindex(proc_col.index)
            imputed = int((aligned.isna() & proc_col.notna()).sum())
        rows.append({
            "kolumna": c,
            "wierszy": n,
            "realne": real,
            "braki_raw": missing,
            "uzupelnione": imputed,
            "pokrycie_%": round(100 * real / n, 1) if n else 0,
        })
    return pd.DataFrame(rows).set_index("kolumna")


def minmax_scaled_frame(store, df_proc, feature_columns):
    """Ramka cech po MinMax wg parametrów skalera zapisanych w processed (snapshot bazy)."""
    proc = store.get("processed")
    if not proc or not feature_columns:
        return pd.DataFrame()
    feats = [c for c in feature_columns if c in proc.get("columns", [])]
    if not feats:
        return pd.DataFrame()
    sc = proc.get("scaler") or {}
    cols = sc.get("columns", feats)
    mins = sc.get("min", [])
    maxs = sc.get("max", [])
    if not mins or not maxs:
        return pd.DataFrame()
    df = df_proc[feats].copy()
    for i, c in enumerate(cols):
        if c not in df.columns:
            continue
        lo, hi = float(mins[i]), float(maxs[i])
        rng = (hi - lo) or 1.0
        df[c] = (df[c] - lo) / rng
    return df
