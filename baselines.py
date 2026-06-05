# -*- coding: utf-8 -*-
"""Baseline'y prognozy ASQI na tym samym splitcie co LSTM."""
import numpy as np

import air_quality
from model_eval import compute_metrics, metrics_by_horizon


def _flat_truth_pred(y_aqi, time_steps, horizon, seq_start, seq_end, predictor):
    """Spłaszcza prawdę i prognozę baseline dla sekwencji [seq_start, seq_end)."""
    y_aqi = np.asarray(y_aqi, dtype=float)
    n_seq = len(y_aqi) - time_steps - horizon + 1
    seq_end = min(seq_end, n_seq)
    truths, preds = [], []
    for i in range(seq_start, seq_end):
        truth = y_aqi[i + time_steps:i + time_steps + horizon]
        pred = predictor(i, y_aqi, time_steps, horizon)
        truths.append(truth)
        preds.append(pred)
    if not truths:
        return np.array([]), np.array([])
    return np.concatenate(truths), np.concatenate(preds)


def _predict_persistence(i, y_aqi, time_steps, horizon):
    last = float(y_aqi[i + time_steps - 1])
    return np.full(horizon, last, dtype=float)


def _predict_ma(i, y_aqi, time_steps, horizon, ma_hours=24):
    w = y_aqi[i:i + time_steps]
    tail = w[-min(ma_hours, len(w)):]
    val = float(np.nanmean(tail))
    return np.full(horizon, val, dtype=float)


def _predict_seasonal_lag(i, y_aqi, time_steps, horizon, lag_hours):
    """Prognoza sezonowa: ta sama wartość sprzed lag_hours (np. 24 h lub 168 h)."""
    out = np.empty(horizon, dtype=float)
    fallback = float(y_aqi[max(0, i + time_steps - 1)])
    lag = int(lag_hours)
    for h in range(horizon):
        t = i + time_steps + h
        if t >= lag:
            out[h] = float(y_aqi[t - lag])
        else:
            out[h] = fallback
    return out


def evaluate_baselines(df_proc, time_steps, horizon, test_seq_start, test_seq_end):
    """Metryki baseline na zbiorze testowym (indeksy sekwencji)."""
    if air_quality.GIOS_AQI_COL not in df_proc.columns:
        return {}
    y_aqi = df_proc[air_quality.GIOS_AQI_COL].values
    t_steps, hor = int(time_steps), int(horizon)
    specs = {
        "persistence": lambda i, y, ts, h: _predict_persistence(i, y, ts, h),
        "ma_24h": lambda i, y, ts, h: _predict_ma(i, y, ts, h, 24),
        "ma_168h": lambda i, y, ts, h: _predict_ma(i, y, ts, h, 168),
        "seasonal_24h": lambda i, y, ts, h: _predict_seasonal_lag(i, y, ts, h, 24),
        "seasonal_168h": lambda i, y, ts, h: _predict_seasonal_lag(i, y, ts, h, 168),
    }
    out = {}
    for name, fn in specs.items():
        yt, yp = _flat_truth_pred(y_aqi, t_steps, hor, test_seq_start, test_seq_end, fn)
        if len(yt) == 0:
            continue
        out[name] = {
            "metrics": compute_metrics(yt, yp),
            "metrics_by_horizon": metrics_by_horizon(
                yt.reshape(-1, hor), yp.reshape(-1, hor),
            ) if hor > 1 else [],
        }
    return out
