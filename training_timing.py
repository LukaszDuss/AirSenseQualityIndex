# -*- coding: utf-8 -*-
"""Pomiar i szacowanie czasu treningu LSTM."""
import time

import numpy as np


def format_duration(seconds):
    """Czytelny czas faktyczny (s / min / h) — po treningu."""
    if seconds is None or seconds < 0:
        return "—"
    sec = float(seconds)
    if sec < 60:
        return f"{sec:.0f} s"
    if sec < 3600:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.2f} h"


def _historical_sec_per_epoch():
    """Mediana sekund/epoka z zapisanych modeli (kalibracja lokalna)."""
    import model_registry

    per_epoch = []
    for name in model_registry.list_models():
        meta = model_registry.load_meta(name)
        if not meta:
            continue
        dur = meta.get("training_duration_sec")
        ep = meta.get("epochs_run") or meta.get("epochs_max")
        if dur and ep and int(ep) > 0:
            per_epoch.append(float(dur) / int(ep))
    if not per_epoch:
        return None
    return float(np.median(per_epoch))


def estimate_training_seconds(
    n_rows,
    time_steps,
    forecast_horizon,
    epochs_max,
    lstm_units,
    n_features,
):
    """Szacunek w sekundach. Zwraca (sekundy, czy_kalibracja_z_historii)."""
    n_rows = max(0, int(n_rows))
    time_steps = max(1, int(time_steps))
    forecast_horizon = max(1, int(forecast_horizon))
    epochs_max = max(1, int(epochs_max))
    lstm_units = max(1, int(lstm_units))
    n_features = max(1, int(n_features))

    n_seq = max(0, n_rows - time_steps - forecast_horizon + 1)
    n_train = max(1, int(n_seq * 0.8))
    effective_epochs = max(5, epochs_max * 0.72)

    hist_spe = _historical_sec_per_epoch()
    if hist_spe is not None:
        scale = (
            (lstm_units / 64.0) ** 1.35
            * (time_steps / 48.0) ** 0.85
            * (n_features / 8.0) ** 0.4
            * (n_train / 400.0) ** 0.55
        )
        sec = hist_spe * effective_epochs * scale + 4.0
        return max(8.0, sec), True

    per_epoch = (
        0.035
        * (n_train / 100.0)
        * (lstm_units / 64.0) ** 1.3
        * (time_steps / 48.0) ** 0.9
        * (n_features / 8.0) ** 0.35
    )
    sec = per_epoch * effective_epochs + 6.0
    return max(10.0, sec), False


def _minutes_range_label(lo_sec, hi_sec):
    """Z przedziału sekund robi etykietę typu '<1 min', '3–5 min'."""
    lo_sec = max(0.0, float(lo_sec))
    hi_sec = max(lo_sec, float(hi_sec))
    hi_m = int(np.ceil(hi_sec / 60.0))
    lo_m = int(lo_sec // 60.0)

    if hi_m < 1:
        return "<1 min"
    if lo_m < 1:
        if hi_m <= 2:
            return "<1–2 min"
        if hi_m <= 5:
            return "<1–5 min"
        return f"<1–{hi_m} min"
    if lo_m >= hi_m:
        hi_m = lo_m + 1
    if lo_m == hi_m:
        return f"{lo_m}–{hi_m + 1} min"
    return f"{lo_m}–{hi_m} min"


def estimate_training_range_label(
    n_rows,
    time_steps,
    forecast_horizon,
    epochs_max,
    lstm_units,
    n_features,
):
    """Przybliżony przedział czasu przed treningiem (bez dokładnych sekund)."""
    sec, calibrated = estimate_training_seconds(
        n_rows, time_steps, forecast_horizon, epochs_max, lstm_units, n_features,
    )
    if calibrated:
        lo, hi = sec * 0.8, sec * 1.35
    else:
        lo, hi = sec * 0.45, sec * 2.2
    return _minutes_range_label(lo, hi)
