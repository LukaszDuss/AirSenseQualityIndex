# -*- coding: utf-8 -*-
"""Wspólne UI, ładowanie danych i trening LSTM."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from keras.losses import Loss
from datetime import timedelta, datetime
import time

import plotly.graph_objects as go

from data_fetch import (
    update_and_load, prepare_xy, fetch_forecast_bundle, WEATHER_COLS,
    OFFICIAL_EU_COL, OFFICIAL_US_COL,
)
import feature_engineering
import baselines
import data_store
import air_quality
import config
import model_registry
import model_eval
import training_timing
import ui_icons as mi

AQI_COL = air_quality.AQI_COL
GIOS_AQI_COL = air_quality.GIOS_AQI_COL
OM_COMPOSITE_COL = air_quality.OM_COMPOSITE_COL
ASQI_SHORT = air_quality.ASQI_SHORT
ASQI_LABEL = air_quality.ASQI_LABEL
GIOS_AQI_LABEL = air_quality.GIOS_AQI_LABEL
OM_COMPOSITE_LABEL = air_quality.OM_COMPOSITE_LABEL
CHART_TZ = "Europe/Warsaw"
EU_COL = air_quality.EU_COL
INDEX_COLS = [GIOS_AQI_COL, EU_COL, "EuropeanClass"]
PL_WEEKDAYS = {0: "pon", 1: "wt", 2: "śr", 3: "czw", 4: "pt", 5: "sob", 6: "niedz"}

EXTRA_AQI_SERIES = {
    "Indeks europejski (nasz)": (EU_COL, "#0ea5e9"),
    "Oficjalny europejski AQI (CAMS)": (OFFICIAL_EU_COL, "#a855f7"),
    "Oficjalny US AQI (EPA)": (OFFICIAL_US_COL, "#f59e0b"),
}


def _normalize_chart_index(index, tz=CHART_TZ):
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is None:
        return idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    return idx.tz_convert(tz)


def _prepare_chart_df(df, tz=CHART_TZ):
    if df is None or df.empty:
        return df
    out = df.copy()
    out.index = _normalize_chart_index(out.index, tz)
    return out.sort_index()


def _chart_now(tz=CHART_TZ):
    return pd.Timestamp.now(tz=tz)


def _clip_time_range(df, t_start, t_end):
    if df is None or df.empty:
        return df
    return df.loc[(df.index >= t_start) & (df.index <= t_end)]


def forecast_future_only(df_fc, now=None):
    """Tylko godziny prognozy (po „teraz”) — np. kafelki 7-dniowe u klienta."""
    if df_fc is None or df_fc.empty:
        return df_fc
    df_fc = _prepare_chart_df(df_fc)
    now = now or _chart_now()
    return df_fc.loc[df_fc.index > now]


def build_hourly_forecast_figure(
    df_fc,
    df_proc,
    *,
    horizon_days=None,
    extra_labels=(),
    show_openmeteo_aqi=True,
    show_gios_aqi=True,
    lstm_model_name=None,
):
    """Wykres godzinowy ±horizon_days: GIOŚ AQI wstecz, Open-Meteo wstecz/wprzód, ASQI LSTM wprzód."""
    fig = go.Figure()
    uses_secondary = False
    df_fc = _prepare_chart_df(df_fc if df_fc is not None else pd.DataFrame())
    df_proc = _prepare_chart_df(df_proc if df_proc is not None else pd.DataFrame())
    horizon_days = horizon_days or config.CLIENT_FORECAST_DAYS
    horizon_h = int(horizon_days * 24)

    now = _chart_now()
    t_start = now - pd.Timedelta(hours=horizon_h)
    t_end = now + pd.Timedelta(hours=horizon_h)
    df_fc = _clip_time_range(df_fc, t_start, t_end)

    if show_gios_aqi and not df_proc.empty and GIOS_AQI_COL in df_proc.columns:
        gios_past = df_proc.loc[
            (df_proc.index >= t_start) & (df_proc.index <= now), GIOS_AQI_COL
        ].dropna()
        if not gios_past.empty:
            fig.add_trace(go.Scatter(
                x=gios_past.index,
                y=gios_past.values,
                mode="lines",
                name=GIOS_AQI_LABEL,
                line=dict(color="#38bdf8", width=3),
            ))

    if show_openmeteo_aqi and not df_fc.empty and OM_COMPOSITE_COL in df_fc.columns:
        om = df_fc[OM_COMPOSITE_COL].dropna(how="all")
        past_om = om.loc[(om.index >= t_start) & (om.index <= now)].dropna()
        future_om = om.loc[(om.index > now) & (om.index <= t_end)].dropna()
        if not past_om.empty:
            fig.add_trace(go.Scatter(
                x=past_om.index,
                y=past_om.values,
                mode="lines",
                name=f"{OM_COMPOSITE_LABEL} — wstecz",
                line=dict(color="#14b8a6", width=2, dash="dot"),
            ))
        if not future_om.empty:
            fig.add_trace(go.Scatter(
                x=future_om.index,
                y=future_om.values,
                mode="lines",
                name=f"{OM_COMPOSITE_LABEL} — wprzód",
                line=dict(color="#14b8a6", width=3),
            ))

    for lbl in extra_labels:
        spec = EXTRA_AQI_SERIES.get(lbl)
        if not spec or df_fc.empty:
            continue
        col, color = spec
        if col not in df_fc.columns:
            continue
        is_us = col == OFFICIAL_US_COL
        uses_secondary = uses_secondary or is_us
        fig.add_trace(go.Scatter(
            x=df_fc.index,
            y=df_fc[col],
            mode="lines",
            name=lbl,
            line=dict(color=color, width=1.6, dash="dot"),
            yaxis="y2" if is_us else "y",
        ))

    if lstm_model_name and not df_proc.empty:
        meta = model_registry.load_meta(lstm_model_name)
        fc = lstm_forecast(lstm_model_name, meta, df_proc) if meta else None
        if fc is not None:
            fc = fc.copy()
            fc.index = _normalize_chart_index(fc.index)
            fc = fc.loc[(fc.index > now) & (fc.index <= t_end)]
            if not fc.empty:
                fig.add_trace(go.Scatter(
                    x=fc.index,
                    y=fc.values,
                    mode="lines+markers",
                    name=f"{ASQI_SHORT} (LSTM)",
                    line=dict(color="#ef4444", width=3),
                ))

    layout = dict(
        template="plotly_dark",
        hovermode="x unified",
        xaxis=dict(type="date", range=[t_start, t_end]),
        yaxis=dict(title="Indeks 0–100"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        shapes=[dict(
            type="line",
            x0=now,
            x1=now,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#64748b", width=1, dash="dash"),
        )],
        annotations=[dict(
            x=now,
            y=1,
            yref="paper",
            text="Teraz",
            showarrow=False,
            font=dict(color="#94a3b8", size=11),
            yanchor="bottom",
        )],
    )
    if uses_secondary:
        layout["yaxis2"] = dict(title="US AQI", overlaying="y", side="right")
    fig.update_layout(**layout)
    return fig


def qp_get(key, default=""):
    v = st.query_params.get(key)
    if isinstance(v, list):
        return v[0] if v else default
    return v if v else default


def inject_css():
    st.markdown("""
    <style>
    .aq-header { background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        color: white; padding: 22px; border-radius: 15px; margin-bottom: 18px; }
    .aq-badge { padding: 6px 14px; border-radius: 10px; color: white; font-weight: 700;
        display: inline-block; }
    .aq-card { background:#1e293b; border-radius:16px; padding:20px 24px; color:#e2e8f0; }
    .aq-now { font-size:64px; font-weight:800; line-height:1; }
    .aq-daycol { background:#1e293b; border-radius:12px; padding:10px 4px;
        text-align:center; color:#e2e8f0; }
    .aq-chip { border-radius:8px; padding:4px 0; font-weight:700; color:#0b1220; margin:6px 4px; }
    div[data-testid="stSidebar"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }
    .aq-model-actions [data-testid="column"] { min-width: 0; padding: 0 2px; }
    .aq-model-actions button {
        white-space: nowrap !important;
        font-size: 0.78rem !important;
        padding: 0.35rem 0.5rem !important;
    }
    .aq-llm-recommend [data-testid="stBaseButton-primary"] {
        background-color: #dc2626 !important;
        border-color: #b91c1c !important;
        color: #fff !important;
    }
    .aq-llm-recommend [data-testid="stBaseButton-primary"]:hover {
        background-color: #b91c1c !important;
        border-color: #991b1b !important;
    }
    .aq-model-row-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        min-height: 2.35rem;
        margin-bottom: 0.1rem;
    }
    .aq-model-name {
        flex: 1 1 auto;
        min-width: 0;
        font-weight: 600;
        font-size: 0.92rem;
        color: #f1f5f9;
        line-height: 1.35;
    }
    .aq-model-tags {
        flex: 0 0 auto;
        text-align: right;
        white-space: nowrap;
        line-height: 1.35;
        padding-top: 0;
    }
    .aq-model-tag {
        font-size: 0.72rem !important;
        padding: 4px 10px !important;
        margin-left: 6px;
    }
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    .material-icons { font-family: 'Material Icons'; font-weight: normal;
        font-style: normal; line-height: 1; letter-spacing: normal;
        text-transform: none; display: inline-block; white-space: nowrap;
        word-wrap: normal; direction: ltr; -webkit-font-smoothing: antialiased; }
    </style>
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    """, unsafe_allow_html=True)


_nav_client: object | None = None
_nav_admin: object | None = None


def set_nav_pages(client_page, admin_page):
    global _nav_client, _nav_admin
    _nav_client = client_page
    _nav_admin = admin_page


def nav_to_admin(label=None):
    if label is None:
        label = mi.lbl(mi.ADMIN, "Panel admin →")
    if st.button(label, use_container_width=True, key="nav_btn_admin"):
        if _nav_admin is not None:
            st.switch_page(_nav_admin)
        else:
            st.warning("Nawigacja niedostępna — uruchom ponownie aplikację.")


def nav_to_client(label=None):
    if label is None:
        label = mi.lbl(mi.PERSON, "← Widok klienta")
    if st.button(label, use_container_width=True, key="nav_btn_client"):
        if _nav_client is not None:
            st.switch_page(_nav_client)
        else:
            st.warning("Nawigacja niedostępna — uruchom ponownie aplikację.")


def status_button(label, is_active, key, *, disabled_when_active=True):
    """Przycisk dwustanowy: primary + check gdy włączony; secondary gdy wyłączony."""
    text = mi.lbl(mi.CHECK, label) if is_active else label
    return st.button(
        text,
        key=key,
        type="primary" if is_active else "secondary",
        use_container_width=True,
        disabled=is_active and disabled_when_active,
    )


def page_header(subtitle=""):
    st.markdown(f"""
    <div class="aq-header">
        <h1 style='margin:0;'>{mi.html_title("air", "AirSense Quality AI")}</h1>
        <p style='margin:0; opacity:0.85;'>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def class_badge(value):
    klasa = air_quality.european_class(value)
    return (f"<span class='aq-badge' style='background:{air_quality.class_color(klasa)}'>"
            f"{air_quality.class_label(klasa)}</span>")


@st.cache_data(ttl=3600)
def load_processed_data(station_id, lat, lon):
    return update_and_load(station_id, lat, lon)


@st.cache_data(ttl=1800)
def load_forecast(lat, lon, days=config.CLIENT_FORECAST_DAYS):
    return fetch_forecast_bundle(lat, lon, days=days)


def feature_split(df_proc):
    feat = [c for c in df_proc.columns if c not in INDEX_COLS]
    cal = set(feature_engineering.CALENDAR_FEATURE_COLS)
    weather = [c for c in feat if c in WEATHER_COLS]
    calendar = [c for c in feat if c in cal]
    poll = [c for c in feat if c not in weather and c not in calendar]
    return feat, poll, weather


def load_lstm_model(filepath):
    """Ładuje zapisany model LSTM do inferencji (bez rekompilacji custom loss)."""
    return load_model(filepath, custom_objects=_keras_custom_objects(), compile=False)


def _keras_custom_objects():
    return {
        "HorizonWeightedMSE": HorizonWeightedMSE,
        "airsense>HorizonWeightedMSE": HorizonWeightedMSE,
    }


@keras.saving.register_keras_serializable(package="airsense")
class HorizonWeightedMSE(Loss):
    """MSE z malejącą wagą wzdłuż horyzontu (bliższe godziny ważniejsze)."""

    def __init__(self, horizon=168, **kwargs):
        super().__init__(**kwargs)
        self.horizon = int(horizon)

    def call(self, y_true, y_pred):
        h = max(1, self.horizon)
        weights = tf.linspace(1.0, 0.35, h)
        sq = tf.square(y_true - y_pred)
        w = weights / tf.reduce_mean(weights)
        return tf.reduce_mean(sq * w)

    def get_config(self):
        config = super().get_config()
        config["horizon"] = self.horizon
        return config


def lstm_forecast(name, meta, df_source):
    feats = list(meta["feature_columns"])
    t_steps, horizon = int(meta["time_steps"]), int(meta["horizon"])
    if len(df_source) < t_steps or any(c not in df_source.columns for c in feats):
        return None
    feat_min = np.asarray(meta["feat_min"], dtype=float)
    feat_max = np.asarray(meta["feat_max"], dtype=float)
    rng = feat_max - feat_min
    rng[rng == 0] = 1.0
    y_min, y_max = float(meta["y_min"]), float(meta["y_max"])
    yr = (y_max - y_min) or 1.0
    model = load_lstm_model(model_registry.model_path(name))
    inp = df_source[feats].iloc[-t_steps:].values
    scaled = (inp - feat_min) / rng
    pred = model.predict(scaled.reshape(1, t_steps, len(feats)), verbose=0)[0]
    pred_real = np.clip(pred * yr + y_min, 0.0, 100.0)
    last = pd.Timestamp(df_source.index[-1])
    dates = [last + timedelta(hours=i) for i in range(1, horizon + 1)]
    return pd.Series(pred_real, index=dates, name=AQI_COL)


def training_epoch_summary(meta):
    """Tekst ostatniej epoki (loss/val) do komunikatu po zapisie modelu."""
    if not meta:
        return ""
    ep_run = int(meta.get("epochs_run") or 0)
    ep_max = int(meta.get("epochs_max") or ep_run or 0)
    hist = meta.get("train_history") or {}
    loss_hist = hist.get("loss") or []
    val_hist = hist.get("val_loss") or []
    loss = float(loss_hist[-1]) if loss_hist else 0.0
    val = float(val_hist[-1]) if val_hist else 0.0
    elapsed = training_timing.format_duration(meta.get("training_fit_sec"))
    return (
        f"**Epoka {ep_run}/{ep_max}** | loss: `{loss:.4f}` "
        f"| val: `{val:.4f}` | czas: **{elapsed}**"
    )


class StreamlitKerasCallback(Callback):
    def __init__(self, progress_bar, status_text, epochs, t0=None):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.epochs = epochs
        self.t0 = t0 if t0 is not None else time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.progress((epoch + 1) / self.epochs)
        elapsed = training_timing.format_duration(time.perf_counter() - self.t0)
        self.status_text.markdown(
            f"**Epoka {epoch + 1}/{self.epochs}** | loss: `{logs.get('loss', 0):.4f}` "
            f"| val: `{logs.get('val_loss', 0):.4f}` | czas: **{elapsed}**"
        )


def _build_lstm_model(time_steps, n_features, lstm_units, lstm_layers, dropout, horizon):
    """Sekwencyjny LSTM: 1 lub 2 warstwy + gęsta wyjściowa."""
    layers = lstm_layers
    drop = float(dropout)
    model = Sequential()
    if layers <= 1:
        model.add(LSTM(int(lstm_units), input_shape=(time_steps, n_features)))
        model.add(Dropout(drop))
    else:
        model.add(LSTM(int(lstm_units), return_sequences=True, input_shape=(time_steps, n_features)))
        model.add(Dropout(drop))
        model.add(LSTM(int(lstm_units), return_sequences=False))
        model.add(Dropout(drop))
    model.add(Dense(int(horizon)))
    return model


def train_lstm(df_proc, station_name, station_id, feature_cols, time_steps, forecast_horizon,
               epochs, lstm_units, progress_ui, *, lstm_layers=None, dropout=None,
               horizon_weighted_loss=None):
    """Trenuje model; zwraca (model_name, meta) lub (None, error_msg)."""
    pb, st_txt = progress_ui
    lstm_layers = int(lstm_layers if lstm_layers is not None else config.DEFAULT_LSTM_LAYERS)
    dropout = float(dropout if dropout is not None else config.DEFAULT_DROPOUT)
    horizon_weighted_loss = (
        config.DEFAULT_HORIZON_WEIGHTED_LOSS
        if horizon_weighted_loss is None else bool(horizon_weighted_loss)
    )
    tf.keras.backend.clear_session()
    n_features = len(feature_cols)
    n_rows = len(df_proc)
    ok, err = config.validate_training_params(n_rows, time_steps, forecast_horizon)
    if not ok:
        return None, err

    train_rows = max(1, int(n_rows * 0.8))
    feat_min = df_proc[feature_cols].iloc[:train_rows].min().values.astype(float)
    feat_max = df_proc[feature_cols].iloc[:train_rows].max().values.astype(float)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    y_min = float(df_proc[GIOS_AQI_COL].iloc[:train_rows].min())
    y_max = float(df_proc[GIOS_AQI_COL].iloc[:train_rows].max())
    y_range = (y_max - y_min) or 1.0

    feat_scaled = (df_proc[feature_cols].values - feat_min) / feat_range
    target_scaled = (df_proc[GIOS_AQI_COL].values - y_min) / y_range
    X, y = prepare_xy(feat_scaled, target_scaled, time_steps, forecast_horizon)
    if len(X) < 5:
        return None, (
            f"Za mało sekwencji ({len(X)}). Okno {time_steps} h + horyzont {forecast_horizon} h "
            f"przy {n_rows} h danych — zmniejsz parametry."
        )

    total = len(X)
    tr, vl = int(total * 0.8), int(total * 0.1)
    X_train, y_train = X[:tr], y[:tr]
    X_val, y_val = X[tr:tr + vl], y[tr:tr + vl]
    X_test, y_test = X[tr + vl:], y[tr + vl:]

    model = _build_lstm_model(
        time_steps, n_features, lstm_units, lstm_layers, dropout, forecast_horizon,
    )
    loss_fn = HorizonWeightedMSE(forecast_horizon) if horizon_weighted_loss else "mse"
    model.compile(optimizer="adam", loss=loss_fn)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)
    t_train = time.perf_counter()
    history = model.fit(X_train, y_train, batch_size=16, epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=[StreamlitKerasCallback(pb, st_txt, epochs, t_train), early_stop],
                        verbose=0)
    fit_sec = time.perf_counter() - t_train
    epochs_run = len(history.history.get("loss", [])) or int(epochs)
    model_name = model_registry.unique_model_name(model_registry.build_model_name(
        station_name, time_steps, forecast_horizon, epochs_run, lstm_units))

    model.save(model_registry.model_path(model_name))
    t_pred = time.perf_counter()
    y_pred = model.predict(X_test, verbose=0)
    predict_sec = time.perf_counter() - t_pred
    training_duration_sec = fit_sec + predict_sec
    y_test_real = y_test * y_range + y_min
    y_pred_real = y_pred * y_range + y_min
    split = {"train": tr, "val": vl, "test": len(X_test), "total_seq": total}
    ev = model_eval._bundle_from_arrays(
        y_test_real, y_pred_real, int(forecast_horizon), split,
    )
    baseline_metrics = baselines.evaluate_baselines(
        df_proc, time_steps, forecast_horizon, tr + vl, total,
    )

    meta = {
        "feature_columns": feature_cols,
        "feat_min": feat_min, "feat_max": feat_max,
        "y_min": y_min, "y_max": y_max,
        "time_steps": int(time_steps), "horizon": int(forecast_horizon),
        "epochs_run": int(epochs_run), "epochs_max": int(epochs),
        "lstm_units": int(lstm_units),
        "lstm_layers": int(lstm_layers),
        "dropout": float(dropout),
        "horizon_weighted_loss": bool(horizon_weighted_loss),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "station": station_name, "station_id": int(station_id), "n_rows": int(n_rows),
        "metrics": ev["metrics"],
        "baseline_metrics": baseline_metrics,
        "eval_split": split,
        "eval_y_true": ev["y_true"].tolist(),
        "eval_y_pred": ev["y_pred"].tolist(),
        "train_history": {
            k: [float(v) for v in history.history[k]]
            for k in ("loss", "val_loss")
            if k in history.history
        },
        "training_duration_sec": float(training_duration_sec),
        "training_fit_sec": float(fit_sec),
        "training_predict_sec": float(predict_sec),
        "n_train_sequences": int(tr),
    }
    np.save(model_registry.meta_path(model_name), np.array(meta, dtype=object))
    return model_name, meta
