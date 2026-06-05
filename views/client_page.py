# -*- coding: utf-8 -*-
"""Widok klienta — / (widget Google Pogoda + ASQI z LSTM)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import air_quality
import config
import model_registry
import stations_settings
import ui_common as ui
import ui_icons as mi

GIOS_AQI_COL = ui.GIOS_AQI_COL
ASQI_MIN, ASQI_MAX = 0.0, 100.0
HOURS_PER_DAY = 24
_PL_LONG = (
    "poniedziałek", "wtorek", "środa", "czwartek",
    "piątek", "sobota", "niedziela",
)


def _inject_client_css():
    st.markdown("""
    <style>
    .wx-shell {
        background: #202124; border-radius: 16px; padding: 24px 28px 20px;
        color: #e8eaed; margin-bottom: 10px;
    }
    .wx-topbar {
        display: flex; justify-content: space-between; align-items: flex-start;
        gap: 24px; flex-wrap: wrap;
    }
    .wx-hero { display: flex; align-items: center; gap: 16px; flex: 1; min-width: 260px; }
    .wx-icon-wrap {
        width: 56px; height: 56px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .wx-icon-wrap .material-icons { font-size: 30px; color: #fff; }
    .wx-main-line { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
    .wx-main-val { font-size: 44px; font-weight: 400; line-height: 1; letter-spacing: -1px; }
    .wx-main-class { font-size: 22px; font-weight: 400; color: #e8eaed; }
    .wx-main-sub { font-size: 12px; color: #9aa0a6; margin-top: 4px; }
    .wx-meta { font-size: 12px; color: #9aa0a6; margin-top: 8px; line-height: 1.55; }
    .wx-meta b { color: #bdc1c6; font-weight: 500; }
    .wx-title-block { text-align: right; flex-shrink: 0; min-width: 150px; }
    .wx-title { font-size: 20px; font-weight: 400; }
    .wx-subtitle { font-size: 13px; color: #9aa0a6; margin-top: 4px; }
    .wx-day-range { font-size: 13px; color: #bdc1c6; margin-top: 4px; }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        background: transparent !important; color: #9aa0a6 !important;
        font-size: 13px !important; padding: 8px 14px !important;
        border: none !important; border-radius: 0 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color: #e8eaed !important; border-bottom: 2px solid #fbbc04 !important;
    }
    div[data-testid="stTabs"] {
        background: #202124; border-radius: 0 0 16px 16px;
        padding: 0 24px 6px; margin-top: -10px;
    }
    .wx-picker-shell {
        background: #202124; border-radius: 16px 16px 0 0; padding: 16px 16px 10px;
        margin-bottom: 0;
    }
    .wx-picker-label {
        font-size: 11px; color: #9aa0a6; letter-spacing: 0.04em;
        text-transform: uppercase; margin-bottom: 0;
    }
    div[data-testid="stMarkdownContainer"]:has(.wx-day-row-marker)
        + div[data-testid="stHorizontalBlock"] {
        background: #202124 !important;
        border-radius: 0 0 16px 16px !important;
        padding: 6px 12px 16px !important;
        margin-top: 0 !important;
        margin-bottom: 10px !important;
    }
    div[data-testid="stMarkdownContainer"]:has(.wx-day-row-marker)
        + div[data-testid="stHorizontalBlock"] [data-testid="column"] {
        position: relative !important;
        padding: 0 4px !important;
    }
    div[data-testid="stMarkdownContainer"]:has(.wx-day-row-marker)
        + div[data-testid="stHorizontalBlock"] [data-testid="column"] [data-testid="stButton"] {
        position: absolute !important;
        inset: 0 !important;
        z-index: 3 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stMarkdownContainer"]:has(.wx-day-row-marker)
        + div[data-testid="stHorizontalBlock"] [data-testid="column"] [data-testid="stButton"] button {
        opacity: 0 !important;
        width: 100% !important;
        height: 100% !important;
        min-height: 92px !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        cursor: pointer !important;
    }
    div[data-testid="stMarkdownContainer"]:has(.wx-day-row-marker)
        + div[data-testid="stHorizontalBlock"] [data-testid="column"]:has(button:hover) .wx-day-tile {
        background: #2d2e31;
        border-color: #484a4d;
    }
    div[data-testid="stMarkdownContainer"]:has(.wx-day-row-marker)
        + div[data-testid="stHorizontalBlock"] [data-testid="column"]:has(button:hover) .wx-day-tile.is-active {
        background: #434549;
    }
    div[data-testid="stMarkdownContainer"]:has(.wx-day-row-marker)
        + div[data-testid="stHorizontalBlock"] [data-testid="column"]:has(button:active) .wx-day-tile {
        transform: scale(0.97);
    }
    div[data-testid="stMarkdownContainer"]:has(.wx-day-row-marker)
        + div[data-testid="stHorizontalBlock"] [data-testid="column"] [data-testid="stMarkdownContainer"] {
        margin: 0 !important;
    }
    .wx-day-wrap {
        position: relative; width: 100%;
    }
    .wx-day-tile {
        border-radius: 14px;
        padding: 10px 6px 12px;
        text-align: center;
        background: transparent;
        border: 1px solid transparent;
        transition: background 160ms ease-out, border-color 160ms ease-out,
            box-shadow 160ms ease-out, transform 100ms ease-out;
        pointer-events: none;
        min-height: 92px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 2px;
    }
    .wx-day-tile.is-active {
        background: #3c4043;
        border-color: #5f6368;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 2px 8px rgba(0,0,0,0.28);
    }
    .wx-day-tile.is-active::before {
        content: "";
        display: block;
        width: 28px;
        height: 3px;
        border-radius: 999px;
        background: var(--aq-color, #9aa0a6);
        margin: 0 auto 6px;
    }
    .wx-day-wd {
        font-size: 12px; font-weight: 600; color: #9aa0a6;
        letter-spacing: 0.02em;
    }
    .wx-day-tile.is-active .wx-day-wd { color: #e8eaed; }
    .wx-day-date { font-size: 10px; color: #80868b; margin-bottom: 2px; }
    .wx-day-dot {
        font-size: 18px; line-height: 1; margin: 2px 0;
        color: var(--aq-color, #9aa0a6);
    }
    .wx-day-hi {
        font-size: 16px; font-weight: 500; color: #e8eaed; line-height: 1.1;
    }
    .wx-day-lo {
        font-size: 12px; color: #80868b; line-height: 1.1; margin-top: 1px;
    }
    </style>
    """, unsafe_allow_html=True)


def _pick_client_station(stations):
    choices = stations_settings.client_station_choices(stations)
    if not choices:
        return None, None
    labels = [lbl for lbl, _ in choices]
    label_to_name = {lbl: full for lbl, full in choices}
    picked = st.selectbox("Miasto:", labels, key="client_station")
    return label_to_name[picked], stations[label_to_name[picked]]


def _align(data):
    if data is None or (hasattr(data, "empty") and data.empty):
        return data
    if isinstance(data, pd.Series):
        out = data.copy()
        out.index = ui._normalize_chart_index(out.index, ui.CHART_TZ)
        return out.sort_index()
    return ui._prepare_chart_df(data, ui.CHART_TZ)


def _clip_asqi(series):
    """ASQI ma skalę 0–100; LSTM może wyjść poza przed clip."""
    s = _align(series).astype(float)
    return s.clip(ASQI_MIN, ASQI_MAX)


def _future_series(series):
    s = _clip_asqi(series)
    now = ui._chart_now()
    part = s.loc[s.index > now]
    return part if not part.empty else s


def _forecast_windows(series, n_days=None):
    """Kolejne okna 24 h od pierwszej godziny prognozy (ciągłość między dniami)."""
    n_days = n_days or config.CLIENT_FORECAST_DAYS
    s = _future_series(series)
    if s.empty:
        return []
    windows = []
    for i in range(n_days):
        chunk = s.iloc[i * HOURS_PER_DAY:(i + 1) * HOURS_PER_DAY]
        if len(chunk) < 1:
            break
        start, end = chunk.index[0], chunk.index[-1]
        windows.append({
            "idx": i,
            "start": start,
            "end": end,
            "wd": ui.PL_WEEKDAYS[start.weekday()],
            "date_label": start.strftime("%d.%m"),
            "hours": chunk,
            "max": float(chunk.max()),
            "min": float(chunk.min()),
            "headline": float(chunk.iloc[0]),
        })
    return windows


def _slice_window(df, start, end):
    if df is None or (hasattr(df, "empty") and df.empty):
        return df
    df = _align(df)
    return df.loc[(df.index >= start) & (df.index <= end)]


def _selected_day_idx(sid, n_days):
    key = f"client_day_{sid}"
    if key not in st.session_state:
        st.session_state[key] = 0
    return max(0, min(int(st.session_state[key]), max(n_days - 1, 0)))


def _asqi_style(value):
    val = float(np.clip(value, ASQI_MIN, ASQI_MAX))
    klasa = air_quality.european_class(val)
    return val, air_quality.class_label(klasa), air_quality.class_color(klasa)


def _area_chart(times, values, line_color, fill_color, *, y_range=None, value_fmt="{:.0f}"):
    vals = np.asarray(values, dtype=float)
    labels = [value_fmt.format(v) if pd.notna(v) else "" for v in vals]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=vals,
        mode="lines+text+markers",
        text=labels, textposition="top center",
        textfont=dict(size=11, color="#e8eaed"),
        marker=dict(size=5, color=line_color),
        line=dict(color=line_color, width=2, shape="linear"),
        fill="tozeroy", fillcolor=fill_color,
    ))
    yaxis = dict(showgrid=False, visible=False, fixedrange=True)
    if y_range:
        yaxis["range"] = list(y_range)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=24, b=8), height=220, showlegend=False,
        xaxis=dict(showgrid=False, tickformat="%H:%M", color="#9aa0a6"),
        yaxis=yaxis,
    )
    return fig


def _aqi_icon_html(value):
    _, _, color = _asqi_style(value)
    return (
        f'<div class="wx-icon-wrap" style="background:{color};">'
        f'{mi.html_icon("air", size_px=30)}</div>'
    )


def _day_tile_html(w, *, active):
    aq_color = air_quality.class_color(air_quality.european_class(w["max"]))
    active_cls = " is-active" if active else ""
    return (
        f'<div class="wx-day-tile{active_cls}" style="--aq-color:{aq_color};">'
        f'<div class="wx-day-wd">{w["wd"]}</div>'
        f'<div class="wx-day-date">{w["date_label"]}</div>'
        f'<div class="wx-day-dot">●</div>'
        f'<div class="wx-day-hi">{w["max"]:.0f}</div>'
        f'<div class="wx-day-lo">{w["min"]:.0f}</div>'
        f"</div>"
    )


def _render_day_picker(windows, sid):
    sel = _selected_day_idx(sid, len(windows))
    st.markdown(
        f'<div class="wx-shell wx-picker-shell">'
        f'<div class="wx-picker-label">Prognoza {ui.ASQI_SHORT} · wybierz dzień</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="wx-day-row-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
    cols = st.columns(len(windows))
    key = f"client_day_{sid}"
    for i, (col, w) in enumerate(zip(cols, windows)):
        with col:
            if st.button(
                "\u200b",
                key=f"client_day_btn_{sid}_{i}",
                type="primary" if i == sel else "secondary",
                use_container_width=True,
            ):
                st.session_state[key] = i
                st.rerun()
            st.markdown(
                f'<div class="wx-day-wrap">{_day_tile_html(w, active=(i == sel))}</div>',
                unsafe_allow_html=True,
            )


def _load_lstm_forecast(station_id, df_proc):
    name = model_registry.get_active(station_id)
    if not name:
        return None, None
    meta = model_registry.load_meta(name)
    if not meta:
        return None, None
    fc = ui.lstm_forecast(name, meta, df_proc)
    return fc, name


def render(stations):
    ui.inject_css()
    _inject_client_css()
    ui.page_header("Prognoza jakości powietrza — widok klienta")

    bar_l, bar_r = st.columns([4, 1])
    with bar_l:
        picked, station_meta = _pick_client_station(stations)
    if not picked:
        st.warning("Brak włączonych stacji.")
        return
    with bar_r:
        ui.nav_to_admin()

    city = stations_settings.station_city_label(picked)
    sid = station_meta["id"]

    with st.spinner(f"Ładowanie ({city})…"):
        df_proc = ui.load_processed_data(sid, station_meta["lat"], station_meta["lon"])
    if df_proc.empty:
        st.error("Brak danych o zanieczyszczeniach dla tej stacji.")
        return

    fc_raw, model_name = _load_lstm_forecast(sid, df_proc)
    if fc_raw is None or fc_raw.empty:
        st.error("Brak aktywnego modelu LSTM — wytrenuj i ustaw jako aktywny w panelu admin.")
        return

    with st.spinner("Prognoza pogody…"):
        df_fc = ui.load_forecast(station_meta["lat"], station_meta["lon"])
    df_future = ui.forecast_future_only(df_fc) if not df_fc.empty else pd.DataFrame()

    windows = _forecast_windows(fc_raw)
    if not windows:
        st.error("Brak prognozy ASQI.")
        return

    sel = _selected_day_idx(sid, len(windows))
    day = windows[sel]
    asqi_hours = day["hours"]
    hero_val, hero_label, hero_color = _asqi_style(day["headline"])

    _, poll, _ = ui.feature_split(df_proc)
    gios_now = float(df_proc[GIOS_AQI_COL].iloc[-1])
    sub_now = air_quality.compute_subindices(df_proc.iloc[[-1]][poll])
    dominant = sub_now.iloc[-1].idxmax() if not sub_now.empty else "—"
    humidity = float(df_proc["Wilgotnosc"].iloc[-1]) if "Wilgotnosc" in df_proc.columns else None
    wind = float(df_proc["Wiatr"].iloc[-1]) if "Wiatr" in df_proc.columns else None

    meta = [f"{ui.GIOS_AQI_LABEL}: <b>{gios_now:.0f}</b>"]
    if humidity is not None:
        meta.append(f"Wilgotność: <b>{humidity:.0f}%</b>")
    if wind is not None:
        meta.append(f"Wiatr: <b>{wind:.0f} km/h</b>")
    meta.append(f"Dominujący: <b>{dominant}</b>")

    t_start = day["start"].strftime("%H:%M")
    t_end = day["end"].strftime("%H:%M")
    header_day = _PL_LONG[day["start"].weekday()]

    st.markdown(f"""
    <div class="wx-shell">
        <div class="wx-topbar">
            <div class="wx-hero">
                {_aqi_icon_html(hero_val)}
                <div>
                    <div class="wx-main-line">
                        <span class="wx-main-val" style="color:{hero_color};">{hero_val:.0f}</span>
                        <span class="wx-main-class">{hero_label}</span>
                    </div>
                    <div class="wx-main-sub">{ui.ASQI_LABEL} · prognoza LSTM</div>
                    <div class="wx-meta">{mi.html_icon('location_on', size_px=13)} {city} · {" · ".join(meta)}</div>
                </div>
            </div>
            <div class="wx-title-block">
                <div class="wx-title">Jakość powietrza</div>
                <div class="wx-subtitle">{header_day}, {day['date_label']} · {t_start}–{t_end}</div>
                <div class="wx-day-range">max {day['max']:.0f} · min {day['min']:.0f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fill = f"rgba({int(hero_color[1:3], 16)},{int(hero_color[3:5], 16)},{int(hero_color[5:7], 16)},0.25)"
    wx_win = _slice_window(df_future, day["start"], day["end"])

    tab_asqi, tab_temp, tab_hum, tab_wind = st.tabs([
        ui.ASQI_SHORT, "Temperatura", "Wilgotność", "Wiatr",
    ])

    with tab_asqi:
        fig = _area_chart(
            asqi_hours.index, asqi_hours.values,
            hero_color, fill, y_range=(ASQI_MIN, ASQI_MAX),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab_temp:
        if wx_win.empty or "Temperatura" not in wx_win.columns:
            st.caption("Brak prognozy temperatury w tym oknie.")
        else:
            fig = _area_chart(
                wx_win.index, wx_win["Temperatura"].values,
                "#fbbc04", "rgba(251,188,4,0.22)",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab_hum:
        if wx_win.empty or "Wilgotnosc" not in wx_win.columns:
            st.caption("Brak prognozy wilgotności w tym oknie.")
        else:
            fig = _area_chart(
                wx_win.index, wx_win["Wilgotnosc"].values,
                "#8ab4f8", "rgba(138,180,248,0.22)",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab_wind:
        if wx_win.empty or "Wiatr" not in wx_win.columns:
            st.caption("Brak prognozy wiatru w tym oknie.")
        else:
            fig = _area_chart(
                wx_win.index, wx_win["Wiatr"].values,
                "#81c995", "rgba(129,201,149,0.22)",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    _render_day_picker(windows, sid)
