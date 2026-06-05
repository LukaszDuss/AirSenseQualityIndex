# -*- coding: utf-8 -*-
"""Panel administracyjny — /admin (zakładki przez ?tab=)."""
import os
import html
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import config
import data_store
import data_fetch
import model_registry
import llm_eval
import normalization_stats as norm_stats
import model_eval
import training_timing
import stations_settings
import gios_audit
import ui_common as ui
import ui_icons as mi
import feature_engineering

ADMIN_TABS = [
    ("dane_surowe", "Dane surowe"),
    ("dane_przetworzone", "Dane przetworzone"),
    ("trening", "Trening modeli"),
    ("modele", "Zarządzanie modelami"),
    ("aktualizacja", "Aktualizacja i ewaluacja"),
    ("podglad", "Podgląd i porównanie"),
    ("ustawienia", "Ustawienia"),
]


def _admin_pick_station(stations):
    """Wspólna lista stacji (selectbox) — synchronizacja z ?station_id=."""
    names = list(stations.keys())
    sid_qp = ui.qp_get("station_id", "")
    default_idx = 0
    if sid_qp:
        for i, n in enumerate(names):
            if str(stations[n]["id"]) == sid_qp:
                default_idx = i
                break
    picked = st.selectbox("Stacja:", names, index=default_idx, key="admin_station_pick")
    meta = stations[picked]
    st.query_params["station_id"] = str(meta["id"])
    return picked, meta


def _nav_tabs(current):
    cols = st.columns(len(ADMIN_TABS))
    for col, (key, label) in zip(cols, ADMIN_TABS):
        with col:
            if st.button(label, type="primary" if key == current else "secondary",
                         use_container_width=True, key=f"nav_{key}"):
                st.query_params["tab"] = key
                st.rerun()


def render():
    ui.inject_css()
    model_registry.delete_invalid_models()
    ui.page_header("Panel administracyjny — /admin")

    ui.nav_to_client()

    tab = ui.qp_get("tab", "dane_przetworzone")
    if tab in ("dane", "normalizacja"):
        tab = "dane_przetworzone"
    if tab == "prognoza":
        tab = "aktualizacja"
    if tab not in [t[0] for t in ADMIN_TABS]:
        tab = "dane_przetworzone"
    _nav_tabs(tab)

    stations = stations_settings.enabled_stations()

    if tab == "ustawienia":
        _tab_settings(stations)
        return

    if not stations:
        st.warning("Brak włączonych stacji — włącz je w zakładce Ustawienia.")
        return

    picked, meta = _admin_pick_station(stations)

    if tab == "modele":
        _tab_models(picked, meta)
        return

    with st.spinner(f"Ładowanie danych ({picked})..."):
        df_proc = ui.load_processed_data(meta["id"], meta["lat"], meta["lon"])
    store = data_store.load_store(meta["id"])
    if not df_proc.empty:
        feat_cols, poll_cols, weather_cols = ui.feature_split(df_proc)
        cal_cols = [c for c in feat_cols if c in feature_engineering.CALENDAR_FEATURE_COLS]
    else:
        feat_cols, poll_cols, weather_cols, cal_cols = [], [], [], []
    n_days = len(df_proc) / 24.0 if not df_proc.empty else 0

    if tab == "podglad":
        _tab_preview_compare(picked, meta, df_proc)
        return

    needs_processed = tab in ("dane_przetworzone", "trening", "aktualizacja")
    if df_proc.empty and needs_processed:
        st.error(
            "Brak danych przetworzonych dla tej stacji — "
            "odśwież w zakładce **Dane surowe**."
        )
        return

    handlers = {
        "dane_surowe": lambda: _tab_data_raw(picked, meta, store),
        "dane_przetworzone": lambda: _tab_data_processed(
            picked, meta, store, df_proc, poll_cols, weather_cols, cal_cols, n_days, feat_cols,
        ),
        "trening": lambda: _tab_train(picked, meta, df_proc, feat_cols, n_days),
        "aktualizacja": lambda: _tab_update_eval(picked, meta, df_proc, store),
    }
    handlers[tab]()


_RAW_DESC = (
    "#### RAW (surowe)\n"
    "- **Źródła:** GIOŚ (zanieczyszczenia) + Open-Meteo (pogoda pomocnicza)\n"
    "- **Bez** uzupełniania braków i **bez** wskaźnika ASQI\n"
    "- Puste komórki = **brak realnego pomiaru** w tej godzinie\n"
    "- Pogoda: dopasowanie do najbliższej godziny GIOŚ (±30 min)\n"
    "- Służy do audytu i pokrycia danych (tabela poniżej)"
)

_PROCESSED_DESC = (
    "#### PROCESSED (do modelu)\n"
    "- Liczone **z całej historii raw** przy każdym odświeżeniu\n"
    "- **Imputacja** braków (interpolacja liniowa na osi GIOŚ)\n"
    "- **Cechy czasowe:** godzina i dzień tygodnia (sin/cos), sezon grzewczy (paź–kwi)\n"
    "- **Pogoda:** temperatura, wilgotność, wiatr, opady (kolumna tylko gdy API zwróci dane)\n"
    "- **GiosAQI (GIOŚ AQI)**, EuropeanIndex, klasa EAQI\n"
    "- Obcięcie outlierów (percentyle 1–99% na kolumnie — widać na wykresie vs raw)\n"
    "- **Braki zmiennych:** brak całego sensora → kolumna znika; luki godzinowe → interpolacja; "
    "pogoda/kalendarz → mediana w kolumnie; wiersz bez kompletnych stężeń EAQI → **odrzucony**\n"
    "- To **nie** jest kopia raw — wartości mogą się różnić po obróbce"
)

_BASELINE_LABELS = {
    "persistence": "Persistence (ostatnia wartość)",
    "ma_24h": "Średnia dobowa (24 h)",
    "ma_168h": "Średnia tygodniowa (168 h)",
    "seasonal_24h": "Sezonowość dobowa (lag 24 h)",
    "seasonal_168h": "Sezonowość tygodniowa (lag 168 h)",
}


def _data_fetch_buttons(name, meta):
    """Pobieranie / reset — tylko w Dane surowe (stąd powstaje też processed)."""
    b_ref, b_reset = st.columns(2)
    with b_ref:
        if st.button(mi.lbl(mi.REFRESH, "Odśwież dane"), use_container_width=True, key="btn_refresh_data"):
            ui.load_processed_data.clear()
            ui.load_forecast.clear()
            st.rerun()
    with b_reset:
        if st.button(
            mi.lbl(mi.DELETE, "Wyczyść bazę i pobierz od zera"),
            type="primary",
            use_container_width=True,
            key="btn_reset_data",
        ):
            with st.spinner(f"Pobieranie archiwum dla {name}…"):
                ui.load_processed_data.clear()
                data_fetch.reset_and_load(meta["id"], meta["lat"], meta["lon"])
            st.success("Baza wyczyszczona i pobrana od zera (raw + przeliczone processed).")
            st.rerun()


def _tab_data_raw(name, meta, store):
    st.header("Dane surowe")
    st.caption(
        "Pobieranie z GIOŚ i Open-Meteo. Po każdym odświeżeniu warstwa **processed** "
        "jest przeliczana automatycznie — zobacz zakładkę **Dane przetworzone**."
    )
    _data_fetch_buttons(name, meta)

    df_raw = data_store.raw_to_dataframe(store)
    _, poll_series = data_store.store_to_series(store)
    raw_days = len(df_raw) / 24.0 if not df_raw.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wiersze RAW", len(df_raw))
    c2.metric("Zakres RAW (dni)", f"{raw_days:.1f}")
    c3.metric("Parametry zaniecz.", len(poll_series))
    c4.metric("Ostatnia aktualizacja", (store.get("last_updated") or "—")[:16])

    st.markdown(_RAW_DESC)
    cov = data_store.get_coverage(store)
    if cov:
        st.markdown("**Pokrycie realnych odczytów (GIOŚ / Open-Meteo)**")
        cdf = pd.DataFrame(cov)
        cdf["od"] = pd.to_datetime(cdf["od"]).dt.strftime("%Y-%m-%d %H:%M")
        cdf["do"] = pd.to_datetime(cdf["do"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(cdf.set_index("parametr"), use_container_width=True)
    if df_raw.empty:
        st.warning("Brak warstwy raw — użyj „Odśwież dane”.")
    else:
        st.dataframe(df_raw, use_container_width=True, height=420)
        st.download_button(
            "Pobierz CSV (raw)",
            df_raw.to_csv().encode("utf-8-sig"),
            file_name=f"{name}_raw.csv",
        )


def _tab_data_processed(name, meta, store, df_proc, poll, weather, calendar, n_days, feat_cols):
    st.header("Dane przetworzone")
    st.caption(
        "Warstwa gotowa do treningu i wykresów (imputacja, indeksy, skaler MinMax). "
        "Powstaje z raw przy odświeżeniu w **Dane surowe**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wiersze processed", len(df_proc))
    c2.metric("Zakres (dni)", f"{n_days:.1f}")
    c3.metric("Kolumny (cechy + indeksy)", len(df_proc.columns) if not df_proc.empty else 0)
    c4.metric("Ostatnia aktualizacja", (store.get("last_updated") or "—")[:16])

    st.markdown(_PROCESSED_DESC)
    if df_proc.empty:
        st.warning("Brak warstwy processed — odśwież dane w zakładce **Dane surowe**.")
        return

    st.markdown(
        f"**Cechy modelu ({len(feat_cols)}):** zanieczyszczenia: {', '.join(poll) or '—'} | "
        f"pogoda: {', '.join(weather) or '—'} | kalendarz: {', '.join(calendar) or '—'}"
    )
    st.dataframe(df_proc, use_container_width=True, height=360)
    st.download_button(
        "Pobierz CSV (processed)",
        df_proc.to_csv().encode("utf-8-sig"),
        file_name=f"{name}_processed.csv",
    )

    st.divider()
    st.subheader("Normalizacja i imputacja")
    st.caption(
        "Porównanie z warstwą raw: co było puste w GIOŚ, co uzupełniono w processed, "
        "oraz podgląd skali MinMax (0–1) zapisanej w bazie."
    )
    df_raw, df_p = norm_stats.raw_vs_processed_frames(store, df_proc)
    st.markdown("**Statystyki braków i uzupełnień**")
    st.dataframe(norm_stats.column_stats(df_raw, df_p), use_container_width=True)
    t_mm, t_cmp = st.tabs(["MinMax (0–1)", "Porównanie raw vs processed"])
    with t_mm:
        scaled = norm_stats.minmax_scaled_frame(store, df_p, feat_cols)
        if scaled.empty:
            st.info("Brak zapisanego skalera w bazie — odśwież dane w **Dane surowe**.")
        else:
            st.dataframe(scaled, use_container_width=True, height=320)
    with t_cmp:
        st.caption(
            "Czerwona linia (processed) może być **niższa od szczytów raw** (niebieska kreska): "
            "przed zapisem processed stosujemy **przycinanie percentyli 1–99%** na każdej kolumnie "
            "(funkcja `remove_outliers` w `data_fetch.py`). Wartości powyżej 99. percentyla "
            "całego szeregu są obcinane do tego limitu — stąd „ścięte” piki PM10. "
            "To stabilizuje trening LSTM; **nie** oznacza błędu pobierania z GIOŚ."
        )
        col = st.selectbox("Kolumna:", list(df_p.columns), key="norm_cmp_col")
        if col in df_raw.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p.index, y=df_raw[col], name="raw",
                                     line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p[col], name="processed"))
            fig.update_layout(template="plotly_dark", title=col)
            st.plotly_chart(fig, use_container_width=True)


def _model_preview_row(mname, mmeta):
    """Wiersz podglądu / kontekstu LLM dla istniejącego modelu."""
    m = mmeta.get("metrics", {})
    ep_run = mmeta.get("epochs_run")
    ep_max = mmeta.get("epochs_max")
    return {
        "model": mname,
        "MAE": m.get("mae"),
        "RMSE": m.get("rmse"),
        "R2": m.get("r2"),
        "okno_h": mmeta.get("time_steps"),
        "horyzont_h": mmeta.get("horizon"),
        "epoki_run": ep_run,
        "epoki_max": ep_max,
        "neurony": mmeta.get("lstm_units"),
        "warstwy": mmeta.get("lstm_layers"),
        "dropout": mmeta.get("dropout"),
        "loss_wazony": mmeta.get("horizon_weighted_loss"),
    }


def _train_slider_keys(station_id):
    p = f"train_{station_id}_"
    return {
        "time_steps": p + "time_steps",
        "forecast_horizon": p + "forecast_horizon",
        "epochs": p + "epochs",
        "lstm_units": p + "lstm_units",
        "lstm_layers": p + "lstm_layers",
        "dropout": p + "dropout",
        "horizon_weighted_loss": p + "horizon_weighted_loss",
    }


def _slider_help(description, min_v, max_v, *, suffix="", extra=""):
    """Tekst pod ikoną ? — zakres min/max i krótki opis."""
    text = (
        f"{description} Dozwolony zakres: min {min_v}{suffix}, max {max_v}{suffix}."
    )
    if extra:
        text += f" {extra}"
    return text


def _init_train_sliders(keys):
    defaults = {
        keys["time_steps"]: config.DEFAULT_WINDOW_HOURS,
        keys["forecast_horizon"]: config.DEFAULT_FORECAST_HOURS,
        keys["epochs"]: config.DEFAULT_EPOCHS,
        keys["lstm_units"]: config.DEFAULT_LSTM_UNITS,
        keys["lstm_layers"]: config.DEFAULT_LSTM_LAYERS,
        keys["dropout"]: config.DEFAULT_DROPOUT,
        keys["horizon_weighted_loss"]: config.DEFAULT_HORIZON_WEIGHTED_LOSS,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _clear_train_outcome(station_id):
    st.session_state.pop(f"train_outcome_{station_id}", None)


def _current_train_params(keys):
    return {
        "time_steps": int(st.session_state.get(keys["time_steps"], 0)),
        "forecast_horizon": int(st.session_state.get(keys["forecast_horizon"], 0)),
        "epochs": int(st.session_state.get(keys["epochs"], 0)),
        "lstm_units": int(st.session_state.get(keys["lstm_units"], 0)),
        "lstm_layers": int(st.session_state.get(keys["lstm_layers"], 0)),
        "dropout": float(st.session_state.get(keys["dropout"], 0)),
        "horizon_weighted_loss": bool(st.session_state.get(keys["horizon_weighted_loss"], False)),
    }


def _sync_train_outcome_with_params(station_id, keys):
    """Ukrywa wynik treningu, gdy slidery nie zgadzają się z parametrami zapisu."""
    outcome = st.session_state.get(f"train_outcome_{station_id}")
    if not outcome:
        return
    saved = outcome.get("params")
    if saved and saved != _current_train_params(keys):
        _clear_train_outcome(station_id)


def _tab_train(name, meta, df_proc, feat_cols, n_days):
    st.header("Trening modeli")
    if n_days < config.MIN_TRAIN_DAYS:
        st.warning(f"~{n_days:.1f} dni danych (min. {config.MIN_TRAIN_DAYS}).")
    st.caption(
        f"Model widzi **{len(feat_cols)}** cech (zanieczyszczenia + pogoda + kalendarz). "
        f"Docelowy horyzont: **{config.DEFAULT_FORECAST_DAYS} dni** (min. {config.MIN_FORECAST_DAYS} dni), "
        f"domyślne okno: **{config.DEFAULT_WINDOW_DAYS} dni**. "
        "Po treningu porównujemy LSTM z baseline na zbiorze testowym."
    )

    sid = meta["id"]
    keys = _train_slider_keys(sid)
    pending = st.session_state.pop(f"pending_train_params_{sid}", None)
    if pending:
        for field in (
            "time_steps", "forecast_horizon", "epochs", "lstm_units",
            "lstm_layers", "dropout",
        ):
            if field in pending and pending[field] is not None:
                st.session_state[keys[field]] = int(pending[field]) if field != "dropout" else float(pending[field])
        if "horizon_weighted_loss" in pending and pending["horizon_weighted_loss"] is not None:
            st.session_state[keys["horizon_weighted_loss"]] = bool(pending["horizon_weighted_loss"])
        st.session_state[f"llm_train_flash_{sid}"] = dict(pending)
    _init_train_sliders(keys)

    n_hours = len(df_proc) if not df_proc.empty else int(n_days * 24)
    fh_init = int(st.session_state.get(keys["forecast_horizon"], config.DEFAULT_FORECAST_HOURS))
    ts_init = int(st.session_state.get(keys["time_steps"], config.DEFAULT_WINDOW_HOURS))
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Parametry treningu")
        st.caption(
            f"Dane w bazie (processed): **{n_hours} h** (~{n_hours / 24:.1f} dni) — "
            "górne limity sliderów liczone z tej liczby wierszy."
        )
        min_win, max_win = config.effective_window_bounds(n_hours)
        ts_cur = int(st.session_state.get(keys["time_steps"], ts_init))
        if ts_cur < min_win or ts_cur > max_win:
            st.session_state[keys["time_steps"]] = min(
                max(ts_cur, min_win),
                min(max_win, config.DEFAULT_WINDOW_HOURS),
            )
        time_steps = st.slider(
            "Okno wejściowe LSTM (godz.)",
            min_value=min_win,
            max_value=max_win,
            key=keys["time_steps"],
            on_change=_clear_train_outcome,
            args=(sid,),
            help=_slider_help(
                f"Ile ostatnich godzin historii widzi sieć (max {max_win} h ≈ {max_win // 24} dni "
                f"przy {n_hours} h danych w bazie). Zmiana horyzontu nie zmienia tego okna — "
                f"limit prognozy dostosowuje się do okna.",
                min_win, max_win, suffix=" h",
            ),
        )
        min_fh, max_fh = config.effective_forecast_bounds(n_hours, time_steps)
        fh_cur = int(st.session_state.get(keys["forecast_horizon"], fh_init))
        if fh_cur < min_fh or fh_cur > max_fh:
            st.session_state[keys["forecast_horizon"]] = min(
                max(fh_cur, min_fh),
                min(max_fh, config.DEFAULT_FORECAST_HOURS),
            )
        forecast_horizon = st.slider(
            "Horyzont prognozy (godz.)",
            min_value=min_fh,
            max_value=max_fh,
            key=keys["forecast_horizon"],
            on_change=_clear_train_outcome,
            args=(sid,),
            help=_slider_help(
                f"Prognoza do przodu (max {max_fh} h przy oknie {time_steps} h i {n_hours} h w bazie).",
                min_fh, max_fh, suffix=" h",
                extra=f"Docelowo {config.DEFAULT_FORECAST_DAYS} dni, jeśli starczy danych.",
            ),
        )
        lstm_units = st.slider(
            "Neurony LSTM",
            min_value=config.MIN_LSTM_UNITS,
            max_value=config.MAX_LSTM_UNITS,
            step=8,
            key=keys["lstm_units"],
            on_change=_clear_train_outcome,
            args=(sid,),
            help=_slider_help(
                "Liczba jednostek w warstwach LSTM — więcej = większa pojemność, dłuższy trening.",
                config.MIN_LSTM_UNITS, config.MAX_LSTM_UNITS,
            ),
        )
        epochs = st.slider(
            "Maks. epoki treningu",
            min_value=config.MIN_EPOCHS,
            max_value=config.MAX_EPOCHS,
            key=keys["epochs"],
            on_change=_clear_train_outcome,
            args=(sid,),
            help=_slider_help(
                "Górny limit epok; early stopping może zakończyć wcześniej przy braku poprawy walidacji.",
                config.MIN_EPOCHS, config.MAX_EPOCHS,
            ),
        )
        lstm_layers = st.slider(
            "Warstwy LSTM",
            min_value=config.MIN_LSTM_LAYERS,
            max_value=config.MAX_LSTM_LAYERS,
            key=keys["lstm_layers"],
            on_change=_clear_train_outcome,
            args=(sid,),
            help=_slider_help(
                "1 warstwa — szybszy, prostszy model; 2 warstwy — większa pojemność.",
                config.MIN_LSTM_LAYERS, config.MAX_LSTM_LAYERS,
            ),
        )
        dropout = st.slider(
            "Dropout",
            min_value=config.MIN_DROPOUT,
            max_value=config.MAX_DROPOUT,
            step=0.05,
            key=keys["dropout"],
            on_change=_clear_train_outcome,
            args=(sid,),
            help=_slider_help(
                "Regularyzacja między warstwami LSTM — wyższe wartości ograniczają przeuczenie.",
                config.MIN_DROPOUT, config.MAX_DROPOUT,
            ),
        )
        horizon_weighted_loss = st.checkbox(
            "Loss ważony (bliższe prognozy ważniejsze)",
            key=keys["horizon_weighted_loss"],
            on_change=_clear_train_outcome,
            args=(sid,),
            help="MSE z malejącą wagą wzdłuż horyzontu — model skupia się na najbliższych godzinach.",
        )
        n_seq = config.training_sequence_count(n_hours, time_steps, forecast_horizon)
        train_ok, train_err = config.validate_training_params(n_hours, time_steps, forecast_horizon)
        st.caption(
            f"Sekwencje przy tych ustawieniach: **{n_seq}** "
            f"(min. {config.MIN_TRAIN_SEQUENCES} przy {n_hours} h w bazie)."
        )
        if not train_ok:
            st.warning(train_err)
        st.caption(
            f"Podgląd nazwy: `{model_registry.build_model_name(name, time_steps, forecast_horizon, epochs, lstm_units)}`"
        )
        est_range = training_timing.estimate_training_range_label(
            len(df_proc), time_steps, forecast_horizon, epochs, lstm_units, len(feat_cols),
        )
        st.caption(f"Szac. czas treningu: **{est_range}**")
        btn_col_train, btn_col_llm = st.columns(2)
        with btn_col_train:
            go_train = st.button(mi.lbl(mi.ROCKET, "Trenuj"), type="primary", use_container_width=True)
        with btn_col_llm:
            go_llm_train = st.button(
                "Zaproponuj ustawienia treningu",
                key=f"btn_llm_train_{sid}",
                use_container_width=True,
            )

        if go_llm_train:
            _clear_train_outcome(sid)
            station_models = [
                _model_preview_row(mname, mmeta)
                for mname, mmeta in model_registry.models_for_station(sid)
            ]
            min_w_llm, max_w_llm = config.effective_window_bounds(n_hours)
            ts_llm = int(st.session_state.get(keys["time_steps"], ts_init))
            min_fh_llm, max_fh_llm = config.effective_forecast_bounds(n_hours, ts_llm)
            constraints = {
                "min_train_days": config.MIN_TRAIN_DAYS,
                "seed_days": config.SEED_PAST_DAYS,
                "min_forecast_hours": config.MIN_FORECAST_HOURS,
                "max_forecast_hours": max_fh_llm,
                "max_forecast_days": config.MAX_FORECAST_DAYS,
                "default_forecast_hours": config.DEFAULT_FORECAST_HOURS,
                "default_window_hours": config.DEFAULT_WINDOW_HOURS,
                "min_window": min_w_llm,
                "max_window": max_w_llm,
                "min_window_days": config.MIN_WINDOW_DAYS,
                "min_lstm_units": config.MIN_LSTM_UNITS,
                "max_lstm_units": config.MAX_LSTM_UNITS,
                "min_epochs": config.MIN_EPOCHS,
                "max_epochs": config.MAX_EPOCHS,
                "min_lstm_layers": config.MIN_LSTM_LAYERS,
                "max_lstm_layers": config.MAX_LSTM_LAYERS,
                "min_dropout": config.MIN_DROPOUT,
                "max_dropout": config.MAX_DROPOUT,
            }
            data_info = {
                "stacja": name,
                "dni_danych": round(n_days, 1),
                "godzin_processed": n_hours,
                "wierszy_processed": len(df_proc),
                "liczba_cech": len(feat_cols),
                "typ_cech": "zanieczyszczenia + pogoda + kalendarz",
                "min_okno_h": min_w_llm,
                "max_okno_h": max_w_llm,
                "domyslne_okno_dni": config.DEFAULT_WINDOW_DAYS,
                "min_horyzont_dni": config.MIN_FORECAST_DAYS,
                "docelowy_horyzont_dni": config.DEFAULT_FORECAST_DAYS,
                "max_horyzont_h": max_fh_llm,
                "godzin_w_bazie": n_hours,
            }
            with st.spinner("Analiza..."):
                try:
                    text, params = llm_eval.recommend_training_settings(
                        station_models, data_info, constraints,
                    )
                    st.session_state[f"llm_train_text_{sid}"] = llm_eval.strip_training_json_block(text)
                    if params:
                        st.session_state[f"pending_train_params_{sid}"] = params
                        st.rerun()
                    else:
                        st.markdown(llm_eval.strip_training_json_block(text))
                        st.warning(
                            "Nie udało się odczytać liczb z JSON — ustaw slidery ręcznie "
                            "(LLM powinien zwrócić blok ```json na końcu odpowiedzi)."
                        )
                except Exception as e:
                    st.error(str(e))

        flash = st.session_state.pop(f"llm_train_flash_{sid}", None)
        if flash:
            st.success(
                "Slidery z rekomendacji AI: "
                + ", ".join(f"{k}={v}" for k, v in flash.items())
            )

        if go_train:
            pb = st.progress(0)
            txt = st.empty()
            mname, result = ui.train_lstm(
                df_proc, name, meta["id"], feat_cols,
                time_steps, forecast_horizon, epochs, lstm_units, (pb, txt),
                lstm_layers=lstm_layers,
                dropout=dropout,
                horizon_weighted_loss=horizon_weighted_loss,
            )
            pb.empty()
            txt.empty()
            if mname:
                model_registry.refresh_best_for_station(meta["id"])
                llm_txt, llm_err = None, None
                with st.spinner("Ocena LLM po treningu…"):
                    llm_txt, llm_err = llm_eval.evaluate_after_training(
                        mname, result, name,
                    )
                st.session_state[f"train_outcome_{sid}"] = {
                    "mname": mname,
                    "meta": result,
                    "epoch_summary": ui.training_epoch_summary(result),
                    "params": {
                        "time_steps": int(time_steps),
                        "forecast_horizon": int(forecast_horizon),
                        "epochs": int(epochs),
                        "lstm_units": int(lstm_units),
                        "lstm_layers": int(lstm_layers),
                        "dropout": float(dropout),
                        "horizon_weighted_loss": bool(horizon_weighted_loss),
                    },
                    "llm_eval": llm_txt,
                    "llm_err": llm_err,
                }
            else:
                st.session_state.pop(f"train_outcome_{sid}", None)
                st.error(result)

        _sync_train_outcome_with_params(sid, keys)
        outcome = st.session_state.get(f"train_outcome_{sid}")
        if outcome and outcome.get("mname"):
            mname_out = outcome["mname"]
            train_meta = outcome.get("meta") or {}
            dur = training_timing.format_duration(train_meta.get("training_duration_sec"))
            epoch_line = outcome.get("epoch_summary") or ui.training_epoch_summary(train_meta)
            st.success(
                f"Zapisano model **{mname_out}**. Czas treningu: **{dur}**\n\n{epoch_line}"
            )
            with st.expander(f"Wynik treningu — {mname_out}", expanded=False):
                if outcome.get("llm_err"):
                    st.warning(f"Ocena LLM: {outcome['llm_err']}")
                elif outcome.get("llm_eval"):
                    st.markdown("#### Ocena LLM")
                    st.markdown(outcome["llm_eval"])
                st.markdown("#### Statystyki i wykresy (zbiór testowy)")
                _render_model_stats(mname_out, train_meta, df_proc)

    with col_b:
        st.subheader("Rekomendacja AI — ustawienia treningu")
        station_models = [
            _model_preview_row(mname, mmeta)
            for mname, mmeta in model_registry.models_for_station(sid)
        ]
        if station_models:
            st.caption("Istniejące modele tej stacji (podgląd — LLM widzi te same kolumny):")
            st.dataframe(pd.DataFrame(station_models), use_container_width=True, height=160)
        else:
            st.caption("Brak wytrenowanych modeli dla tej stacji.")
        if st.session_state.get(f"llm_train_text_{sid}"):
            st.markdown(
                llm_eval.strip_training_json_block(st.session_state[f"llm_train_text_{sid}"])
            )


def _render_baseline_comparison(metrics, baseline_metrics):
    """Tabela MAE / R² LSTM vs baseline na zbiorze testowym."""
    if not baseline_metrics:
        return
    lstm_mae = metrics.get("mae")
    lstm_r2 = metrics.get("r2")
    rows = [{
        "Model": "LSTM",
        "MAE": lstm_mae,
        "R²": lstm_r2,
    }]
    for key, block in baseline_metrics.items():
        m = (block or {}).get("metrics") or {}
        rows.append({
            "Model": _BASELINE_LABELS.get(key, key),
            "MAE": m.get("mae"),
            "R²": m.get("r2"),
        })
    st.markdown("**Porównanie z baseline (zbiór testowy)**")
    for r in rows:
        if r.get("MAE") is not None:
            r["MAE"] = round(float(r["MAE"]), 2)
        if r.get("R²") is not None:
            r["R²"] = round(float(r["R²"]), 3)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if lstm_mae is not None:
        better = [
            _BASELINE_LABELS.get(k, k)
            for k, b in baseline_metrics.items()
            if (b or {}).get("metrics", {}).get("mae") is not None
            and (b["metrics"]["mae"] < lstm_mae)
        ]
        if better:
            st.caption(
                "Baseline lepszy od LSTM (niższe MAE): "
                + ", ".join(better)
                + " — rozważ dłuższe okno lub więcej danych."
            )


def _render_model_stats(mname, mmeta, df_proc):
    """Szczegółowe metryki i wykresy błędów dla jednego modelu."""
    ev = model_eval.resolve_evaluation(mname, mmeta, df_proc)
    if ev.get("error"):
        st.warning(ev["error"])
        return
    metrics = ev.get("metrics") or mmeta.get("metrics") or {}
    baseline_metrics = mmeta.get("baseline_metrics") or ev.get("baseline_metrics")
    if not baseline_metrics and df_proc is not None and not df_proc.empty:
        split = ev.get("split") or mmeta.get("eval_split")
        baseline_metrics = model_eval.baseline_metrics_for_split(mmeta, df_proc, split)
    rows = [(k, lb) for k, lb in model_eval.METRIC_LABELS if metrics.get(k) is not None]
    for start in range(0, len(rows), 4):
        cols = st.columns(4)
        for col, (key, label) in zip(cols, rows[start:start + 4]):
            val = metrics[key]
            with col:
                if key == "n_samples":
                    col.metric(label, f"{int(val)}")
                elif key == "r2":
                    col.metric(label, f"{val:.3f}")
                elif key == "mape_pct":
                    col.metric(label, f"{val:.1f}%")
                else:
                    col.metric(label, f"{val:.2f}")

    _render_baseline_comparison(metrics, baseline_metrics)

    split = ev.get("split") or mmeta.get("eval_split") or {}
    if split:
        st.caption(
            f"Podział sekwencji: trening **{split.get('train', '—')}** | "
            f"walidacja **{split.get('val', '—')}** | test **{split.get('test', '—')}** "
            f"(łącznie {split.get('total_seq', '—')})"
        )

    by_h = ev.get("metrics_by_horizon") or []
    if by_h:
        st.markdown("**Metryki per krok horyzontu**")
        st.dataframe(pd.DataFrame(by_h), use_container_width=True, hide_index=True)

    figs = model_eval.build_chart_figures(ev, mmeta)
    if not figs:
        st.info("Brak danych do wykresów.")
        return
    chart_tabs = st.tabs(list(figs.keys()))
    for tab, title in zip(chart_tabs, figs.keys()):
        with tab:
            st.plotly_chart(figs[title], use_container_width=True)
    if not mmeta.get("train_history"):
        st.caption(
            "Brak krzywej uczenia (epoki) — model wytrenowany przed zapisem `train_history`. "
            "Wytrenuj ponownie, aby zobaczyć wykres loss / val_loss."
        )


def _render_llm_client_column(sname, sid, models):
    """Prawa kolumna — rekomendacja modelu na widok klienta."""
    st.subheader("Rekomendacja dla klienta (LLM)")
    st.markdown('<div class="aq-llm-recommend">', unsafe_allow_html=True)
    if st.button(
        "Rekomenduj model",
        key=f"btn_llm_client_{sid}",
        type="primary",
        use_container_width=True,
    ):
        rows = [_model_preview_row(mname, mmeta) for mname, mmeta in models]
        with st.spinner("Analiza modeli…"):
            try:
                text, chosen = llm_eval.recommend_client_model(
                    rows, {"stacja": sname, "id": sid},
                )
                st.session_state[f"llm_client_text_{sid}"] = text
                if chosen:
                    model_registry.set_llm_recommended(sid, chosen)
                    st.session_state.pop(f"llm_client_pick_{sid}", None)
                    st.rerun()
                st.markdown(llm_eval.strip_client_recommendation_text(text))
                st.warning(f"Nie rozpoznano nazwy modelu — ustaw {mi.STAR} ręcznie.")
            except Exception as e:
                st.error(str(e))
    st.markdown("</div>", unsafe_allow_html=True)
    rec_stale = model_registry.llm_recommendation_stale(sid)
    if rec_stale:
        st.session_state.pop(f"llm_client_text_{sid}", None)
        meta = model_registry.get_llm_recommendation_meta(sid) or {}
        when = meta.get("recommended_at")
        when_part = f" (z {when})" if when else ""
        st.warning(
            f"Lista modeli zmieniła się od ostatniej rekomendacji LLM{when_part}. "
            "Uruchom **Rekomenduj model** ponownie."
        )
    else:
        llm_txt = st.session_state.get(f"llm_client_text_{sid}")
        if llm_txt:
            st.markdown(llm_eval.strip_client_recommendation_text(llm_txt))


def _model_tag_suffix(mname, active, llm_rec, best):
    """Tekst tagów do etykiety selectboxa (np. · AKTYWNY · NAJLEPSZY)."""
    tags = []
    if mname == active:
        tags.append("AKTYWNY")
    if llm_rec and mname == llm_rec:
        tags.append("REKOMENDOWANY")
    if mname == best:
        tags.append("NAJLEPSZY")
    if not tags:
        return ""
    return " · " + " · ".join(tags)


def _model_selectbox(label, station_id, model_names, *, key, on_change=None, args=()):
    """Lista modeli z tagami statusu; domyślnie aktywny model klienta."""
    if not model_names:
        return None
    active = model_registry.get_active(station_id)
    best = model_registry.get_best(station_id)
    llm_rec = model_registry.effective_llm_recommended(station_id)
    options = [
        f"{n}{_model_tag_suffix(n, active, llm_rec, best)}" for n in model_names
    ]
    default_idx = model_names.index(active) if active in model_names else 0
    pick = st.selectbox(
        label, options, index=default_idx, key=key,
        on_change=on_change, args=args,
    )
    return model_names[options.index(pick)]


def _model_row_badges_html(mname, active, llm_rec, best):
    """Odznaki statusu modelu — wyrównane do prawej, oddzielone od nazwy."""
    tags = []
    if mname == active:
        tags.append(("check_circle", "AKTYWNY", "#059669"))
    if llm_rec and mname == llm_rec:
        tags.append(("star", "REKOMENDOWANY", "#d97706"))
    if mname == best:
        tags.append(("analytics", "NAJLEPSZY", "#2563eb"))
    if not tags:
        return ""
    parts = []
    for icon, label, color in tags:
        parts.append(
            f'<span class="aq-badge aq-model-tag" style="background:{color};">'
            f'<span class="material-icons" style="font-size:14px;vertical-align:middle;'
            f'margin-right:3px;">{icon}</span>{label}</span>'
        )
    return f'<div class="aq-model-tags">{"".join(parts)}</div>'


def _model_row_header_html(title, badge_html):
    """Nazwa modelu i odznaki w jednej linii (niezależnie od rozwinięcia expandera)."""
    safe_title = html.escape(str(title))
    tags = badge_html or ""
    return (
        f'<div class="aq-model-row-head">'
        f'<span class="aq-model-name">{safe_title}</span>{tags}'
        f"</div>"
    )


def _render_client_status_banner(active):
    """Status klienta — tylko gdy brak aktywnego modelu (aktywny = tag na liście)."""
    if not active:
        st.warning(
            f"Brak aktywnego — ustaw **{mi.lbl(mi.CHECK_CIRCLE, 'Aktywny')}** przy modelu poniżej."
        )


def _tab_models(sname, smeta):
    st.header("Zarządzanie modelami")
    sid = smeta["id"]
    st.caption(
        f"Stacja: **{sname}** (id {sid}). **Najlepszy** ({mi.ANALYTICS}) przeliczany przy wejściu "
        f"(MAE, RMSE, R²). Lista po lewej, rekomendacja LLM po prawej."
    )

    models = model_registry.models_for_station(sid)
    active = model_registry.get_active(sid)
    best = model_registry.refresh_best_for_station(sid)
    llm_rec = model_registry.effective_llm_recommended(sid)
    if not models:
        st.info("Brak modeli dla tej stacji — wytrenuj w zakładce **Trening modeli**.")
        return

    df_proc = ui.load_processed_data(smeta["id"], smeta["lat"], smeta["lon"])
    col_models, col_llm = st.columns([2, 1], gap="large")

    with col_llm:
        _render_llm_client_column(sname, sid, models)

    with col_models:
        _render_client_status_banner(active)
        for mname, mmeta in models:
            met = mmeta.get("metrics", {})
            mae_v = met.get("mae")
            r2_v = met.get("r2")
            title = (
                f"{mname} — MAE {mae_v:.2f}  R² {r2_v:.3f}"
                if mae_v is not None else mname
            )
            badge_html = _model_row_badges_html(mname, active, llm_rec, best)
            st.markdown(_model_row_header_html(title, badge_html), unsafe_allow_html=True)
            with st.expander("Szczegóły modelu", expanded=(mname == active)):
                t1, t2, t3, t4, t5, t6, t7 = st.columns(7)
                t1.metric("Okno (h)", mmeta.get("time_steps", "—"))
                t2.metric("Horyzont (h)", mmeta.get("horizon", "—"))
                t3.metric("Epoki", f"{mmeta.get('epochs_run', '—')}/{mmeta.get('epochs_max', '—')}")
                t4.metric("LSTM", mmeta.get("lstm_units", "—"))
                t5.metric("Wierszy", mmeta.get("n_rows", "—"))
                t6.metric("Czas treningu", training_timing.format_duration(
                    mmeta.get("training_duration_sec")))
                t7.metric("Data", mmeta.get("trained_at", "—"))

                is_active = mname == active
                saved_eval, eval_at = model_registry.get_llm_evaluation(mname)
                if saved_eval is None:
                    saved_eval = mmeta.get("llm_evaluation")
                    eval_at = mmeta.get("llm_evaluated_at")

                llm_btn = (
                    mi.lbl(mi.REFRESH, "Ponów ocenę LLM")
                    if saved_eval else mi.lbl(mi.EDIT_NOTE, "Ocena LLM (ręcznie)")
                )
                st.markdown('<div class="aq-model-actions">', unsafe_allow_html=True)
                b1, b2, b3 = st.columns([2, 1, 1])
                with b1:
                    if st.button(
                        llm_btn,
                        key=f"llmmet_{mname}_{sid}",
                        use_container_width=True,
                        type="secondary" if saved_eval else "primary",
                    ):
                        with st.spinner("Generowanie oceny LLM…"):
                            _t, err = llm_eval.evaluate_after_training(
                                mname, mmeta, sname,
                            )
                            if err:
                                st.error(err)
                            else:
                                st.rerun()
                with b2:
                    if ui.status_button(mi.lbl(mi.CHECK_CIRCLE, "Aktywny"), is_active, f"act_{mname}_{sid}"):
                        model_registry.set_active(sid, mname)
                        st.rerun()
                with b3:
                    if st.button(mi.lbl(mi.DELETE, "Usuń"), key=f"del_{mname}_{sid}", use_container_width=True):
                        model_registry.delete_model(mname)
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

                if saved_eval:
                    st.markdown(f"**Ocena LLM** ({eval_at or 'zapisana'}):")
                    st.markdown(saved_eval)

                st.markdown("#### Statystyki i wykresy (zbiór testowy)")
                _render_model_stats(mname, mmeta, df_proc)


def _clear_update_outcome(station_id):
    st.session_state.pop(f"update_outcome_{station_id}", None)


def _update_params_from_meta(mmeta):
    """Hiperparametry skopiowane z metadanych modelu źródłowego."""
    return {
        "time_steps": int(mmeta["time_steps"]),
        "forecast_horizon": int(mmeta["horizon"]),
        "epochs": int(mmeta.get("epochs_max") or mmeta.get("epochs_run") or config.DEFAULT_EPOCHS),
        "lstm_units": int(mmeta["lstm_units"]),
        "lstm_layers": int(mmeta.get("lstm_layers") or config.DEFAULT_LSTM_LAYERS),
        "dropout": float(
            mmeta["dropout"] if mmeta.get("dropout") is not None else config.DEFAULT_DROPOUT
        ),
        "horizon_weighted_loss": bool(
            mmeta.get("horizon_weighted_loss", config.DEFAULT_HORIZON_WEIGHTED_LOSS)
        ),
        "feature_columns": list(mmeta["feature_columns"]),
    }


def _metric_pct_delta(old, new):
    """Zmiana względna w % (2 miejsca po przecinku) do st.metric(delta=...)."""
    if old is None or new is None:
        return None
    old_f = float(old)
    if old_f == 0:
        return None
    return f"{((float(new) - old_f) / abs(old_f)) * 100:.2f}%"


def _render_update_comparison(source_name, old_metrics, new_name, new_meta):
    """Tabela porównawcza MAE / R² przed i po aktualizacji (ta sama baza danych)."""
    new_metrics = (new_meta or {}).get("metrics") or {}
    old_mae = old_metrics.get("mae")
    old_r2 = old_metrics.get("r2")
    new_mae = new_metrics.get("mae")
    new_r2 = new_metrics.get("r2")

    st.markdown("**Porównanie modeli (zbiór testowy na bieżących danych)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption(f"Źródło: `{source_name}`")
        if old_mae is not None:
            st.metric("MAE (przed)", f"{old_mae:.2f}")
        if old_r2 is not None:
            st.metric("R² (przed)", f"{old_r2:.3f}")
    with c2:
        st.caption(f"Nowy: `{new_name}`")
        if new_mae is not None:
            mae_delta = _metric_pct_delta(old_mae, new_mae)
            st.metric("MAE (po)", f"{new_mae:.2f}", delta=mae_delta, delta_color="inverse")
        if new_r2 is not None:
            r2_delta = _metric_pct_delta(old_r2, new_r2)
            st.metric("R² (po)", f"{new_r2:.3f}", delta=r2_delta)
    with c3:
        if old_mae is not None and new_mae is not None:
            pct = (old_mae - new_mae) / old_mae * 100 if old_mae else 0.0
            if new_mae < old_mae:
                st.success(f"MAE lepsze o **{pct:.2f}%**")
            elif new_mae > old_mae:
                st.warning(f"MAE gorsze o **{abs(pct):.2f}%** — rozważ zostawienie starego modelu.")
            else:
                st.info("MAE bez zmian.")


def _render_update_llm_comparison(
    sid, outcome, source_name, source_meta, new_name, new_meta, station_name,
):
    """Sekcja porównania LLM stary vs nowy model."""
    st.markdown("#### Porównanie LLM (stary → nowy)")
    llm_txt = outcome.get("llm_compare")
    llm_err = outcome.get("llm_compare_err")
    if not llm_txt:
        saved, saved_at, saved_src = model_registry.get_llm_update_comparison(new_name)
        if saved and saved_src == source_name:
            llm_txt = saved
            if saved_at:
                st.caption(f"Zapisane: {saved_at}")

    c1, c2 = st.columns([1, 3])
    with c1:
        regen = st.button(
            mi.lbl(mi.REFRESH, "Porównaj ponownie (LLM)"),
            key=f"upd_llm_regen_{sid}_{new_name}",
            use_container_width=True,
        )
    if regen:
        with st.spinner("Analiza LLM…"):
            text, err = llm_eval.compare_models_after_update(
                source_name,
                source_meta,
                outcome.get("old_metrics") or {},
                new_name,
                new_meta,
                station_name,
            )
            outcome["llm_compare"] = text
            outcome["llm_compare_err"] = err
            st.session_state[f"update_outcome_{sid}"] = outcome
            st.rerun()

    if llm_err and not llm_txt:
        st.warning(f"Porównanie LLM: {llm_err}")
    elif llm_txt:
        st.markdown(llm_txt)
    else:
        st.info("Brak porównania LLM — użyj przycisku obok, aby wygenerować.")


def _tab_update_eval(name, meta, df_proc, store):
    st.header("Aktualizacja i ewaluacja")
    st.caption(
        "Retrening **tych samych parametrów** co wybrany model, na **aktualnej** warstwie processed. "
        "Prognoza operacyjna jest na widoku klienta — tutaj odświeżasz model i porównujesz jakość."
    )
    station_models = [n for n, m in model_registry.models_for_station(meta["id"])]
    if not station_models:
        st.warning("Brak modeli — najpierw wytrenuj w zakładce **Trening modeli**.")
        return

    sid = meta["id"]
    sel = _model_selectbox(
        "Model źródłowy (parametry + baza do porównania):",
        sid,
        station_models,
        key=f"upd_model_{sid}",
        on_change=_clear_update_outcome,
        args=(sid,),
    )
    mmeta = model_registry.load_meta(sel)
    params = _update_params_from_meta(mmeta)
    n_hours = len(df_proc)

    st.subheader("Dane treningowe")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Wiersze processed", n_hours)
    d2.metric("Zakres (dni)", f"{n_hours / 24:.1f}")
    d3.metric("Ostatnia aktualizacja danych", (store.get("last_updated") or "—")[:16])
    d4.metric("Trening źródła", mmeta.get("trained_at", "—"))

    rf1, rf2 = st.columns([1, 2])
    with rf1:
        refresh_before = st.checkbox(
            "Odśwież dane z GIOŚ przed treningiem",
            value=True,
            key=f"upd_refresh_{sid}",
        )
    with rf2:
        if st.button(mi.lbl(mi.REFRESH, "Odśwież dane teraz"), key=f"upd_btn_refresh_{sid}"):
            ui.load_processed_data.clear()
            ui.load_forecast.clear()
            _clear_update_outcome(sid)
            st.rerun()

    st.subheader("Parametry aktualizacji (z modelu źródłowego)")
    p1, p2, p3, p4, p5, p6, p7 = st.columns(7)
    p1.metric("Okno (h)", params["time_steps"])
    p2.metric("Horyzont (h)", params["forecast_horizon"])
    p3.metric("Epoki (max)", params["epochs"])
    p4.metric("Neurony LSTM", params["lstm_units"])
    p5.metric("Warstwy", params["lstm_layers"])
    p6.metric("Dropout", f"{params['dropout']:.2f}")
    p7.metric("Loss ważony", "tak" if params["horizon_weighted_loss"] else "nie")

    feats = [c for c in params["feature_columns"] if c in df_proc.columns]
    missing = [c for c in params["feature_columns"] if c not in df_proc.columns]
    if missing:
        st.error(f"Brak cech w processed: {missing}. Odśwież dane lub wytrenuj od zera.")
        return

    n_seq = config.training_sequence_count(
        n_hours, params["time_steps"], params["forecast_horizon"],
    )
    train_ok, train_err = config.validate_training_params(
        n_hours, params["time_steps"], params["forecast_horizon"],
    )
    st.caption(
        f"Sekwencje przy tych ustawieniach: **{n_seq}** "
        f"(min. {config.MIN_TRAIN_SEQUENCES}). "
        f"Podgląd nazwy: `{model_registry.build_model_name(name, params['time_steps'], params['forecast_horizon'], params['epochs'], params['lstm_units'])}`"
    )
    if not train_ok:
        st.warning(train_err)

    old_eval = model_eval.resolve_evaluation(sel, mmeta, df_proc)
    if old_eval.get("error"):
        st.warning(f"Ewaluacja źródła na bieżących danych: {old_eval['error']}")
    old_metrics = old_eval.get("metrics") or mmeta.get("metrics") or {}

    st.subheader("Stan przed aktualizacją")
    b1, b2 = st.columns(2)
    if old_metrics.get("mae") is not None:
        b1.metric("MAE (model źródłowy)", f"{old_metrics['mae']:.2f}")
    if old_metrics.get("r2") is not None:
        b2.metric("R² (model źródłowy)", f"{old_metrics['r2']:.3f}")

    est_range = training_timing.estimate_training_range_label(
        n_hours,
        params["time_steps"],
        params["forecast_horizon"],
        params["epochs"],
        params["lstm_units"],
        len(feats),
    )
    st.caption(f"Szac. czas aktualizacji: **{est_range}**")

    if st.button(
        mi.lbl(mi.ROCKET, "Aktualizuj model"),
        type="primary",
        disabled=not train_ok,
        key=f"upd_train_{sid}",
    ):
        df_train = df_proc
        if refresh_before:
            with st.spinner("Pobieranie danych z GIOŚ i Open-Meteo…"):
                ui.load_processed_data.clear()
                ui.load_forecast.clear()
                df_train = data_fetch.update_and_load(meta["id"], meta["lat"], meta["lon"])
        if df_train.empty:
            st.error("Brak danych processed po odświeżeniu.")
        else:
            old_eval_run = model_eval.resolve_evaluation(sel, mmeta, df_train)
            old_metrics_run = old_eval_run.get("metrics") or old_metrics
            pb = st.progress(0)
            txt = st.empty()
            mname, result = ui.train_lstm(
                df_train,
                name,
                meta["id"],
                feats,
                params["time_steps"],
                params["forecast_horizon"],
                params["epochs"],
                params["lstm_units"],
                (pb, txt),
                lstm_layers=params["lstm_layers"],
                dropout=params["dropout"],
                horizon_weighted_loss=params["horizon_weighted_loss"],
            )
            pb.empty()
            txt.empty()
            if mname:
                model_registry.refresh_best_for_station(sid)
                ui.load_processed_data.clear()
                llm_txt, llm_err = None, None
                with st.spinner("Porównanie LLM stary vs nowy…"):
                    llm_txt, llm_err = llm_eval.compare_models_after_update(
                        sel,
                        mmeta,
                        old_metrics_run,
                        mname,
                        result,
                        name,
                    )
                st.session_state[f"update_outcome_{sid}"] = {
                    "source": sel,
                    "old_metrics": old_metrics_run,
                    "new_name": mname,
                    "new_meta": result,
                    "epoch_summary": ui.training_epoch_summary(result),
                    "llm_compare": llm_txt,
                    "llm_compare_err": llm_err,
                }
                st.rerun()
            else:
                st.error(result)

    outcome = st.session_state.get(f"update_outcome_{sid}")
    if outcome and outcome.get("source") == sel and outcome.get("new_name"):
        new_name = outcome["new_name"]
        new_meta = outcome.get("new_meta") or {}
        dur = training_timing.format_duration(new_meta.get("training_duration_sec"))
        st.success(
            f"Zaktualizowano: **`{new_name}`** (z `{sel}`). Czas treningu: **{dur}**\n\n"
            f"{outcome.get('epoch_summary') or ui.training_epoch_summary(new_meta)}"
        )
        _render_update_comparison(
            sel, outcome.get("old_metrics") or {}, new_name, new_meta,
        )
        source_meta = model_registry.load_meta(sel) or {}
        _render_update_llm_comparison(
            sid, outcome, sel, source_meta, new_name, new_meta, name,
        )
        a1, a2 = st.columns(2)
        with a1:
            if st.button(
                mi.lbl(mi.CHECK_CIRCLE, "Ustaw nowy jako aktywny"),
                type="primary",
                key=f"upd_act_{sid}_{new_name}",
            ):
                model_registry.set_active(sid, new_name)
                st.rerun()
        with a2:
            if st.button("Zostaw obecny aktywny", key=f"upd_keep_{sid}"):
                _clear_update_outcome(sid)
                st.rerun()

        st.divider()
        st.subheader("Ewaluacja nowego modelu")
        df_eval = ui.load_processed_data(meta["id"], meta["lat"], meta["lon"])
        _render_model_stats(new_name, new_meta, df_eval)
    elif outcome and outcome.get("source") != sel:
        st.info("Wynik aktualizacji dotyczy innego modelu źródłowego — wybierz go z listy lub uruchom ponownie.")


def _tab_preview_compare(name, meta, df_proc):
    """Podgląd wykresu klienta + porównanie Open-Meteo vs LSTM (admin)."""
    st.header("Podgląd i porównanie")
    st.caption(
        f"Wykres godzinowy ±{config.CLIENT_FORECAST_DAYS} dni względem **teraz**: "
        f"**{ui.GIOS_AQI_LABEL}** wstecz (pomiary stacji), **Open-Meteo** wstecz i wprzód, "
        f"opcjonalnie **{ui.ASQI_SHORT} (LSTM)** wprzód."
    )
    sid = meta["id"]

    with st.spinner(f"Open-Meteo ±{config.CLIENT_FORECAST_DAYS} dni…"):
        df_fc = ui.load_forecast(meta["lat"], meta["lon"])
    if df_fc.empty:
        st.error("Nie udało się pobrać prognozy Open-Meteo.")
        return

    if df_proc is None or df_proc.empty:
        st.warning(
            "Brak danych processed — wykres pokaże tylko Open-Meteo. "
            "Odśwież dane w **Dane surowe** lub użyj przycisku poniżej."
        )
        df_proc = df_proc if df_proc is not None else pd.DataFrame()

    station_models = [n for n, m in model_registry.models_for_station(sid)]
    active = model_registry.get_active(sid)

    extra_available = [
        lbl for lbl in ui.EXTRA_AQI_SERIES
        if ui.EXTRA_AQI_SERIES[lbl][0] in df_fc.columns
    ]

    if st.button(
        mi.lbl(mi.REFRESH, "Odśwież dane"),
        key=f"prev_refresh_{sid}",
    ):
        ui.load_processed_data.clear()
        ui.load_forecast.clear()
        st.rerun()

    model_col, ind_col = st.columns(2)
    with model_col:
        lstm_model = None
        if station_models:
            lstm_model = _model_selectbox(
                "Model LSTM na wykresie:",
                sid,
                station_models,
                key=f"prev_lstm_{sid}",
            )
        else:
            st.caption("Brak modeli LSTM — wytrenuj w zakładce Trening.")

    lstm_label = f"{ui.ASQI_SHORT} (LSTM)"
    indicator_options = [ui.GIOS_AQI_LABEL, ui.OM_COMPOSITE_LABEL]
    if lstm_model:
        indicator_options.append(lstm_label)
    indicator_options.extend(extra_available)

    default_indicators = [ui.GIOS_AQI_LABEL, ui.OM_COMPOSITE_LABEL]
    if lstm_model:
        default_indicators.append(lstm_label)

    with ind_col:
        selected = st.multiselect(
            "Wskaźniki:",
            indicator_options,
            default=default_indicators,
            key=f"prev_ind_{sid}",
        )
    show_gios = ui.GIOS_AQI_LABEL in selected
    show_om = ui.OM_COMPOSITE_LABEL in selected
    show_lstm = lstm_model and lstm_label in selected
    extra = [lbl for lbl in selected if lbl in extra_available]

    if active and lstm_model == active:
        st.caption(f"Model LSTM = **aktywny** na widoku klienta (`{active}`).")
    elif active:
        st.caption(
            f"Aktywny u klienta: `{active}`. Na wykresie: `{lstm_model or '—'}`."
        )

    fig = ui.build_hourly_forecast_figure(
        df_fc,
        df_proc,
        horizon_days=config.CLIENT_FORECAST_DAYS,
        extra_labels=extra,
        show_openmeteo_aqi=show_om,
        show_gios_aqi=show_gios,
        lstm_model_name=lstm_model if show_lstm else None,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Niebieski** — {ui.GIOS_AQI_LABEL} z pomiarów stacji (wstecz). "
        f"**Turkus (kropki)** — {ui.OM_COMPOSITE_LABEL} (wstecz). "
        f"**Turkus (ciągła)** — ta sama seria wprzód. "
        f"**Czerwony** — prognoza {ui.ASQI_SHORT} (LSTM, wprzód). "
        "Pionowa linia = teraz. Skala 0–100."
    )


def _tab_settings(stations):
    st.header("Ustawienia")
    settings = stations_settings.load_settings()
    catalog = stations_settings.all_stations_catalog(settings)
    enabled_ids = set(int(x) for x in settings.get("enabled_station_ids", []))
    in_app_ids = stations_settings.configured_station_ids()

    tab_list, tab_add, tab_audit, tab_openai = st.tabs([
        "Moje stacje", "Dodaj z listy GIOŚ", "Kompletność GIOŚ", "OpenAI",
    ])
    audit_cache = gios_audit.load_audit_cache()

    with tab_list:
        st.caption(
            "Widoczna w aplikacji — stacja w selectboxie klienta/admin. "
            "Usuń — trwale usuwa stację i plik danych `data_store/station_<id>.json`."
        )
        if not catalog:
            st.info("Brak stacji w aplikacji — dodaj stację w drugiej zakładce.")
        for sname, sm in sorted(catalog.items(), key=lambda x: x[0]):
            sid = int(sm["id"])
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                on = sid in enabled_ids
                new_on = st.checkbox(
                    f"**{sname}** — id {sid}  ({sm['lat']:.4f}, {sm['lon']:.4f})",
                    value=on,
                    key=f"en_{sid}",
                )
                if new_on != on:
                    stations_settings.set_station_enabled(sid, new_on)
                    st.rerun()
            with c3:
                if st.button("Usuń", key=f"del_st_{sid}", type="secondary"):
                    st.session_state["confirm_del_station"] = sid
            if st.session_state.get("confirm_del_station") == sid:
                st.warning(f"Czy na pewno usunąć **{sname}** (id {sid})?")
                y, n = st.columns(2)
                if y.button("Tak, usuń", key=f"yes_del_{sid}", type="primary"):
                    stations_settings.remove_station(sid, delete_data=True)
                    ui.load_processed_data.clear()
                    ui.load_forecast.clear()
                    st.session_state.pop("confirm_del_station", None)
                    st.success(f"Usunięto stację {sname}.")
                    st.rerun()
                if n.button("Anuluj", key=f"no_del_{sid}"):
                    st.session_state.pop("confirm_del_station", None)
                    st.rerun()

    with tab_audit:
        st.subheader("Które stacje mają pełny komplet?")
        st.markdown(
            f"**Pełne sensory EAQI** — wszystkie 5 parametrów: "
            f"{', '.join(config.REQUIRED_EAQI_POLLUTANTS)}.  \n"
            f"**Pełne dane 30 dni** — każdy z tych parametrów ma ≥ "
            f"**{config.MIN_ARCHIVAL_POINTS}** punktów godzinowych w archiwum GIOŚ "
            f"(~70% z {config.EXPECTED_HOURLY_30D} h)."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button(mi.lbl(mi.SEARCH, "Skanuj sensory (wszystkie stacje, ~1–2 min)"), type="primary"):
                gios_list_scan = stations_settings.fetch_gios_station_list()
                prog = st.progress(0)
                txt = st.empty()

                def _prog(i, total, name):
                    prog.progress(i / total)
                    txt.caption(f"{i}/{total}: {name}")

                gios_audit.scan_all_sensors(gios_list_scan, on_progress=_prog)
                st.success("Skan sensorów zakończony.")
                st.rerun()
        with c2:
            if st.button(mi.lbl(mi.ANALYTICS, "Skanuj dane 30 dni (tylko stacje z 5/5 sensorów)")):
                cache_pre = gios_audit.load_audit_cache()
                n = sum(1 for e in cache_pre.get("stations", {}).values() if e.get("full_sensors"))
                if n == 0:
                    st.warning("Najpierw uruchom skan sensorów.")
                else:
                    prog = st.progress(0)
                    txt = st.empty()

                    def _prog2(i, total, name):
                        prog.progress(i / total)
                        txt.caption(f"{i}/{total}: {name}")

                    with st.spinner(f"Pobieranie archiwum dla {n} stacji (może potrwać)…"):
                        gios_audit.scan_data_full_sensor_stations(on_progress=_prog2)
                    st.success("Skan danych zakończony.")
                    st.rerun()
        audit_cache = gios_audit.load_audit_cache()
        if audit_cache.get("sensor_scan_at"):
            st.caption(
                f"Ostatni skan sensorów: {audit_cache['sensor_scan_at']} | "
                f"skan danych: {audit_cache.get('data_scan_at') or '—'}"
            )
        rows = gios_audit.audit_table_rows(audit_cache)
        if not rows:
            st.info("Brak wyników audytu — uruchom skan sensorów.")
        else:
            df_audit = pd.DataFrame(rows)
            f1, f2, f3 = st.columns(3)
            only_sens = f1.checkbox("Tylko pełne sensory (5/5)", value=False)
            only_data = f2.checkbox("Tylko pełne dane 30 dni", value=False)
            q_a = f3.text_input("Filtr nazwy/miasta:", key="audit_q")
            if only_sens:
                df_audit = df_audit[df_audit["pełne_sensory"] == mi.YES]
            if only_data:
                df_audit = df_audit[df_audit["pełne_dane_30d"] == mi.YES]
            if q_a.strip():
                ql = q_a.strip().lower()
                df_audit = df_audit[
                    df_audit["nazwa"].str.lower().str.contains(ql, na=False)
                    | df_audit["miasto"].str.lower().str.contains(ql, na=False)
                    | (df_audit["id"].astype(str) == ql)
                ]
            st.metric("Stacji w tabeli", len(df_audit))
            n_full = int((df_audit["pełne_dane_30d"] == mi.YES).sum()) if "pełne_dane_30d" in df_audit else 0
            n_sens = int((df_audit["pełne_sensory"] == mi.YES).sum())
            st.caption(f"Pełne sensory: **{n_sens}** | Pełne dane 30d: **{n_full}**")
            st.dataframe(df_audit, use_container_width=True, height=420)
            st.download_button(
                "Pobierz CSV audytu",
                df_audit.to_csv(index=False).encode("utf-8-sig"),
                file_name="gios_audit.csv",
            )

    with tab_add:
        @st.cache_data(ttl=3600, show_spinner="Pobieranie listy stacji GIOŚ…")
        def _gios_catalog():
            return stations_settings.fetch_gios_station_list()

        gios_list = _gios_catalog()
        audit_cache = gios_audit.load_audit_cache()
        if not gios_list:
            st.error("Nie udało się pobrać listy z API GIOŚ.")
        else:
            st.caption(f"Dostępnych stacji w GIOŚ: **{len(gios_list)}** (możesz filtrować poniżej).")
            if not audit_cache.get("sensor_scan_at"):
                st.warning(
                    "Uruchom **Kompletność GIOŚ → Skanuj sensory**, aby filtrować stacje "
                    "z pełnym kompletem EAQI."
                )
            fa, fb, fc = st.columns(3)
            only_full_sens = fa.checkbox("Tylko 5/5 sensory EAQI", key="add_full_sens")
            only_full_data = fb.checkbox("Tylko pełne dane 30d", key="add_full_data")
            q = fc.text_input("Szukaj:", key="gios_search")
            q_l = q.strip().lower()
            filtered = []
            for r in gios_list:
                if r["id"] in in_app_ids:
                    continue
                aud = gios_audit.get_station_audit(r["id"], audit_cache)
                if only_full_sens and not (aud and aud.get("full_sensors")):
                    continue
                if only_full_data and not (aud and aud.get("full_data_30d")):
                    continue
                badge = gios_audit.format_station_badge(r["id"], audit_cache)
                badge_s = f" [{badge}]" if badge else ""
                label = f"{r['name']} — {r.get('city') or '?'} (id {r['id']}){badge_s}"
                if q_l and q_l not in label.lower() and q_l != str(r["id"]):
                    continue
                filtered.append((label, r))
            if not filtered:
                st.info(
                    "Brak wyników — zmień filtr albo usuń stację z listy „Moje stacje”, "
                    "jeśli chcesz dodać ją ponownie."
                )
            else:
                labels = [x[0] for x in filtered]
                pick = st.selectbox("Wybierz stację:", labels, key="gios_pick")
                chosen = dict(filtered)[pick]
                st.write(
                    f"Współrzędne: **{chosen['lat']:.4f}**, **{chosen['lon']:.4f}**"
                )
                aud = gios_audit.get_station_audit(chosen["id"], audit_cache)
                if aud:
                    st.json({
                        "sensory_EAQI": aud.get("eaqi_present"),
                        "brakuje": aud.get("eaqi_missing"),
                        "pełne_sensory": aud.get("full_sensors"),
                        "punkty_30d": aud.get("points"),
                        "pełne_dane_30d": aud.get("full_data_30d"),
                    })
                if st.button("Sprawdź tę stację (sensory + archiwum 30d)", key="audit_one"):
                    with st.spinner("Pobieranie…"):
                        s = gios_audit.audit_sensors(chosen["id"])
                        d = gios_audit.audit_archival_data(chosen["id"])
                        cache = gios_audit.load_audit_cache()
                        ent = cache["stations"].setdefault(str(chosen["id"]), {})
                        ent.update(s)
                        ent.update(d)
                        ent.update({
                            "name": chosen["name"],
                            "city": chosen.get("city", ""),
                            "lat": chosen["lat"],
                            "lon": chosen["lon"],
                        })
                        gios_audit.save_audit_cache(cache)
                    st.rerun()
                if st.button(mi.lbl(mi.ADD, "Dodaj do aplikacji"), type="primary"):
                    stations_settings.add_custom_station(
                        chosen["name"], chosen["id"], chosen["lat"], chosen["lon"]
                    )
                    ui.load_processed_data.clear()
                    st.success(f"Dodano: {chosen['name']} (id {chosen['id']}).")
                    st.rerun()

        with st.expander("Dodaj ręcznie (ID + współrzędne)"):
            with st.form("manual_station"):
                n = st.text_input("Nazwa")
                sid = st.number_input("ID stacji", min_value=1, step=1)
                lat = st.number_input("Szerokość", format="%.4f")
                lon = st.number_input("Długość", format="%.4f")
                if st.form_submit_button("Dodaj ręcznie"):
                    stations_settings.add_custom_station(n, sid, lat, lon)
                    ui.load_processed_data.clear()
                    st.rerun()

    with tab_openai:
        st.subheader("OpenAI")
        st.caption(
            "Klucz i model używane we wszystkich funkcjach LLM (trening, modele, ocena). "
            "Pusty klucz = `OPENAI_API_KEY` z `.env`."
        )
        key = st.text_input(
            "Klucz API:",
            type="password",
            value=settings.get("openai_api_key", ""),
            key="openai_key_input",
        )
        cur_model = settings.get("openai_model") or "gpt-4o"
        if cur_model not in stations_settings.OPENAI_MODEL_CHOICES:
            cur_model = stations_settings.OPENAI_MODEL_CHOICES[0]
        model_pick = st.selectbox(
            "Model LLM:",
            stations_settings.OPENAI_MODEL_CHOICES,
            index=stations_settings.OPENAI_MODEL_CHOICES.index(cur_model),
            key="openai_model_pick",
            help="gpt-5.5 = wolniejszy, głębsze rozumowanie; gpt-4o = szybszy, domyślny.",
        )
        if st.button("Zapisz ustawienia OpenAI", key="save_openai_settings"):
            settings["openai_api_key"] = key.strip()
            settings["openai_model"] = model_pick
            stations_settings.save_settings(settings)
            st.success(f"Zapisano. Model: **{model_pick}**")
            st.rerun()
        st.info(f"Aktualnie: model **{stations_settings.resolve_openai_model()}**")
        st.subheader("Stałe projektu")
        st.markdown(
            f"- Min. dni treningu: **{config.MIN_TRAIN_DAYS}**\n"
            f"- Max horyzont prognozy: **{config.MAX_FORECAST_HOURS}** h "
            f"({config.MAX_FORECAST_DAYS} dni)\n"
            f"- Okno LSTM: **{config.MIN_WINDOW_HOURS}–{config.MAX_WINDOW_HOURS}** h\n"
            f"- Neurony LSTM: **{config.MIN_LSTM_UNITS}–{config.MAX_LSTM_UNITS}**"
        )
