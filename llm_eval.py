# -*- coding: utf-8 -*-
"""Ocena metryk modelu z pomocą LLM (OpenAI).

Bierze metryki MAE/MSE/RMSE/R² wskaźnika AirSenseQualityIndex / ASQI (skala 0–100) i prosi model
językowy o interpretację po polsku: co znaczą liczby, czy model jest dobry i co poprawić.
Moduł niezależny od Streamlita.
"""
import os

import config

DEFAULT_MODEL = "gpt-4o"
# Najmocniejszy model (czerwiec 2026) — model rozumujący z konfigurowalnym wysiłkiem.
STRONG_MODEL = "gpt-5.5"

EVAL_BASELINE_LABELS = {
    "persistence": "Persistence (ostatnia wartość)",
    "ma_24h": "Średnia dobowa (24 h)",
    "ma_168h": "Średnia tygodniowa (168 h)",
    "seasonal_24h": "Sezonowość dobowa (lag 24 h)",
    "seasonal_168h": "Sezonowość tygodniowa (lag 168 h)",
}

SYSTEM_PROMPT = (
    "Jesteś analitykiem ML w projekcie AirSense — prognoza jakości powietrza w Polsce "
    "(GIOŚ + Open-Meteo, dane godzinowe).\n\n"
    "**Cel produktu:** przewidywać wskaźnik **AirSenseQualityIndex (ASQI)** (0–100; niżej = czystsze powietrze) "
    f"na minimum {config.MIN_FORECAST_DAYS} dni, docelowo {config.DEFAULT_FORECAST_DAYS} dni do przodu.\n\n"
    "**Model:** LSTM wielowymiarowe → wyjście wielokrokowe (cały horyzont naraz).\n"
    "**Wejście (cechy):** stężenia zanieczyszczeń ze stacji, pogoda (temperatura, wilgotność, "
    "wiatr, opady), kalendarz (godzina/dzień tygodnia sin/cos, sezon grzewczy paź–kwi).\n"
    "**Trening:** okno wejściowe domyślnie ok. 21 dni wstecz; split sekwencji 80/10/10; "
    "early stopping; opcjonalnie 1–2 warstwy LSTM, dropout, loss ważony (bliższe kroki horyzontu).\n\n"
    "**Benchmark:** na zbiorze testowym porównujemy LSTM z baseline: persistence, średnia "
    "dobowa/tygodniowa, sezonowość 24 h / 168 h. Jeśli LSTM nie bije prostych baseline, "
    "wskaż to w werdykcie.\n\n"
    "Odpowiadasz po polsku, zwięźle, w markdown. Sugestie poprawy muszą być praktyczne "
    "w ramach tego pipeline (dane, okno, horyzont, architektura, cechy) — bez ogólników."
)


def resolve_api_key(explicit_key=None):
    """Zwraca klucz API: jawny argument, ustawienia aplikacji lub .env."""
    key = (explicit_key or "").strip()
    if not key:
        try:
            import stations_settings
            key = stations_settings.resolve_openai_key()
        except Exception:
            pass
    if not key:
        try:
            import env_config
            env_config.load_env()
            key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        except Exception:
            pass
    return key


def build_eval_context(meta, station_name):
    """Kontekst treningu i baseline'ów do promptu oceny modelu."""
    meta = meta or {}
    metrics = meta.get("metrics") or {}
    ctx = {
        "stacja": station_name,
        "okno_wejsciowe_h": meta.get("time_steps"),
        "horyzont_h": meta.get("horizon"),
        "epoki": f"{meta.get('epochs_run')}/{meta.get('epochs_max')}",
        "neurony_lstm": meta.get("lstm_units"),
        "warstwy_lstm": meta.get("lstm_layers"),
        "dropout": meta.get("dropout"),
        "loss_wazony_horyzont": meta.get("horizon_weighted_loss"),
        "liczba_cech": len(meta.get("feature_columns") or []),
        "wiersze_processed": meta.get("n_rows"),
    }
    ts = meta.get("time_steps")
    hor = meta.get("horizon")
    if ts:
        ctx["okno_wejsciowe_dni"] = round(int(ts) / 24, 1)
    if hor:
        ctx["horyzont_dni"] = round(int(hor) / 24, 1)
    split = meta.get("eval_split") or {}
    if split:
        ctx["split_test_sekwencji"] = (
            f"train {split.get('train')} / val {split.get('val')} / test {split.get('test')}"
        )
    for key, block in (meta.get("baseline_metrics") or {}).items():
        m = (block or {}).get("metrics") or {}
        label = EVAL_BASELINE_LABELS.get(key, key)
        if m.get("mae") is not None:
            ctx[f"{label} — MAE"] = round(float(m["mae"]), 2)
        if m.get("r2") is not None:
            ctx[f"{label} — R²"] = round(float(m["r2"]), 3)
    if metrics.get("mae") is not None:
        ctx["LSTM — MAE"] = round(float(metrics["mae"]), 2)
    if metrics.get("r2") is not None:
        ctx["LSTM — R²"] = round(float(metrics["r2"]), 3)
    return {k: v for k, v in ctx.items() if v is not None}


def build_prompt(metrics, context):
    """Buduje treść zapytania na podstawie metryk i kontekstu treningu."""
    ctx_lines = [f"- {k}: {v}" for k, v in (context or {}).items()]
    extra_metrics = []
    for key, label in (
        ("median_ae", "Mediana |błąd|"),
        ("bias", "Bias (śr. błąd)"),
        ("mape_pct", "MAPE %"),
        ("p90_abs_error", "P90 |błąd|"),
    ):
        if metrics.get(key) is not None:
            extra_metrics.append(f"- {label}: {metrics[key]}")
    extra_block = ("\n" + "\n".join(extra_metrics)) if extra_metrics else ""
    return (
        "Oceń model LSTM prognozujący **ASQI / AirSenseQualityIndex** (0–100, niżej = lepsze powietrze).\n\n"
        "### Kontekst treningu i porównanie z baseline\n"
        + "\n".join(ctx_lines)
        + "\n\n### Metryki LSTM na zbiorze testowym (10%, jednostki wskaźnika 0–100)\n"
        f"- MAE:  {metrics.get('mae')}\n"
        f"- MSE:  {metrics.get('mse')}\n"
        f"- RMSE: {metrics.get('rmse')}\n"
        f"- R²:   {metrics.get('r2')}"
        + extra_block
        + "\n\n"
        "Napisz krótko (maks. ~220 słów):\n"
        "1. **Interpretacja** — co oznaczają MAE/R² dla użytkownika końcowego (skala 0–100).\n"
        "2. **vs baseline** — czy LSTM bije persistence / średnie / sezonowość; jeśli nie, dlaczego to ważne.\n"
        "3. **Werdykt** — słaby / przeciętny / dobry względem celu "
        f"({config.MIN_FORECAST_DAYS}–{config.DEFAULT_FORECAST_DAYS} dni prognozy).\n"
        "4. **Sugestie** — 2–3 konkretne kroki (okno, horyzont, dropout, warstwy, cechy, więcej danych).\n"
        "Używaj nagłówków markdown."
    )


def resolve_openai_model(explicit=None):
    if explicit:
        return explicit
    try:
        import stations_settings
        return stations_settings.resolve_openai_model()
    except Exception:
        return DEFAULT_MODEL


def _api_kwargs(model_name, messages):
    """Parametry wywołania API zależne od rodziny modelu."""
    kwargs = {"model": model_name, "messages": messages}
    if model_name.startswith("gpt-5") or model_name.startswith("o"):
        kwargs["reasoning_effort"] = "high"
    else:
        kwargs["temperature"] = 0.3
    return kwargs


def evaluate_after_training(model_name, meta, station_name, api_key=None, model=None):
    """Ocena LLM zaraz po treningu; zapis w meta modelu. Zwraca (tekst, błąd)."""
    if not resolve_api_key(api_key):
        return None, "Brak klucza OpenAI (Ustawienia lub OPENAI_API_KEY)."
    import model_registry

    metrics = (meta or {}).get("metrics") or {}
    ctx = build_eval_context(meta, station_name)
    try:
        text = evaluate_metrics(metrics, ctx, api_key=api_key, model=model)
        model_registry.save_llm_evaluation(model_name, text)
        return text, None
    except Exception as e:
        return None, str(e)


def evaluate_metrics(metrics, context=None, api_key=None, model=None):
    """Zwraca tekstową ocenę metryk od LLM."""
    key = resolve_api_key(api_key)
    if not key:
        raise ValueError("Brak klucza OpenAI. Podaj klucz w polu lub ustaw zmienną OPENAI_API_KEY.")

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Pakiet 'openai' nie jest zainstalowany.") from e

    model_name = resolve_openai_model(model)
    client = OpenAI(api_key=key)
    kwargs = _api_kwargs(
        model_name,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(metrics, context)},
        ],
    )
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


UPDATE_COMPARE_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\n\n**Twoje zadanie:** porównaj **model źródłowy** (przed aktualizacją) z **modelem "
    "po retreningu** na świeższych danych — **identyczne hiperparametry** (okno, horyzont, "
    "architektura). Oba modele oceniasz na **tej samej** aktualnej bazie processed (zbiór testowy).\n"
    "Na końcu podaj jednoznaczną rekomendację operacyjną dla admina."
)


def _metrics_lines(metrics, prefix=""):
    """Krótki blok metryk do promptu porównawczego."""
    metrics = metrics or {}
    lines = []
    for key, label in (
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("r2", "R²"),
        ("mape_pct", "MAPE %"),
        ("bias", "Bias"),
    ):
        val = metrics.get(key)
        if val is not None:
            lines.append(f"- {prefix}{label}: {val}")
    return lines


def build_update_compare_prompt(
    source_name,
    source_meta,
    old_metrics,
    new_name,
    new_meta,
    station_name,
):
    """Prompt użytkownika: stary vs nowy model po aktualizacji."""
    source_meta = source_meta or {}
    new_meta = new_meta or {}
    old_metrics = old_metrics or {}
    new_metrics = new_meta.get("metrics") or {}

    def _delta(old, new):
        if old is None or new is None:
            return "—"
        return round(float(new) - float(old), 4)

    ctx_old = build_eval_context(source_meta, station_name)
    ctx_new = build_eval_context(new_meta, station_name)

    old_lines = _metrics_lines(old_metrics, "Stary — ")
    new_lines = _metrics_lines(new_metrics, "Nowy — ")
    delta_mae = _delta(old_metrics.get("mae"), new_metrics.get("mae"))
    delta_r2 = _delta(old_metrics.get("r2"), new_metrics.get("r2"))

    return (
        f"Porównaj aktualizację modelu LSTM dla stacji **{station_name}**.\n\n"
        f"### Model źródłowy (przed)\n"
        f"- Nazwa: `{source_name}`\n"
        f"- Wytrenowany: {source_meta.get('trained_at', '—')}\n"
        f"- Wiersze processed przy treningu: {source_meta.get('n_rows', '—')}\n"
        + "\n".join(f"- {k}: {v}" for k, v in ctx_old.items())
        + "\n\n### Metryki na bieżących danych (model źródłowy, zbiór testowy)\n"
        + ("\n".join(old_lines) if old_lines else "- brak metryk")
        + f"\n\n### Model po aktualizacji\n"
        f"- Nazwa: `{new_name}`\n"
        f"- Wytrenowany: {new_meta.get('trained_at', '—')}\n"
        f"- Wiersze processed: {new_meta.get('n_rows', '—')}\n"
        + "\n".join(f"- {k}: {v}" for k, v in ctx_new.items())
        + "\n\n### Metryki po retreningu (zbiór testowy)\n"
        + ("\n".join(new_lines) if new_lines else "- brak metryk")
        + f"\n\n### Zmiana (nowy − stary)\n"
        f"- Δ MAE: {delta_mae}\n"
        f"- Δ R²: {delta_r2}\n\n"
        "Napisz po polsku (maks. ~280 słów), w markdown:\n"
        "1. **Kontekst** — co się zmieniło (dane vs wagi; te same parametry).\n"
        "2. **Porównanie metryk** — MAE/R² i baseline nowego modelu; czy poprawa jest istotna "
        "praktycznie na skali AQI 0–100.\n"
        "3. **Ryzyka** — np. gorsze R² przy lepszym MAE, overfit, słaby horyzont dalszych dni.\n"
        "4. **Rekomendacja** — jedna linia: "
        "**Aktywuj nowy** / **Zostaw stary** / **Trenuj od zera z innymi parametrami** "
        "(krótkie uzasadnienie).\n"
        "Używaj nagłówków markdown."
    )


def compare_models_after_update(
    source_name,
    source_meta,
    old_metrics,
    new_name,
    new_meta,
    station_name,
    *,
    api_key=None,
    model=None,
    save=True,
):
    """Porównanie LLM stary vs nowy po aktualizacji. Zwraca (tekst, błąd)."""
    if not resolve_api_key(api_key):
        return None, "Brak klucza OpenAI (Ustawienia lub OPENAI_API_KEY)."
    try:
        from openai import OpenAI
    except ImportError as e:
        return None, "Pakiet 'openai' nie jest zainstalowany."

    import model_registry

    try:
        client = OpenAI(api_key=resolve_api_key(api_key))
        user = build_update_compare_prompt(
            source_name, source_meta, old_metrics, new_name, new_meta, station_name,
        )
        text = _chat(client, UPDATE_COMPARE_SYSTEM_PROMPT, user, model)
        if save and text:
            model_registry.save_llm_update_comparison(new_name, source_name, text)
        return text, None
    except Exception as e:
        return None, str(e)


def project_context_brief():
    """Bardzo skrótowy kontekst AirSense — wspólny dla promptów LLM."""
    return (
        "**AirSense** — prognoza jakości powietrza w Polsce (dane godzinowe: GIOŚ + Open-Meteo). "
        "Model **LSTM** przewiduje **AirSenseQualityIndex (ASQI)** (0–100; niżej = czyściej), "
        "liczony ze stężeń zanieczyszczeń (metodyka zbliżona do EAQI).\n"
        "**Wejście:** stężenia dostępne na stacji + pogoda (temp., wilgotność, wiatr, opady) + "
        "kalendarz (godzina/dzień tyg. sin/cos, sezon grzewczy paź–kwi).\n"
        f"**Cel produktu:** prognoza min. **{config.MIN_FORECAST_DAYS} dni**, docelowo "
        f"**{config.DEFAULT_FORECAST_DAYS} dni**; okno historii domyślnie **{config.DEFAULT_WINDOW_DAYS} dni**. "
        "Trening na ~30 dniach archiwum; ewaluacja 80/10/10 + baseline (persistence, średnie, sezonowość)."
    )


TRAIN_RECO_SYSTEM_PROMPT = (
    "Jesteś inżynierem ML doradzającym **ustawienia kolejnego treningu** w AirSense.\n\n"
    + project_context_brief()
    + "\n\n"
    "**Twoje zadanie:** zaproponuj liczby pod slidery UI (okno, horyzont, neurony, epoki, "
    "warstwy, dropout, loss ważony). Nie wybieraj najlepszego modelu dla klienta — tylko parametry nowego runu.\n\n"
    "**Zasady:**\n"
    f"- **Okno wejściowe (time_steps)** ≠ horyzont — okno to historia wstecz (domyślnie "
    f"{config.DEFAULT_WINDOW_HOURS} h ≈ {config.DEFAULT_WINDOW_DAYS} dni).\n"
    f"- **Horyzont (forecast_horizon):** min. {config.MIN_FORECAST_HOURS} h "
    f"({config.MIN_FORECAST_DAYS} dni), docelowo {config.DEFAULT_FORECAST_HOURS} h "
    f"({config.DEFAULT_FORECAST_DAYS} dni), jeśli starczy danych i sekwencji.\n"
    "- **Architektura:** 1–2 warstwy LSTM, dropout 0–0.5; przy długim horyzoncie rozważ "
    "loss ważony (bliższe godziny ważniejsze).\n"
    "- Analizuj istniejące modele stacji: unikaj duplikatów; popraw słabe MAE/R² lub "
    "zwiększ horyzont/okno, jeśli poprzednie runy były zbyt krótkie.\n"
    "Odpowiadaj po polsku, zwięźle, w markdown."
)


CLIENT_RECO_SYSTEM_PROMPT = (
    "Jesteś inżynierem ML wybierającym **jeden gotowy model** na widok klienta AirSense.\n\n"
    + project_context_brief()
    + "\n\n"
    "Porównuj metryki testowe (MAE, R², horyzont, okno). Preferuj horyzont "
    f"≥ {config.MIN_FORECAST_DAYS} dni, idealnie {config.DEFAULT_FORECAST_DAYS} dni, "
    "przy porównywalnej jakości. Odpowiadaj po polsku, w markdown."
)

RECO_SYSTEM_PROMPT = TRAIN_RECO_SYSTEM_PROMPT  # kompatybilność wsteczna


def _chat(client, system, user, model=None):
    model_name = resolve_openai_model(model)
    kwargs = _api_kwargs(
        model_name,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return client.chat.completions.create(**kwargs).choices[0].message.content


def _clamp_int(value, lo, hi):
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, v))


def strip_training_json_block(markdown_text):
    """Usuwa blok JSON parametrów sliderów z tekstu LLM (wartości są już w UI)."""
    import re

    if not markdown_text:
        return markdown_text
    out = re.sub(
        r"```json\s*\{.*?\}\s*```",
        "",
        markdown_text,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out = re.sub(
        r'^\s*\{\s*"time_steps"[^\}]+\}\s*$',
        "",
        out,
        flags=re.MULTILINE | re.DOTALL,
    )
    return out.strip()


def strip_client_recommendation_text(markdown_text):
    """Usuwa blok JSON z recommended_model z odpowiedzi LLM (nazwa jest w rejestrze / tagu)."""
    import re

    text = strip_training_json_block(markdown_text)
    if not text:
        return text
    return re.sub(
        r'^\s*\{\s*"recommended_model"[^\}]+\}\s*$',
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    ).strip()


def _clamp_float(value, lo, hi):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return max(float(lo), min(float(hi), v))


def parse_training_params(markdown_text, constraints=None):
    """Wyciąga parametry sliderów treningu z odpowiedzi LLM (JSON lub tekst)."""
    import json as _json
    import re

    constr = constraints or {}
    bounds = {
        "time_steps": (
            constr.get("min_window", config.MIN_WINDOW_HOURS),
            constr.get("max_window", config.MAX_WINDOW_HOURS),
        ),
        "forecast_horizon": (
            constr.get("min_forecast_hours", config.MIN_FORECAST_HOURS),
            constr.get("max_forecast_hours", config.MAX_FORECAST_HOURS),
        ),
        "lstm_units": (
            constr.get("min_lstm_units", config.MIN_LSTM_UNITS),
            constr.get("max_lstm_units", config.MAX_LSTM_UNITS),
        ),
        "lstm_layers": (
            constr.get("min_lstm_layers", config.MIN_LSTM_LAYERS),
            constr.get("max_lstm_layers", config.MAX_LSTM_LAYERS),
        ),
        "epochs": (
            constr.get("min_epochs", config.MIN_EPOCHS),
            constr.get("max_epochs", config.MAX_EPOCHS),
        ),
    }
    float_bounds = {
        "dropout": (
            constr.get("min_dropout", config.MIN_DROPOUT),
            constr.get("max_dropout", config.MAX_DROPOUT),
        ),
    }
    found = {}

    if markdown_text:
        m = re.search(r"```json\s*(\{.*?\})\s*```", markdown_text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                blob = _json.loads(m.group(1))
                for key in bounds:
                    if key in blob:
                        found[key] = _clamp_int(blob[key], *bounds[key])
                for key in float_bounds:
                    if key in blob:
                        found[key] = _clamp_float(blob[key], *float_bounds[key])
                if "horizon_weighted_loss" in blob:
                    found["horizon_weighted_loss"] = bool(blob["horizon_weighted_loss"])
            except (_json.JSONDecodeError, TypeError):
                pass

    patterns = {
        "time_steps": [
            r"okno[^0-9]{0,40}?(\d{1,3})\s*h",
            r"time[_\s-]?steps?[:\s]+(\d{1,3})",
        ],
        "forecast_horizon": [
            r"horyzont[^0-9]{0,40}?(\d{1,3})\s*h",
            r"forecast[_\s-]?horizon[:\s]+(\d{1,3})",
        ],
        "lstm_units": [
            r"neuron[^0-9]{0,30}?(\d{1,3})",
            r"lstm[_\s-]?units?[:\s]+(\d{1,3})",
        ],
        "lstm_layers": [
            r"warstw[^0-9]{0,20}?(\d)",
            r"lstm[_\s-]?layers?[:\s]+(\d)",
        ],
        "epochs": [
            r"epok[^0-9]{0,20}?(\d{1,3})",
            r"epochs?[:\s]+(\d{1,3})",
        ],
    }
    text = (markdown_text or "").lower()
    for key, pats in patterns.items():
        if key in found and found[key] is not None:
            continue
        for pat in pats:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                found[key] = _clamp_int(m.group(1), *bounds[key])
                break

    return {k: v for k, v in found.items() if v is not None}


def build_training_recommendation_user(models_table, data_info, constraints=None):
    """Treść user promptu dla rekomendacji ustawień treningu."""
    import json as _json

    models_str = _json.dumps(models_table or [], ensure_ascii=False, indent=2, default=str)
    data_str = "\n".join(f"- {k}: {v}" for k, v in (data_info or {}).items())
    constr = constraints or {}
    min_w = int(constr.get("min_window", config.MIN_WINDOW_HOURS))
    max_w = int(constr.get("max_window", config.MAX_WINDOW_HOURS))
    min_fh = int(constr.get("min_forecast_hours", config.MIN_FORECAST_HOURS))
    max_fh = int(constr.get("max_forecast_hours", config.MAX_FORECAST_HOURS))
    seed_days = int(constr.get("seed_days", config.SEED_PAST_DAYS))
    def_w = int(constr.get("default_window_hours", config.DEFAULT_WINDOW_HOURS))
    def_fh = int(constr.get("default_forecast_hours", config.DEFAULT_FORECAST_HOURS))
    return (
        "### Kontekst projektu (skrót)\n"
        + project_context_brief()
        + "\n\n"
        "### Istniejące modele tej stacji (podgląd — NIE wybieraj najlepszego dla klienta)\n"
        f"```json\n{models_str}\n```\n\n"
        f"### Dane i limity dla tej stacji\n{data_str}\n\n"
        f"### Dozwolone zakresy sliderów (1 h)\n"
        f"- archiwum treningowe: typowo ~{seed_days} dni (~{seed_days * 24} h)\n"
        f"- **time_steps** (okno wstecz): {min_w}–{max_w} h "
        f"(≈ {min_w // 24}–{max_w // 24} dni); **domyślnie {def_w} h** "
        f"(≈ {def_w // 24} dni)\n"
        f"- **forecast_horizon** (prognoza do przodu): {min_fh}–{max_fh} h "
        f"({config.MIN_FORECAST_DAYS}–{config.MAX_FORECAST_DAYS} dni); **docelowo {def_fh} h** "
        f"({config.DEFAULT_FORECAST_DAYS} dni)\n"
        f"- **lstm_units**: {constr.get('min_lstm_units', config.MIN_LSTM_UNITS)}–"
        f"{constr.get('max_lstm_units', config.MAX_LSTM_UNITS)}\n"
        f"- **lstm_layers**: {constr.get('min_lstm_layers', config.MIN_LSTM_LAYERS)}–"
        f"{constr.get('max_lstm_layers', config.MAX_LSTM_LAYERS)}\n"
        f"- **dropout**: {constr.get('min_dropout', config.MIN_DROPOUT)}–"
        f"{constr.get('max_dropout', config.MAX_DROPOUT)}\n"
        f"- **epochs**: {constr.get('min_epochs', config.MIN_EPOCHS)}–"
        f"{constr.get('max_epochs', config.MAX_EPOCHS)}\n"
        f"- **horizon_weighted_loss**: true/false (checkbox w UI)\n\n"
        "**Uwaga:** time_steps i forecast_horizon to różne parametry. "
        "Jeśli poprzednie modele miały krótki horyzont (np. 24 h), rozważ run z 7 dniami.\n\n"
        "### Twoja odpowiedź\n"
        "Tylko ustawienia **nowego** treningu (markdown, po polsku):\n"
        "1. Krótkie uzasadnienie w kontekście istniejących modeli i danych stacji.\n"
        "2. Konkretne liczby: okno h, horyzont h, neurony, epoki, warstwy, dropout, loss ważony.\n"
        "3. Na końcu **obowiązkowy** JSON (liczby mieszczące się w zakresach powyżej):\n"
        "```json\n"
        f'{{"time_steps": {def_w}, "forecast_horizon": {def_fh}, '
        f'"lstm_units": {config.DEFAULT_LSTM_UNITS}, "lstm_layers": {config.DEFAULT_LSTM_LAYERS}, '
        f'"dropout": {config.DEFAULT_DROPOUT}, "epochs": {config.DEFAULT_EPOCHS}, '
        f'"horizon_weighted_loss": {str(config.DEFAULT_HORIZON_WEIGHTED_LOSS).lower()}}}\n'
        "```"
    )


def parse_recommended_model(markdown_text, valid_names=None):
    """Wyciąga nazwę modelu z odpowiedzi LLM (JSON lub wzorzec aq_*)."""
    import json as _json
    import re

    valid = set(valid_names or [])
    if markdown_text:
        m = re.search(r"```json\s*(\{.*?\})\s*```", markdown_text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                blob = _json.loads(m.group(1))
                for key in ("recommended_model", "model", "nazwa_modelu"):
                    if key in blob and blob[key]:
                        name = str(blob[key]).strip()
                        if not valid or name in valid:
                            return name
            except (_json.JSONDecodeError, TypeError):
                pass
        for name in re.findall(r"\b(aq_[a-z0-9_]+)\b", markdown_text, re.IGNORECASE):
            if not valid or name in valid:
                return name
    return None


def recommend_training_settings(models_table, data_info, constraints=None,
                              api_key=None, model=None):
    """LLM proponuje wyłącznie ustawienia kolejnego treningu. Zwraca (markdown, params dict)."""
    key = resolve_api_key(api_key)
    if not key:
        raise ValueError("Brak klucza OpenAI. Podaj klucz w polu lub ustaw zmienną OPENAI_API_KEY.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Pakiet 'openai' nie jest zainstalowany.") from e

    import json as _json

    constr = constraints or {}
    client = OpenAI(api_key=key)
    user = build_training_recommendation_user(models_table, data_info, constr)
    text = _chat(client, TRAIN_RECO_SYSTEM_PROMPT, user, model)
    return text, parse_training_params(text, constr)


def recommend_client_model(models_table, station_info=None,
                           api_key=None, model=None):
    """LLM wybiera model na widok klienta. Zwraca (markdown, nazwa_modelu lub None)."""
    key = resolve_api_key(api_key)
    if not key:
        raise ValueError("Brak klucza OpenAI. Podaj klucz w polu lub ustaw zmienną OPENAI_API_KEY.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Pakiet 'openai' nie jest zainstalowany.") from e

    import json as _json
    models_str = _json.dumps(models_table or [], ensure_ascii=False, indent=2, default=str)
    info_str = "\n".join(f"- {k}: {v}" for k, v in (station_info or {}).items())
    valid_names = [r.get("model") for r in (models_table or []) if r.get("model")]

    user = (
        "Wybierz JEDEN model do pokazywania prognozy na widoku klienta.\n"
        "Metryki na zbiorze testowym (ASQI 0–100; MAE/RMSE niżej = lepiej, R² bliżej 1 = lepiej). "
        f"Preferuj modele z horyzontem ≥ {config.MIN_FORECAST_DAYS} dni, idealnie "
        f"{config.DEFAULT_FORECAST_DAYS} dni, jeśli jakość metryk jest porównywalna.\n"
        f"```json\n{models_str}\n```\n\n"
        f"Stacja:\n{info_str}\n\n"
        "Odpowiedz po polsku w markdown:\n"
        "### Rekomendowany model dla klienta\n"
        "Podaj dokładną nazwę z tabeli i uzasadnienie (2–4 zdania): metryki, horyzont, okno, stabilność.\n\n"
        "Na końcu JSON:\n"
        "```json\n"
        '{"recommended_model": "dokładna_nazwa_z_tabeli"}\n'
        "```"
    )

    client = OpenAI(api_key=key)
    text = _chat(client, CLIENT_RECO_SYSTEM_PROMPT, user, model)
    chosen = parse_recommended_model(text, valid_names)
    return text, chosen
