# -*- coding: utf-8 -*-
"""Lokalna baza danych AirSense (JSON) — dwuwarstwowa.

WARSTWA `raw`       — surowe, REALNE odczyty (zanieczyszczenia + pogoda pomocnicza).
                      Służą do prowenancji i PRZYROSTOWEGO dociągania (min. 30 dni przy
                      pierwszym uruchomieniu, potem tylko nowe godziny).
WARSTWA `processed` — przeliczana W CAŁOŚCI przy każdej aktualizacji: dane oczyszczone,
                      uzupełnione (imputacja) i z policzonym wskaźnikiem GiosAQI (GIOŚ AQI)
                      oraz EuropeanIndex. Zapisujemy też parametry normalizacji (skaler
                      fit na całym snapshocie) — normalizacja = zastosuj skaler. Pełne
                      przeliczanie gwarantuje, że normalizacja nie "dryfuje" w czasie.

Jeden plik na stację: data_store/station_<id>.json. Moduł niezależny od Streamlita.
"""
import json
import os

import numpy as np
import pandas as pd

import air_quality

SCHEMA_VERSION = 2
STORE_DIR = "data_store"
TZ_NAME = "Europe/Warsaw"


def _store_path(station_id):
    return os.path.join(STORE_DIR, f"station_{station_id}.json")


def _empty_store(station_id, lat=None, lon=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "station_id": station_id,
        "lat": lat,
        "lon": lon,
        "last_updated": None,
        "timezone": TZ_NAME,
        "raw": {"weather": {}, "pollutants": {}},
        "processed": None,
    }


def _migrate_v1(store):
    """Stary format (v1: weather/pollutants na najwyższym poziomie) -> v2 (raw/processed)."""
    if "raw" not in store:
        store["raw"] = {
            "weather": store.pop("weather", {}) or {},
            "pollutants": store.pop("pollutants", {}) or {},
        }
    store.setdefault("processed", None)
    store["schema_version"] = SCHEMA_VERSION
    return store


def load_store(station_id, lat=None, lon=None):
    """Wczytuje bazę stacji. Przy braku/uszkodzeniu pliku zwraca pusty szkielet."""
    path = _store_path(station_id)
    if not os.path.exists(path):
        return _empty_store(station_id, lat, lon)
    try:
        with open(path, "r", encoding="utf-8") as f:
            store = json.load(f)
        if not isinstance(store, dict):
            return _empty_store(station_id, lat, lon)
        store = _migrate_v1(store)
        store.setdefault("raw", {"weather": {}, "pollutants": {}})
        store["raw"].setdefault("weather", {})
        store["raw"].setdefault("pollutants", {})
        store.setdefault("processed", None)
        if lat is not None and store.get("lat") is None:
            store["lat"] = lat
        if lon is not None and store.get("lon") is None:
            store["lon"] = lon
        return store
    except (json.JSONDecodeError, OSError, ValueError, KeyError):
        return _empty_store(station_id, lat, lon)


def save_store(store):
    """Atomiczny zapis bazy (zapis do .tmp + os.replace)."""
    os.makedirs(STORE_DIR, exist_ok=True)
    path = _store_path(store["station_id"])
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# ----------------------------------------------------------------------------
# Warstwa RAW: serie surowych odczytów
# ----------------------------------------------------------------------------
def _section_to_series(section):
    out = {}
    for name, mapping in (section or {}).items():
        if not mapping:
            continue
        idx = pd.to_datetime(list(mapping.keys()), errors="coerce")
        vals = pd.to_numeric(list(mapping.values()), errors="coerce")
        s = pd.Series(vals, index=idx).dropna()
        s = s[~s.index.isna()].sort_index()
        if not s.empty:
            out[name] = s
    return out


def store_to_series(store):
    """(weather_series, pollutant_series) jako dict[str, pd.Series] z warstwy raw."""
    raw = store.get("raw", {})
    return _section_to_series(raw.get("weather")), _section_to_series(raw.get("pollutants"))


def _series_to_section(series_dict):
    section = {}
    for name, s in series_dict.items():
        s = s.dropna().sort_index()
        if s.empty:
            continue
        section[name] = {ts.isoformat(): float(v) for ts, v in s.items()}
    return section


def merge_series(old, new):
    """Union dwóch serii po znaczniku czasu; przy konflikcie wygrywa NOWA wartość."""
    parts = [s for s in (old, new) if s is not None and not s.empty]
    if not parts:
        return pd.Series(dtype="float64")
    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def merge_series_prefer_dense(a, b):
    """Union serii z dwóch sensorów tego samego parametru.

    Przy nakładających się godzinach wygrywa seria z większą liczbą punktów
    (główny sensor). Krótszy / rzadszy sensor tylko uzupełnia brakujące godziny.
    """
    if a is None or (hasattr(a, "empty") and a.empty):
        return b.sort_index() if b is not None and not b.empty else pd.Series(dtype="float64")
    if b is None or (hasattr(b, "empty") and b.empty):
        return a.sort_index()
    primary, secondary = (a, b) if len(a) >= len(b) else (b, a)
    return primary.combine_first(secondary).sort_index()


def clear_store(station_id, lat=None, lon=None):
    """Usuwa plik bazy stacji (następne update_and_load zrobi seed archiwalny)."""
    path = _store_path(station_id)
    if os.path.exists(path):
        os.remove(path)
    return _empty_store(station_id, lat, lon)


def update_raw(store, new_weather, new_pollutants):
    """Wmerguj nowe surowe serie do warstwy raw (in-place + zwrot)."""
    cur_weather, cur_pollutants = store_to_series(store)
    for name, s in (new_weather or {}).items():
        cur_weather[name] = merge_series(cur_weather.get(name), s)
    for name, s in (new_pollutants or {}).items():
        cur_pollutants[name] = merge_series(cur_pollutants.get(name), s)
    store["raw"]["weather"] = _series_to_section(cur_weather)
    store["raw"]["pollutants"] = _series_to_section(cur_pollutants)
    store["last_updated"] = pd.Timestamp.now().isoformat(timespec="seconds")
    return store


def last_timestamp(store, source):
    """Najświeższy znacznik czasu w raw 'weather'/'pollutants' lub None."""
    section = store.get("raw", {}).get(source) or {}
    latest = None
    for mapping in section.values():
        for ts in mapping.keys():
            t = pd.to_datetime(ts, errors="coerce")
            if pd.isna(t):
                continue
            if latest is None or t > latest:
                latest = t
    return latest


def is_pollutants_empty(store):
    """Czy baza nie ma jeszcze żadnych realnych danych o zanieczyszczeniach."""
    return not any((store.get("raw", {}).get("pollutants") or {}).values())


def get_coverage(store):
    """Podsumowanie pokrycia warstwy raw: liczba realnych punktów i zakres dat."""
    weather, pollutants = store_to_series(store)
    rows = []
    for kind, series_dict in (("zanieczyszczenia", pollutants), ("pogoda (pomocnicza)", weather)):
        for name, s in series_dict.items():
            rows.append({
                "parametr": name,
                "typ": kind,
                "punkty": int(len(s)),
                "od": s.index.min(),
                "do": s.index.max(),
            })
    return rows


# ----------------------------------------------------------------------------
# Warstwa PROCESSED: oczyszczone + uzupełnione + indeksy + parametry normalizacji
# ----------------------------------------------------------------------------
def save_processed(store, processed_df, feature_columns):
    """Zapisuje snapshot przetworzonych danych + parametry normalizacji (skaler na całości).

    processed_df : DataFrame z kolumnami cech (zanieczyszczenia + pogoda) ORAZ
                   policzonymi GiosAQI / EuropeanIndex / EuropeanClass.
    feature_columns : kolumny będące cechami modelu (bez kolumn wskaźnika).
    """
    df = processed_df.copy()
    cols = list(df.columns)

    # Parametry normalizacji liczone na CAŁYM snapshocie (poziom bazy). Model trenuje
    # własny skaler na części treningowej (bez wycieku) — to są dwie różne rzeczy.
    feats = [c for c in feature_columns if c in df.columns]
    fmin = df[feats].min().tolist() if feats else []
    fmax = df[feats].max().tolist() if feats else []

    store["processed"] = {
        "columns": cols,
        "feature_columns": feats,
        "index": [ts.isoformat() for ts in pd.to_datetime(df.index)],
        "values": np.asarray(df.values, dtype=float).round(4).tolist(),
        "scaler": {"columns": feats, "min": fmin, "max": fmax},
        "n_rows": int(df.shape[0]),
    }
    return store


def weather_past_days_for_store(store, default_days=30, max_days=92):
    """Ile dni wstecz pobrać z Open-Meteo, by pokryć zakres godzin zanieczyszczeń w raw."""
    _, pollutants = store_to_series(store)
    if not pollutants:
        return default_days
    tmin = min(s.index.min() for s in pollutants.values() if not s.empty)
    if pd.isna(tmin):
        return default_days
    days = int((pd.Timestamp.now(tz=None) - pd.Timestamp(tmin)).total_seconds() // 86400) + 2
    return max(default_days, min(days, max_days))


def raw_to_dataframe(store):
    """Ramka SUROWYCH realnych odczytów z warstwy raw (bez imputacji i wskaźników).

    Oś czasu = godziny z pomiarów GIOŚ. Pogoda (Open-Meteo) jest dopinana do najbliższej
    godziny (±30 min) — nadal bez interpolacji liniowej jak w processed.
    """
    weather, pollutants = store_to_series(store)
    base_index = None
    for s in pollutants.values():
        if s is not None and not s.empty:
            base_index = s.index if base_index is None else base_index.union(s.index)
    if base_index is None:
        for s in weather.values():
            base_index = s.index if base_index is None else base_index.union(s.index)
    if base_index is None:
        return pd.DataFrame()
    base_index = base_index.sort_values()
    df = pd.DataFrame(index=base_index)
    for name, s in pollutants.items():
        df[name] = s.reindex(base_index)
    tol = pd.Timedelta("30min")
    for name, s in weather.items():
        aligned = s.sort_index().reindex(base_index, method="nearest", tolerance=tol)
        df[name] = aligned
    df.index.name = "Data"
    return df


def load_processed(store):
    """Odtwarza przetworzony DataFrame (jednostki rzeczywiste) z warstwy processed."""
    proc = store.get("processed")
    if not proc or not proc.get("index"):
        return pd.DataFrame()
    idx = pd.to_datetime(proc["index"])
    df = pd.DataFrame(proc["values"], columns=proc["columns"], index=idx)
    df.index.name = "Data"
    return air_quality.rename_legacy_index_columns(df)
