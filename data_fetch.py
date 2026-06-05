# -*- coding: utf-8 -*-
"""Warstwa danych AirSense (bez zależności od Streamlit) — pobieranie i czyszczenie.

ROZDZIELCZOŚĆ GODZINOWA: model uczy się na danych godzinowych. Pogoda (Open-Meteo)
daje ~30 dni wstecz. Dane GIOŚ o zanieczyszczeniach z bieżącego endpointu są dostępne
tylko za ostatnie ~3 dni — dlatego akumulujemy je w lokalnym magazynie JSON
(data_store.py), a przy pierwszym uruchomieniu stacji seedujemy 30 dni z endpointu
ARCHIWALNEGO GIOŚ. Kolejne uruchomienia dociągają tylko nowe punkty.

Przepływ:
    update_and_load(station)  ->  load_store  ->  fetch tylko nowe (weather + GIOŚ)
                              ->  merge do magazynu  ->  save_store
                              ->  assemble_clean_frame(z PEŁNEJ zakumulowanej historii)

Surowe, realne obserwacje trzymamy w JSON. Interpolacja i usuwanie outlierów są liczone
TYLKO w pamięci (assemble_clean_frame), żeby nie utrwalać wartości zmyślonych.
"""
import time

import numpy as np
import pandas as pd
import requests

import data_store
import air_quality
import config
import feature_engineering

GIOS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7',
    'Connection': 'keep-alive',
    'Referer': 'https://powietrze.gios.gov.pl/'
}

WEATHER_COLS = ("Temperatura", "Wilgotnosc", "Wiatr", "Opady")


# ----------------------------------------------------------------------------
# Pomocnicze: czyszczenie i budowa sekwencji
# ----------------------------------------------------------------------------
def remove_outliers(df):
    """Ograniczanie wartości odstających metodą percentyli (1./99.)."""
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            lower_limit = df_cleaned[col].quantile(0.01)
            upper_limit = df_cleaned[col].quantile(0.99)
            df_cleaned[col] = np.clip(df_cleaned[col], lower_limit, upper_limit)
    return df_cleaned


def prepare_sequences(data, time_steps, forecast_horizon):
    """Buduje nakładające się sekwencje wejście->wyjście (wielowymiarowe wyjście)."""
    X, y = [], []
    for i in range(len(data) - time_steps - forecast_horizon + 1):
        X.append(data[i:(i + time_steps), :])
        y.append(data[(i + time_steps):(i + time_steps + forecast_horizon), :])
    return np.array(X), np.array(y)


def prepare_xy(feature_array, target_array, time_steps, forecast_horizon):
    """Sekwencje: wejście WIELOWYMIAROWE (cechy) -> wyjście JEDNOWYMIAROWE (target AQI).

    X: (n, time_steps, n_features), y: (n, forecast_horizon).
    """
    feature_array = np.asarray(feature_array, dtype=float)
    target_array = np.asarray(target_array, dtype=float).reshape(-1)
    X, y = [], []
    for i in range(len(feature_array) - time_steps - forecast_horizon + 1):
        X.append(feature_array[i:(i + time_steps), :])
        y.append(target_array[(i + time_steps):(i + time_steps + forecast_horizon)])
    return np.array(X), np.array(y)


def _first_list(obj):
    """Zwraca listę z odpowiedzi API. GIOŚ v1 pakuje dane w słownik z kluczem 'Lista ...'."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list) and 'lista' in str(k).lower():
                return v
        for v in obj.values():
            if isinstance(v, list):
                return v
    return []


def _pick(cols, *frags):
    """Nazwa kolumny/klucza zawierająca wszystkie fragmenty (niewrażliwe na wielkość liter)."""
    for c in cols:
        cl = str(c).lower()
        if all(f in cl for f in frags):
            return c
    return None


# ----------------------------------------------------------------------------
# Pobieranie SUROWE — pogoda
# ----------------------------------------------------------------------------
def fetch_weather_raw(lat, lon, past_days=30):
    """Surowa pogoda godzinowa (Open-Meteo): {nazwa: pd.Series}. Pusty dict przy błędzie."""
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&past_days={past_days}&forecast_days=1&timezone=Europe/Warsaw"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
    )
    try:
        res = requests.get(weather_url, timeout=15)
        res.raise_for_status()
        j = res.json()
        idx = pd.to_datetime(j['hourly']['time'])
        mapping = {
            'Temperatura': j['hourly']['temperature_2m'],
            'Wilgotnosc': j['hourly']['relative_humidity_2m'],
            'Wiatr': j['hourly']['wind_speed_10m'],
            'Opady': j['hourly']['precipitation'],
        }
        out = {}
        for name, vals in mapping.items():
            s = pd.Series(pd.to_numeric(vals, errors='coerce'), index=idx).dropna().sort_index()
            if not s.empty:
                out[name] = s
        return out
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Pobieranie SUROWE — zanieczyszczenia (GIOŚ)
# ----------------------------------------------------------------------------
def list_gios_sensors(station_id):
    """Zwraca listę (param_code, sensor_id) dla stacji. Pusta lista przy błędzie."""
    url = f"https://api.gios.gov.pl/pjp-api/v1/rest/station/sensors/{station_id}"
    try:
        res = requests.get(url, headers=GIOS_HEADERS, timeout=10)
        if res.status_code != 200:
            return []
        sensors = _first_list(res.json())
    except Exception:
        return []

    out = []
    for sensor in sensors:
        if not isinstance(sensor, dict):
            continue
        code_key = _pick(sensor.keys(), 'kod')
        if code_key:
            param_code = str(sensor[code_key]).replace(".", "")
            id_key = (_pick(sensor.keys(), 'stanowiska')
                      or _pick(sensor.keys(), 'identyfikator') or 'id')
            sensor_id = sensor.get(id_key)
        elif isinstance(sensor.get('param'), dict) and 'paramCode' in sensor['param']:
            param_code = str(sensor['param']['paramCode']).replace(".", "")
            sensor_id = sensor.get('id')
        else:
            continue
        if sensor_id is not None:
            out.append((param_code, sensor_id))
    return out


def _parse_gios_series(rows):
    """Lista rekordów GIOŚ -> pd.Series(indeks=datetime). None gdy brak użytecznych danych."""
    if not rows:
        return None
    df_s = pd.DataFrame(rows)
    date_key = _pick(df_s.columns, 'data') or _pick(df_s.columns, 'date')
    val_key = _pick(df_s.columns, 'warto') or _pick(df_s.columns, 'value')
    if not (date_key and val_key):
        return None
    df_s[date_key] = pd.to_datetime(df_s[date_key], errors='coerce')
    df_s[val_key] = pd.to_numeric(df_s[val_key], errors='coerce')
    df_s = df_s.dropna(subset=[date_key])
    s = df_s.set_index(date_key)[val_key].sort_index()
    s = s.groupby(s.index).mean()  # duplikaty godzin -> średnia
    s = s.dropna()
    return s if not s.empty else None


def fetch_pollutants_raw(station_id, mode='recent'):
    """Surowe zanieczyszczenia GIOŚ: {param_code: pd.Series}.

    mode='recent'   -> getData (ostatnie ~3 dni) — do dociągania przyrostowego.
    mode='archival' -> archivalData (dayNumber=30) — seed przy pierwszym uruchomieniu.
    """
    sensors = list_gios_sensors(station_id)
    out = {}
    for param_code, sensor_id in sensors:
        try:
            if mode == 'archival':
                url = (f"https://api.gios.gov.pl/pjp-api/v1/rest/archivalData/"
                       f"getDataBySensor/{sensor_id}?dayNumber=30&size=2000")
            else:
                url = (f"https://api.gios.gov.pl/pjp-api/v1/rest/data/"
                       f"getData/{sensor_id}?size=500")
            res = requests.get(url, headers=GIOS_HEADERS, timeout=30)
            if res.status_code == 200:
                s = _parse_gios_series(_first_list(res.json()))
                if s is not None:
                    # Ta sama stacja może mieć WIELE sensorów tego samego kodu (np. dwa PM2.5).
                    # Wcześniej ostatni nadpisywał poprzedni — stąd „dziura” (zostawała seria
                    # z gorszego sensora). Łączymy wszystkie pomiary po czasie.
                    if param_code in out:
                        out[param_code] = data_store.merge_series_prefer_dense(
                            out[param_code], s
                        )
                    else:
                        out[param_code] = s
        except Exception:
            pass
        time.sleep(0.2)
    return out


# ----------------------------------------------------------------------------
# Budowa czystej ramki cech (czyszczenie + imputacja, TYLKO w pamięci)
# ----------------------------------------------------------------------------
def assemble_clean_frame(weather_series, pollutant_series):
    """Z zakumulowanych SUROWYCH serii buduje oczyszczoną i uzupełnioną ramkę CECH.

    Oś czasu bazuje na zanieczyszczeniach (to jest aplikacja jakości powietrza); pogoda
    jest dorównywana jako cecha pomocnicza. Kolejność kolumn: najpierw zanieczyszczenia,
    potem pogoda. Wartości są interpolowane (imputacja) i przycinane z outlierów.
    """
    # Oś czasu = suma znaczników czasu zanieczyszczeń. Gdy ich brak — pusty wynik.
    base_index = None
    for s in pollutant_series.values():
        base_index = s.index if base_index is None else base_index.union(s.index)
    if base_index is None:
        # Brak zanieczyszczeń: bez nich nie ma czego mierzyć (zwracamy pustą ramkę).
        return pd.DataFrame()
    # Uwzględnij też znaczniki pogody, by mieć pełną godzinową oś.
    for s in weather_series.values():
        base_index = base_index.union(s.index)
    base_index = base_index.sort_values()

    df_final = pd.DataFrame(index=base_index)
    # Zanieczyszczenia (najpierw — to one są przedmiotem prognozy)
    for p_name, s in pollutant_series.items():
        if isinstance(s, pd.Series) and not s.dropna().empty:
            df_final[p_name] = s.reindex(base_index)
    # Pogoda pomocnicza
    for c in WEATHER_COLS:
        if c in weather_series:
            df_final[c] = weather_series[c].reindex(base_index)
    df_final.index.name = 'Data'

    df_final = df_final.interpolate(method='linear', limit_direction='both')
    df_final = df_final.dropna(axis=1, how='all')
    df_final = remove_outliers(df_final)
    df_final = df_final[df_final.index <= pd.Timestamp.now()]
    return df_final


def build_processed_frame(weather_series, pollutant_series):
    """Oczyszczona ramka cech + policzone kolumny GiosAQI / EuropeanIndex.

    Zwraca (processed_df, feature_columns). feature_columns to kolumny cech modelu
    (zanieczyszczenia + pogoda), bez kolumn wskaźnika.
    """
    features = assemble_clean_frame(weather_series, pollutant_series)
    if features.empty or not air_quality.has_index_pollutants(features.columns):
        return pd.DataFrame(), []
    features = feature_engineering.add_calendar_features(features)
    features = feature_engineering.finalize_features_for_model(features)
    if features.empty:
        return pd.DataFrame(), []
    feature_columns = list(features.columns)
    processed = air_quality.compute_indices(features)
    return processed, feature_columns


# ----------------------------------------------------------------------------
# Orkiestracja: magazyn + przyrostowe dociąganie
# ----------------------------------------------------------------------------
def reset_and_load(station_id, lat, lon, past_days=config.SEED_PAST_DAYS):
    """Czyści lokalną bazę stacji i pobiera dane od zera (archiwum 30 dni GIOŚ)."""
    data_store.clear_store(station_id, lat, lon)
    return update_and_load(station_id, lat, lon, past_days=past_days)


def update_and_load(station_id, lat, lon, past_days=config.SEED_PAST_DAYS):
    """Aktualizuje bazę JSON i zwraca PROCESSED DataFrame (cechy + GiosAQI).

    Pierwsze uruchomienie stacji: seed zanieczyszczeń z archiwum GIOŚ (30 dni).
    Kolejne: dociąganie bieżące (getData ~3 dni) i akumulacja w warstwie raw.
    Warstwa processed (czyszczenie + imputacja + normalizacja + wskaźniki) jest
    przeliczana w całości i zapisywana przy każdej aktualizacji.
    """
    store = data_store.load_store(station_id, lat, lon)

    # Najpierw GIOŚ (oś czasu aplikacji), potem pogoda Open-Meteo na ten sam (lub szerszy) zakres.
    mode = 'archival' if data_store.is_pollutants_empty(store) else 'recent'
    new_pollutants = fetch_pollutants_raw(station_id, mode=mode)
    data_store.update_raw(store, {}, new_pollutants)

    weather_days = data_store.weather_past_days_for_store(store, default_days=past_days)
    new_weather = fetch_weather_raw(lat, lon, past_days=weather_days)
    data_store.update_raw(store, new_weather, {})

    # Pełne przeliczenie warstwy processed z zakumulowanej historii raw.
    weather_series, pollutant_series = data_store.store_to_series(store)
    processed, feature_columns = build_processed_frame(weather_series, pollutant_series)
    if not processed.empty:
        data_store.save_processed(store, processed, feature_columns)

    try:
        data_store.save_store(store)
    except OSError:
        pass  # brak możliwości zapisu nie może blokować działania aplikacji

    return air_quality.rename_legacy_index_columns(processed)


def fetch_real_climate_data(station_id, lat, lon, past_days=30):
    """Zachowane dla wstecznej zgodności — deleguje do update_and_load."""
    return update_and_load(station_id, lat, lon, past_days=past_days)


# ----------------------------------------------------------------------------
# PROGNOZA 7-DNIOWA (widok klienta): pogoda + jakość powietrza z Open-Meteo
# ----------------------------------------------------------------------------
# Open-Meteo Air Quality API zwraca prognozę stężeń + GOTOWE oficjalne indeksy
# (european_aqi wg CAMS/EEA, us_aqi wg EPA) — używamy ich do porównania z naszymi.
_AQ_FORECAST_POLLUTANTS = {
    "pm2_5": "PM25", "pm10": "PM10", "nitrogen_dioxide": "NO2",
    "sulphur_dioxide": "SO2", "ozone": "O3", "carbon_monoxide": "CO",
}
OFFICIAL_EU_COL = "OfficialEuropeanAQI"
OFFICIAL_US_COL = "OfficialUS_AQI"


def fetch_weather_forecast(lat, lon, days=7, past_days=None):
    """Prognoza pogody godzinowa (Open-Meteo): Temperatura/Wilgotnosc/Wiatr. Pusto przy błędzie."""
    past_days = days if past_days is None else past_days
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&forecast_days={days}&past_days={past_days}&timezone=Europe/Warsaw"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
    )
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        j = res.json()
        return pd.DataFrame({
            "Temperatura": j["hourly"]["temperature_2m"],
            "Wilgotnosc": j["hourly"]["relative_humidity_2m"],
            "Wiatr": j["hourly"]["wind_speed_10m"],
            "Opady": j["hourly"]["precipitation"],
        }, index=pd.to_datetime(j["hourly"]["time"])).sort_index()
    except Exception:
        return pd.DataFrame()


def fetch_air_quality_forecast(lat, lon, days=7, past_days=None):
    """Prognoza zanieczyszczeń (Open-Meteo AQ) + oficjalne indeksy european_aqi / us_aqi."""
    past_days = days if past_days is None else past_days
    fields = list(_AQ_FORECAST_POLLUTANTS.keys()) + ["european_aqi", "us_aqi"]
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
        f"&forecast_days={days}&past_days={past_days}&timezone=Europe/Warsaw&hourly={','.join(fields)}"
    )
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        h = res.json()["hourly"]
        df = pd.DataFrame(index=pd.to_datetime(h["time"]))
        for api_name, our_code in _AQ_FORECAST_POLLUTANTS.items():
            if api_name in h:
                df[our_code] = pd.to_numeric(h[api_name], errors="coerce")
        if "european_aqi" in h:
            df[OFFICIAL_EU_COL] = pd.to_numeric(h["european_aqi"], errors="coerce")
        if "us_aqi" in h:
            df[OFFICIAL_US_COL] = pd.to_numeric(h["us_aqi"], errors="coerce")
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def fetch_forecast_bundle(lat, lon, days=config.CLIENT_FORECAST_DAYS, past_days=None):
    """Łączna prognoza ±days: pogoda + zanieczyszczenia + nasze i oficjalne wskaźniki.

    Domyślnie past_days=days — symetryczne okno wstecz/przód względem teraz.
    Zwraca DataFrame z kolumnami: pogoda, stężenia zanieczyszczeń,
    OpenMeteoCompositeIndex (wzór jak GiosAQI ze stężeń prognozy), EuropeanIndex (nasz),
    OfficialEuropeanAQI, OfficialUS_AQI. Pusty przy braku danych AQ.
    """
    past_days = days if past_days is None else past_days
    aq = fetch_air_quality_forecast(lat, lon, days=days, past_days=past_days)
    if aq.empty:
        return pd.DataFrame()
    weather = fetch_weather_forecast(lat, lon, days=days, past_days=past_days)
    df = aq.join(weather, how="left") if not weather.empty else aq

    # Ten sam wzór co GiosAQI, ale ze stężeń prognozowanych (Open-Meteo) — osobna kolumna.
    indexed = air_quality.compute_indices(df)
    df[air_quality.OM_COMPOSITE_COL] = indexed[air_quality.GIOS_AQI_COL]
    for col in (air_quality.EU_COL, "EuropeanClass"):
        df[col] = indexed[col]
    df.index.name = "Data"
    return df
