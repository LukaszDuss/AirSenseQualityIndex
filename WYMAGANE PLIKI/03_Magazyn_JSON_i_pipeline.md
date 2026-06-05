# 3. Magazyn JSON i pipeline danych

## 3.1. Plik per stacja

Ścieżka: `data_store/station_<id>.json`

Moduł: `data_store.py`, `SCHEMA_VERSION = 2`.

## 3.2. Struktura dokumentu

```json
{
  "schema_version": 2,
  "station_id": 550,
  "lat": 50.31,
  "lon": 18.77,
  "last_updated": "2026-06-04T12:00:00",
  "timezone": "Europe/Warsaw",
  "raw": {
    "weather": { "Temperatura": { "2026-06-01T10:00:00": 18.2, ... } },
    "pollutants": { "PM25": { ... }, "NO2": { ... } }
  },
  "processed": {
    "columns": [...],
    "feature_columns": [...],
    "data": [[timestamp, val1, val2, ...], ...],
    "feat_min": [...],
    "feat_max": [...],
    "y_min": 12.5,
    "y_max": 78.3
  }
}
```

### Warstwa `raw`

- **Tylko realne odczyty** z API.
- Format: słownik `nazwa_serii → { ISO_timestamp → float }`.
- Służy do prowenancji i przyrostowego dociągania.
- Interpolacja **nie jest** zapisywana do raw.

### Warstwa `processed`

- Snapshot przeliczany **w całości** przy każdej aktualizacji.
- Zawiera: oczyszczone cechy + `GiosAQI`, `EuropeanIndex`, `EuropeanClass`.
- Zapisuje parametry skalera (min/max cech i targetu) — używane przy inferencji LSTM.
- Pełne przeliczenie zapobiega „dryfowi” normalizacji w czasie.

## 3.3. Migracja v1 → v2

`_migrate_v1()`: stare pliki miały `weather`/`pollutants` na top level → przeniesione do `raw`.

## 3.4. Zapis atomowy

`save_store()`:
1. Zapis do `station_<id>.json.tmp`
2. `os.replace(tmp, final)` — brak uszkodzonego pliku przy przerwaniu zapisu.

## 3.5. Przepływ `update_and_load` (krok po kroku)

```
load_store
    │
    ├─ is_pollutants_empty? ──TAK──► fetch archival (30 dni)
    │                         NIE──► fetch recent (~3 dni)
    ├─ update_raw(pollutants)
    ├─ weather_past_days_for_store()
    ├─ fetch_weather_raw(past_days)
    ├─ update_raw(weather)
    │
    ├─ store_to_series()
    ├─ build_processed_frame()
    │       ├─ assemble_clean_frame()
    │       ├─ add_calendar_features()
    │       ├─ finalize_features_for_model()
    │       └─ compute_indices()
    ├─ save_processed()
    └─ save_store()
```

## 3.6. `assemble_clean_frame` — zasady

1. **Oś czasu** = unia znaczników zanieczyszczeń (+ pogoda).
2. Brak zanieczyszczeń → pusty DataFrame (nie ma czego mierzyć).
3. Kolumny: najpierw pollutanty, potem `WEATHER_COLS`.
4. `interpolate(linear, limit_direction='both')`.
5. `remove_outliers` — clip do percentyli 1% i 99%.
6. `index <= now()` — bez przyszłości w historii.

## 3.7. `merge_series_prefer_dense`

Przy aktualizacji raw: nowe punkty są **łączone** ze starą serią. Gdy dwa sensory tego samego kodu — wybierana gęstsza seria lub merge po czasie z preferencją większej liczby nie-NaN.

## 3.8. Cache Streamlit

`ui_common.load_processed_data()` — `@st.cache_data(ttl=3600)` → wywołuje `update_and_load`.

`load_forecast()` — `ttl=1800` → `fetch_forecast_bundle`.

Invalidacja: przyciski „Odśwież dane” w adminie czyszczą cache.

## 3.9. Co NIE jest w magazynie

| Element | Gdzie |
|---|---|
| Modele LSTM | `models/*.keras` + `*_meta.npy` |
| Ustawienia app | `settings/app_settings.json` |
| Active/best model | `models/active_models.json`, `best_models.json` |
| Audyt GIOŚ cache | `settings/gios_audit_cache.json` |
