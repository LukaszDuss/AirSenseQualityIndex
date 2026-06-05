# 2. Źródła danych i API

## 2.1. Dwa niezależne źródła

| Źródło | Co daje | Rola w AirSense |
|---|---|---|
| **GIOŚ API v1** | Stężenia zanieczyszczeń ze stacji | Oś czasu aplikacji, target treningu (`GiosAQI`) |
| **Open-Meteo** | Pogoda + prognoza AQ | Cechy pomocnicze + porównanie w adminie |

## 2.2. GIOŚ — nagłówki i endpointy

Moduł: `data_fetch.py`, stała `GIOS_HEADERS`.

Serwer GIOŚ odrzuca część zapytań bez nagłówków przeglądarki. Wysyłamy m.in.:
- `User-Agent` (Chrome)
- `Referer: https://powietrze.gios.gov.pl/`
- `Accept: application/json`

### Lista sensorów stacji

```
GET https://api.gios.gov.pl/pjp-api/v1/rest/station/sensors/{station_id}
```

Funkcja: `list_gios_sensors(station_id)` → lista par `(param_code, sensor_id)`.

Parsowanie odpowiedzi:
- `_first_list()` — API pakuje listę w klucz typu „Lista …”.
- `_pick()` — elastyczne dopasowanie polskich nazw pól (`kod`, `wartość`, `data`).

Kody parametrów normalizowane: `PM2.5` → `PM25`.

### Dane bieżące (~3 dni)

```
GET .../rest/data/getData/{sensor_id}?size=500
```

`fetch_pollutants_raw(station_id, mode='recent')` — do **przyrostowego** dociągania.

### Dane archiwalne (30 dni)

```
GET .../rest/archivalData/getDataBySensor/{sensor_id}?dayNumber=30&size=2000
```

`mode='archival'` — **seed** przy pierwszym uruchomieniu stacji (gdy `raw.pollutants` puste).

### Wiele sensorów tego samego kodu

Ta sama stacja może mieć kilka sensorów PM2.5. Wcześniejszy kod nadpisywał serię — teraz `data_store.merge_series_prefer_dense()` łączy serie, preferując gęstsze pokrycie godzinowe.

Między sensorami: `time.sleep(0.2)` — łagodzenie rate limit.

## 2.3. Open-Meteo — pogoda historyczna

`fetch_weather_raw(lat, lon, past_days=30)`:

```
GET https://api.open-meteo.com/v1/forecast
  ?latitude=...&longitude=...
  &past_days=30&forecast_days=1
  &timezone=Europe/Warsaw
  &hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation
```

Mapowanie kolumn:

| API | Kolumna w projekcie | Jednostka |
|---|---|---|
| `temperature_2m` | `Temperatura` | °C |
| `relative_humidity_2m` | `Wilgotnosc` | % |
| `wind_speed_10m` | `Wiatr` | km/h |
| `precipitation` | `Opady` | mm |

`past_days` jest **adaptacyjny** — `data_store.weather_past_days_for_store()` dopasowuje zakres do zakresu zanieczyszczeń w magazynie.

## 2.4. Open-Meteo — prognoza (widok klienta + admin)

### Pogoda

`fetch_weather_forecast(lat, lon, days=7, past_days=7)` — symetryczne okno ±7 dni w adminie (wykres porównawczy).

### Jakość powietrza

`fetch_air_quality_forecast()` — endpoint:

```
https://air-quality-api.open-meteo.com/v1/air-quality
```

Pola: `pm2_5`, `pm10`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`, `carbon_monoxide`, plus gotowe `european_aqi`, `us_aqi`.

### Bundle prognozy

`fetch_forecast_bundle()` łączy AQ + pogodę, liczy:
- `OpenMeteoCompositeIndex` (nasz wzór ze stężeń),
- `EuropeanIndex`, `OfficialEuropeanAQI`, `OfficialUS_AQI`.

## 2.5. Strefa czasowa

Wszędzie: **`Europe/Warsaw`**. W UI wykresów: `ui_common.CHART_TZ` i `_normalize_chart_index()` — unikanie błędów tz-aware vs naive.

## 2.6. Ograniczenia źródeł (ważne dla projektu)

| Ograniczenie | Skutek | Obejście w AirSense |
|---|---|---|
| GIOŚ recent ~3 dni | Krótka historia z API | Akumulacja w JSON + seed archiwalny 30 dni |
| Braki sensorów | Luki godzinowe | Interpolacja + merge sensorów |
| Open-Meteo ≠ pomiar GIOŚ | Rozjazd indeksów | Osobne kolumny; OM tylko do porównania |
| Rate limit GIOŚ | Wolne pełne skanowanie | `gios_audit` z cache, sleep między sensorami |

## 2.7. Orkiestracja pobierania

Główna funkcja: `data_fetch.update_and_load(station_id, lat, lon)`:

1. `load_store`
2. GIOŚ: `archival` jeśli brak pollutantów, inaczej `recent`
3. `update_raw` (pollutants)
4. Pogoda: `fetch_weather_raw` z adaptacyjnym `past_days`
5. `update_raw` (weather)
6. `build_processed_frame` z **całej** historii raw
7. `save_processed` + `save_store`
8. Zwraca `processed` DataFrame

Reset: `reset_and_load()` → `clear_store` + od nowa.

Narzędzie CLI: `scripts/refetch_data.py` — przebudowa wszystkich stacji z katalogu.
