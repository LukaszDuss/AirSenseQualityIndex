# 11. Panel administracyjny (`/admin`)

Moduł: `views/admin_router.py`. Nawigacja: `?tab=<klucz>&station_id=<id>`.

## 11.1. Zakładki (`ADMIN_TABS`)

| Klucz | Etykieta | Funkcja |
|---|---|---|
| `dane_surowe` | Dane surowe | `_tab_data_raw` |
| `dane_przetworzone` | Dane przetworzone | `_tab_data_processed` |
| `trening` | Trening modeli | `_tab_train` |
| `modele` | Zarządzanie modelami | `_tab_models` |
| `aktualizacja` | Aktualizacja i ewaluacja | `_tab_update_eval` |
| `podglad` | Podgląd i porównanie | `_tab_preview_compare` |
| `ustawienia` | Ustawienia | `_tab_settings` |

Domyślna zakładka: `dane_przetworzone`. Aliasy stare URL: `dane` → przetworzone, `prognoza` → aktualizacja.

## 11.2. Dane surowe

- Przyciski: **Odśwież dane**, **Reset i pobierz od zera** (`reset_and_load`).
- Statystyki pokrycia: liczba punktów per seria w `raw`.
- Podgląd tabeli surowej (`data_store.raw_to_dataframe`).
- Eksport CSV.
- Informacja o trybie GIOŚ: archival vs recent.

## 11.3. Dane przetworzone

- Liczba wierszy/godzin, zakres dat.
- `feature_split` — lista pollutantów, pogody, kalendarza.
- Tabela `column_stats` — NaN%, min, max.
- Wykresy raw vs processed (`normalization_stats`).
- Podgląd ramki po MinMax (diagnostyka skali).
- Eksport processed CSV.

## 11.4. Trening modeli

**Lewa kolumna — slidery:**

| Slider | Bounds z `config` |
|---|---|
| Okno wejściowe LSTM | `effective_window_bounds(n_hours)` |
| Horyzont prognozy | `effective_forecast_bounds(n_hours, time_steps)` |
| Neurony LSTM | 16–256, krok 8 |
| Maks. epoki | 5–200 |
| Warstwy LSTM | 1–2 |
| Dropout | 0.0–0.5, krok 0.05 |
| Loss ważony | checkbox |

**Podgląd:**
- Liczba sekwencji + walidacja (`validate_training_params`).
- Podgląd nazwy pliku modelu.
- **Szac. czas treningu** (`training_timing`).
- Przyciski: **Trenuj**, **Zaproponuj ustawienia treningu** (LLM).

**Po treningu:**
- Metryki LSTM vs baseline (tabela).
- Wykresy ewaluacji.
- Opcja: ustaw jako aktywny, zapisz ocenę LLM.

## 11.5. Zarządzanie modelami

- Lista modeli dla wybranej stacji.
- Metryki, data treningu, parametry.
- Akcje: aktywuj, usuń, pełna ewaluacja + wykresy, **ocena LLM**.
- Oznaczenia: aktywny / best / LLM recommended.

## 11.6. Aktualizacja i ewaluacja

Retrening **z hiperparametrami modelu źródłowego** na świeżych danych:

1. Wybór modelu bazowego.
2. `train_lstm` z tymi samymi `time_steps`, `horizon`, `units`, …
3. Porównanie metryk stary vs nowy (delta MAE/RMSE/R²).
4. LLM: `compare_models_after_update` — tekstowa analiza.
5. Opcja przełączenia active na nowy model.

## 11.7. Podgląd i porównanie

Wykres godzinowy ±7 dni (`ui.build_hourly_forecast_figure`):

| Seria | Kierunek |
|---|---|
| GIOŚ AQI | wstecz (processed) |
| Open-Meteo composite | wstecz + wprzód |
| ASQI LSTM | wprzód (wybrany model) |
| Oficjalne EU/US AQI | opcjonalnie |

Sterowanie **2 rzędami**:
1. Odśwież prognozę | wybór modelu LSTM.
2. Multiselect wskaźników: GIOŚ AQI, OM, ASQI LSTM, dodatkowe indeksy.

Linia pionowa „Teraz”. `past_days=7` w `fetch_forecast_bundle`.

## 11.8. Ustawienia

Podzakładki:

### Moje stacje
- Włącz/wyłącz stację na liście klienta.
- Usuń stację (z katalogu aplikacji).

### Dodaj z listy GIOŚ
- `fetch_gios_station_list()` — import katalogu.
- Filtr, wyszukiwarka, badge audytu.
- Dodaj z `lat/lon` i ID.

### Kompletność GIOŚ (`gios_audit.py`)
- Skan sensorów: które mają 5 pollutantów EAQI.
- Skan archiwum 30 dni: `MIN_ARCHIVAL_POINTS` (~70% godzin).
- Cache wyników, eksport tabeli.

### OpenAI
- Klucz API, wybór modelu (`gpt-4o`, `gpt-5.5`, …).
- Zapis w `settings/app_settings.json`.
- Podgląd stałych z `config` (granice sliderów).

## 11.9. Wspólne elementy admina

- `ui.page_header`, `ui.inject_css`, Material Icons.
- `model_registry.delete_invalid_models()` na starcie.
- Spinner przy ładowaniu danych stacji.
- Query params synchronizują stację między zakładkami.
