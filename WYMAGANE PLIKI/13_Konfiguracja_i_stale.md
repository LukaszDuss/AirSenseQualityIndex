# 13. Konfiguracja i stałe (`config.py`)

Jedno źródło prawdy dla całego projektu.

## 13.1. Dane i seed

| Stała | Wartość | Znaczenie |
|---|---|---|
| `MIN_TRAIN_DAYS` | 30 | Minimalny oczekiwany zakres do treningu |
| `SEED_PAST_DAYS` | 30 | Dni archiwum GIOŚ przy pierwszym uruchomieniu |
| `EXPECTED_HOURLY_30D` | 720 | Nominalna liczba godzin w 30 dniach |
| `MIN_ARCHIVAL_POINTS` | ~504 | 70% × 720 — próg audytu GIOŚ |

## 13.2. Horyzont prognozy

| Stała | Wartość |
|---|---|
| `MIN_FORECAST_DAYS` | 3 |
| `MIN_FORECAST_HOURS` | 72 |
| `MAX_FORECAST_DAYS` | 7 |
| `MAX_FORECAST_HOURS` | 168 |
| `DEFAULT_FORECAST_DAYS` | 7 |
| `DEFAULT_FORECAST_HOURS` | 168 |

## 13.3. Okno LSTM (lookback)

| Stała | Wartość |
|---|---|
| `MIN_WINDOW_DAYS` | 3 |
| `MIN_WINDOW_HOURS` | 72 |
| `DEFAULT_WINDOW_DAYS` | 21 |
| `DEFAULT_WINDOW_HOURS` | 504 |
| `MAX_WINDOW_HOURS` | 720 |
| `MIN_TRAIN_SEQUENCES` | 48 |

## 13.4. Slidery treningu

| Stała | Min | Max | Default |
|---|---|---|---|
| `LSTM_UNITS` | 16 | 256 | 64 |
| `EPOCHS` | 5 | 200 | 60 |
| `LSTM_LAYERS` | 1 | 2 | 2 |
| `DROPOUT` | 0.0 | 0.5 | 0.2 |
| `DEFAULT_HORIZON_WEIGHTED_LOSS` | — | — | True |

## 13.5. Klient i modele

| Stała | Wartość |
|---|---|
| `CLIENT_FORECAST_DAYS` | 7 |
| `MODELS_DIR` | `"models"` |
| `ACTIVE_MODELS_FILE` | `models/active_models.json` |
| `BEST_MODELS_FILE` | `models/best_models.json` |
| `LLM_RECOMMENDED_FILE` | `models/llm_recommended.json` |

## 13.6. Funkcje pomocnicze

### `effective_window_bounds(n_hourly_rows)`

Zwraca `(min_okno, max_okno)` w godzinach.

```
cap = n_rows - MIN_FORECAST_HOURS - MIN_TRAIN_SEQUENCES
max_okno = min(cap, n_rows - MIN_FORECAST_HOURS - 1)
min_okno = MIN_WINDOW_HOURS (lub 24 jeśli mało danych)
```

Rezerwuje miejsce na minimalny horyzont i 48 sekwencji.

### `effective_forecast_bounds(n_hourly_rows, time_steps)`

```
cap = n_rows - time_steps - MIN_TRAIN_SEQUENCES
max_horyzont = min(MAX_FORECAST_HOURS, cap)
min_horyzont = MIN_FORECAST_HOURS (lub mniej gdy cap mały)
```

### `training_sequence_count(n_rows, time_steps, forecast_horizon)`

```
n_rows - time_steps - forecast_horizon + 1
```

### `validate_training_params(...)`

Zwraca `(True, None)` lub `(False, komunikat)` gdy `n_seq < 48`.

## 13.7. Audyt GIOŚ

`REQUIRED_EAQI_POLLUTANTS = ("PM25", "PM10", "NO2", "O3", "SO2")` — stacja „kompletna” pod EAQI musi mieć wszystkie pięć w archiwum.

## 13.8. Gdzie stałe są wyświetlane

Zakładka admin **Ustawienia → OpenAI** — tabela statycznych granic z `config` (informacja dla operatora).
