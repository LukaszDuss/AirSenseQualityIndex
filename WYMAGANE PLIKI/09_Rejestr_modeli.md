# 9. Rejestr modeli

Moduł: `model_registry.py`. Katalog: `models/`.

## 9.1. Pliki per model

| Plik | Zawartość |
|---|---|
| `<name>.keras` | Wagi Keras |
| `<name>_meta.npy` | Słownik Python (pickle via numpy) |

## 9.2. Konwencja nazewnictwa

`build_model_name(station, time_steps, horizon, epochs_run, lstm_units)`:

```
aq_{slug_stacji}_o{okno}_h{horyzont}_e{epoki_faktyczne}_n{neurony}_{YYYYMMDD}
```

Przykład: `aq_zabrze_o504_h168_e37_n64_20260604`

- `epochs_run` — faktyczna liczba epok (po early stopping), nie `epochs_max`.
- `unique_model_name()` dodaje `_2`, `_3` przy kolizji.

## 9.3. Kluczowe pola meta

| Pole | Rola |
|---|---|
| `feature_columns` | Kolumny wejścia — muszą istnieć w `df_proc` przy inferencji |
| `feat_min`, `feat_max` | Skaler cech |
| `y_min`, `y_max` | Skaler targetu |
| `time_steps`, `horizon` | Okno i horyzont |
| `lstm_units`, `lstm_layers`, `dropout` | Architektura |
| `horizon_weighted_loss` | Czy użyto weighted MSE |
| `metrics` | MAE/MSE/RMSE/R² |
| `baseline_metrics` | Porównanie z baseline |
| `training_duration_sec` | Kalibracja czasu (rozdz. 7) |
| `train_history` | `loss`, `val_loss` per epoka |
| `station_id` | Powiązanie ze stacją |

## 9.4. Mapowania JSON

| Plik | Zawartość |
|---|---|
| `active_models.json` | `{ "550": "aq_zabrze_...", ... }` — **model na widoku klienta** |
| `best_models.json` | Najlepszy RMSE per `station_id` |
| `llm_recommended.json` | Rekomendacja LLM dla klienta |

Funkcje: `set_active/get_active`, `set_best/get_best`, `set_llm_recommended/get_llm_recommended`.

## 9.5. Zarządzanie w adminie

Zakładka „Zarządzanie modelami”:

- Tabela: stacja, okno, horyzont, epoki, neurony, RMSE, data.
- Badge: **aktywny**, **best**, **rekomendowany LLM**.
- Akcje: ustaw aktywny, usuń, ocena LLM, wykresy metryk.
- `delete_invalid_models()` przy wejściu admina — usuwa `.keras` bez poprawnego meta.

## 9.6. Zgodność meta

`load_meta()` zwraca `None` gdy brak `feature_columns` — model trafia do listy „niezgodnych” (wymaga retreningu).

## 9.7. Zapis ocen LLM

`save_llm_evaluation(model_name, text)` → `models/llm_evaluations/<name>.txt`

## 9.8. Przepływ wyboru modelu na kliencie

```
client_page._load_lstm_forecast()
    → model_registry.get_active(station_id)
    → load_meta(name)
    → ui.lstm_forecast(name, meta, df_proc)
```

Brak aktywnego modelu → komunikat błędu na `/`.
