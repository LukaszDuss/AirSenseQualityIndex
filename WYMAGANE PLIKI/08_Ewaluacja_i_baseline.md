# 8. Ewaluacja i baseline

Moduły: `model_eval.py`, `baselines.py`.

## 8.1. Metryki globalne

`compute_metrics(y_true, y_pred)` — scikit-learn:

| Metryka | Znaczenie |
|---|---|
| **MAE** | Średni błąd bezwzględny (w punktach ASQI 0–100) |
| **MSE** | Średni błąd kwadratowy |
| **RMSE** | Pierwiastek MSE — główna metryka porównań modeli |
| **R²** | Wyjaśniona wariancja (1 = idealnie) |

Liczone na **spłaszczonych** wartościach testowych (wszystkie kroki horyzontu × wszystkie sekwencje testowe).

## 8.2. Metryki per horyzont

`metrics_by_horizon(y_true_2d, y_pred_2d)` — osobno dla kroku 1h, 2h, … `horizon`.

Pokazuje degradację dokładności wraz z oddalaniem prognozy (typowe dla LSTM).

## 8.3. Zbiór testowy

Indeksy sekwencji testowych: od `tr + vl` do końca (`eval_split` w meta).

- `train` = 80% sekwencji
- `val` = 10%
- `test` = 10%

Bez losowego mieszania — ostatnie 10% chronologicznie.

## 8.4. Baseline'y

`baselines.evaluate_baselines()` — te same sekwencje testowe co LSTM:

| Klucz | Metoda |
|---|---|
| `persistence` | Ostatnia wartość ASQI z okna wejściowego, powtórzona na cały horyzont |
| `ma_24h` | Średnia z ostatnich 24 h okna |
| `ma_168h` | Średnia z ostatnich 168 h (tydzień) |
| `seasonal_24h` | Wartość sprzed 24 h (lag dobowy) |
| `seasonal_168h` | Wartość sprzed 168 h (lag tygodniowy) |

Cel: udowodnić, że LSTM bije naiwne prognozy. Wyniki w `meta["baseline_metrics"]`.

## 8.5. Bundle ewaluacji

`_bundle_from_arrays()` zbiera:
- `metrics`, `metrics_by_horizon`
- `y_true`, `y_pred` (do wykresów)
- `split` — liczności zbiorów

Używane po treningu i przy „Aktualizacji modelu”.

## 8.6. Wykresy Plotly (`build_chart_figures`)

| Wykres | Opis |
|---|---|
| Scatter true vs pred | Punkty wokół linii y=x |
| Residual | Błąd w funkcji czasu/kroku |
| Horizon bars | MAE/RMSE per krok horyzontu |

Cache w `st.session_state` admina — unikanie ponownego liczenia przy przełączaniu zakładek.

## 8.7. `resolve_evaluation`

Ładuje meta modelu, odtwarza split testowy z `df_proc` i liczy metryki na żądanie (zakładka Modele / Aktualizacja).

## 8.8. Wybór „best” modelu

`model_registry.refresh_best_for_station()` — najniższy RMSE na teście wśród modeli danej stacji → `best_models.json`.

Osobno od **active** (wybór ręczny pod klienta) i **LLM recommended**.

## 8.9. Interpretacja w kontekście produktu

- ASQI 0–100: MAE=5 oznacza średnio ±5 punktów indeksu.
- Porównuj z baseline persistence — jeśli LSTM gorszy, model nie ma wartości.
- Per-horizon: akceptowalne gorsze R² przy 168 h, ważniejsze kroki 1–24 h (stąd opcjonalny weighted loss).
