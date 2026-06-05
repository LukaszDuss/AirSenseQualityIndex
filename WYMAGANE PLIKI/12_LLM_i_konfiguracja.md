# 12. LLM i konfiguracja

Moduł: `llm_eval.py`. Ustawienia: `stations_settings.py`, `env_config.py`.

## 12.1. Kiedy używany LLM

LLM (**OpenAI**) jest **opcjonalny**. Bez klucza API aplikacja działa; funkcje AI w adminie zwracają komunikat o braku pakietu/klucza.

## 12.2. Skąd bierze się klucz

Kolejność (`stations_settings.resolve_openai_key()`):

1. `settings/app_settings.json` → pole `openai_api_key` (zapisane w panelu admin).
2. Zmienna środowiskowa `OPENAI_API_KEY` (z `.env` przez `env_config.load_env()`).

Model: `resolve_openai_model()` — domyślnie `gpt-4o`, dostępny też `gpt-5.5` (`STRONG_MODEL`).

## 12.3. Funkcje LLM

| Funkcja | Kiedy wywołana | Co robi |
|---|---|---|
| `evaluate_after_training` | Po treningu / w zakładce Modele | Interpretacja metryk MAE/RMSE/R² po polsku |
| `recommend_training_settings` | Przycisk „Zaproponuj ustawienia treningu” | JSON z propozycją sliderów |
| `compare_models_after_update` | Po aktualizacji modelu | Porównanie stary vs nowy |
| `recommend_client_model` | Wybór modelu dla klienta | Rekomendacja z listy modeli stacji |
| `parse_training_params` | Po odpowiedzi LLM | Mapowanie tekstu/JSON na liczby sliderów |

## 12.4. Prompt systemowy (skrót)

`SYSTEM_PROMPT` w `llm_eval.py` zawiera:
- kontekst AirSense, GIOŚ, Open-Meteo,
- definicję ASQI 0–100,
- opis LSTM wielokrokowego,
- listę cech wejściowych,
- ograniczenia z `config` (min/max okno, horyzont, sekwencje).

## 12.5. `recommend_training_settings` — wejście

`build_training_recommendation_user()` przekazuje LLM:
- tabelę istniejących modeli stacji,
- `data_info`: `n_hours`, `n_features`, lista kolumn,
- `constraints`: `min_window`, `max_window`, `min_window_days`, `default_window_hours`, …

LLM musi zaproponować parametry **w dozwolonych granicach** — potem `parse_training_params` + walidacja `validate_training_params`.

## 12.6. Zapis wyników LLM

- Tekst oceny → `model_registry.save_llm_evaluation(name, text)`.
- Rekomendacja modelu klienta → `set_llm_recommended(station_id, model_name)`.

## 12.7. `env_config.py`

Lekki parser `.env` **bez** zależności `python-dotenv`:
- pomija komentarze i puste linie,
- obsługuje `export KEY=val`,
- nie nadpisuje istniejących zmiennych OS (chyba że `override=True`).

Wywołanie: `env_config.load_env()` w `app.py` przy starcie.

## 12.8. `stations_settings.py`

| Plik | Zawartość |
|---|---|
| `settings/app_settings.json` | OpenAI, lista stacji custom, enabled, removed |

Funkcje:
- `enabled_stations()` — słownik stacji widocznych w app,
- `add_custom_station`, `remove_station`, `toggle_station`,
- `fetch_gios_station_list()` — katalog do dodawania stacji.

Domyślny katalog zawiera m.in. Zabrze (550), Warszawa (544), Kraków (400), Gdańsk (738), Wrocław (265).

## 12.9. Bezpieczeństwo

- `.env` i `app_settings.json` w `.gitignore`.
- Szablon: `.env.example` (tylko komentarz OPENAI_API_KEY).
- Klucz w UI admin: pole password.
