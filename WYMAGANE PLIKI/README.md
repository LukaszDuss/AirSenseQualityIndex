# Dokumentacja techniczna AirSense (MD)

Zestaw opisów **zsynchronizowany z kodem** (czerwiec 2026). Uzupełnia i zastępuje fragmenty starego `Dokumentacja_techniczna_AirSense.pdf` (faza „Weather AI”).

## Spis rozdziałów

| # | Plik | Zawartość |
|---|---|---|
| 0 | [00_SPIS_I_TODO.md](00_SPIS_I_TODO.md) | Checklista + status prac |
| 1 | [01_Wprowadzenie_i_architektura.md](01_Wprowadzenie_i_architektura.md) | Cel, moduły, routing |
| 2 | [02_Zrodla_danych_i_API.md](02_Zrodla_danych_i_API.md) | GIOŚ + Open-Meteo |
| 3 | [03_Magazyn_JSON_i_pipeline.md](03_Magazyn_JSON_i_pipeline.md) | raw → processed |
| 4 | [04_Wskazniki_i_metodyka.md](04_Wskazniki_i_metodyka.md) | GiosAQI, ASQI, EAQI |
| 5 | [05_Cechy_i_przetwarzanie.md](05_Cechy_i_przetwarzanie.md) | Czyszczenie, kalendarz |
| 6 | [06_Sekwencje_i_trening_LSTM.md](06_Sekwencje_i_trening_LSTM.md) | Model, loss, zapis |
| 7 | [07_Szacowanie_czasu_treningu.md](07_Szacowanie_czasu_treningu.md) | `training_timing.py` |
| 8 | [08_Ewaluacja_i_baseline.md](08_Ewaluacja_i_baseline.md) | Metryki, baseline |
| 9 | [09_Rejestr_modeli.md](09_Rejestr_modeli.md) | active / best / LLM |
| 10 | [10_Widok_klienta.md](10_Widok_klienta.md) | UI `/` |
| 11 | [11_Panel_admina.md](11_Panel_admina.md) | UI `/admin` |
| 12 | [12_LLM_i_konfiguracja.md](12_LLM_i_konfiguracja.md) | OpenAI |
| 13 | [13_Konfiguracja_i_stale.md](13_Konfiguracja_i_stale.md) | `config.py` |
| 14 | [14_Uruchamianie_i_pliki.md](14_Uruchamianie_i_pliki.md) | venv, struktura repo |

## Wykresy

Regeneracja diagramów:

```powershell
.\.venv\Scripts\python.exe "WYMAGANE PLIKI\assets\make_charts.py"
```

| Plik | Opis |
|---|---|
| `assets/pipeline.png` | Przepływ danych end-to-end |
| `assets/architecture.png` | Architektura LSTM |
| `assets/sequences.png` | Dlaczego odejmujemy okno + horyzont |
| `assets/training_time.png` | Wpływ `lstm_units` na szacunek czasu |
| `assets/indices.png` | EuropeanIndex vs GiosAQI |
| `assets/client_windows.png` | Okna 24 h w widoku klienta |

## Powiązane dokumenty w repo

- [METODYKA_AirSenseQuality.md](../METODYKA_AirSenseQuality.md) — uzasadnienie metodyki wskaźnika
- [URUCHAMIANIE.md](../URUCHAMIANIE.md) — instalacja i start aplikacji
