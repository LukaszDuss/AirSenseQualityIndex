# AirSense — spis dokumentacji i checklist (TODO)

> Dokumentacja techniczna w folderze `WYMAGANE PLIKI/` — analogiczna do `Dokumentacja_techniczna_AirSense.pdf`, ale **zsynchronizowana z aktualnym kodem** (czerwiec 2026).
> Status: `[x]` gotowe · `[~]` w trakcie · `[ ]` do zrobienia

---

## Jak czytać ten zestaw

| Plik | Temat | Status |
|---|---|---|
| [README.md](README.md) | Indeks + jak regenerować wykresy | [x] |
| [01_Wprowadzenie_i_architektura.md](01_Wprowadzenie_i_architektura.md) | Cel, nazewnictwo, moduły, routing | [x] |
| [02_Zrodla_danych_i_API.md](02_Zrodla_danych_i_API.md) | GIOŚ, Open-Meteo, nagłówki, endpointy | [x] |
| [03_Magazyn_JSON_i_pipeline.md](03_Magazyn_JSON_i_pipeline.md) | raw/processed, update_and_load | [x] |
| [04_Wskazniki_i_metodyka.md](04_Wskazniki_i_metodyka.md) | GiosAQI, ASQI, EAQI, Open-Meteo | [x] |
| [05_Cechy_i_przetwarzanie.md](05_Cechy_i_przetwarzanie.md) | Imputacja, kalendarz, outliery | [x] |
| [06_Sekwencje_i_trening_LSTM.md](06_Sekwencje_i_trening_LSTM.md) | prepare_xy, architektura, loss | [x] |
| [07_Szacowanie_czasu_treningu.md](07_Szacowanie_czasu_treningu.md) | training_timing.py — pełny opis | [x] |
| [08_Ewaluacja_i_baseline.md](08_Ewaluacja_i_baseline.md) | Metryki, wykresy, baseline | [x] |
| [09_Rejestr_modeli.md](09_Rejestr_modeli.md) | models/, active/best/LLM | [x] |
| [10_Widok_klienta.md](10_Widok_klienta.md) | UI, zakładki, okna 24 h | [x] |
| [11_Panel_admina.md](11_Panel_admina.md) | Wszystkie zakładki admin | [x] |
| [12_LLM_i_konfiguracja.md](12_LLM_i_konfiguracja.md) | OpenAI, prompty, ustawienia | [x] |
| [13_Konfiguracja_i_stale.md](13_Konfiguracja_i_stale.md) | config.py — każda stała | [x] |
| [14_Uruchamianie_i_pliki.md](14_Uruchamianie_i_pliki.md) | venv, skrypty, struktura repo | [x] |

---

## Wyczerpująca lista TODO (opis projektu)

### A. Kontekst produktu
- [x] A1. Cel: prognoza **ASQI** (AirSenseQualityIndex), nie pogody
- [x] A2. Rozróżnienie: GIOŚ AQI (pomiar) vs ASQI (LSTM) vs Open-Meteo composite
- [x] A3. Dwa widoki: `/` klient, `/admin` administrator
- [x] A4. Streamlit multipage + ukryta nawigacja (`app.py`, `app_pages.py`)
- [x] A5. Stacje: katalog domyślny + własne z listy GIOŚ (`stations_settings.py`)

### B. Źródła danych
- [x] B1. GIOŚ API v1 — lista sensorów, getData (~3 dni), archivalData (30 dni)
- [x] B2. Nagłówki HTTP przeglądarki (User-Agent, Referer)
- [x] B3. Parsowanie polskich kluczy API (`_pick`, `_first_list`)
- [x] B4. Łączenie wielu sensorów tego samego kodu (`merge_series_prefer_dense`)
- [x] B5. Open-Meteo pogoda: temperature_2m, humidity, wind, precipitation
- [x] B6. Open-Meteo AQ: stężenia + european_aqi + us_aqi
- [x] B7. Strefa czasowa Europe/Warsaw wszędzie
- [x] B8. Prognoza ±7 dni (`past_days` symetrycznie w adminie)

### C. Magazyn lokalny
- [x] C1. Plik `data_store/station_<id>.json`
- [x] C2. Warstwa `raw` — tylko realne odczyty
- [x] C3. Warstwa `processed` — snapshot po każdej aktualizacji
- [x] C4. Migracja schema v1 → v2
- [x] C5. Zapis atomowy (.tmp + replace)
- [x] C6. `update_and_load` — tryb archival vs recent
- [x] C7. `weather_past_days_for_store` — adaptacyjny zakres pogody
- [x] C8. Parametry skalera w processed (feat_min/max, y_min/max)

### D. Czyszczenie i cechy
- [x] D1. Oś czasu z zanieczyszczeń (pogoda pomocnicza)
- [x] D2. Interpolacja liniowa `limit_direction='both'`
- [x] D3. Winsoryzacja percentyle 1% / 99%
- [x] D4. Obcięcie do `now` — bez przyszłości w historii
- [x] D5. Cechy kalendarzowe: sin/cos godzina, sin/cos dzień tyg., sezon grzewczy
- [x] D6. `finalize_features_for_model` — drop wierszy bez kompletnych EAQI pollutantów
- [x] D7. Mediana dla braków pogody/opadów

### E. Wskaźniki
- [x] E1. Sub-indeksy: interpolacja w pasmach EAQI (`INDEX_BREAKPOINTS`)
- [x] E2. EuropeanIndex = max(sub)
- [x] E3. GiosAQI = 0.7·max + 0.3·mean (wagi W_MAX, W_MEAN)
- [x] E4. Klasy 1–6 + kolory + etykiety PL
- [x] E5. OpenMeteoCompositeIndex — ten sam wzór ze stężeń prognozy OM
- [x] E6. ASQI — tylko wyjście LSTM (nie kolumna processed)
- [x] E7. Migracja legacy: AirSenseQuality → GiosAQI

### F. Sekwencje i trening
- [x] F1. Wzór liczby sekwencji: `n_rows - time_steps - horizon + 1`
- [x] F2. MIN_TRAIN_SEQUENCES = 48
- [x] F3. `prepare_xy`: X wielowymiarowe, y jednowymiarowe (GiosAQI)
- [x] F4. MinMax na train_rows (pierwsze 80% wierszy processed, nie sekwencji!)
- [x] F5. Podział sekwencji 80/10/10 chronologiczny
- [x] F6. Architektura LSTM 1–2 warstwy + Dense(horizon)
- [x] F7. HorizonWeightedMSE — wagi 1.0 → 0.35 po horyzoncie
- [x] F8. EarlyStopping patience=12
- [x] F9. batch_size=16, optimizer adam
- [x] F10. Zapis .keras + .npy meta
- [x] F11. Pomiar czasu: fit_sec + predict_sec → training_duration_sec

### G. Szacowanie czasu treningu
- [x] G1. `estimate_training_seconds` — tryb kalibracji vs heurystyka
- [x] G2. Mediana sec/epoch z metadanych starych modeli
- [x] G3. Skale: units^1.35, time_steps^0.85, features^0.4, n_train^0.55
- [x] G4. effective_epochs = max(5, epochs_max × 0.72)
- [x] G5. Przedział UI: kalibracja 0.8–1.35×, heurystyka 0.45–2.2×
- [x] G6. `format_duration` po treningu
- [x] G7. StreamlitKerasCallback — live epoka/loss/czas

### H. Ewaluacja
- [x] H1. MAE, MSE, RMSE, R² globalnie
- [x] H2. Metryki per horyzont (1h, 24h, …)
- [x] H3. Baseline: persistence, MA 24h/168h, seasonal lag 24h/168h
- [x] H4. Wykresy Plotly: scatter, residual, horizon bars
- [x] H5. Cache ewaluacji w session_state admina

### I. Rejestr modeli
- [x] I1. Nazewnictwo: `{stacja}_{okno}h_{horyzont}h_ep{epoki}_{units}u`
- [x] I2. active_models.json — model na widoku klienta
- [x] I3. best_models.json — najlepszy RMSE
- [x] I4. llm_recommended.json — rekomendacja LLM
- [x] I5. llm_evaluations/ — zapis tekstów ocen

### J. Widok klienta
- [x] J1. Hero: ASQI headline + klasa + kolor
- [x] J2. Meta: GIOŚ AQI teraz, wilgotność/wiatr z ostatniego processed
- [x] J3. Zakładki: ASQI (LSTM), Temp/Wilg/Wiatr (Open-Meteo prognoza)
- [x] J4. Okna 24 h sekwencyjne od pierwszej godziny prognozy LSTM
- [x] J5. Clip ASQI 0–100
- [x] J6. Picker dni — kafelki HTML + niewidoczny st.button
- [x] J7. Strefa czasowa wykresów: Europe/Warsaw

### K. Panel admina
- [x] K1. Dane surowe — refresh, reset, coverage, CSV
- [x] K2. Dane przetworzone — diagnostyka, min-max, raw vs processed
- [x] K3. Trening — slidery, sekwencje, szac. czas, LLM propose, trenuj
- [x] K4. Modele — aktywuj, usuń, LLM review, wykresy
- [x] K5. Aktualizacja — retrain z meta źródłowego modelu
- [x] K6. Podgląd — wykres ±7 dni GIOŚ/OM/LSTM
- [x] K7. Ustawienia — stacje, audit GIOŚ, OpenAI

### L. LLM
- [x] L1. Klucz: .env lub panel admin
- [x] L2. evaluate_after_training — interpretacja metryk PL
- [x] L3. recommend_training_settings — propozycja sliderów
- [x] L4. compare_models_after_update — stary vs nowy
- [x] L5. recommend_client_model — wybór modelu dla klienta

### M. Konfiguracja i uruchamianie
- [x] M1. Wszystkie stałe config.py z opisem
- [x] M2. effective_window_bounds / effective_forecast_bounds
- [x] M3. setup.ps1 / run.ps1 / requirements.txt
- [x] M4. env_config.py — loader .env bez python-dotenv

### N. Wykresy dokumentacji
- [x] N1. pipeline.png
- [x] N2. architecture.png
- [x] N3. sequences.png
- [x] N4. training_time.png
- [x] N5. indices.png
- [x] N6. client_windows.png

### O. Do ewentualnego rozszerzenia (poza kodem)
- [ ] O1. Hindcast LSTM wstecz (omawiane, nie zaimplementowane)
- [ ] O2. Eksport PDF z MD (skrypt build jak w `dokumenty/`)
- [ ] O3. Wykresy z realnych metryk modeli (auto z `models/`)

---

## Regeneracja wykresów

```powershell
.\.venv\Scripts\python.exe "WYMAGANE PLIKI\assets\make_charts.py"
```

Wykresy trafiają do `WYMAGANE PLIKI/assets/*.png` i są linkowane w rozdziałach MD.
