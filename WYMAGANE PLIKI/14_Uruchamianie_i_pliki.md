# 14. Uruchamianie i struktura plików

Pełna instrukcja operacyjna: [URUCHAMIANIE.md](../URUCHAMIANIE.md).

## 14.1. Szybki start (Windows)

```text
setup.bat    # pierwsza instalacja — tworzy .venv
run.bat      # start Streamlit z .venv
```

Adresy:
- http://localhost:8501 — klient
- http://localhost:8501/admin — admin

## 14.2. Wymagania

- Python **3.10 – 3.12**
- Zależności: [requirements.txt](../requirements.txt)
- **Nie** uruchamiaj przez Anaconda `(base)` bez TensorFlow — użyj `.venv`.

## 14.3. Struktura repozytorium

```
app.py                 # entrypoint Streamlit
app_pages.py           # definicje stron / i /admin
env_config.py          # loader .env

config.py              # stałe globalne
air_quality.py         # wskaźniki
data_fetch.py          # API + pipeline
data_store.py          # JSON magazyn
feature_engineering.py # cechy kalendarzowe

ui_common.py           # UI wspólne + LSTM train/infer
ui_icons.py            # ikony Material
views/
  client_page.py       # widok klienta
  admin_router.py      # panel admin

model_registry.py      # rejestr modeli
model_eval.py          # ewaluacja
baselines.py           # baseline'y
training_timing.py     # szacunek czasu treningu
llm_eval.py            # OpenAI
stations_settings.py   # stacje + ustawienia
gios_audit.py          # audyt GIOŚ
normalization_stats.py # diagnostyka processed

scripts/refetch_data.py  # CLI przebudowy danych

data_store/            # generowane: station_*.json
models/                # generowane: *.keras, *_meta.npy, *.json
settings/              # app_settings.json, cache audytu

WYMAGANE PLIKI/        # ta dokumentacja + wykresy
dokumenty/             # stare skrypty PDF/PPTX
```

## 14.4. Pliki generowane lokalnie (gitignore)

| Ścieżka | Opis |
|---|---|
| `data_store/` | Bazy JSON stacji |
| `models/` | Modele i meta |
| `.env` | Sekrety |
| `settings/app_settings.json` | Ustawienia UI |
| `.venv/` | Środowisko Python |

## 14.5. Skrypty pomocnicze

| Skrypt | Rola |
|---|---|
| `setup.ps1` / `setup.bat` | `python -m venv` + `pip install -r requirements.txt` |
| `run.ps1` / `run.bat` | `streamlit run app.py` z `.venv` |
| `scripts/refetch_data.py` | Wymuszenie przebudowy danych wszystkich stacji |
| `WYMAGANE PLIKI/assets/make_charts.py` | Diagramy do dokumentacji |
| `dokumenty/build_pdfs.py` | Generator starego PDF (faza Weather AI) |

## 14.6. Typowy workflow operatora

1. `run.bat` — start aplikacji.
2. `/admin` → Ustawienia → dodaj/włącz stację.
3. Dane surowe → Odśwież (zbuduj 30 dni historii).
4. Dane przetworzone → sprawdź pokrycie i NaN.
5. Trening → ustaw slidery → Trenuj → ustaw aktywny model.
6. `/` — sprawdź prognozę ASQI na kliencie.
7. Podgląd → porównaj GIOŚ / OM / LSTM na wykresie ±7 dni.

## 14.7. Rozwiązywanie problemów

| Problem | Rozwiązanie |
|---|---|
| `No module named 'tensorflow'` | Użyj `run.bat`, nie systemowego `streamlit` |
| Za mało sekwencji | Zmniejsz okno/horyzont lub odśwież dane (więcej godzin) |
| Brak ASQI na kliencie | Ustaw model aktywny w adminie |
| LLM nie działa | Klucz OpenAI w ustawieniach lub `.env` |

## 14.8. Dokumentacja powiązana

| Dokument | Temat |
|---|---|
| `METODYKA_AirSenseQuality.md` | Uzasadnienie wzoru wskaźnika |
| `URUCHAMIANIE.md` | Instalacja szczegółowa |
| `WYMAGANE PLIKI/*.md` | Dokumentacja techniczna pełna |
| `Dokumentacja_techniczna_AirSense.pdf` | Wersja PDF (częściowo nieaktualna) |
