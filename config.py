# -*- coding: utf-8 -*-
"""Centralne stałe projektu AirSense — jedno źródło prawdy dla całego kodu.

Ujednolicone założenia:
  * do treningu bierzemy MINIMUM 30 dni danych (seed archiwalny + akumulacja),
  * okno wejściowe LSTM: domyślnie 21 dni; max ~30 dni historii seed,
  * horyzont prognozy: minimum 3 dni, docelowo 7 dni (= 168 h), rozdzielczość godzinowa.
"""

# --- Dane / trening ---
MIN_TRAIN_DAYS = 30                      # minimalny zakres danych do treningu
SEED_PAST_DAYS = 30                      # ile dni seedujemy przy pierwszym uruchomieniu

# --- Kompletność stacji GIOŚ (EAQI: 5 zanieczyszczeń, archiwum 30 dni) ---
REQUIRED_EAQI_POLLUTANTS = ("PM25", "PM10", "NO2", "O3", "SO2")
EXPECTED_HOURLY_30D = SEED_PAST_DAYS * 24          # ~720 punktów / parametr
MIN_ARCHIVAL_POINTS = int(EXPECTED_HOURLY_30D * 0.7)  # min. ~70% godzin w 30 dniach

# --- Horyzont prognozy (godziny) ---
MIN_FORECAST_DAYS = 3
MIN_FORECAST_HOURS = MIN_FORECAST_DAYS * 24   # 72
MAX_FORECAST_DAYS = 7
MAX_FORECAST_HOURS = MAX_FORECAST_DAYS * 24   # 168
DEFAULT_FORECAST_DAYS = 7
DEFAULT_FORECAST_HOURS = DEFAULT_FORECAST_DAYS * 24   # 168

# --- Okno wejściowe LSTM (ile godzin wstecz widzi sieć) — nie mylić z horyzontem prognozy ---
MIN_WINDOW_DAYS = 3
MIN_WINDOW_HOURS = MIN_WINDOW_DAYS * 24   # min. kilka dni historii (LSTM)
DEFAULT_WINDOW_DAYS = 21
DEFAULT_WINDOW_HOURS = DEFAULT_WINDOW_DAYS * 24   # 504
MAX_WINDOW_HOURS = SEED_PAST_DAYS * 24    # teoretycznie całe archiwum (~720 h)
# Minimalna liczba sekwencji treningowych przy danym oknie (ogranicza slider w UI).
MIN_TRAIN_SEQUENCES = 48


def effective_window_bounds(n_hourly_rows, forecast_horizon=None):
    """(min_okno, max_okno) w godzinach — max z liczby wierszy w bazie (processed).

    Górny limit okna nie zależy od aktualnego horyzontu w UI — rezerwujemy tylko
    min. horyzont (72 h). Maks. horyzont liczy ``effective_forecast_bounds`` z okna.
    """
    rows = max(0, int(n_hourly_rows))
    reserve_h = MIN_FORECAST_HOURS
    cap = rows - reserve_h - MIN_TRAIN_SEQUENCES
    if rows > reserve_h + 1:
        cap = min(cap, rows - reserve_h - 1)
    max_w = max(0, int(cap))
    min_w = MIN_WINDOW_HOURS
    if max_w < min_w:
        min_w = max(24, max_w)
    max_w = max(min_w, max_w)
    return int(min_w), int(max_w)


def effective_forecast_bounds(n_hourly_rows, time_steps):
    """(min_horyzont, max_horyzont) w godzinach — max z liczby wierszy i okna wejściowego."""
    rows = max(0, int(n_hourly_rows))
    ts = max(1, int(time_steps))
    cap = rows - ts - MIN_TRAIN_SEQUENCES
    max_fh = min(MAX_FORECAST_HOURS, max(1, int(cap)))
    min_fh = MIN_FORECAST_HOURS
    if max_fh < min_fh:
        min_fh = max(1, max_fh)
    max_fh = max(min_fh, max_fh)
    return int(min_fh), int(max_fh)


def effective_max_window_hours(n_hourly_rows, forecast_horizon):
    """Górny limit okna (kompatybilność wsteczna)."""
    return effective_window_bounds(n_hourly_rows, forecast_horizon)[1]


def training_sequence_count(n_hourly_rows, time_steps, forecast_horizon):
    """Liczba nakładających się sekwencji treningowych."""
    rows = max(0, int(n_hourly_rows))
    ts, fh = int(time_steps), int(forecast_horizon)
    if rows < ts + fh:
        return 0
    return rows - ts - fh + 1


def validate_training_params(n_hourly_rows, time_steps, forecast_horizon):
    """Czy da się zbudować min. liczbę sekwencji. Zwraca (ok, komunikat_błędu)."""
    n_rows = max(0, int(n_hourly_rows))
    ts, fh = int(time_steps), int(forecast_horizon)
    n_seq = training_sequence_count(n_rows, ts, fh)
    if n_seq < MIN_TRAIN_SEQUENCES:
        return False, (
            f"Za mało danych dla okna/horyzontu: {n_rows} h w bazie, okno {ts} h, "
            f"horyzont {fh} h → tylko {n_seq} sekwencji (min. {MIN_TRAIN_SEQUENCES}). "
            "Zmniejsz okno lub horyzont albo odśwież dane surowe."
        )
    if n_seq < 5:
        return False, "Za mało sekwencji (min. 5) — zmniejsz okno lub horyzont."
    return True, None

# --- Trening LSTM (zakresy sliderów w UI) ---
MIN_LSTM_UNITS = 16
MAX_LSTM_UNITS = 256
DEFAULT_LSTM_UNITS = 64
MIN_EPOCHS = 5
MAX_EPOCHS = 200
DEFAULT_EPOCHS = 60

# --- Architektura LSTM (slider w UI) ---
MIN_LSTM_LAYERS = 1
MAX_LSTM_LAYERS = 2
DEFAULT_LSTM_LAYERS = 2
MIN_DROPOUT = 0.0
MAX_DROPOUT = 0.5
DEFAULT_DROPOUT = 0.2
DEFAULT_HORIZON_WEIGHTED_LOSS = True

# --- Prognoza w widoku klienta ---
CLIENT_FORECAST_DAYS = MAX_FORECAST_DAYS      # 7 dni

# --- Rejestr modeli ---
MODELS_DIR = "models"
ACTIVE_MODELS_FILE = "models/active_models.json"   # model na widoku klienta (LSTM)
BEST_MODELS_FILE = "models/best_models.json"       # najniższe MAE per stacja (admin)
LLM_RECOMMENDED_FILE = "models/llm_recommended.json"  # rekomendacja LLM dla klienta
