# 5. Cechy i przetwarzanie

Moduły: `data_fetch.py` (czyszczenie), `feature_engineering.py` (cechy modelu).

## 5.1. Podział kolumn (`ui_common.feature_split`)

Z `df_proc` wyciągane są:

| Grupa | Przykłady | Rola |
|---|---|---|
| **poll** | PM25, PM10, NO2, O3, SO2, CO | Sub-indeksy, cechy LSTM |
| **weather** | Temperatura, Wilgotnosc, Wiatr, Opady | Cechy pomocnicze |
| **calendar** | Godzina_sin/cos, DzienTyg_sin/cos, Sezon_grzewczy | Cechy cykliczne |
| **index** | GiosAQI, EuropeanIndex, EuropeanClass | Target + diagnostyka, **nie** wejście LSTM |

`feature_columns` zapisane w meta modelu = poll + weather + calendar (bez kolumn indeksu).

## 5.2. Cechy kalendarzowe

`add_calendar_features(df)`:

| Kolumna | Wzór / logika |
|---|---|
| `Godzina_sin`, `Godzina_cos` | `sin/cos(2π × (hour + min/60) / 24)` |
| `DzienTyg_sin`, `DzienTyg_cos` | `sin/cos(2π × dayofweek / 7)` |
| `Sezon_grzewczy` | 1.0 dla miesięcy paź–kwi, else 0.0 |

Uzasadnienie: sezon grzewczy koreluje ze wzrostem PM w Polsce; cykliczność doby/tygodnia bez „skoku” na północy.

## 5.3. Finalizacja pod model

`finalize_features_for_model()`:

1. **Pollutanty EAQI** — wiersz odrzucony jeśli którykolwiek z obecnych pollutantów ma NaN (`dropna(subset=poll_cols)`).
2. **Pozostałe kolumny** (pogoda, opady, kalendarz) — braki wypełniane **medianą** kolumny (fallback 0).

## 5.4. Usuwanie outlierów

`remove_outliers()` — winsoryzacja per kolumna numeryczna:
- dolna granica: percentyl 1%
- górna granica: percentyl 99%

Cel: pojedyncze błędy czujników GIOŚ bez kasowania całych rekordów.

## 5.5. Interpolacja

`interpolate(method='linear', limit_direction='both')` na pełnej osi czasu.

**Ważne:** interpolacja jest tylko w warstwie processed (w pamięci), nie w raw.

## 5.6. Normalizacja przy treningu

W `train_lstm()` (nie w `data_store`):

- Zakres min/max liczony z **pierwszych 80% wiersów** `df_proc` (`train_rows`).
- Cechy: `(x - feat_min) / (feat_max - feat_min)`.
- Target `GiosAQI`: `(y - y_min) / y_range`.
- Parametry zapisywane w meta modelu — **ten sam** skaler przy inferencji.

## 5.7. Diagnostyka w adminie

Zakładka „Dane przetworzone” (`normalization_stats.py`):

- `raw_vs_processed_frames()` — pokrycie kolumn przed/po.
- `column_stats()` — min, max, NaN%, liczba wierszy.
- `minmax_scaled_frame()` — podgląd wartości po skali 0–1.

Wykresy porównawcze raw vs processed (Plotly) — wykrywanie „dziur” po imputacji.

## 5.8. Dominujący pollutant (widok klienta)

`air_quality.compute_subindices(df_proc.iloc[[-1]])` → `idxmax()` na ostatnim wierszu.

Pokazywane w meta hero: który sub-indeks jest najwyższy „teraz”.
