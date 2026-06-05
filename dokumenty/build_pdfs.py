# -*- coding: utf-8 -*-
"""Buduje trzy dokumenty PDF: dokumentację techniczną, raport indywidualny i scenariusz prezentacji."""
import os
from reportlab.lib.units import cm
from pdf_common import (H1, H2, H3, P, CAP, SP, BULLETS, CODE, IMG, TABLE,
                        title_page, build, CONTENT_W, lead)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
def A(n): return os.path.join(ASSETS, n)


# =========================================================================
#  1. DOKUMENTACJA TECHNICZNA
# =========================================================================
def build_techniczna():
    s = title_page("DOKUMENTACJA TECHNICZNA",
                   "Opis każdego etapu systemu: źródła danych, czyszczenie, model i predykcja")

    s += [H1("1. Wprowadzenie"),
          P("AirSense Weather AI to prototyp systemu uczenia maszynowego, który pobiera dane z rzeczywistych, "
            "publicznych API, przetwarza je i prognozuje krótkoterminowo (godzinowo) parametry pogodowe oraz jakości powietrza. "
            "Aplikacja zbudowana jest w Streamlit i obejmuje pełny cykl ML: pobieranie danych, preprocessing, "
            "analizę eksploracyjną (EDA), budowę i trening sieci LSTM, ewaluację oraz inferencję."),
          P("Niniejszy dokument opisuje krok po kroku każdy etap przetwarzania — od momentu pobrania surowych "
            "danych, przez ich czyszczenie, aż po architekturę modelu i sposób generowania prognozy.")]

    s += [H1("2. Architektura i pipeline"),
          P("System realizuje klasyczny pipeline ML w jednej aplikacji. Dane z dwóch źródeł są łączone w jeden "
            "zbiór dzienny, czyszczone, przekształcane w sekwencje czasowe i podawane do modelu LSTM, który "
            "generuje prognozę. Jakość modelu mierzona jest na wydzielonym zbiorze testowym."),
          SP(2), IMG(A("pipeline.png"), CONTENT_W),
          CAP("Rys. 1. Pipeline przetwarzania danych — od API do predykcji.")]

    s += [H1("3. Źródła danych"),
          P("Aplikacja korzysta z dwóch niezależnych, darmowych API. Dla każdej z pięciu obsługiwanych stacji "
            "zapisany jest identyfikator GIOŚ oraz współrzędne geograficzne potrzebne do Open-Meteo.")]

    s += [H2("3.1. Dane pogodowe — Open-Meteo"),
          P("Z Open-Meteo pobierane są temperatura, wilgotność względna i prędkość wiatru jako dane "
            "<b>godzinowe</b> (~720 punktów na 30 dni). Model uczy się właśnie na tej rozdzielczości — "
            "agregacja do dni dawała tylko ~30 punktów, czyli zbiór zbyt mały do trenowania. Pobieramy "
            "30 dni historii oraz bieżący dzień (<b>forecast_days=1</b>); dane są obcinane do bieżącej "
            "godziny, więc nie trafiają do nich godziny „w przyszłość”. Parametr <b>timezone=Europe/Warsaw</b> "
            "dopasowuje czas do strefy polskiej."),
          CODE('weather_url = (\n'
               '    f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"\n'
               '    f"&past_days=30&forecast_days=1&timezone=Europe/Warsaw"\n'
               '    f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"\n'
               ')\n'
               "df_weather = df_weather.set_index('Data').sort_index()   # rozdzielczosc godzinowa")]

    s += [H2("3.2. Dane o jakości powietrza — GIOŚ API v1"),
          P("Z GIOŚ pobierana jest lista sensorów stacji, a następnie pomiary z każdego z nich. Serwer potrafi "
            "odrzucać zapytania bez nagłówków przeglądarki, dlatego dołączane są nagłówki HTTP (User-Agent, "
            "Referer, Accept). <b>Ważne:</b> API v1 zwraca klucze po polsku (np. „Wskaźnik - kod”, „Wartość”, "
            "„Identyfikator stanowiska”) opakowane w słownik — kod wykrywa je elastycznie (z fallbackiem do "
            "starszego, angielskiego API). Endpoint udostępnia tylko ostatnie ~3 dni pomiarów godzinowych."),
          CODE("headers = {\n"
               "    'User-Agent': 'Mozilla/5.0 ... Chrome/122.0.0.0 Safari/537.36',\n"
               "    'Accept': 'application/json, text/plain, */*',\n"
               "    'Referer': 'https://powietrze.gios.gov.pl/'\n"
               "}\n"
               'url = f"https://api.gios.gov.pl/pjp-api/v1/rest/station/sensors/{station_id}"\n'
               "s_res = requests.get(url, headers=headers, timeout=10)")]

    s += [H2("3.3. Obsługiwane stacje pomiarowe"),
          TABLE([["Stacja", "ID GIOŚ", "Szer. geogr.", "Dł. geogr."],
                 ["Zabrze", "550", "50.3124", "18.7711"],
                 ["Warszawa — Marszałkowska", "544", "52.2287", "21.0122"],
                 ["Kraków — Al. Krasińskiego", "400", "50.0574", "19.9261"],
                 ["Gdańsk — Lektykarska", "738", "54.3497", "18.6548"],
                 ["Wrocław — Korzeniowskiego", "265", "51.1294", "17.0292"]],
                col_w=[7.4 * cm, 2.6 * cm, 3.5 * cm, 3.5 * cm])]

    s += [H1("4. Czyszczenie i przygotowanie danych"),
          P("Dane z rzeczywistych stacji nie są idealne — zawierają luki i pojedyncze błędne piki. Zastosowano "
            "następujące kroki czyszczenia:"),
          BULLETS([
              "<b>Interpolacja liniowa</b> braków w obie strony (limit_direction='both').",
              "<b>Ograniczanie wartości odstających</b> metodą percentyli — przycinanie do 1. i 99. percentyla "
              "(winsoryzacja), co usuwa błędne piki czujników bez kasowania całych rekordów.",
              "<b>Łączenie źródeł</b> po dacie dziennej; brakujące kolumny pollutantów są dołączane dynamicznie.",
              "<b>Obcięcie do dnia dzisiejszego</b> — nie przechowujemy dni „w przyszłość”.",
          ]),
          CODE("def remove_outliers(df):\n"
               "    df_cleaned = df.copy()\n"
               "    for col in df_cleaned.columns:\n"
               "        if pd.api.types.is_numeric_dtype(df_cleaned[col]):\n"
               "            lower = df_cleaned[col].quantile(0.01)\n"
               "            upper = df_cleaned[col].quantile(0.99)\n"
               "            df_cleaned[col] = np.clip(df_cleaned[col], lower, upper)\n"
               "    return df_cleaned"),
          SP(2),
          P("Po interpolacji i usunięciu outlierów zbiór jest obcinany do bieżącej daty:"),
          CODE("df_final = df_final.interpolate(method='linear', limit_direction='both')\n"
               "df_final = remove_outliers(df_final)\n"
               "df_final = df_final[df_final.index <= date.today()]   # bez dni \"w przyszlosc\"")]

    s += [H1("5. Sekwencje czasowe i skalowanie"),
          P("Model uczy się przewidywać kolejne <b>godziny</b> na podstawie okna poprzednich godzin. Dane są "
            "najpierw skalowane do przedziału [0, 1] (MinMaxScaler), a następnie przekształcane w nakładające się "
            "sekwencje wejście→wyjście. Okno wejściowe (domyślnie 48 h) i horyzont prognozy (domyślnie 6 h) są "
            "parametrami konfigurowalnymi w aplikacji."),
          CODE("def prepare_sequences(data, time_steps, forecast_horizon):\n"
               "    X, y = [], []\n"
               "    for i in range(len(data) - time_steps - forecast_horizon + 1):\n"
               "        X.append(data[i:(i + time_steps), :])\n"
               "        y.append(data[(i + time_steps):(i + time_steps + forecast_horizon), :])\n"
               "    return np.array(X), np.array(y)")]

    s += [H1("6. Model LSTM (Sequence-to-Sequence)"),
          P("Sercem systemu jest dwuwarstwowa sieć LSTM typu sequence-to-sequence z regularyzacją Dropout. "
            "Model przyjmuje okno cech z kilku dni i zwraca prognozę wszystkich cech na kilka dni naprzód. "
            "Liczba cech jest dynamiczna — dopasowana do liczby aktywnych sensorów danej stacji."),
          IMG(A("architecture.png"), CONTENT_W * 0.62),
          CAP("Rys. 2. Architektura modelu LSTM Seq2Seq."),
          CODE("model = Sequential([\n"
               "    LSTM(lstm_units, return_sequences=True, input_shape=(time_steps, N_FEATURES)),\n"
               "    Dropout(0.2),\n"
               "    LSTM(lstm_units, return_sequences=False),\n"
               "    Dropout(0.2),\n"
               "    Dense(forecast_horizon * N_FEATURES),\n"
               "    Reshape((forecast_horizon, N_FEATURES)),\n"
               "])\n"
               "model.compile(optimizer='adam', loss='mse')"),
          SP(4),
          H3("Hiperparametry i podział zbioru"),
          TABLE([["Parametr", "Wartość"],
                 ["Okno wejściowe (time_steps)", "12–96 h (domyślnie 48)"],
                 ["Horyzont prognozy (forecast_horizon)", "1–24 h (domyślnie 6)"],
                 ["Liczba neuronów LSTM", "50–100 (domyślnie 64)"],
                 ["Liczba epok", "do 150 + early stopping (patience 12)"],
                 ["Batch size", "16"],
                 ["Dropout", "0.2"],
                 ["Optymalizator / funkcja straty", "Adam / MSE"],
                 ["Podział zbioru (chronologiczny)", "80% trening / 10% walidacja / 10% test"]],
                col_w=[9.5 * cm, CONTENT_W - 9.5 * cm]),
          SP(4),
          P("Podział jest <b>chronologiczny</b> (bez tasowania) — to istotne dla szeregów czasowych, aby model "
            "nie „widział” przyszłości podczas treningu:"),
          CODE("total = len(X)\n"
               "train_len = int(total * 0.8)\n"
               "val_len   = int(total * 0.1)\n"
               "X_train, y_train = X[:train_len], y[:train_len]\n"
               "X_val,   y_val   = X[train_len:train_len+val_len], y[train_len:train_len+val_len]\n"
               "X_test,  y_test  = X[train_len+val_len:], y[train_len+val_len:]")]

    s += [H1("7. Ewaluacja modelu"),
          P("Ewaluacja odbywa się na 10% zbiorze testowym, którego model nie widział podczas treningu. "
            "Metryki liczone są w <b>jednostkach rzeczywistych</b> — predykcje i wartości oczekiwane są najpierw "
            "odskalowane (inverse_transform), a dopiero potem porównywane. Dzięki temu MAE i RMSE są "
            "interpretowalne (np. stopnie, µg/m³), a nie wyrażone w bezwymiarowej skali [0, 1]."),
          CODE("y_test_real = scaler.inverse_transform(y_test.reshape(-1, N_FEATURES))\n"
               "y_pred_real = scaler.inverse_transform(y_pred.reshape(-1, N_FEATURES))\n"
               "mae  = mean_absolute_error(y_test_real, y_pred_real)\n"
               "rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))\n"
               "r2   = r2_score(y_test_real, y_pred_real)   # uśrednione po cechach"),
          SP(4),
          P("Metryki liczone są też w rozbiciu na cechy. Poniższy wykres pokazuje realny wynik (stacja Zabrze, "
            "dane godzinowe, prognoza +1 h) jako średni błąd względny nMAE = MAE / średnia."),
          IMG(A("metrics_per_feature.png"), CONTENT_W * 0.82),
          CAP("Rys. 3. Realny błąd prognozy (nMAE) w rozbiciu na cechy — stacja Zabrze."),
          H3("Realne metryki (zbiór testowy, jednostki rzeczywiste)"),
          TABLE([["Cecha", "MAE", "R²"],
                 ["Ogółem", "6.22", "−0.25"],
                 ["Temperatura", "3.18 °C", "−1.31"],
                 ["Wilgotność", "8.54 %", "0.43"],
                 ["Wiatr", "2.57", "0.11"],
                 ["PM10", "3.91 µg/m³", "−0.25"],
                 ["PM2.5", "2.78 µg/m³", "−0.64"],
                 ["O3", "18.9 µg/m³", "−0.01"]],
                col_w=[5 * cm, 5 * cm, CONTENT_W - 10 * cm]),
          SP(4),
          P("<b>Interpretacja (uczciwa):</b> MAE jest stabilne i interpretowalne — np. temperatura myli się średnio "
            "o ~3°C. R² jest jednak bliskie zera, a często ujemne i niestabilne między uruchomieniami, co oznacza, "
            "że model jest na poziomie prostego baseline'u. Przyczyny: bardzo mały zbiór testowy (~3 dni) oraz dane "
            "GIOŚ o zanieczyszczeniach dostępne tylko za ~3 dni (reszta okna jest interpolowana). Jest to działający "
            "prototyp i kompletny pipeline, ale jeszcze nie dokładny predyktor — najlepiej radzi sobie z "
            "krótkoterminową prognozą pogody (wilgotność, temperatura). Główny kierunek rozwoju: więcej danych "
            "(dłuższa historia, archiwalne dane GIOŚ).")]

    s += [H1("8. Predykcja i inferencja"),
          P("W zakładce predykcji użytkownik wybiera zapisany model, a aplikacja odtwarza parametry skalera z "
            "metadanych. Macierz wejściowa (ostatnie dni) jest edytowalna. Przy odtwarzaniu skalera zabezpieczono "
            "się przed cechami stałymi — bez tego dzielenie przez zero dawałoby prognozę NaN:"),
          CODE("data_range = scaler_max - scaler_min\n"
               "data_range[data_range == 0] = 1.0   # cechy stałe -> brak dzielenia przez zero\n"
               "trained_scaler.scale_ = 1.0 / data_range\n"
               "trained_scaler.min_   = -scaler_min * trained_scaler.scale_"),
          SP(3),
          P("Prognoza zaczyna się godzinę po ostatnim realnym pomiarze, a oś czasu na wykresie to prawdziwe znaczniki czasu:"),
          CODE("last_date = pd.Timestamp(df_live.index[-1])\n"
               "future_dates = [last_date + timedelta(hours=i) for i in range(1, f_horizon + 1)]"),
          SP(3),
          P("Do oceny jakości najlepiej posłużyć się wykresem predykcja vs rzeczywistość na zbiorze testowym "
            "(poniżej, dla temperatury, prognoza +1 h) — model śledzi ogólny kształt, ale z wyraźnymi błędami."),
          IMG(A("prediction.png"), CONTENT_W * 0.78),
          CAP("Rys. 4. Predykcja vs rzeczywistość — temperatura, zbiór testowy (+1 h).")]

    s += [H1("9. Dashboard i analiza eksploracyjna (EDA)"),
          P("Pierwsza zakładka prezentuje surowe odczyty godzinowe wraz z 24-godzinną średnią kroczącą (trend "
            "dobowy). Trend liczony jest jako osobna seria, aby nie zmieniać liczby cech przekazywanych do modelu. "
            "Pionowa linia „Dziś” wyraźnie oddziela dane historyczne od przyszłości (za którą odpowiada już prognoza)."),
          IMG(A("eda_trend.png"), CONTENT_W * 0.86),
          CAP("Rys. 5. Wykres EDA: odczyty godzinowe, 24-godzinna średnia krocząca i linia „Dziś”.")]

    s += [H1("10. Ostatnie usprawnienia jakości"),
          BULLETS([
              "<b>Naprawiono parser GIOŚ v1</b> (polskie klucze) — wcześniej żaden wskaźnik zanieczyszczeń się "
              "nie wczytywał i widoczna była tylko pogoda.",
              "<b>Przejście na modelowanie godzinowe</b> zamiast dziennego — realnie trenowalny zbiór "
              "(~720 zamiast ~30 punktów).",
              "Dodano early stopping — model bywał niedouczony przy małej liczbie epok.",
              "Dodano pionową linię „Dziś” na wykresach EDA i predykcji.",
              "Wyłączono pobieranie godzin „w przyszłość” (forecast_days=1 + obcięcie do teraz).",
              "Metryki liczone w jednostkach rzeczywistych oraz w rozbiciu na cechy.",
              "Zabezpieczenie odtwarzania skalera przed cechami stałymi (brak prognoz NaN); .bfill().ffill().",
          ])]

    s += [H1("11. Uruchomienie"),
          CODE("pip install -r requirements.txt\n"
               "streamlit run app.py"),
          P("Aplikacja uruchamia się w przeglądarce. W panelu bocznym wybiera się stację; w zakładkach kolejno: "
            "analiza EDA, trening modelu i predykcja.")]

    s += [H1("12. Struktura plików"),
          TABLE([["Plik / katalog", "Znaczenie"],
                 ["app.py", "Główna aplikacja Streamlit: EDA, trening LSTM, ewaluacja, predykcja."],
                 ["train_model.py", "Pomocniczy skrypt treningowy (podstawowy LSTM na danych AQI)."],
                 ["requirements.txt", "Lista zależności (Streamlit, TensorFlow, scikit-learn, Plotly...)."],
                 ["airsense_lstm.keras", "Zapisany przykładowy model Keras."],
                 ["metrics.npy, scaler_params.npy", "Metryki i parametry skalera przykładowego modelu."],
                 ["models/", "Modele wytrenowane w aplikacji (.keras + _meta.npy)."],
                 ["dokumenty/", "Dokumentacja, raport, prezentacja i wykresy."]],
                col_w=[5.6 * cm, CONTENT_W - 5.6 * cm])]

    s += [H1("13. Ograniczenia i dalszy rozwój"),
          BULLETS([
              "Dane GIOŚ o zanieczyszczeniach dostępne tylko za ~3 dni → pollutanty są w dużej części interpolowane "
              "i słabo przewidywane (R² ≤ 0).",
              "Mały zbiór testowy → R² niestabilne (bliskie 0); model jest na poziomie baseline'u. Najlepiej działa "
              "krótkoterminowa prognoza pogody (wilgotność, temperatura, błąd ~13–21%).",
              "Kierunki rozwoju: archiwalne dane GIOŚ (dłuższa historia zanieczyszczeń), więcej miast i automatyczna "
              "detekcja stacji, modelowanie pojedynczych cech lub ważenie strat, porównanie z GRU / Prophet / "
              "XGBoost / Transformerami, automatyczny re-trening i baza historii predykcji.",
          ])]

    build(os.path.join(HERE, "Dokumentacja_techniczna_AirSense.pdf"),
          "AirSense Weather AI — Dokumentacja techniczna", s)


# =========================================================================
#  2. RAPORT INDYWIDUALNY
# =========================================================================
def build_raport():
    s = title_page("RAPORT INDYWIDUALNY",
                   "Szczegółowy opis wkładu każdego członka zespołu (Grupa 5)")

    s += [H1("1. Zespół"),
          P("Projekt AirSense Weather AI zrealizował czteroosobowy zespół. Praca została podzielona zgodnie z "
            "rolami w typowym projekcie ML — od inżynierii danych, przez analizę i model, po integrację i "
            "ewaluację. Poniższa tabela podsumowuje role, a kolejne sekcje opisują indywidualny wkład każdej osoby."),
          TABLE([["Osoba", "Rola", "Główny obszar"],
                 ["Jakub Moszyński", "Data Engineer", "Pobieranie i czyszczenie danych, sekwencje"],
                 ["Damian Tomczyk", "Data Analyst", "EDA, dashboard i wizualizacje"],
                 ["Nikol Olszewska", "Deep Learning Engineer", "Architektura i trening modelu LSTM"],
                 ["Łukasz Duss", "ML Analyst / Project Manager", "Integracja, ewaluacja, rejestr modeli"]],
                col_w=[4.2 * cm, 4.8 * cm, CONTENT_W - 9.0 * cm])]

    s += [H1("2. Wkład indywidualny")]

    # --- Jakub ---
    s += [H2("2.1. Jakub Moszyński — Data Engineer"),
          P("<b>Zakres odpowiedzialności:</b> integracja z zewnętrznymi API, pozyskanie surowych danych i ich "
            "wstępne przygotowanie do dalszej analizy."),
          P("<b>Kluczowy wkład:</b>"),
          BULLETS([
              "Implementacja funkcji <b>fetch_real_climate_data</b> łączącej dwa źródła (Open-Meteo i GIOŚ).",
              "Pobieranie danych godzinowych z Open-Meteo i agregacja do wartości dziennych.",
              "Obsługa GIOŚ API v1: nagłówki HTTP imitujące przeglądarkę oraz elastyczne parsowanie polskich "
              "kluczy odpowiedzi („Wskaźnik - kod”, „Wartość”, „Identyfikator stanowiska”) i pobieranie pomiarów per sensor.",
              "Czyszczenie danych: interpolacja liniowa braków oraz funkcja <b>remove_outliers</b> "
              "(przycinanie do 1./99. percentyla).",
              "Przygotowanie sekwencji czasowych (<b>prepare_sequences</b>) jako wejścia do modelu.",
          ]),
          P("<b>Wyzwania i decyzje:</b> serwery GIOŚ odrzucały zapytania bez nagłówków — rozwiązaniem było "
            "dołączenie User-Agent i Referer oraz odporna na różne formaty obsługa JSON. Braki i błędne piki "
            "rozwiązano interpolacją i winsoryzacją zamiast usuwania rekordów."),
          P("<b>Rezultat:</b> spójny, dzienny zbiór danych gotowy do analizy i treningu, budowany na żywo z "
            "publicznych API.")]

    # --- Damian ---
    s += [H2("2.2. Damian Tomczyk — Data Analyst"),
          P("<b>Zakres odpowiedzialności:</b> analiza eksploracyjna danych (EDA) oraz warstwa wizualizacji w "
            "interfejsie."),
          P("<b>Kluczowy wkład:</b>"),
          BULLETS([
              "Zakładka „EDA i Trendy”: wybór parametru i interaktywny wykres odczytów w Plotly.",
              "Obliczanie 24-godzinnej średniej kroczącej (trend dobowy) jako osobnej serii (bez zmiany liczby cech modelu).",
              "Dodanie pionowej linii z dzisiejszą datą („Dziś”) oddzielającej historię od prognozy.",
              "Czytelne formatowanie wykresów (ciemny motyw, hovermode), spójne z resztą aplikacji.",
          ]),
          P("<b>Wyzwania i decyzje:</b> trend liczony jest poza macierzą wejściową modelu, co rozwiązuje problem "
            "niezgodności wymiarów. Po uwadze recenzenta usunięto pobieranie dni „w przyszłość”, dzięki czemu "
            "linia „Dziś” trafia na koniec realnej historii."),
          P("<b>Rezultat:</b> przejrzysty dashboard, który pozwala szybko zrozumieć trendy i sezonowość danych "
            "dla wybranej stacji.")]

    # --- Nikol ---
    s += [H2("2.3. Nikol Olszewska — Deep Learning Engineer"),
          P("<b>Zakres odpowiedzialności:</b> projekt, implementacja i trening sieci neuronowej."),
          P("<b>Kluczowy wkład:</b>"),
          BULLETS([
              "Architektura LSTM typu sequence-to-sequence: dwie warstwy LSTM, Dropout 0.2, warstwa Dense i Reshape.",
              "Skalowanie danych (MinMaxScaler) oraz zapis parametrów skalera do późniejszej inferencji.",
              "Dynamiczna liczba cech — model dopasowuje kształt wejścia/wyjścia do liczby sensorów stacji.",
              "Konfiguracja treningu (Adam, MSE) i integracja paska postępu epok w interfejsie (callback Keras).",
          ]),
          P("<b>Wyzwania i decyzje:</b> przy stosunkowo małym zbiorze zastosowano Dropout, aby ograniczyć "
            "przeuczenie. Architektura Seq2Seq pozwala prognozować kilka dni naprzód dla wszystkich cech jednocześnie."),
          P("<b>Rezultat:</b> elastyczny model, który trenuje się od nowa dla dowolnej stacji i dowolnej liczby "
            "dostępnych parametrów.")]

    # --- Łukasz ---
    s += [H2("2.4. Łukasz Duss — ML Analyst / Project Manager"),
          P("<b>Zakres odpowiedzialności:</b> integracja całej aplikacji, ewaluacja modelu oraz zarządzanie "
            "projektem i przepływem danych między zakładkami."),
          P("<b>Kluczowy wkład:</b>"),
          BULLETS([
              "Spięcie etapów w jedną aplikację Streamlit (zakładki EDA / trening / predykcja).",
              "Chronologiczny podział zbioru 80 / 10 / 10 (trening / walidacja / test).",
              "Ewaluacja modelu: MAE, MSE, RMSE i R² liczone w jednostkach rzeczywistych oraz w rozbiciu na cechy.",
              "Rejestr modeli: zapis i odczyt modeli oraz metadanych (_meta.npy), wybór modelu do predykcji.",
              "Utrzymywanie wyników w Session State, odtwarzanie skalera i generowanie prognozy na realne daty.",
          ]),
          P("<b>Wyzwania i decyzje:</b> metryki przeniesiono ze skali [0, 1] do jednostek rzeczywistych, co "
            "uczyniło je interpretowalnymi. Zabezpieczono odtwarzanie skalera przed cechami stałymi (brak "
            "prognoz NaN). Session State rozwiązuje problem znikania wyników po każdej interakcji w Streamlit."),
          P("<b>Rezultat:</b> kompletna, działająca aplikacja end-to-end z czytelnym raportem ewaluacji i "
            "powtarzalną predykcją.")]

    s += [H1("3. Macierz wkładu"),
          P("Oznaczenia: ● — odpowiedzialność główna, ○ — wsparcie."),
          TABLE([["Etap", "Jakub", "Damian", "Nikol", "Łukasz"],
                 ["Pobieranie danych (API)", "●", "", "", ""],
                 ["Czyszczenie / outliery", "●", "", "", ""],
                 ["Sekwencje czasowe", "●", "", "○", ""],
                 ["EDA / wizualizacje", "", "●", "", ""],
                 ["Architektura LSTM", "", "", "●", ""],
                 ["Skalowanie danych", "", "", "●", "○"],
                 ["Trening + podział 80/10/10", "", "", "●", "●"],
                 ["Ewaluacja (metryki)", "", "", "○", "●"],
                 ["Predykcja / inferencja", "", "○", "", "●"],
                 ["Integracja i UI", "", "○", "", "●"],
                 ["Rejestr modeli / Session State", "", "", "", "●"]],
                col_w=[CONTENT_W - 4 * 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm])]

    s += [H1("4. Wspólne usprawnienia jakości"),
          P("Po recenzji zespół wprowadził poprawki, które dotknęły kilku obszarów odpowiedzialności:"),
          BULLETS([
              "Naprawa parsera GIOŚ v1 (polskie klucze) — bez niej brakowało wszystkich wskaźników zanieczyszczeń "
              "(widoczna była tylko pogoda) — Jakub.",
              "Przejście na modelowanie godzinowe zamiast dziennego (realnie trenowalny zbiór) — Nikol, Jakub.",
              "Linia „Dziś” i rezygnacja z godzin „w przyszłość” — Damian (wizualizacja) i Jakub (dane).",
              "Metryki w jednostkach rzeczywistych i w rozbiciu na cechy + early stopping — Łukasz, Nikol.",
              "Ciągła trajektoria/predykcja vs rzeczywistość oraz zabezpieczenie skalera — Łukasz, Nikol.",
          ])]

    build(os.path.join(HERE, "Raport_indywidualny_AirSense.pdf"),
          "AirSense Weather AI — Raport indywidualny", s)


# =========================================================================
#  3. SCENARIUSZ PREZENTACJI
# =========================================================================
def build_scenariusz():
    s = title_page("SCENARIUSZ PREZENTACJI",
                   "Plan wystąpienia na ocenę — 15 slajdów, ok. 21 min + pytania")

    s += [H1("Jak korzystać z tego scenariusza"),
          P("Dokument przypisuje każdy slajd do mówcy, podaje orientacyjny czas i najważniejsze punkty do "
            "powiedzenia oraz zdanie-przejście do kolejnej części. Łączny czas treści to ok. 21 minut, co z "
            "sesją pytań mieści się w limicie 20–25 minut. Jeśli zabraknie czasu, najłatwiej skrócić demo "
            "(slajd 12) oraz slajd o wynikach.")]

    s += [H1("Budżet czasowy"),
          TABLE([["Część", "Slajdy", "Mówca", "Czas"],
                 ["Wprowadzenie i problem", "1–4", "Łukasz", "~3:30"],
                 ["Dane i preprocessing", "5–7", "Jakub", "~4:30"],
                 ["Model i trening", "8–9", "Nikol", "~3:30"],
                 ["Wyniki i wizualizacje", "10–11", "Łukasz + Damian", "~3:15"],
                 ["Demo aplikacji", "12", "Damian", "~3:00"],
                 ["Wnioski i rozwój", "13–14", "Łukasz", "~2:30"],
                 ["Zakończenie i pytania", "15", "Cały zespół", "~3:00"]],
                col_w=[5.6 * cm, 2.4 * cm, 4.8 * cm, CONTENT_W - 12.8 * cm])]

    def slide(num, title, speaker, time, points, transition=None):
        out = [H2("Slajd %s — %s" % (num, title)),
               P("<b>Mówca:</b> %s &nbsp;&nbsp;|&nbsp;&nbsp; <b>Czas:</b> %s" % (speaker, time))]
        out.append(BULLETS(points))
        if transition:
            out.append(P("<i>Przejście:</i> „%s”" % transition))
        out.append(SP(3))
        return out

    s += [H1("Przebieg slajd po slajdzie")]

    s += slide("1", "Tytuł", "Łukasz", "0:45",
               ["Przywitanie, przedstawienie zespołu (Grupa 5) i tematu.",
                "Jedno zdanie haczyk: „Łączymy realne dane o pogodzie i smogu, by prognozować kolejne dni”."],
               "Zobaczmy, co dziś pokażemy.")
    s += slide("2", "Plan prezentacji", "Łukasz", "0:30",
               ["Krótko wymienić 6 punktów wystąpienia.",
                "Zaznaczyć, że na końcu będzie demo na żywo."],
               "Zacznijmy od problemu, który rozwiązujemy.")
    s += slide("3", "Zespół i podział pracy", "Łukasz", "0:30",
               ["Po jednym zdaniu o roli każdej osoby.",
                "Zapowiedzieć, że każdy poprowadzi swoją część."],
               "Dlaczego w ogóle zajęliśmy się tym tematem?")
    s += slide("4", "Problem", "Łukasz", "1:45",
               ["Smog (PM10, PM2.5, NO2) to realny problem zdrowotny, zwłaszcza zimą.",
                "Potrzebna jest prognoza na kolejne dni, a nie tylko bieżący odczyt.",
                "Dane są rozproszone — pogoda osobno, zanieczyszczenia osobno.",
                "Nasz cel: prototyp łączący te źródła i prognozujący je siecią LSTM."],
               "Przekażę głos Jakubowi — opowie, skąd bierzemy dane.")
    s += slide("5", "Pełny pipeline ML", "Jakub", "1:00",
               ["Pokazać, że to kompletny pipeline, nie sam model.",
                "Wymienić etapy: dane → preprocessing → sekwencje → LSTM → ewaluacja → predykcja."],
               "Przyjrzyjmy się dwóm źródłom danych.")
    s += slide("6", "Skąd pochodzą dane", "Jakub", "1:30",
               ["Open-Meteo: temperatura, wilgotność, wiatr; dane godzinowe (~720 pkt na 30 dni).",
                "GIOŚ API v1: zanieczyszczenia; polskie klucze w JSON (trzeba było je sparsować); tylko ~3 dni historii.",
                "5 stacji: Zabrze, Warszawa, Kraków, Gdańsk, Wrocław."],
               "Surowe dane nie są idealne — zobaczmy wyzwania preprocessingu.")
    s += slide("7", "Wyzwania preprocessingu", "Jakub", "2:00",
               ["Braki → interpolacja liniowa; błędne piki → przycinanie do 1./99. percentyla.",
                "Różna liczba sensorów → dynamiczna liczba cech.",
                "Blokady GIOŚ → nagłówki HTTP i obsługa błędów.",
                "Nasza poprawka: brak dni „w przyszłość” (forecast_days=1) + linia „Dziś”."],
               "Mamy czyste dane — Nikol opowie o modelu.")
    s += slide("8", "Architektura LSTM", "Nikol", "2:00",
               ["Dlaczego LSTM: szeregi czasowe, pamięć krótkiej historii.",
                "Seq2Seq: z okna 48 h prognoza kolejnych godzin dla wszystkich cech.",
                "Dwie warstwy LSTM(64) + Dropout 0.2, Dense + Reshape na wyjściu.",
                "Dynamiczne wejście dopasowane do liczby sensorów."],
               "Jak ten model trenujemy?")
    s += slide("9", "Trening i konfiguracja", "Nikol", "1:30",
               ["Chronologiczny podział 80/10/10 (ważne dla szeregów czasowych).",
                "Okno 48 h, horyzont +1–24 h, Adam + MSE, early stopping.",
                "Skaler zapisujemy, by odtworzyć skalę przy predykcji."],
               "Skoro mamy wytrenowany model — jakie są wyniki?")
    s += slide("10", "Wyniki — metryki", "Łukasz", "1:45",
               ["Metryki realne (dane godzinowe, prognoza +1 h), w jednostkach rzeczywistych.",
                "Najlepiej: wilgotność (błąd ~13%) i temperatura (MAE ~3°C).",
                "Uczciwie: R² bliskie 0 i niestabilne — mały zbiór testowy + tylko ~3 dni danych GIOŚ; to prototyp, nie produkcja."],
               "Zobaczmy, jak wygląda to na wykresach.")
    s += slide("11", "Wizualizacje", "Damian", "1:30",
               ["Zakładka EDA: dane godzinowe + 24-godzinna średnia krocząca + linia „Dziś”.",
                "Predykcja vs rzeczywistość na zbiorze testowym (+1 h): model śledzi kształt, ale z błędami."],
               "Najlepiej zobaczyć to na żywo — przechodzimy do demo.")
    s += slide("12", "Demo aplikacji", "Damian", "3:00",
               ["Uruchomić: streamlit run app.py.",
                "Krok 1: wybrać stację w panelu bocznym, pokazać wykres EDA z linią „Dziś”.",
                "Krok 2: w zakładce 2 ustawić parametry i odpalić krótki trening (mało epok).",
                "Krok 3: w zakładce 3 pokazać metryki i wygenerować prognozę na kolejne godziny.",
                "Plan awaryjny: zrzuty ekranu / nagranie, gdyby zabrakło internetu."],
               "Podsumujmy, co udało się osiągnąć.")
    s += slide("13", "Wnioski", "Łukasz", "1:30",
               ["Działający prototyp klasy MLOps na realnych danych godzinowych z API.",
                "Krótkoterminowo pogoda jest częściowo przewidywalna; zanieczyszczenia słabo (tylko ~3 dni GIOŚ).",
                "Naprawiony parser GIOŚ + dane godzinowe + uczciwa ewaluacja; baza do rozwoju, nie model produkcyjny."],
               "Co dalej z projektem?")
    s += slide("14", "Kierunki rozwoju", "Łukasz", "1:00",
               ["Dłuższa historia, więcej miast, automatyczna detekcja stacji.",
                "Porównanie z GRU, Prophet, XGBoost, Transformerami.",
                "Automatyczny re-trening, baza predykcji, rejestr modeli."],
               "To wszystko z naszej strony.")
    s += slide("15", "Zakończenie i pytania", "Cały zespół", "0:30 + Q&A",
               ["Podziękowanie za uwagę.",
                "Otwarcie na pytania; każdy odpowiada w swoim obszarze.",
                "Przygotować odpowiedzi: dlaczego LSTM, dlaczego R² bliskie 0, skąd dane, ograniczenia projektu."])

    s += [H1("Wskazówki praktyczne"),
          BULLETS([
              "Przećwiczyć demo wcześniej i mieć wytrenowany model „w zapasie” (krótki trening na żywo bywa ryzykowny).",
              "Mieć zrzuty ekranu jako plan awaryjny przy braku internetu lub problemach z API.",
              "Pilnować czasu — jeśli demo się przedłuża, skrócić omawianie slajdów 13–14.",
              "Mówić do publiczności, nie do slajdów; slajdy są tłem dla narracji.",
          ])]

    build(os.path.join(HERE, "Scenariusz_prezentacji.pdf"),
          "AirSense Weather AI — Scenariusz prezentacji", s)


if __name__ == "__main__":
    build_techniczna()
    build_raport()
    build_scenariusz()
    print("Gotowe.")
