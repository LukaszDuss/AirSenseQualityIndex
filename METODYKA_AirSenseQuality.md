# Metodyka wskaźnika AirSenseQuality

Dokument projektowy AirSense. Wyjaśnia, **co liczymy, jak liczymy i dlaczego** —
w szczególności rolę zmiennych pogodowych oraz definicję autorskiego wskaźnika
**AirSenseQuality** i jego porównanie z indeksem europejskim (EAQI).

---

## 1. Cel projektu

Aplikacja prognozuje **jakość powietrza**, a nie pogodę. Celem prognozy jest jeden,
zagregowany wskaźnik **AirSenseQuality** (nasz odpowiednik AQI), liczony ze stężeń
zanieczyszczeń. Temperatura, wilgotność i prędkość wiatru są **wyłącznie cechami
pomocniczymi modelu**.

---

## 2. Decyzja: pogoda jest cechą modelu, a NIE składnikiem wzoru wskaźnika

**Decyzja:** temperatura, wilgotność i wiatr wchodzą do modelu jako cechy wejściowe,
ale **nie** wchodzą do wzoru na wartość AirSenseQuality.

### Uzasadnienie

1. **Definicja indeksu jakości powietrza.** Wszystkie uznane indeksy (europejski EAQI,
   amerykański US EPA AQI, polski indeks GIOŚ) liczone są **wyłącznie ze stężeń
   zanieczyszczeń** (PM2.5, PM10, NO₂, SO₂, O₃, …). Pogoda nie jest zanieczyszczeniem i
   nie ma zdrowotnych progów stężeń — nie da się jej sensownie „wymieszać" do indeksu.

2. **Unikamy podwójnego liczenia.** Pogoda wpływa na powietrze **pośrednio**: wiatr
   rozprasza zanieczyszczenia, inwersja termiczna i wysoka wilgotność sprzyjają ich
   kumulacji i tworzeniu aerozoli wtórnych. Ten wpływ jest **już zapisany w zmierzonych
   stężeniach**. Gdybyśmy dorzucili pogodę jeszcze raz do wzoru, policzylibyśmy ten sam
   efekt dwa razy.

3. **Porównywalność.** Chcemy zestawiać AirSenseQuality z indeksem europejskim (EAQI).
   Porównanie ma sens tylko wtedy, gdy oba liczą **to samo zjawisko** (stężenia
   zanieczyszczeń). Wrzucenie pogody do naszego wzoru zerwałoby porównywalność.

4. **Interpretowalność.** „AirSenseQuality = 60" musi znaczyć konkretny poziom
   zanieczyszczenia. Domieszka temperatury rozmyłaby tę interpretację.

5. **Wartość predykcyjna pozostaje wykorzystana.** Pogoda jest cenna jako **cecha
   prognostyczna**: nadchodzący silny wiatr zapowiada spadek stężeń, a spadek
   temperatury z bezwietrzną nocą — wzrost smogu. Model LSTM uczy się tych zależności
   z sekwencji wejściowej. To realizuje rolę „pomocniczego wskaźnika, który pomaga
   określić przyszły stan powietrza", bez psucia samej definicji indeksu.

**Wniosek:** pogoda → wejście modelu. Indeks → tylko zanieczyszczenia. To rozwiązanie
zgodne ze standardami, porównywalne i interpretowalne.

---

## 3. Decyzja: źródło danych pogodowych — Open-Meteo (pozostaje)

Skoro pogoda jest potrzebna jako cecha, zostawiamy **Open-Meteo** jako jej źródło:
to jedyne łatwo dostępne, darmowe, **godzinowe** źródło temperatury, wilgotności i
wiatru dla współrzędnych stacji, spójne czasowo z danymi GIOŚ. Rezygnacja z pogody
osłabiłaby zdolność predykcyjną modelu (utrata sygnału o dyspersji), więc ją zostawiamy.

---

## 4. Definicja AirSenseQuality (skala ciągła 0–100)

Liczymy **dwa** wskaźniki na tej samej skali 0–100, dla porównania:

### 4.1. Sub-indeksy per zanieczyszczenie (wspólna podstawa)

Dla każdego zanieczyszczenia liczymy sub-indeks przez **interpolację liniową stężenia
w pasmach europejskich (EAQI)** w µg/m³ (bez konwersji jednostek — pasuje do danych
GIOŚ). Pasma (górne granice 6 klas EAQI):

| Zanieczyszczenie | Pasma stężeń [µg/m³] |
|---|---|
| PM2.5 | 0 – 10 – 20 – 25 – 50 – 75 – 800 |
| PM10  | 0 – 20 – 40 – 50 – 100 – 150 – 1200 |
| NO₂   | 0 – 40 – 90 – 120 – 230 – 340 – 1000 |
| O₃    | 0 – 50 – 100 – 130 – 240 – 380 – 800 |
| SO₂   | 0 – 100 – 200 – 350 – 500 – 750 – 1250 |

Granice pasm odwzorowujemy na punkty indeksu `[0, 16.7, 33.3, 50, 66.7, 83.3, 100]`
i interpolujemy liniowo. Stężenia powyżej najwyższego progu → 100.

> CO i C₆H₆ (benzen) nie są częścią EAQI, więc nie wchodzą do sub-indeksów (mogą być
> dostępne w danych i służyć jako dodatkowe cechy modelu).

### 4.2. Indeks europejski (EAQI) — wariant porównawczy

`EuropeanIndex = max(sub_indeksy)` — **decyduje najgorszy parametr** (zasada EAQI).
Dodatkowo wyznaczamy klasę 1–6:

| Klasa | Zakres 0–100 | Opis |
|---|---|---|
| 1 | 0–16.7 | Bardzo dobry |
| 2 | 16.7–33.3 | Dobry |
| 3 | 33.3–50 | Umiarkowany |
| 4 | 50–66.7 | Dostateczny |
| 5 | 66.7–83.3 | Zły |
| 6 | 83.3–100 | Bardzo zły |

### 4.3. AirSenseQuality (autorski kompozyt)

Różni się od EAQI tym, że uwzględnia **całe obciążenie** powietrza, nie tylko najgorszy
parametr:

```
AirSenseQuality = 0.7 · max(sub_indeksy) + 0.3 · średnia(sub_indeksy)
```

- Składnik `max` (waga 0.7) zachowuje zdrowotną zasadę „najgorszy parametr rządzi".
- Składnik `średnia` (waga 0.3) podnosi wskaźnik, gdy **wiele** zanieczyszczeń jest
  podwyższonych jednocześnie (sytuacja realnie gorsza niż pojedynczy wysoki parametr).

Dzięki temu `AirSenseQuality ≥ EuropeanIndex` i różnica między nimi pokazuje, jak bardzo
„skumulowane" jest zanieczyszczenie. Wagi (0.7/0.3) to parametry projektowe — łatwe do
zmiany w `air_quality.py`.

---

## 5. Co prognozujemy

Model (LSTM) dostaje na wejściu **sekwencję wszystkich zmiennych** (zanieczyszczenia +
pogoda pomocnicza) z okna `t_steps` godzin i prognozuje **wyłącznie trajektorię
AirSenseQuality** na `forecast_horizon` godzin do przodu. Nie prognozujemy każdej zmiennej
z osobna — zgodnie z założeniem „na bazie wszystkich zmiennych wyciągamy jeden wskaźnik".

---

## 6. Przetwarzanie i baza danych (JSON, dwie warstwy)

- **`raw`** — surowe, realne odczyty (zanieczyszczenia + pogoda). Służą do prowenancji i
  **przyrostowego dociągania** (min. 30 dni przy pierwszym uruchomieniu, potem tylko nowe
  godziny).
- **`processed`** — przeliczana **w całości przy każdej aktualizacji**: dane oczyszczone
  (outliery), **uzupełnione** (imputacja braków), **znormalizowane** (z zapisanymi
  parametrami skalera) oraz policzone kolumny `AirSenseQuality` i `EuropeanIndex`.
  Przeliczanie od zera gwarantuje, że normalizacja nie „dryfuje" w miarę napływu danych.
