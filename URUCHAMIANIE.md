# Uruchamianie AirSense

Aplikacja **musi** dzialac z wirtualnego srodowiska `.venv` w katalogu projektu.
Uruchomienie `streamlit` z Anacondy lub systemowego Pythona bez TensorFlow konczy sie bledem:

```text
ModuleNotFoundError: No module named 'tensorflow'
```

---

## Wymagania

| Element | Wersja / uwagi |
|---|---|
| Python | **3.10 – 3.12** (sprawdzono na 3.12) |
| System | Windows (skrypty `.bat` / `.ps1`); recznie tez Linux/macOS |
| Siec | dostep do API GIOŚ, Open-Meteo; opcjonalnie OpenAI (panel admin) |

---

## Pierwsza instalacja (Windows)

W katalogu projektu (`New folder (3)`):

**Opcja A — dwuklik**

```text
setup.bat
```

**Opcja B — PowerShell**

```powershell
cd "C:\Users\lukas\Desktop\New folder (3)"
.\setup.ps1
```

Skrypt:

1. tworzy `.venv` (jesli nie istnieje),
2. instaluje pakiety z `requirements.txt` (m.in. TensorFlow — moze potrwac kilka minut).

---

## Uruchomienie aplikacji

**Opcja A — dwuklik**

```text
run.bat
```

**Opcja B — PowerShell**

```powershell
cd "C:\Users\lukas\Desktop\New folder (3)"
.\run.ps1
```

**Opcja C — recznie (po aktywacji venv)**

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Po starcie otworz w przegladarce:

| Adres | Opis |
|---|---|
| http://localhost:8501 | Widok klienta (prognoza ASQI) |
| http://localhost:8501/admin | Panel administratora (trening LSTM, dane, ustawienia) |

Zatrzymanie: **Ctrl+C** w terminalu.

---

## Cursor / VS Code

1. **Python: Select Interpreter** → wybierz  
   `...\New folder (3)\.venv\Scripts\python.exe`
2. Terminal w IDE: `.\run.ps1` albo aktywuj `.venv` i `streamlit run app.py`.

Nie uzywaj interpretera `(base)` z Anacondy do tego projektu.

---

## Zaleznosci (`requirements.txt`)

Pakiet | Rola |
|---|---|
| streamlit | interfejs web |
| plotly | wykresy |
| pandas, numpy | dane godzinowe |
| tensorflow | model LSTM (trening + prognoza) |
| scikit-learn | metryki ewaluacji |
| requests | API GIOŚ / Open-Meteo |
| openai | opcjonalna ocena LLM w adminie |

Stary plik `requirements.txt` zawieral pelny freeze venv (torch, stylegan itd.) — **nie sa potrzebne** do AirSense i utrudnialy instalacje.

---

## Konfiguracja opcjonalna (OpenAI)

1. Skopiuj `.env.example` → `.env`
2. Ustaw `OPENAI_API_KEY=...`  
   albo zapisz klucz w panelu admin → zakladka ustawien OpenAI.

Bez klucza aplikacja dziala; funkcje LLM w adminie beda niedostepne.

---

## Typowe problemy

### `ModuleNotFoundError: No module named 'tensorflow'`

Uruchamiasz Streamlit **poza** `.venv`. Uzyj `run.bat` / `run.ps1` albo aktywuj `.venv` przed `streamlit run`.

### `ExecutionPolicy` w PowerShell

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

albo uruchamiaj przez `setup.bat` / `run.bat` (omijaja polityke dla jednego skryptu).

### Ponowna instalacja pakietow

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Usuniecie i czysta instalacja venv

```powershell
Remove-Item -Recurse -Force .venv
.\setup.ps1
```

---

## Struktura skryptow

| Plik | Dzialanie |
|---|---|
| `setup.ps1` / `setup.bat` | tworzy `.venv`, `pip install -r requirements.txt` |
| `run.ps1` / `run.bat` | start `streamlit run app.py` z `.venv` |
| `app.py` | punkt wejscia aplikacji |
