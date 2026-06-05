# Uruchamia Streamlit z .venv projektu (nie z Anaconda/base Python)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Streamlit = Join-Path $Root ".venv\Scripts\streamlit.exe"

if (-not (Test-Path $Streamlit)) {
    Write-Host "Brak .venv — uruchamiam instalacje ..."
    & (Join-Path $Root "setup.ps1")
}

# Szybka kontrola TensorFlow (typowy blad przy zlym interpreterze)
& $VenvPython -c "import tensorflow" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "TensorFlow niedostepny w .venv — ponowna instalacja pakietow ..."
    & $VenvPython -m pip install -r requirements.txt
}

Write-Host "Start AirSense: http://localhost:8501"
& $Streamlit run app.py
