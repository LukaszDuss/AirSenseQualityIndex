# Tworzy .venv i instaluje zależności z requirements.txt
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Tworzenie wirtualnego srodowiska .venv ..."
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        $py = Get-Command py -ErrorAction SilentlyContinue
        if ($py) {
            & py -3.12 -m venv .venv
        } else {
            throw "Nie znaleziono Pythona. Zainstaluj Python 3.10–3.12 i dodaj do PATH."
        }
    } else {
        & python -m venv .venv
    }
    if (-not (Test-Path $VenvPython)) {
        throw "Nie udalo sie utworzyc .venv"
    }
}

Write-Host "Instalacja pakietow (moze potrwac kilka minut — TensorFlow) ..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

Write-Host ""
Write-Host "Gotowe. Uruchom aplikacje: .\run.ps1"
