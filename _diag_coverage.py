# -*- coding: utf-8 -*-
"""Diagnostyka pokrycia: realne punkty per zmienna (PRZED interpolacja) - produkcyjna sciezka."""
import time
import pandas as pd
import requests

from data_fetch import (GIOS_HEADERS, WEATHER_COLS, _first_list, _pick,
                        _fetch_sensor_series)

STATION = {"name": "Zabrze", "id": 550, "lat": 50.3124, "lon": 18.7711}
PAST_DAYS = 30


def fetch_weather(lat, lon, past_days):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&past_days={past_days}&forecast_days=1&timezone=Europe/Warsaw"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    j = requests.get(url, timeout=15).json()
    return pd.DataFrame({
        'Data': pd.to_datetime(j['hourly']['time']),
        'Temperatura': j['hourly']['temperature_2m'],
        'Wilgotnosc': j['hourly']['relative_humidity_2m'],
        'Wiatr': j['hourly']['wind_speed_10m'],
    }).set_index('Data').sort_index()


def report(name, s):
    s = s.dropna()
    if s.empty:
        print(f"  {name:<14} BRAK danych")
        return
    span_h = (s.index.max() - s.index.min()).total_seconds() / 3600
    print(f"  {name:<14} n={len(s):>4}  od {s.index.min()}  do {s.index.max()}  (~{span_h/24:.1f} dni)")


if __name__ == "__main__":
    print(f"=== {STATION['name']} (past_days={PAST_DAYS}) ===\n")
    w = fetch_weather(STATION['lat'], STATION['lon'], PAST_DAYS)
    print("POGODA (Open-Meteo):")
    for c in w.columns:
        report(c, w[c])

    print("\nZANIECZYSZCZENIA (GIOS, archiwum getDataBySensor):")
    sess = requests.Session()
    sensors = _first_list(sess.get(
        f"https://api.gios.gov.pl/pjp-api/v1/rest/station/sensors/{STATION['id']}",
        headers=GIOS_HEADERS, timeout=10).json())
    pol = {}
    for sensor in sensors:
        code_key = _pick(sensor.keys(), 'kod')
        if not code_key:
            continue
        code = str(sensor[code_key]).replace(".", "")
        id_key = (_pick(sensor.keys(), 'stanowiska') or _pick(sensor.keys(), 'identyfikator') or 'id')
        sid = sensor.get(id_key)
        s = _fetch_sensor_series(sid, PAST_DAYS, sess)
        if s is not None:
            prev = pol.get(code)
            if prev is None or s.dropna().size > prev.dropna().size:
                pol[code] = s
        time.sleep(0.3)
    for code, s in pol.items():
        report(code, s)

    df = pd.DataFrame(index=w.index)
    for c in WEATHER_COLS:
        df[c] = w[c]
    for code, s in pol.items():
        df[code] = s
    print("\nPRZED interpolacja - realne pokrycie w zlaczonej ramce (os = pogoda):")
    for c in df.columns:
        print(f"  {c:<14} realnych={df[c].notna().sum():>4} / {len(df)}")

    df_i = df.interpolate(method='linear', limit_direction='both').dropna()
    print(f"\nPO interpolacji + dropna(): {df_i.shape[0]} wierszy x {df_i.shape[1]} kolumn")
    if not df_i.empty:
        print(f"  zakres: {df_i.index.min()} .. {df_i.index.max()}")
