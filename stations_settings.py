# -*- coding: utf-8 -*-
"""Konfiguracja stacji GIOŚ — włączone w UI, lista z API, ustawienia aplikacji."""
import json
import os

import requests

import data_store
from data_fetch import GIOS_HEADERS, _first_list, _pick

SETTINGS_FILE = os.path.join("settings", "app_settings.json")

# Domyślne stacje projektu (Grupa 5)
DEFAULT_STATIONS = {
    "Zabrze": {"id": 550, "lat": 50.3124, "lon": 18.7711},
    "Warszawa - Marszałkowska": {"id": 544, "lat": 52.2287, "lon": 21.0122},
    "Kraków - Al. Krasińskiego": {"id": 400, "lat": 50.0574, "lon": 19.9261},
    "Gdańsk - Lektykarska": {"id": 738, "lat": 54.3497, "lon": 18.6548},
    "Wrocław - Korzeniowskiego": {"id": 265, "lat": 51.1294, "lon": 17.0292},
}


OPENAI_MODEL_CHOICES = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5.5",
)


def _default_settings():
    return {
        "enabled_station_ids": [m["id"] for m in DEFAULT_STATIONS.values()],
        "custom_stations": {},
        "removed_station_ids": [],
        "openai_api_key": "",
        "openai_model": "gpt-4o",
    }


def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return _default_settings()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = _default_settings()
        base.update({k: data.get(k, base[k]) for k in base})
        base["custom_stations"] = data.get("custom_stations", {}) or {}
        base["removed_station_ids"] = data.get("removed_station_ids", []) or []
        return base
    except (OSError, json.JSONDecodeError):
        return _default_settings()


def save_settings(settings):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SETTINGS_FILE)


def _removed_ids(settings=None):
    settings = settings or load_settings()
    return set(int(x) for x in settings.get("removed_station_ids", []))


def all_stations_catalog(settings=None):
    """Słownik name -> {id, lat, lon} = domyślne + custom, bez usuniętych."""
    settings = settings or load_settings()
    removed = _removed_ids(settings)
    catalog = {}
    for name, meta in DEFAULT_STATIONS.items():
        if int(meta["id"]) not in removed:
            catalog[name] = dict(meta)
    for name, meta in (settings.get("custom_stations") or {}).items():
        if int(meta["id"]) not in removed:
            catalog[name] = dict(meta)
    return catalog


def configured_station_ids():
    """Wszystkie ID stacji w katalogu (włączone i wyłączone, bez usuniętych)."""
    return {int(m["id"]) for m in all_stations_catalog().values()}


def enabled_stations():
    """Stacje widoczne w UI (klient + admin)."""
    settings = load_settings()
    enabled_ids = set(int(x) for x in settings.get("enabled_station_ids", []))
    out = {}
    for name, meta in all_stations_catalog().items():
        if int(meta["id"]) in enabled_ids:
            out[name] = meta
    return out


def station_city_label(station_name):
    """Skrót do miasta (np. 'Warszawa - Marszałkowska' → 'Warszawa')."""
    s = str(station_name).strip()
    if " - " in s:
        return s.split(" - ", 1)[0].strip()
    return s


def client_station_choices(stations):
    """Etykiety selectboxa klienta (miasto) → pełna nazwa stacji w katalogu."""
    counts = {}
    choices = []
    for full_name in stations:
        city = station_city_label(full_name)
        counts[city] = counts.get(city, 0) + 1
    seen = {}
    for full_name in stations:
        city = station_city_label(full_name)
        if counts[city] > 1:
            seen[city] = seen.get(city, 0) + 1
            label = f"{city} ({seen[city]})"
        else:
            label = city
        choices.append((label, full_name))
    return choices


def set_station_enabled(station_id, enabled):
    settings = load_settings()
    ids = set(int(x) for x in settings.get("enabled_station_ids", []))
    sid = int(station_id)
    if enabled:
        ids.add(sid)
    else:
        ids.discard(sid)
    settings["enabled_station_ids"] = sorted(ids)
    save_settings(settings)


def _unique_catalog_name(base_name, station_id, catalog):
    """Unikalna nazwa w katalogu (gdy ta sama nazwa, inne ID)."""
    name = str(base_name).strip()
    if name not in catalog or int(catalog[name]["id"]) == int(station_id):
        return name
    alt = f"{name} (id {station_id})"
    if alt not in catalog:
        return alt
    return f"{name} #{station_id}"


def add_custom_station(name, station_id, lat, lon):
    settings = load_settings()
    removed = set(int(x) for x in settings.get("removed_station_ids", []))
    sid = int(station_id)
    removed.discard(sid)
    settings["removed_station_ids"] = sorted(removed)

    custom = settings.get("custom_stations", {}) or {}
    catalog = all_stations_catalog(settings)
    catalog = {k: v for k, v in catalog.items() if int(v["id"]) != sid}
    label = _unique_catalog_name(name, sid, catalog)
    custom = {k: v for k, v in custom.items() if int(v["id"]) != sid}
    custom[label] = {"id": sid, "lat": float(lat), "lon": float(lon)}
    settings["custom_stations"] = custom

    ids = set(int(x) for x in settings.get("enabled_station_ids", []))
    ids.add(sid)
    settings["enabled_station_ids"] = sorted(ids)
    save_settings(settings)


def remove_station(station_id, delete_data=True):
    """Usuwa stację z aplikacji (katalog + włączenie); opcjonalnie plik danych."""
    settings = load_settings()
    sid = int(station_id)
    removed = set(int(x) for x in settings.get("removed_station_ids", []))
    removed.add(sid)
    settings["removed_station_ids"] = sorted(removed)

    custom = settings.get("custom_stations", {}) or {}
    settings["custom_stations"] = {
        k: v for k, v in custom.items() if int(v["id"]) != sid
    }

    ids = set(int(x) for x in settings.get("enabled_station_ids", []))
    ids.discard(sid)
    settings["enabled_station_ids"] = sorted(ids)
    save_settings(settings)

    if delete_data:
        data_store.clear_store(sid)


def _parse_gios_station_row(row):
    if not isinstance(row, dict):
        return None
    sid = row.get("Identyfikator stacji") or row.get("id")
    name = row.get("Nazwa stacji") or row.get("name") or row.get("stationName")
    lat = row.get("WGS84 φ N") or row.get("Gęograficzna szerokość") or row.get("lat")
    lon = row.get("WGS84 λ E") or row.get("Gęograficzna długość") or row.get("lon")
    if sid is None or not name:
        id_k = _pick(row.keys(), "identyfikator", "stacji")
        name_k = _pick(row.keys(), "nazwa", "stacji")
        lat_k = _pick(row.keys(), "szerok") or _pick(row.keys(), "wgs84")
        lon_k = _pick(row.keys(), "dlug") or _pick(row.keys(), "wgs84")
        if id_k:
            sid = row.get(id_k)
        if name_k:
            name = row.get(name_k)
        if lat_k and lat is None:
            lat = row.get(lat_k)
        if lon_k and lon is None:
            lon = row.get(lon_k)
    city = row.get("Nazwa miasta") or row.get("Miasto") or ""
    if sid and name:
        try:
            return {
                "id": int(sid),
                "name": str(name).strip(),
                "city": str(city).strip() if city else "",
                "lat": float(str(lat).replace(",", ".")) if lat is not None else None,
                "lon": float(str(lon).replace(",", ".")) if lon is not None else None,
            }
        except (TypeError, ValueError):
            return None
    return None


def fetch_gios_station_list():
    """Pełna lista stacji GIOŚ (findAll + paginacja)."""
    import time
    out = []
    seen = set()
    page = 0
    total_pages = 1
    try:
        while page < total_pages:
            url = f"https://api.gios.gov.pl/pjp-api/v1/rest/station/findAll?page={page}"
            r = requests.get(url, headers=GIOS_HEADERS, timeout=25)
            if r.status_code != 200:
                break
            payload = r.json()
            if page == 0:
                total_pages = int(payload.get("totalPages") or 1)
            for row in _first_list(payload):
                parsed = _parse_gios_station_row(row)
                if not parsed or parsed["id"] in seen:
                    continue
                if parsed.get("lat") is None or parsed.get("lon") is None:
                    continue
                seen.add(parsed["id"])
                out.append(parsed)
            page += 1
            time.sleep(0.15)
    except Exception:
        pass
    out.sort(key=lambda x: (x.get("city", ""), x["name"]))
    return out


def resolve_openai_key():
    """Klucz z ustawień (jeśli wpisany) lub ze zmiennej / .env."""
    key = (load_settings().get("openai_api_key") or "").strip()
    if key:
        return key
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def resolve_openai_model():
    """Model OpenAI z ustawień aplikacji."""
    m = (load_settings().get("openai_model") or "gpt-4o").strip()
    return m if m in OPENAI_MODEL_CHOICES else OPENAI_MODEL_CHOICES[0]
