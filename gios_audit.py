# -*- coding: utf-8 -*-
"""Audyt kompletności stacji GIOŚ — sensory EAQI i pokrycie archiwum 30 dni."""
import json
import os
import time
from datetime import datetime

import config
import data_fetch
import stations_settings
import ui_icons as mi

AUDIT_CACHE_FILE = os.path.join("settings", "gios_audit_cache.json")


def _audit_yes_no(flag):
    return mi.YES if flag else mi.NO
REQUIRED = config.REQUIRED_EAQI_POLLUTANTS


def _empty_cache():
    return {
        "sensor_scan_at": None,
        "data_scan_at": None,
        "stations": {},
    }


def load_audit_cache():
    if not os.path.exists(AUDIT_CACHE_FILE):
        return _empty_cache()
    try:
        with open(AUDIT_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("stations", {})
        return data
    except (OSError, json.JSONDecodeError):
        return _empty_cache()


def save_audit_cache(cache):
    os.makedirs(os.path.dirname(AUDIT_CACHE_FILE), exist_ok=True)
    tmp = AUDIT_CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, AUDIT_CACHE_FILE)


def audit_sensors(station_id):
    """Czy stacja ma sensory dla wszystkich parametrów EAQI (bez pobierania szeregów)."""
    sensors = data_fetch.list_gios_sensors(station_id)
    codes = {c for c, _ in sensors}
    present = [p for p in REQUIRED if p in codes]
    missing = [p for p in REQUIRED if p not in codes]
    return {
        "station_id": int(station_id),
        "param_codes": sorted(codes),
        "eaqi_present": present,
        "eaqi_missing": missing,
        "eaqi_count": len(present),
        "full_sensors": len(missing) == 0 and len(codes) > 0,
    }


def audit_archival_data(station_id):
    """Pokrycie archiwum 30 dni (wolne — jedno wywołanie fetch_pollutants_raw na stację)."""
    pollutants = data_fetch.fetch_pollutants_raw(station_id, mode="archival")
    points = {p: len(pollutants[p]) if p in pollutants and pollutants[p] is not None else 0
              for p in REQUIRED}
    min_pts = min(points.values()) if points else 0
    full = all(points.get(p, 0) >= config.MIN_ARCHIVAL_POINTS for p in REQUIRED)
    return {
        "station_id": int(station_id),
        "points": points,
        "min_points": int(min_pts),
        "full_data_30d": full,
    }


def _station_meta(station_row):
    return {
        "name": station_row.get("name", ""),
        "city": station_row.get("city", ""),
        "lat": station_row.get("lat"),
        "lon": station_row.get("lon"),
    }


def scan_all_sensors(station_list=None, on_progress=None):
    """Szybki skan sensorów dla listy stacji (domyślnie cały katalog GIOŚ)."""
    station_list = station_list or stations_settings.fetch_gios_station_list()
    cache = load_audit_cache()
    cache["sensor_scan_at"] = datetime.now().isoformat(timespec="seconds")
    total = len(station_list)
    for i, row in enumerate(station_list):
        sid = row["id"]
        rec = audit_sensors(sid)
        rec.update(_station_meta(row))
        entry = cache["stations"].setdefault(str(sid), {})
        entry.update(rec)
        if on_progress:
            on_progress(i + 1, total, row.get("name", sid))
        time.sleep(0.08)
    save_audit_cache(cache)
    return cache


def scan_data_for_stations(station_ids, station_lookup=None, on_progress=None):
    """Skan archiwum 30 dni — tylko podane ID (np. stacje z pełnymi sensorami)."""
    station_lookup = station_lookup or {}
    cache = load_audit_cache()
    cache["data_scan_at"] = datetime.now().isoformat(timespec="seconds")
    ids = [int(x) for x in station_ids]
    total = len(ids)
    for i, sid in enumerate(ids):
        rec = audit_archival_data(sid)
        entry = cache["stations"].setdefault(str(sid), {"station_id": sid})
        entry["points"] = rec["points"]
        entry["min_points"] = rec["min_points"]
        entry["full_data_30d"] = rec["full_data_30d"]
        meta = station_lookup.get(sid, {})
        if meta:
            entry.update(_station_meta(meta))
        if on_progress:
            on_progress(i + 1, total, meta.get("name", sid))
    save_audit_cache(cache)
    return cache


def scan_data_full_sensor_stations(on_progress=None):
    """Archiwum 30 dni tylko dla stacji z pełnym kompletem sensorów EAQI."""
    cache = load_audit_cache()
    ids = [
        int(sid) for sid, e in cache.get("stations", {}).items()
        if e.get("full_sensors")
    ]
    if not ids:
        return cache
    lookup = {
        int(sid): e for sid, e in cache["stations"].items()
    }
    return scan_data_for_stations(ids, lookup, on_progress)


def audit_table_rows(cache=None):
    """Wiersze do tabeli w UI."""
    cache = cache or load_audit_cache()
    rows = []
    for sid, e in sorted(cache.get("stations", {}).items(), key=lambda x: int(x[0])):
        pts = e.get("points") or {}
        rows.append({
            "id": int(sid),
            "nazwa": e.get("name", ""),
            "miasto": e.get("city", ""),
            "sensory_EAQI": f"{e.get('eaqi_count', 0)}/5",
            "pełne_sensory": _audit_yes_no(e.get("full_sensors")),
            "brakuje": ", ".join(e.get("eaqi_missing") or []) or "—",
            "PM25": pts.get("PM25", "—"),
            "PM10": pts.get("PM10", "—"),
            "NO2": pts.get("NO2", "—"),
            "O3": pts.get("O3", "—"),
            "SO2": pts.get("SO2", "—"),
            "min_pkt_30d": e.get("min_points", "—"),
            "pełne_dane_30d": _audit_yes_no(e.get("full_data_30d")),
        })
    return rows


def stations_with_full_data(cache=None):
    """Lista metadanych stacji z pełnymi sensorami i danymi 30d."""
    cache = cache or load_audit_cache()
    out = []
    for sid, e in cache.get("stations", {}).items():
        if e.get("full_sensors") and e.get("full_data_30d"):
            out.append({
                "id": int(sid),
                "name": e.get("name", ""),
                "city": e.get("city", ""),
                "lat": e.get("lat"),
                "lon": e.get("lon"),
            })
    return sorted(out, key=lambda x: (x.get("city", ""), x["name"]))


def get_station_audit(station_id, cache=None):
    cache = cache or load_audit_cache()
    return cache.get("stations", {}).get(str(int(station_id)))


def format_station_badge(station_id, cache=None):
    """Krótka etykieta do listy wyboru stacji."""
    e = get_station_audit(station_id, cache)
    if not e:
        return ""
    parts = []
    if e.get("eaqi_count") is not None:
        parts.append(f"EAQI {e.get('eaqi_count', 0)}/5")
    if e.get("full_sensors"):
        parts.append("sensory OK")
    if e.get("full_data_30d"):
        parts.append("dane 30d OK")
    return " | ".join(parts) if parts else ""
