# -*- coding: utf-8 -*-
"""Wyczyść bazę i pobierz dane GIOŚ + pogodę od zera dla wszystkich stacji z katalogu."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_fetch
import data_store
import stations_settings


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    catalog = stations_settings.all_stations_catalog()
    print(f"Stacji w katalogu: {len(catalog)}\n")
    for name, meta in catalog.items():
        sid, lat, lon = meta["id"], meta["lat"], meta["lon"]
        print(f"=== {name!r} (id {sid}) ===")
        data_store.clear_store(sid, lat, lon)
        df = data_fetch.update_and_load(sid, lat, lon)
        store = data_store.load_store(sid)
        cov = data_store.get_coverage(store)
        for r in cov:
            if r["typ"].startswith("zanieczyszczenia"):
                print(
                    f"  {r['parametr']}: {r['punkty']} pkt "
                    f"({r['od']} — {r['do']})"
                )
        print(f"  processed wierszy: {len(df)}\n")


if __name__ == "__main__":
    main()
