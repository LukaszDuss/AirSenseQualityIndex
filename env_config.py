# -*- coding: utf-8 -*-
"""Lekki loader pliku .env (bez zależności od python-dotenv).

Wczytuje pary KLUCZ=WARTOSC z pliku .env w katalogu projektu i ustawia je w
os.environ (o ile nie są już ustawione w środowisku). Obsługuje spacje wokół '='
oraz wartości w cudzysłowach pojedynczych/podwójnych.
"""
import os

_DEF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_env(path=_DEF_PATH, override=False):
    """Wczytuje .env do os.environ. Zwraca liczbę wczytanych zmiennych."""
    if not os.path.exists(path):
        return 0
    loaded = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.lower().startswith("export "):
                    line = line[7:]
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'").strip()
                if not key:
                    continue
                if override or key not in os.environ:
                    os.environ[key] = value
                    loaded += 1
    except OSError:
        return loaded
    return loaded
