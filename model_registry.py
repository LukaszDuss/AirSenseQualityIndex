# -*- coding: utf-8 -*-
"""Rejestr modeli AirSense — metadane, tabela, usuwanie i wybór aktywnego modelu klienta.

Moduł niezależny od TensorFlow/Streamlita (operuje na plikach .npy i JSON). Ładowanie
samego modelu Keras zostaje po stronie aplikacji.
"""
import json
import os
import re
from datetime import datetime

import numpy as np

import config


def _station_slug(station_label):
    """Krótki slug ze nazwy stacji (np. 'Warszawa - Marszałkowska' -> 'warszawa')."""
    word = str(station_label).lower().split()[0]
    return re.sub(r"[^a-z0-9]", "", word) or "stacja"


def build_model_name(station_label, time_steps, horizon, epochs_run, lstm_units, when=None):
    """Nazwa pliku modelu: okno, horyzont, epoki (faktyczne), neurony, data.

    Przykład: aq_zabrze_o48_h24_e37_n64_20260604
    """
    when = when or datetime.now()
    date_part = when.strftime("%Y%m%d")
    return (f"aq_{_station_slug(station_label)}_o{int(time_steps)}_h{int(horizon)}"
            f"_e{int(epochs_run)}_n{int(lstm_units)}_{date_part}")


def unique_model_name(base):
    """Zwraca `base` lub `base_2`, `base_3`… gdy plik .keras już istnieje."""
    if not os.path.exists(model_path(base)):
        return base
    for i in range(2, 100):
        candidate = f"{base}_{i}"
        if not os.path.exists(model_path(candidate)):
            return candidate
    return f"{base}_{datetime.now().strftime('%H%M%S')}"


def model_path(name):
    return os.path.join(config.MODELS_DIR, f"{name}.keras")


def meta_path(name):
    return os.path.join(config.MODELS_DIR, f"{name}_meta.npy")


def list_models():
    """Nazwy wszystkich zapisanych modeli (po pliku .keras)."""
    if not os.path.isdir(config.MODELS_DIR):
        return []
    return sorted(f[:-len(".keras")] for f in os.listdir(config.MODELS_DIR)
                  if f.endswith(".keras"))


def save_meta(name, meta):
    """Zapisuje słownik metadanych modelu."""
    np.save(meta_path(name), np.array(meta, dtype=object))


def load_meta(name):
    """Metadane modelu jako dict albo None dla starego/niekompatybilnego formatu."""
    try:
        meta = np.load(meta_path(name), allow_pickle=True).item()
    except (ValueError, OSError, FileNotFoundError, AttributeError):
        return None
    if not isinstance(meta, dict) or "feature_columns" not in meta:
        return None
    return meta


def split_models():
    """(zgodne, niezgodne) — niezgodne = stary format, wymaga retreningu."""
    valid, invalid = [], []
    for name in list_models():
        (valid if load_meta(name) else invalid).append(name)
    return valid, invalid


def models_table():
    """Lista wierszy do tabeli zarządzania (tylko zgodne modele)."""
    rows = []
    for name in list_models():
        meta = load_meta(name)
        if not meta:
            continue
        m = meta.get("metrics", {})
        rows.append({
            "model": name,
            "stacja": meta.get("station", "—"),
            "okno_h": meta.get("time_steps"),
            "horyzont_h": meta.get("horizon"),
            "epoki": meta.get("epochs_run", "—"),
            "neurony": meta.get("lstm_units", "—"),
            "data": meta.get("trained_at", "—"),
            "wierszy": meta.get("n_rows"),
            "cechy": len(meta.get("feature_columns", [])),
            "MAE": m.get("mae"),
            "MSE": m.get("mse"),
            "RMSE": m.get("rmse"),
            "R2": m.get("r2"),
        })
    return rows


def delete_model(name):
    """Usuwa pliki modelu i czyści go z list aktywnych i best."""
    for p in (model_path(name), meta_path(name)):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
    active = _load_active()
    changed = {k: v for k, v in active.items() if v != name}
    if changed != active:
        _save_active(changed)
    best = _load_best()
    changed_b = {k: v for k, v in best.items() if v != name}
    if changed_b != best:
        _save_best(changed_b)
    llm_rec = _load_llm_recommended()
    changed_l = {
        k: v for k, v in llm_rec.items()
        if _llm_entry_model(v) != name
    }
    if changed_l != llm_rec:
        _save_llm_recommended(changed_l)


def models_for_station(station_id):
    """Modele wytrenowane dla danej stacji (station_id w meta)."""
    sid = int(station_id)
    out = []
    for name in list_models():
        meta = load_meta(name)
        if meta and int(meta.get("station_id", -1)) == sid:
            out.append((name, meta))
    return sorted(out, key=lambda x: x[1].get("trained_at", ""), reverse=True)


def delete_invalid_models():
    """Usuwa modele niezgodne (stary format) oraz osierocone pliki *_meta.npy. Zwraca listę nazw."""
    deleted = []
    _, invalid = split_models()
    for name in invalid:
        delete_model(name)
        deleted.append(name)
    if os.path.isdir(config.MODELS_DIR):
        for fname in os.listdir(config.MODELS_DIR):
            if not fname.endswith("_meta.npy"):
                continue
            name = fname[: -len("_meta.npy")]
            if load_meta(name) is None or not os.path.exists(model_path(name)):
                try:
                    os.remove(meta_path(name))
                    if name not in deleted:
                        deleted.append(name)
                except OSError:
                    pass
    return deleted


# --- Aktywny model klienta (per stacja) ---
def _load_active():
    try:
        with open(config.ACTIVE_MODELS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_active(mapping):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    tmp = config.ACTIVE_MODELS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    os.replace(tmp, config.ACTIVE_MODELS_FILE)


def get_active(station_id):
    """Nazwa aktywnego modelu dla stacji lub None. Weryfikuje, że model wciąż istnieje."""
    name = _load_active().get(str(station_id))
    if name and load_meta(name) is not None:
        return name
    return None


def set_active(station_id, name):
    active = _load_active()
    active[str(station_id)] = name
    _save_active(active)


# --- Najlepszy model (⭐) per stacja ---
def _load_best():
    try:
        with open(config.BEST_MODELS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_best(mapping):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    tmp = config.BEST_MODELS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    os.replace(tmp, config.BEST_MODELS_FILE)


def get_best(station_id):
    name = _load_best().get(str(station_id))
    if name and load_meta(name) is not None:
        return name
    return None


def set_best(station_id, name):
    best = _load_best()
    best[str(station_id)] = name
    _save_best(best)


# --- Rekomendacja LLM (⭐) dla widoku klienta ---
def _llm_entry_model(entry):
    if isinstance(entry, dict):
        return entry.get("model")
    if isinstance(entry, str):
        return entry
    return None


def _load_llm_recommended():
    try:
        with open(config.LLM_RECOMMENDED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_llm_recommended(mapping):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    tmp = config.LLM_RECOMMENDED_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    os.replace(tmp, config.LLM_RECOMMENDED_FILE)


def get_llm_recommended(station_id):
    name = _llm_entry_model(_load_llm_recommended().get(str(station_id)))
    if name and load_meta(name) is not None:
        return name
    return None


def set_llm_recommended(station_id, name):
    sid = str(station_id)
    snapshot = sorted(n for n, _ in models_for_station(station_id))
    rec = _load_llm_recommended()
    rec[sid] = {
        "model": name,
        "models_snapshot": snapshot,
        "recommended_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    _save_llm_recommended(rec)


def llm_recommendation_stale(station_id):
    """True gdy lista modeli stacji różni się od zapisu przy ostatniej rekomendacji LLM."""
    entry = _load_llm_recommended().get(str(station_id))
    if not isinstance(entry, dict):
        return False
    snapshot = entry.get("models_snapshot")
    if not snapshot:
        return False
    current = sorted(n for n, _ in models_for_station(station_id))
    return current != list(snapshot)


def get_llm_recommendation_meta(station_id):
    """Metadane ostatniej rekomendacji LLM (model, data, snapshot) lub None."""
    entry = _load_llm_recommended().get(str(station_id))
    if not isinstance(entry, dict):
        return None
    model = entry.get("model")
    if not model or load_meta(model) is None:
        return None
    return entry


def effective_llm_recommended(station_id):
    """Rekomendowany model tylko gdy lista nie zmieniła się od zapisu snapshotu."""
    if llm_recommendation_stale(station_id):
        return None
    return get_llm_recommended(station_id)


def get_llm_evaluation(model_name):
    meta = load_meta(model_name)
    if not meta:
        return None, None
    return meta.get("llm_evaluation"), meta.get("llm_evaluated_at")


def save_llm_evaluation(model_name, text):
    """Zapisuje ocenę metryk od LLM w metadanych modelu (bez ponownego generowania)."""
    meta = load_meta(model_name)
    if not meta:
        return False
    meta["llm_evaluation"] = text
    meta["llm_evaluated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_meta(model_name, meta)
    return True


def get_llm_update_comparison(model_name):
    """Porównanie LLM stary→nowy z aktualizacji (jeśli zapisane)."""
    meta = load_meta(model_name)
    if not meta:
        return None, None, None
    return (
        meta.get("llm_update_comparison"),
        meta.get("llm_update_compared_at"),
        meta.get("llm_update_source"),
    )


def save_llm_update_comparison(new_model_name, source_name, text):
    """Zapisuje porównanie LLM po aktualizacji w metadanych nowego modelu."""
    meta = load_meta(new_model_name)
    if not meta:
        return False
    meta["llm_update_comparison"] = text
    meta["llm_update_compared_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta["llm_update_source"] = source_name
    save_meta(new_model_name, meta)
    return True


def model_quality_score(metrics):
    """Jeden wynik jakości — wyżej = lepiej (MAE/RMSE↓, R²↑ na zbiorze testowym)."""
    if not metrics:
        return float("-inf")
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    r2 = metrics.get("r2")
    if mae is None and r2 is None:
        return float("-inf")
    mae_v = float(mae) if mae is not None else 50.0
    rmse_v = float(rmse) if rmse is not None else mae_v
    r2_v = float(r2) if r2 is not None else 0.0
    return r2_v * 50.0 - mae_v * 2.0 - rmse_v * 1.0


def compute_best_model_name(station_id):
    """Nazwa najlepszego modelu wg złożonego wyniku (bez zapisu)."""
    candidates = models_for_station(station_id)
    scored = []
    for name, meta in candidates:
        metrics = meta.get("metrics") or {}
        if metrics.get("mae") is None and metrics.get("r2") is None:
            continue
        scored.append((name, model_quality_score(metrics)))
    if not scored:
        return None
    return max(scored, key=lambda x: x[1])[0]


def refresh_best_for_station(station_id):
    """Przelicza i zapisuje najlepszy model (MAE, RMSE, R²). Wywoływać przy liście modeli."""
    best_name = compute_best_model_name(station_id)
    if best_name:
        set_best(station_id, best_name)
    return best_name


def pick_best_by_mae(station_id):
    """Kompatybilność wsteczna — przelicza najlepszy wg metryk złożonych."""
    return refresh_best_for_station(station_id)
