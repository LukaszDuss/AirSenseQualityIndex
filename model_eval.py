# -*- coding: utf-8 -*-
"""Ewaluacja modeli — metryki, serie testowe i wykresy Plotly."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from data_fetch import prepare_xy
import air_quality
import baselines


def _as_1d(a):
    return np.asarray(a, dtype=float).reshape(-1)


def compute_metrics(y_true, y_pred):
    """Pełny zestaw metryk regresji ASQI (0–100)."""
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    if len(yt) == 0:
        return {}
    err = yp - yt
    abs_err = np.abs(err)
    mse = float(mean_squared_error(yt, yp))
    metrics = {
        "n_samples": int(len(yt)),
        "mae": float(mean_absolute_error(yt, yp)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else float("nan"),
        "median_ae": float(median_absolute_error(yt, yp)),
        "max_error": float(max_error(yt, yp)),
        "max_abs_error": float(np.max(abs_err)),
        "bias": float(np.mean(err)),
        "std_residual": float(np.std(err)),
        "explained_variance": float(explained_variance_score(yt, yp))
        if len(yt) > 1 else float("nan"),
    }
    mask = np.abs(yt) > 1.0
    if mask.any():
        metrics["mape_pct"] = float(np.mean(abs_err[mask] / np.abs(yt[mask])) * 100.0)
    metrics["p90_abs_error"] = float(np.percentile(abs_err, 90))
    metrics["p95_abs_error"] = float(np.percentile(abs_err, 95))
    return metrics


def metrics_by_horizon(y_true_2d, y_pred_2d):
    """MAE / RMSE / R² per krok horyzontu (wiersze = sekwencje, kolumny = h)."""
    yt = np.asarray(y_true_2d, dtype=float)
    yp = np.asarray(y_pred_2d, dtype=float)
    if yt.ndim != 2:
        return []
    h = yt.shape[1]
    rows = []
    for i in range(h):
        m = compute_metrics(yt[:, i], yp[:, i])
        rows.append({
            "krok_h": i + 1,
            "mae": m.get("mae"),
            "rmse": m.get("rmse"),
            "r2": m.get("r2"),
        })
    return rows


def _split_test_arrays(meta, df_proc):
    """Buduje X_test, y_test w skali 0–1 oraz parametry denormalizacji."""
    feats = list(meta["feature_columns"])
    missing = [c for c in feats + [air_quality.GIOS_AQI_COL] if c not in df_proc.columns]
    if missing:
        return None, f"Brak kolumn: {', '.join(missing)}"

    t_steps = int(meta["time_steps"])
    horizon = int(meta["horizon"])
    feat_min = np.asarray(meta["feat_min"], dtype=float)
    feat_max = np.asarray(meta["feat_max"], dtype=float)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    y_min, y_max = float(meta["y_min"]), float(meta["y_max"])
    y_range = (y_max - y_min) or 1.0

    feat_scaled = (df_proc[feats].values - feat_min) / feat_range
    target_scaled = (df_proc[air_quality.GIOS_AQI_COL].values - y_min) / y_range
    X, y = prepare_xy(feat_scaled, target_scaled, t_steps, horizon)
    if len(X) < 5:
        return None, "Za mało sekwencji"

    total = len(X)
    tr = int(total * 0.8)
    vl = int(total * 0.1)
    return {
        "X_test": X[tr + vl:],
        "y_test": y[tr + vl:],
        "y_min": y_min,
        "y_max": y_max,
        "y_range": y_range,
        "horizon": horizon,
        "split": {"train": tr, "val": vl, "test": total - tr - vl, "total_seq": total},
    }, None


def baseline_metrics_for_split(meta, df_proc, split):
    """Baseline'y na tym samym zbiorze testowym co LSTM (indeksy sekwencji)."""
    if not split or air_quality.GIOS_AQI_COL not in df_proc.columns:
        return {}
    test_start = int(split.get("train", 0)) + int(split.get("val", 0))
    test_end = int(split.get("total_seq", 0))
    return baselines.evaluate_baselines(
        df_proc,
        int(meta["time_steps"]),
        int(meta["horizon"]),
        test_start,
        test_end,
    )


def run_evaluation(model_name, meta, df_proc):
    """Zwraca dict z metrykami, seriami 2D/1D i ewentualnym błędem."""
    from tensorflow.keras.models import load_model
    import model_registry

    pack, err = _split_test_arrays(meta, df_proc)
    if err:
        return {"error": err}

    model = load_model(model_registry.model_path(model_name), compile=False)
    y_pred = model.predict(pack["X_test"], verbose=0)
    y_test = pack["y_test"]
    y_range, y_min = pack["y_range"], pack["y_min"]
    y_test_real = y_test * y_range + y_min
    y_pred_real = y_pred * y_range + y_min

    bundle = _bundle_from_arrays(
        y_test_real, y_pred_real, pack["horizon"], pack["split"],
    )
    bundle["baseline_metrics"] = baseline_metrics_for_split(meta, df_proc, pack["split"])
    return bundle


def bundle_from_meta(meta):
    """Odtwarza pakiet ewaluacji z zapisanych eval_y_* (stare modele)."""
    yt = meta.get("eval_y_true")
    yp = meta.get("eval_y_pred")
    if not yt or not yp:
        return None
    bl = meta.get("baseline_metrics")
    horizon = int(meta.get("horizon", 1))
    yt_a = np.asarray(yt, dtype=float)
    yp_a = np.asarray(yp, dtype=float)
    n = len(yt_a)
    if horizon > 1 and n % horizon == 0:
        yt_2d = yt_a.reshape(-1, horizon)
        yp_2d = yp_a.reshape(-1, horizon)
    else:
        yt_2d = yt_a.reshape(-1, 1)
        yp_2d = yp_a.reshape(-1, 1)
        horizon = 1
    split = meta.get("eval_split") or {}
    out = _bundle_from_arrays(yt_2d, yp_2d, horizon, split)
    if bl:
        out["baseline_metrics"] = bl
    return out


def _bundle_from_arrays(y_test_real, y_pred_real, horizon, split):
    yt_2d = np.asarray(y_test_real, dtype=float)
    yp_2d = np.asarray(y_pred_real, dtype=float)
    if yt_2d.ndim == 1:
        yt_2d = yt_2d.reshape(-1, 1)
        yp_2d = yp_2d.reshape(-1, 1)
    yt_flat = yt_2d.reshape(-1)
    yp_flat = yp_2d.reshape(-1)
    residuals = yp_flat - yt_flat
    return {
        "metrics": compute_metrics(yt_flat, yp_flat),
        "metrics_by_horizon": metrics_by_horizon(yt_2d, yp_2d) if yt_2d.shape[1] > 1 else [],
        "y_true": yt_flat,
        "y_pred": yp_flat,
        "y_true_2d": yt_2d,
        "y_pred_2d": yp_2d,
        "residuals": residuals,
        "horizon": int(horizon),
        "split": split,
    }


def resolve_evaluation(model_name, meta, df_proc):
    """Meta cache → przeliczenie na bieżących danych."""
    cached = bundle_from_meta(meta)
    if cached:
        return cached
    if df_proc is None or df_proc.empty:
        return {"error": "Brak zapisanej ewaluacji i brak danych processed."}
    return run_evaluation(model_name, meta, df_proc)


def training_history_figure(history):
    """Krzywe loss / val_loss z treningu."""
    if not history:
        return None
    loss = history.get("loss") or []
    val_loss = history.get("val_loss") or []
    if not loss:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss, mode="lines+markers", name="loss (train)"))
    if val_loss:
        fig.add_trace(go.Scatter(y=val_loss, mode="lines+markers", name="val_loss"))
    fig.update_layout(
        template="plotly_dark",
        title="Historia uczenia (epoki)",
        xaxis_title="Epoka",
        yaxis_title="MSE (skala znormalizowana)",
        height=320,
    )
    return fig


def build_chart_figures(ev, meta):
    """Słownik nazwa → Figure Plotly."""
    if not ev or "error" in ev:
        return {}
    yt = ev["y_true"]
    yp = ev["y_pred"]
    res = ev["residuals"]
    n = len(yt)
    x = list(range(n))
    figs = {}

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=x, y=yt, mode="lines", name="Test — prawda", line=dict(width=1.5)))
    fig_ts.add_trace(go.Scatter(x=x, y=yp, mode="lines", name="Test — prognoza", line=dict(width=1.2)))
    fig_ts.update_layout(
        template="plotly_dark", title="Szereg czasowy (zbiór testowy, spłaszczone)",
        xaxis_title="Indeks punktu testowego", yaxis_title=f"{air_quality.ASQI_SHORT} (AirSenseQualityIndex)",
        height=360,
    )
    figs["Szereg czasowy"] = fig_ts

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(x=yt, y=yp, mode="markers", marker=dict(size=4, opacity=0.6),
                                name="punkty"))
    lo, hi = float(np.min(yt)), float(np.max(yt))
    fig_sc.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="idealna 1:1",
                                line=dict(dash="dash", color="#94a3b8")))
    fig_sc.update_layout(
        template="plotly_dark", title="Prawda vs prognoza",
        xaxis_title="Prawda", yaxis_title="Prognoza", height=360,
    )
    figs["Rozrzut 1:1"] = fig_sc

    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=x, y=res, mode="lines", name="residuum"))
    fig_res.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig_res.update_layout(
        template="plotly_dark", title="Residua w czasie (prognoza − prawda)",
        xaxis_title="Indeks", yaxis_title="Błąd", height=320,
    )
    figs["Residua"] = fig_res

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=res, nbinsx=30, name="residua"))
    fig_hist.update_layout(
        template="plotly_dark", title="Histogram residuów",
        xaxis_title="Błąd", height=320,
    )
    figs["Histogram residuów"] = fig_hist

    fig_ae = go.Figure()
    fig_ae.add_trace(go.Histogram(x=np.abs(res), nbinsx=30, name="|błąd|"))
    fig_ae.update_layout(
        template="plotly_dark", title="Histogram |błędu bezwzględnego|",
        xaxis_title="|błąd|", height=320,
    )
    figs["Histogram |błąd|"] = fig_ae

    fig_cdf = go.Figure()
    sorted_ae = np.sort(np.abs(res))
    cdf_y = np.arange(1, len(sorted_ae) + 1) / len(sorted_ae)
    fig_cdf.add_trace(go.Scatter(x=sorted_ae, y=cdf_y, mode="lines", name="CDF |błąd|"))
    fig_cdf.update_layout(
        template="plotly_dark", title="CDF błędu bezwzględnego",
        xaxis_title="|błąd|", yaxis_title="Ułamek próby", height=320,
    )
    figs["CDF |błąd|"] = fig_cdf

    by_h = ev.get("metrics_by_horizon") or []
    if by_h:
        steps = [r["krok_h"] for r in by_h]
        fig_h = make_subplots(rows=1, cols=3, subplot_titles=("MAE", "RMSE", "R²"))
        fig_h.add_trace(go.Bar(x=steps, y=[r["mae"] for r in by_h], name="MAE"), row=1, col=1)
        fig_h.add_trace(go.Bar(x=steps, y=[r["rmse"] for r in by_h], name="RMSE"), row=1, col=2)
        fig_h.add_trace(go.Bar(x=steps, y=[r["r2"] for r in by_h], name="R²"), row=1, col=3)
        fig_h.update_layout(template="plotly_dark", title="Metryki per krok horyzontu",
                            height=340, showlegend=False)
        figs["Horyzont (kroki)"] = fig_h

    hist = meta.get("train_history")
    th = training_history_figure(hist)
    if th:
        figs["Uczenie (epoki)"] = th

    return figs


METRIC_LABELS = [
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("mse", "MSE"),
    ("r2", "R²"),
    ("median_ae", "Mediana |błąd|"),
    ("max_abs_error", "Max |błąd|"),
    ("max_error", "Max błąd (ze znakiem)"),
    ("bias", "Bias (śr. błąd)"),
    ("std_residual", "Std residuów"),
    ("explained_variance", "Expl. variance"),
    ("mape_pct", "MAPE %"),
    ("p90_abs_error", "P90 |błąd|"),
    ("p95_abs_error", "P95 |błąd|"),
    ("n_samples", "Punkty testowe"),
]
