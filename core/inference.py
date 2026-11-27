"""Carga de artefactos del Notebook 1 y probabilidades por ventana."""
import os
import numpy as np
import pandas as pd
import joblib


def out_dir():
    return os.environ.get("OUT_POSTER", "./outputs")


def load_artifacts(d=None):
    d = d or out_dir()
    meta = joblib.load(os.path.join(d, "modelo_final.pkl"))
    df = pd.read_csv(os.path.join(d, "dataset_features.csv"))
    oof = pd.read_csv(os.path.join(d, "OOF_best_model.csv"))
    ts = None
    if meta["kind"] == "seq":
        import torch
        ts = torch.jit.load(os.path.join(d, meta["torchscript"]), map_location="cpu").eval()
    return meta, df, oof, ts


def session_probs(meta, ts, d, oof, source="loso"):
    """P(estrés) por ventana de una sesión `d` (ordenada por Time).
    source='loso'  -> probabilidad fuera de muestra (sujeto no visto, la honesta).
    source='final' -> modelo final entrenado con todos (optimista para estos sujetos).
    NaN donde un modelo secuencial aún no tiene ventanas suficientes."""
    cols = meta["cols"]
    if source == "loso":
        m = oof[oof.Subject == d.Subject.iloc[0]].set_index("Time").p
        return d.Time.map(m).to_numpy(dtype=float)
    if meta["kind"] == "classic":
        return meta["model"].predict_proba(d[cols])[:, 1]
    import torch
    L, t, X = meta["seq_len"], d.Time.values, d[cols].values.astype(float)
    p = np.full(len(d), np.nan)
    for i in range(L - 1, len(d)):
        if np.all(np.diff(t[i - L + 1:i + 1]) == meta["step_sec"]):
            seq = meta["scaler"].transform(X[i - L + 1:i + 1])[None]
            with torch.no_grad():
                p[i] = ts(torch.tensor(seq, dtype=torch.float32)).item()
    return p
