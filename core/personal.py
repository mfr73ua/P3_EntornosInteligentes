"""Personalización a nivel de decisión: umbral propio y baseline que se actualiza.
(La normalización de features por usuario empeoraba en el Notebook 1, sección 6.)"""
import numpy as np
import pandas as pd


class PersonalProfile:
    def __init__(self, calib_X: pd.DataFrame, calib_p, margin=0.15, floor=0.6, ceil=0.9, beta=0.02):
        self.mu = calib_X.mean()
        sd = calib_X.std()
        self.sd = sd.where(sd > 1e-6, 1.0).fillna(1.0)
        self.beta = beta                       # velocidad de actualización del baseline
        p = np.asarray(calib_p, dtype=float)
        p95 = np.nanpercentile(p, 95) if np.isfinite(p).any() else 0.5
        self.base_thr = float(np.clip(p95 + margin, floor, ceil))

    def thr_stress(self, adjust=0.0):
        """Umbral de estrés del usuario: percentil 95 en reposo + margen + ajuste por feedback/sensibilidad."""
        return float(np.clip(self.base_thr + adjust, 0.5, 0.95))

    def zscores(self, x: pd.Series):
        return ((x - self.mu) / self.sd).clip(-5, 5)

    def update(self, x: pd.Series, is_calm: bool):
        """Baseline móvil: solo se actualiza con ventanas de calma."""
        if is_calm:
            self.mu = (1 - self.beta) * self.mu + self.beta * x
