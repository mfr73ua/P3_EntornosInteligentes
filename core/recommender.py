"""Recomendador: reglas según qué señal se desvía + bandit epsilon-greedy con el feedback."""
import numpy as np

ACTIONS = {
    "respiracion": dict(icono="🌬️", titulo="Respiración guiada", tab="🌬️ Respiración", base=1.0,
                        texto="Sigue el círculo: inhala al crecer, exhala al encogerse. 1-2 minutos."),
    "anclaje":     dict(icono="🖐️", titulo="Anclaje 5-4-3-2-1", tab="🖐️ Anclaje 5-4-3-2-1", base=0.9,
                        texto="Nombra cosas que ves, tocas, oyes, hueles y saboreas para volver al presente."),
    "relajacion":  dict(icono="💪", titulo="Relajación muscular", tab="💪 Rutinas guiadas", base=0.8,
                        texto="Tensa y suelta grupos musculares paso a paso."),
    "estiramiento": dict(icono="🤸", titulo="Estiramiento de cuello y hombros", tab="💪 Rutinas guiadas", base=0.8,
                         texto="Cinco movimientos lentos para soltar la zona alta del cuerpo."),
    "visual":      dict(icono="👀", titulo="Descanso visual 20-20-20", tab="💪 Rutinas guiadas", base=0.7,
                        texto="Aparta la vista de la pantalla y relaja los ojos."),
    "descarga":    dict(icono="📝", titulo="Descarga mental", tab="📝 Descarga mental", base=0.7,
                        texto="Escribe lo que te ronda y elige una cosa pequeña que sí puedes hacer."),
    "sonido":      dict(icono="🎧", titulo="Sonidos relajantes", tab="🎧 Sonidos", base=0.8,
                        texto="Ruido marrón, rosa u olas para bajar el ritmo."),
}

FEATURE_LABELS = {
    "EDA_Tonic_Mean": "Activación de la piel (EDA)", "EDA_SCR_count": "Picos de sudoración",
    "EDA_Phasic_AUC": "Respuesta de sudoración", "HRV_RMSSD": "Variabilidad cardíaca (RMSSD)",
    "HRV_MeanNN": "Intervalo entre latidos", "HRV_LFHF": "Equilibrio simpático/parasimpático",
    "RESP_Rate": "Frecuencia respiratoria", "EMG_RMS": "Tensión muscular", "TEMP_Mean": "Temperatura de la piel",
}


def is_moving(z):
    """Mucho movimiento: la señal puede ser actividad física, no tensión."""
    return z.get("ACC_Mean", 0) > 2.5


def pick_action(z, stats, rng, eps=0.15):
    boost, why = {a: 0.0 for a in ACTIONS}, []
    if z.get("HRV_RMSSD", 0) < -1:
        boost["respiracion"] += 1; why.append("tu variabilidad cardíaca está por debajo de lo habitual")
    if z.get("RESP_Rate", 0) > 1:
        boost["respiracion"] += 1; why.append("respiras más rápido de lo habitual")
    if z.get("EDA_Tonic_Mean", 0) > 1 or z.get("EDA_SCR_count", 0) > 1:
        boost["anclaje"] += 1; boost["sonido"] += 0.5; boost["descarga"] += 0.3
        why.append("tu activación en la piel está elevada")
    if z.get("EMG_RMS", 0) > 1:
        boost["relajacion"] += 1; boost["estiramiento"] += 1; why.append("hay tensión muscular")
    # prior por reglas x tasa de éxito aprendida del feedback (suavizada)
    score = {a: (ACTIONS[a]["base"] + boost[a]) * (stats[a]["ok"] + 1) / (stats[a]["n"] + 2) for a in ACTIONS}
    a = rng.choice(list(ACTIONS)) if rng.random() < eps else max(score, key=score.get)
    return a, ("; ".join(why).capitalize() if why else "Tus señales llevan un rato por encima de tu nivel habitual")