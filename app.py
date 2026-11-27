"""Panel de bienestar (espejo inteligente) streamlit run app.py
Reproduce sesiones grabadas de WESAD ventana a ventana."""
import io
import time
import wave
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core import inference as inf
from core.personal import PersonalProfile
from core.recommender import ACTIONS, pick_action, is_moving

st.set_page_config(page_title="Panel de bienestar", page_icon=None, layout="wide")
ss = st.session_state


@st.cache_resource
def artifacts():
    return inf.load_artifacts()


try:
    META, DF, OOF, TS = artifacts()
except FileNotFoundError as e:
    st.error(f"No encuentro los ficheros del Notebook 1 en '{inf.out_dir()}' ({e}). "
             "Define OUT_POSTER o copia la carpeta outputs/ junto a app.py.")
    st.stop()

COLS, STEP = META["cols"], META["step_sec"]
SUJ = sorted(DF.Subject.unique(), key=lambda s: int(s[1:]))
COND_COLORS = {"Baseline": "#4c72b0", "Amusement": "#dd8452", "Meditation": "#55a868", "Stress": "#c44e52"}
STATE_UI = {"calibrando": ("Calibrando tu nivel de reposo…", "#7f8c8d"),
            "calma":      ("Estás en calma",                 "#2ecc71"),
            "incierto":   ("Señales mixtas",                 "#f1c40f"),
            "estres":     ("Señales de tensión",             "#e74c3c")}

# cabecera y barra lateral
st.markdown("""<style>
html, body, [class*="css"] { font-size: 18px; }
.stButton>button{min-height:3.8rem;font-size:1.3rem;border-radius:12px}
button[data-baseweb="tab"]{padding:18px 28px} button[data-baseweb="tab"] p{font-size:1.4rem}
.big-step{font-size:3rem;font-weight:600;text-align:center;padding:56px;border-radius:18px;background:#4c9be822}
.big-step span{font-size:4.4rem;display:block;margin-top:12px}
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label { font-size:1.1rem !important; }
div[data-testid="stCaption"] p { font-size: 1rem !important; }
</style>""", unsafe_allow_html=True)
st.title("Panel de bienestar")
st.caption("Apoyo al bienestar, no diagnóstico. Todas las sugerencias son voluntarias. "
           "Sesión reproducida de datos grabados (WESAD), no sensores en vivo.")
suj = st.selectbox("Persona (sesión grabada)", SUJ)

with st.expander("Ajustes", expanded=False):
    a1, a2, a3 = st.columns(3)
    n_calib = a1.slider("Ventanas de calibración (30 s c/u)", 4, 12, 6)
    sens = a2.slider("Sensibilidad", -0.2, 0.2, 0.0, 0.05, help="Positivo = avisa antes (umbral más bajo)") * -1
    persist = a3.slider("Ventanas seguidas de tensión para avisar", 2, 6, 3)
fuente, cooldown, alpha, demo = "loso", 10, 0.5, True
DELAY = {"calibrando": 0.12, "calma": 0.12, "incierto": 0.7, "estres": 1.8}

# estado
ss.setdefault("users", {})
user = ss.users.setdefault(suj, {"stats": {}, "thr_adj": 0.0})
for _a in ACTIONS:
    user["stats"].setdefault(_a, {"n": 0, "ok": 0})


def reset_session():
    d = DF[DF.Subject == suj].sort_values("Time").reset_index(drop=True)
    P = inf.session_probs(META, TS, d, OOF, fuente)
    ss.update(key=(suj, fuente, n_calib), d=d, P=P, prof=PersonalProfile(d[COLS].iloc[:n_calib], P[:n_calib]),
              i=0, ema=None, hist=[], last_alert=-10**9, next_t=0.0, choice=None, alert=None, playing=False, note="",
              rng=np.random.default_rng(42))


if ss.get("key") != (suj, fuente, n_calib):
    reset_session()


def step():
    i, d = ss.i, ss.d
    if i >= len(d):
        ss.playing = False
        return
    p = ss.P[i]
    if np.isnan(p):
        p = ss.ema if ss.ema is not None else 0.0
    ss.ema = p if ss.ema is None else alpha * p + (1 - alpha) * ss.ema
    x, new_alert, ss.note = d.loc[i, COLS], False, ""
    if i < n_calib:
        state = "calibrando"
    else:
        thr = ss.prof.thr_stress(sens + user["thr_adj"])
        state = "estres" if ss.ema >= thr else "calma" if ss.ema <= max(0.05, thr - 0.2) else "incierto"
        ss.prof.update(x, state == "calma")
        z = ss.prof.zscores(x)
        recent = [h["state"] for h in ss.hist[max(0, len(ss.hist) - (persist - 1)):]] + [state]
        if len(recent) == persist and all(s == "estres" for s in recent) and i - ss.last_alert >= cooldown:
            if is_moving(z):
                ss.note = "Mucho movimiento: no se emite aviso (podría ser actividad física)."
            else:
                a, why = pick_action(z, user["stats"], ss.rng)
                ss.alert, ss.last_alert, new_alert = {"i": i, "action": a, "why": why, "answered": False}, i, True
    ss.hist.append({"i": i, "t": d.Time[i] / 60, "p": p, "ema": ss.ema, "state": state,
                    "cond": d.Cond[i], "alert": new_alert})
    ss.i += 1
    ss.next_t = time.time() + DELAY[state]
    if new_alert and ss.playing:
        ss.playing, ss.choice = False, "ask"


def feedback(kind):
    a = ss.alert
    if kind == "tenso":
        user["thr_adj"] = max(-0.2, user["thr_adj"] - 0.03)
        return
    if not a or a["answered"]:
        return
    s = user["stats"][a["action"]]
    if kind == "util":
        s["n"] += 1; s["ok"] += 1
    elif kind == "nosirvio":
        s["n"] += 1
    elif kind == "falsa":
        user["thr_adj"] = min(0.3, user["thr_adj"] + 0.03)
    a["answered"] = True


def rate(action, ok):
    st_ = user["stats"][action]
    st_["n"] += 1
    st_["ok"] += int(ok)
    st.toast("Gracias, lo tendré en cuenta.")


def pause():
    ss.playing = False


def resume():
    ss.update(choice=None, playing=True, next_t=0.0)


# vista
def render():
    if not ss.hist:
        st.info(f"Pulsa Reproducir para empezar. Los primeros {n_calib * STEP // 60} min son de calibración en reposo.")
        return
    h = ss.hist[-1]
    label, color = STATE_UI[h["state"]]
    thr = ss.prof.thr_stress(sens + user["thr_adj"])
    if ss.choice == "ask":
        st.error("### Reproducción detenida: tensión alta\n¿Qué prefieres hacer?")
        d1, d2 = st.columns(2)
        if d1.button("▶  Continuar la reproducción", key="go_on", width="stretch"):
            resume()
            st.rerun()
        if d2.button("Hacer un método de relajación", key="go_relax", type="primary", width="stretch"):
            ss.choice = "relax"
            st.rerun()
    elif ss.choice == "relax":
        act = ACTIONS[ss.alert["action"]]
        st.success(f"### Elige un método en el Centro de calma (abajo)\nTe sugerimos **{act['titulo']}**, "
                   f"en la pestaña «{act['tab']}». Cuando termines, continúa.")
        if st.button("▶  Ya he terminado, continuar la reproducción", key="go_on2", width="stretch"):
            resume()
            st.rerun()
    left, right = st.columns([1, 2])

    with left:
        st.markdown(f"<div style='padding:24px;border-radius:14px;background:{color}22;border:2px solid {color}'>"
                    f"<div style='font-size:28px;font-weight:600'>{label}</div></div>",
                    unsafe_allow_html=True)
        if h["state"] == "calibrando":
            st.progress(min(1.0, ss.i / n_calib), text=f"Calibración {ss.i}/{n_calib}")
        else:
            st.progress(float(np.clip(h["ema"], 0, 1)), text=f"Nivel de tensión estimado: {h['ema']:.0%}")
        if ss.note:
            st.info(ss.note)

        a = ss.alert
        if a and ss.i - 1 - a["i"] <= 6:
            act = ACTIONS[a["action"]]
            st.warning(f"**{act['titulo']}** (opcional)\n\n{act['texto']}")
            st.caption(f"¿Por qué? {a['why']}.")
            st.info(f"Pruébalo en el Centro de calma, pestaña «{act['tab']}»")
            if not a["answered"]:
                c1, c2, c3 = st.columns(3)
                c1.button("Útil", on_click=feedback, args=("util",), width="stretch")
                c2.button("No me sirvió", on_click=feedback, args=("nosirvio",), width="stretch")
                c3.button("No era tensión", on_click=feedback, args=("falsa",), width="stretch")
        st.button("Me siento tenso/a y no ha avisado", on_click=feedback, args=("tenso",))

    with right:
        H = pd.DataFrame(ss.hist)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        if demo:
            start = 0
            for k in range(1, len(H) + 1):
                if k == len(H) or H.cond[k] != H.cond[start]:
                    ax.axvspan(H.t[start] - STEP / 120, H.t[k - 1] + STEP / 120, color=COND_COLORS[H.cond[start]], alpha=0.15, lw=0)
                    start = k
        ax.plot(H.t, H.p, color="gray", lw=0.8, alpha=0.6, label="P(estrés) ventana")
        ax.plot(H.t, H.ema, color="darkblue", lw=2, label="Suavizada")
        ax.axhline(thr, color="red", ls="--", lw=1, label=f"Tu umbral ({thr:.2f})")
        al = H[H.alert]
        ax.scatter(al.t, al.ema, marker="v", color="black", s=80, zorder=5, label="Aviso")
        ax.set_ylim(0, 1.05); ax.set_xlim(0, ss.d.Time.max() / 60)
        ax.set_xlabel("Minutos", fontsize=13); ax.set_ylabel("P(estrés)", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.axvline(n_calib * STEP / 60, color="black", lw=0.8, ls=":")
        ax.legend(fontsize=11, loc="upper left", ncol=2)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


# controles
b1, b2, b3, _ = st.columns([1, 1, 1, 4])
b1.button("Pausar" if ss.playing else "▶  Reproducir",
          on_click=lambda: ss.update(playing=not ss.playing, choice=None), width="stretch")
b2.button("▶▶  Siguiente", on_click=step, width="stretch")
b3.button("↺  Reiniciar", on_click=reset_session, width="stretch")


@st.fragment(run_every=0.15 if ss.playing else None)
def live():
    if ss.playing and time.time() >= ss.next_t:
        step()
        if not ss.playing:
            st.rerun()
    render()


live()


# Centro de calma
PATTERNS = {"Relajante 4-6": [("Inhala", 4), ("Exhala", 6)],
            "Caja 4-4-4-4": [("Inhala", 4), ("Sostén", 4), ("Exhala", 4), ("Sostén", 4)],
            "4-7-8": [("Inhala", 4), ("Sostén", 7), ("Exhala", 8)]}
ROUTINES = {
    "Relajación muscular": ("relajacion", [("Puños: aprieta fuerte", 5), ("Suelta y nota la diferencia", 8),
        ("Hombros hacia las orejas", 5), ("Suelta despacio", 8), ("Aprieta mandíbula y cara", 5), ("Suelta", 8),
        ("Tensa muslos y pantorrillas", 5), ("Suelta", 8)]),
    "Estiramiento de cuello y hombros": ("estiramiento", [("Inclina la cabeza a la derecha", 15),
        ("Inclina la cabeza a la izquierda", 15), ("Barbilla al pecho", 15), ("Rota los hombros hacia atrás", 20),
        ("Entrelaza las manos y estira los brazos", 15)]),
    "Descanso visual 20-20-20": ("visual", [("Mira algo a más de 6 metros", 20),
        ("Parpadea despacio 10 veces", 10), ("Cierra los ojos y relaja la frente", 20)]),
}
SENSES = [(5, "VES"), (4, "puedes TOCAR"), (3, "OYES"), (2, "HUELES"), (1, "SABOREAS")]


def breathing_html(phases, cycles, run_id):
    total = sum(sec for _, sec in phases)
    t, scale, kf, css, spans = 0, 0.55, ["0%{transform:scale(0.55)}"], "", ""
    for i, (name, secs) in enumerate(phases):
        t0, t = t / total * 100, t + secs
        t1 = t / total * 100
        scale = 1.0 if name == "Inhala" else 0.55 if name == "Exhala" else scale
        kf.append(f"{t1:.2f}%{{transform:scale({scale})}}")
        stops = [(0, 1 if t0 == 0 else 0)]
        if t0 > 0:
            stops += [(t0, 0), (t0 + .01, 1)]
        stops += [(t1, 1)]
        if t1 < 99.99:
            stops += [(t1 + .01, 0), (100, 0)]
        css += f"@keyframes l{run_id}_{i}{{" + "".join(f"{p:.2f}%{{opacity:{o}}}" for p, o in stops) + "}"
        spans += (f"<span style='position:absolute;opacity:0;animation:l{run_id}_{i} {total}s linear {cycles} forwards'>{name}</span>"
                  if run_id else "")
    css += f"@keyframes c{run_id}{{{','.join(kf)}}}"
    anim = f"animation:c{run_id} {total}s ease-in-out {cycles} forwards;" if run_id else ""
    spans = spans or "<span style='position:absolute;font-size:22px'>Pulsa Empezar</span>"
    return (f"<style>{css}</style><div style='height:520px;display:flex;align-items:center;justify-content:center'>"
            f"<div style='position:relative;width:440px;height:440px;border-radius:50%;"
            f"background:linear-gradient(135deg,#4c9be8,#7fd1ae);display:flex;align-items:center;justify-content:center;"
            f"color:white;font:600 44px sans-serif;{anim}'>{spans}</div></div>")


@st.cache_data
def ambient_wav(kind, secs=40, sr=16000):
    rng = np.random.default_rng(0)
    spec = np.fft.rfft(rng.standard_normal(sr * secs))
    f = np.fft.rfftfreq(sr * secs, 1 / sr)
    f[0] = np.inf
    x = np.fft.irfft(spec / f ** (1.0 if kind != "Ruido rosa" else 0.5), n=sr * secs)
    if kind == "Olas":
        x *= 0.55 + 0.45 * np.sin(2 * np.pi * np.arange(len(x)) / sr / 8)
    x = (x / np.abs(x).max() * 0.5 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes())
    return buf.getvalue()


def rate_buttons(action):
    c1, c2, _ = st.columns([1, 1, 2])
    c1.button("Me ha ayudado", key=f"ok_{action}", on_click=rate, args=(action, True))
    c2.button("No me ha servido", key=f"no_{action}", on_click=rate, args=(action, False))


def clear_keys(*keys):
    for k in keys:
        ss[k] = ""


def run_routine(name):
    steps = ROUTINES[name][1]
    total, done, box, bar = sum(s_ for _, s_ in steps), 0, st.empty(), st.progress(0.0)
    for txt, secs in steps:
        for k in range(secs, 0, -1):
            box.markdown(f"<div class='big-step'>{txt}<span>{k} s</span></div>", unsafe_allow_html=True)
            time.sleep(1)
            done += 1
            bar.progress(done / total)
    box.success("Rutina completada.")


st.divider()
st.markdown("""<style>
/* Centro de calma — todo más grande */
div[data-testid="stTabs"] button[data-baseweb="tab"] { padding: 22px 36px; }
div[data-testid="stTabs"] button[data-baseweb="tab"] p { font-size: 1.6rem; }
div[data-testid="stTabs"] div[data-testid="stMarkdownContainer"] p,
div[data-testid="stTabs"] div[data-testid="stText"] p,
div[data-testid="stTabs"] label { font-size: 1.25rem !important; }
div[data-testid="stTabs"] .stButton>button { min-height: 4.4rem; font-size: 1.45rem; }
div[data-testid="stTabs"] div[data-testid="stSlider"] { zoom: 1.15; }
div[data-testid="stTabs"] div[data-testid="stRadio"] label { font-size: 1.2rem !important; }
div[data-testid="stTabs"] div[data-testid="stTextInput"] input,
div[data-testid="stTabs"] div[data-testid="stTextArea"] textarea { font-size: 1.2rem; padding: 14px; }
div[data-testid="stTabs"] div[data-testid="stCaption"] p { font-size: 1.1rem !important; }
div[data-testid="stTabs"] div[data-testid="stProgress"] { zoom: 1.2; }
</style>""", unsafe_allow_html=True)
st.header("Centro de calma")
st.caption("Elige lo que te apetezca; no hace falta esperar a un aviso.")
t_resp, t_anc, t_rut, t_desc, t_son = st.tabs(["Respiración", "Anclaje 5-4-3-2-1", "Rutinas guiadas",
                                                "Descarga mental", "Sonidos"])

with t_resp:
    c1, c2 = st.columns(2)
    pat = c1.radio("Técnica", list(PATTERNS), horizontal=True)
    cyc = c2.slider("Ciclos", 2, 10, 5)
    ss.setdefault("breath_id", 0)
    if st.button("▶  Empezar", key="breath_go", on_click=lambda: ss.update(breath_id=ss.breath_id + 1, playing=False)):
        pass
    st.html(breathing_html(PATTERNS[pat], cyc, ss.breath_id))
    st.caption("Inhala mientras el círculo crece, exhala mientras se encoge.")
    rate_buttons("respiracion")

with t_anc:
    done = 0
    for n, verbo in SENSES:
        v = st.text_input(f"{n} cosas que {verbo}", key=f"anc{n}", placeholder="separa con comas")
        ok = len([w for w in v.split(",") if w.strip()]) >= n
        done += ok
        if ok:
            st.caption("Completado")
    st.progress(done / len(SENSES), text=f"{done}/{len(SENSES)} pasos")
    if done == len(SENSES):
        st.success("Estás aquí y ahora. Respira hondo.")
    st.button("↺  Empezar de nuevo", on_click=clear_keys, args=tuple(f"anc{n}" for n, _ in SENSES))
    rate_buttons("anclaje")

with t_rut:
    rut = st.radio("Rutina", list(ROUTINES))
    st.button("▶  Empezar rutina", key="rut_go", on_click=pause)
    if ss.get("rut_go"):
        run_routine(rut)
    rate_buttons(ROUTINES[rut][0])

with t_desc:
    w = st.text_area("¿Qué te ronda la cabeza? Escríbelo sin filtrar", key="worry", height=140)
    if w.strip():
        st.text_input("¿Qué pequeña cosa sí puedes hacer hoy al respecto?", key="ctrl")
        st.button("Soltarlo", on_click=lambda: (clear_keys("worry", "ctrl"), st.toast("Soltado. Respira hondo.")))
    rate_buttons("descarga")

with t_son:
    kind = st.radio("Sonido", ["Ruido marrón", "Ruido rosa", "Olas"], horizontal=True)
    st.audio(ambient_wav(kind), format="audio/wav", loop=True)
    rate_buttons("sonido")