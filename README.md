# P3_EntornosInteligentes

Práctica 3 de la asignatura **Entornos Inteligentes** del Grado en Ingeniería en Inteligencia Artificial de la Universidad de Alicante.

Este repositorio implementa un servicio de **apoyo al bienestar** para un espejo inteligente: a partir de señales fisiológicas (EDA, HRV, respiración, EMG, temperatura y aceleración) estima si una ventana de 60 s es compatible con estrés y, cuando la señal es sostenida, propone acciones de relajación **voluntarias**. La evaluación se hace con validación *leave-one-subject-out* (LOSO) sobre el dataset WESAD.

> **Importante:** no es una herramienta de diagnóstico médico. El panel incluido **reproduce sesiones grabadas** de WESAD ventana a ventana; no lee sensores en vivo.

## Índice

- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos previos](#requisitos-previos)
- [Preparación del entorno](#preparación-del-entorno)
- [Datos: WESAD](#datos-wesad)
- [Cómo ejecutar todo](#cómo-ejecutar-todo)
- [Archivos generados](#archivos-generados)
- [Referencias y licencia](#referencias-y-licencia)

## Estructura del proyecto

```text
P3_EntornosInteligentes/
├── 01_Entrenamiento.ipynb   # 1. Extracción de features, LOSO, modelos y exportación
├── 02_Inferencia.ipynb      # 2. Análisis visual, simulación del panel e inferencia
├── 03_Figuras_Poster.ipynb  # 3. Figuras del póster (PNG / PDF / SVG)
├── app.py                   # Panel Streamlit: reproducción del "espejo inteligente"
├── core/
│   ├── inference.py         # Carga de artefactos y P(estrés) por ventana
│   ├── personal.py          # Umbral y baseline personal por usuario
│   └── recommender.py       # Reglas por señal + bandit ε-greedy con feedback
├── outputs/                 # Resultados exportados por el notebook 01 (versionados)
├── figuras_poster/          # Figuras exportadas por el notebook 03 (versionadas)
├── requirements.txt         # Dependencias con versiones fijadas
├── .gitignore               # Entornos, cachés y archivos de sistema
└── README.md
```

La lógica compartida está en `core/` para que los notebooks y `app.py` usen exactamente la misma inferencia y el mismo recomendador. `outputs/` y `figuras_poster/` se versionan a propósito: así el panel se puede abrir sin haber entrenado antes; si vuelves a ejecutar el notebook 01, los artefactos se sobrescriben.

## Requisitos previos

- **Python 3.11 o 3.12** recomendado (el entorno de desarrollo usó las versiones fijadas en `requirements.txt`, incluyendo PyTorch, XGBoost y NeuroKit2).
- **Git** y conexión a internet para descargar el dataset (o una copia local de WESAD).
- **JupyterLab / notebooks** y **Streamlit** (se instalan desde `requirements.txt`).
- Varios GB de RAM libres: la extracción de features y la validación LOSO son los pasos más costosos. Una GPU acelera LSTM/CNN, aunque todo funciona en CPU.
- Opcional: cuenta de Kaggle con token de API si la descarga automática pide autenticación.

## Preparación del entorno

Ejecuta siempre desde la raíz del repositorio:

```bash
git clone https://github.com/mfr73ua/P3_EntornosInteligentes.git
cd P3_EntornosInteligentes

python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell: .venv\Scripts\Activate.ps1)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Registrar el intérprete como kernel de Jupyter (y seleccionarlo después en *Kernel → Change Kernel*):

```bash
python -m ipykernel install --user --name p3-entornos --display-name "Python (P3 - Entornos Inteligentes)"
```

Para desactivar el entorno cuando termines: `deactivate`.

## Datos: WESAD

El notebook 01 descarga automáticamente el dataset con `kagglehub`:

```python
kagglehub.dataset_download("orvile/wesad-wearable-stress-affect-detection-dataset")
```

Si ya tienes WESAD en local, define `WESAD_PATH` **antes de abrir Jupyter**; la carpeta debe contener los directorios de sujeto con su fichero `.pkl` dentro, por ejemplo `S2/S2.pkl`, `S3/S3.pkl`, …, `S17/S17.pkl` (no existen S1 ni S12):

```bash
export WESAD_PATH="/ruta/a/WESAD"
```

Si Kaggle pide autenticación, genera un token en *Kaggle → Account → Create New API Token* y guárdalo en `~/.kaggle/kaggle.json` (`chmod 600 ~/.kaggle/kaggle.json`). El token no se versiona.

Para usar otra carpeta de resultados, define `OUT_POSTER` en el mismo shell desde el que lances Jupyter o Streamlit (por defecto es `./outputs`):

```bash
export OUT_POSTER="/ruta/a/otra/carpeta/outputs"
```

Ambas variables deben estar definidas en todos los procesos que lean los resultados. El notebook 03, además, busca `outputs/` automáticamente cerca del directorio de trabajo.

## Cómo ejecutar todo

El orden importa: el notebook 01 genera los archivos que consumen el resto.

### 1. Entrenamiento y exportación

```bash
jupyter lab
```

Abre `01_Entrenamiento.ipynb` y ejecuta **todas las celdas** (*Run → Run All Cells*). Descarga/lee WESAD, extrae ventanas de 60 s con paso de 30 s, compara baselines, modelos clásicos, ablaciones, personalización y modelos secuenciales con LOSO, y escribe `outputs/`. Puede tardar bastante en CPU.

Parámetros principales (celda de configuración): `SEED = 42`, `N_JOBS = 4`, `WINDOW_SEC = 60`, `STEP_SEC = 30`, `SEQ_LEN = 5`, `N_CALIB = 5`, `RUN_WRIST = True` y `USE_CACHE = False`. Tras la primera extracción puedes poner `USE_CACHE = True` para reutilizar los CSV de features; pon `RUN_WRIST = False` para saltarte la comparación pecho vs. muñeca.

### 2. Análisis e inferencia

Abre `02_Inferencia.ipynb` y ejecútalo completo. Contiene gráficos interactivos (widgets), simulación del panel (umbrales, persistencia y cooldown) e inferencia con el modelo final exportado. Las probabilidades grises son LOSO (sujeto no visto); las azules corresponden al modelo final reentrenado con todos los sujetos y son optimistas para esos sujetos.

### 3. Figuras del póster

Abre `03_Figuras_Poster.ipynb` y ejecútalo. Solo lee los CSV de `outputs/` y guarda cada figura en `figuras_poster/` con extensiones `.png`, `.pdf` y `.svg`. No entrena ni modifica los resultados.

### 4. Panel de bienestar (Streamlit)

```bash
streamlit run app.py
```

Abre <http://localhost:8501>. Selecciona una sesión grabada y pulsa **Reproducir** (o **Siguiente** para avanzar ventana a ventana). Tras unos 3 minutos de calibración en reposo (6 ventanas de 30 s), el panel calcula el umbral personal, suaviza la probabilidad, evita avisar si detecta mucho movimiento y ofrece un "Centro de calma" con respiración guiada, anclaje 5-4-3-2-1, rutinas, descarga mental y sonidos. El feedback del usuario ajusta el umbral y pondera el recomendador. Si el puerto está ocupado: `streamlit run app.py --server.port 8502`.

## Archivos generados

| Fichero | Contenido |
| --- | --- |
| `features_chest_60s_30s.csv`, `features_wrist_60s_30s.csv` | Features por ventana (pecho / muñeca) |
| `dataset_features.csv` | Ventanas de pecho con `Subject`, `Time`, `Cond`, `Label` y features |
| `OOF_best_model.csv`, `OOF_modelos_principales.csv` | Probabilidades fuera de muestra (LOSO) del mejor modelo y de los principales |
| `resultados_resumen.csv`, `resultados_por_sujeto.csv` | Métricas medias y por participante |
| `modelo_final.pkl` | Mejor modelo reentrenado con todos los sujetos, escalador y metadatos |
| `modelo_final.pt` | Versión TorchScript, solo si el mejor modelo es LSTM o CNN |
| `outputs/*.png` | Curvas y figuras de análisis del notebook 01 |
| `figuras_poster/*` | Figuras del póster en tres formatos |

## Referencias y licencia

- Schmidt, P., Reiss, A., Dürichen, R., Marberger, C. y Van Laerhoven, K. (2018). *Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection*. ICMI '18, pp. 400-408. DOI: [10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985).
- Los datos de WESAD se usan con fines académicos y no comerciales, citando a sus autores. El código es un trabajo docente de la asignatura Entornos Inteligentes (Universidad de Alicante); consulta al autor antes de reutilizarlo fuera de ese contexto.
- Bibliotecas principales: scikit-learn, PyTorch, XGBoost, NeuroKit2, pandas/NumPy, Matplotlib/Seaborn, Streamlit y KaggleHub (versiones exactas en `requirements.txt`).
