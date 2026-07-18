# CoArrCP IA / YOLO-seg — guía CLI

Repositorio de **código** (sin imágenes, pesos ni documentación pesada).  
Origen: https://github.com/wil2025ux/Imagenes_coralinas_train_yolo

## Requisitos previos

```bash
cd Imagenes_coralinas_train_yolo
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -r coarcp_ia/requirements.txt
# Extra si usas API REST / entrenamiento:
pip install fastapi uvicorn python-multipart
```

Necesitas por tu cuenta (no van en el repo):

- carpeta `images/` con JPG
- `result_coco.json` (export Label Studio COCO)
- pesos `best.pt` tras entrenar (o descargarlos aparte)

En Mac con Apple Silicon suele usarse `--device mps`; en NVIDIA, `--device 0`.

---

## Flujo rápido recomendado

```bash
# 1) COCO → YOLO-seg
python3 prepare_coco_to_yolo_seg.py \
  --coco-json result_coco.json \
  --images-dir images \
  --output-dir benthic_yolo_seg \
  --class-order algas corales almejas esponjas arena

# 2) Entrenar
python3 train_seg.py \
  --data benthic_yolo_seg/dataset.yaml \
  --model yolo11n-seg.pt \
  --epochs 250 --imgsz 640 --batch 2 --device mps \
  --name benthos_yolo11n_seg

# 3) App web CoArrCP IA
python3 run.py \
  --weights runs/segment/benthos_yolo11n_seg/weights/best.pt \
  --host 127.0.0.1 --port 9001
# Abrir http://127.0.0.1:9001/
```

---

## Archivos de la raíz (uso CLI)

### `prepare_coco_to_yolo_seg.py`

Convierte anotaciones COCO (polígonos) a estructura YOLO-seg + `dataset.yaml`.

```bash
python3 prepare_coco_to_yolo_seg.py \
  --coco-json result_coco.json \
  --images-dir images \
  --output-dir benthic_yolo_seg \
  --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 \
  --seed 42 \
  --class-order algas corales almejas esponjas arena
```

### `train_seg.py`

Entrena YOLO de segmentación.

```bash
python3 train_seg.py \
  --data benthic_yolo_seg/dataset.yaml \
  --model yolo11n-seg.pt \
  --epochs 120 \
  --imgsz 1024 \
  --batch 4 \
  --device mps \
  --name mi_corrida
```

### `run_10_corridas_seg.py`

Lanza varias corridas de entrenamiento (experimento reproducible).

```bash
python3 run_10_corridas_seg.py \
  --coco-json result_coco.json \
  --images-dir images \
  --base-model yolo11n-seg.pt \
  --device mps \
  --project-root experimentos_seg3
```

### `analizar_errores_split.py`

Evalúa un split y lista imágenes con fallos / no detectadas.

```bash
python3 analizar_errores_split.py \
  --weights experimentos_seg3/runs/seg3_r012/weights/best.pt \
  --data benthic_yolo_seg/dataset.yaml \
  --split val \
  --conf 0.25 --iou 0.5 --imgsz 640 --device mps \
  --output-csv errores_val.csv \
  --output-misses-txt no_detectadas_val.txt
```

### `infer_webcam_seg.py`

Inferencia en vivo (webcam o video).

```bash
python3 infer_webcam_seg.py \
  --weights experimentos_seg3/runs/seg3_r012/weights/best.pt \
  --source 0 \
  --conf 0.25 --iou 0.5
# --source ruta/video.mp4  para un archivo
```

### `run.py`

Punto de entrada de **CoArrCP IA** (Flask).

```bash
python3 run.py
python3 run.py --weights path/a/best.pt --host 127.0.0.1 --port 9001
python3 run.py --no-model          # UI sin cargar YOLO
python3 run.py --reload            # recarga en desarrollo (si está soportado)
```

### `app_flask.py`

Compatibilidad: delega / arranca la app Flask (preferir `run.py`).

```bash
python3 app_flask.py --weights path/a/best.pt --host 127.0.0.1 --port 9001
```

### `api_rest.py`

API REST local (FastAPI/Uvicorn) para clientes externos (p. ej. Xojo).

```bash
python3 api_rest.py \
  --weights path/a/best.pt \
  --host 127.0.0.1 \
  --port 8000

# Ejemplo:
curl -X POST "http://127.0.0.1:8000/predict/file?conf=0.25&iou=0.5" \
  -F "file=@images/DSCN3074.JPG"
```

### `run_prototipo_v1.py` / `prototipo_v1/app_legacy.py`

Prototipo DCU anterior (puerto por defecto distinto).

```bash
python3 run_prototipo_v1.py
# o
python3 prototipo_v1/app_legacy.py --weights path/a/best.pt --port 9002
```

### Generadores de reportes DOCX

Requieren `python-docx` y rutas locales de documentos/figuras (no versionadas).

```bash
pip install python-docx
python3 generar_reporte_proyecto_final_vision3d.py
python3 generar_reporte_tecnico_final_docx.py
python3 generar_informe_biologo_docx.py
python3 generar_evaluacion_heuristica_docx.py
```

### `generar_tabla_corridas_tex.py`

Consolida métricas de corridas a tabla LaTeX (según salidas en `experimentos_seg3/`).

```bash
python3 generar_tabla_corridas_tex.py
# (revisa argumentos con: python3 generar_tabla_corridas_tex.py -h)
```

---

## Paquete `coarcp_ia/`

| Archivo | Rol | Cómo usarlo |
|---------|-----|-------------|
| `app.py` | App Flask completa | Vía `python3 run.py` (no hace falta llamarlo directo) |
| `requirements.txt` | Dependencias | `pip install -r coarcp_ia/requirements.txt` |
| `retrain_from_points.py` | Export + job de reentreno desde puntos/máscaras | Lo dispara la UI/API de la app; también importable |
| `sam3_assistant.py` | Asistente SAM3 (punto → máscara) | Lo usa la app; requiere repo SAM3 + checkpoint HF |
| `static/app.css` | Estilos | Servido por Flask |
| `templates/index.html` | UI | Servido por Flask |
| `__init__.py` | Paquete Python | — |

Variables útiles para SAM3 (si lo corres en local):

```bash
export SAM3_REPO=/ruta/al/repo/sam3
export HF_HUB_DISABLE_XET=1
```

SAM3 pesado (~3.2 GB) suele probarse en **Google Colab** (notebook fuera de este repo por tamaño).

---

## Config

### `benthic_yolo_seg/dataset.yaml`

Define rutas `train/val/test` y nombres de clase. Se genera/actualiza con `prepare_coco_to_yolo_seg.py`.  
Uso típico:

```bash
python3 train_seg.py --data benthic_yolo_seg/dataset.yaml ...
```

---

## Copia en `ENTREGA_TESIS_SEGMENTACION_BENTONICA/algoritmos/`

Misma familia de scripts que en la raíz (entrega empaquetada). Uso equivalente, por ejemplo:

```bash
cd ENTREGA_TESIS_SEGMENTACION_BENTONICA/algoritmos
python3 run.py --weights ../../experimentos_seg3/runs/seg3_r012/weights/best.pt --port 9001
```

Prefiere los scripts de la **raíz** del repo para desarrollo diario.

---

## Qué no está en Git (a propósito)

`.gitignore` excluye: `*.md` (salvo este README), PDF/Word, `.venv`, `images/`, `runs/`, `experimentos_seg3/`, `*.pt`, zips, notebooks, `result_coco.json`, DB SQLite, etc.
