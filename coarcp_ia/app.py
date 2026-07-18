import argparse
import base64
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for
from ultralytics import YOLO
from ultralytics.utils import ops
from werkzeug.utils import secure_filename

try:
    from retrain_from_points import build_dataset_from_corrections, count_trainable_corrections
    from sam3_assistant import (
        load_sam3,
        mask_overlay_bgr,
        mask_to_data_url,
        predict_mask_from_point,
        sam3_status,
        save_mask_png,
    )
except ImportError:
    from coarcp_ia.retrain_from_points import build_dataset_from_corrections, count_trainable_corrections
    from coarcp_ia.sam3_assistant import (
        load_sam3,
        mask_overlay_bgr,
        mask_to_data_url,
        predict_mask_from_point,
        sam3_status,
        save_mask_png,
    )


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DATA_DIR = APP_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "coarcp_ia.sqlite"
STATIC_RESULTS_DIR = APP_ROOT / "static" / "results"
RETRAIN_DATASETS_DIR = DATA_DIR / "retrain_datasets"
RETRAIN_RUNS_DIR = DATA_DIR / "retrain_runs"
RETRAIN_LOGS_DIR = DATA_DIR / "retrain_logs"
DEFAULT_WEIGHTS = PROJECT_ROOT / "experimentos_seg3/runs/seg3_r012/weights/best.pt"
_retrain_lock = threading.Lock()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = Flask(__name__)
app.secret_key = os.getenv("COARCP_IA_SECRET", "coarcp-ia-dev-secret")

model = None
class_names: List[str] = []
weights_loaded: Optional[str] = None


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    STATIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RETRAIN_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    RETRAIN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RETRAIN_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _table_columns(conn, table: str) -> set:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


def migrate_schema(conn):
    point_cols = _table_columns(conn, "points")
    if "point_name" not in point_cols:
        conn.execute("ALTER TABLE points ADD COLUMN point_name TEXT")
    if "manual_label" not in point_cols:
        conn.execute("ALTER TABLE points ADD COLUMN manual_label INTEGER DEFAULT 0")
    if "mask_path" not in point_cols:
        conn.execute("ALTER TABLE points ADD COLUMN mask_path TEXT")
    if "mask_score" not in point_cols:
        conn.execute("ALTER TABLE points ADD COLUMN mask_score REAL")
    if "mask_source" not in point_cols:
        conn.execute("ALTER TABLE points ADD COLUMN mask_source TEXT")

    conn.execute(
        """
        UPDATE points
        SET point_name = 'P' || point_index
        WHERE point_name IS NULL OR TRIM(point_name) = ''
        """
    )

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS retrain_jobs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          status TEXT NOT NULL DEFAULT 'queued',
          created_at TEXT NOT NULL,
          started_at TEXT,
          finished_at TEXT,
          dataset_path TEXT,
          weights_out TEXT,
          log_path TEXT,
          error TEXT,
          num_images INTEGER DEFAULT 0,
          num_points INTEGER DEFAULT 0,
          epochs INTEGER DEFAULT 30,
          imgsz INTEGER DEFAULT 640,
          batch INTEGER DEFAULT 2,
          radius_ratio REAL DEFAULT 0.04,
          base_weights TEXT
        );
        """
    )


def resolve_weights_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path.resolve()
    candidates = [
        path,
        APP_ROOT / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return path


def db_connect():
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    ensure_dirs()
    with db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL DEFAULT 'CoArrCP IA',
              notes TEXT,
              point_count INTEGER DEFAULT 13,
              conf REAL DEFAULT 0.25,
              iou REAL DEFAULT 0.50,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS images (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              filename TEXT NOT NULL,
              original_name TEXT NOT NULL,
              group_name TEXT,
              original_path TEXT NOT NULL,
              result_path TEXT,
              width INTEGER,
              height INTEGER,
              status TEXT DEFAULT 'pendiente',
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS inference_runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              image_id INTEGER NOT NULL,
              model_weights TEXT,
              conf REAL NOT NULL,
              iou REAL NOT NULL,
              point_count INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              num_detections INTEGER DEFAULT 0,
              processing_ms INTEGER DEFAULT 0,
              error TEXT,
              FOREIGN KEY(image_id) REFERENCES images(id)
            );

            CREATE TABLE IF NOT EXISTS coverage (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id INTEGER NOT NULL,
              class_name TEXT NOT NULL,
              percent REAL NOT NULL,
              FOREIGN KEY(run_id) REFERENCES inference_runs(id)
            );

            CREATE TABLE IF NOT EXISTS detections (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id INTEGER NOT NULL,
              class_name TEXT,
              confidence REAL,
              x1 REAL,
              y1 REAL,
              x2 REAL,
              y2 REAL,
              component_name TEXT,
              accepted INTEGER DEFAULT 0,
              FOREIGN KEY(run_id) REFERENCES inference_runs(id)
            );

            CREATE TABLE IF NOT EXISTS points (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id INTEGER NOT NULL,
              point_index INTEGER NOT NULL,
              x REAL NOT NULL,
              y REAL NOT NULL,
              class_name TEXT,
              confidence REAL DEFAULT 0,
              component_name TEXT,
              status TEXT DEFAULT 'sin_deteccion',
              edited_manual INTEGER DEFAULT 0,
              FOREIGN KEY(run_id) REFERENCES inference_runs(id)
            );

            CREATE TABLE IF NOT EXISTS components (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              class_name TEXT UNIQUE NOT NULL,
              component_name TEXT NOT NULL,
              ggmf TEXT,
              gmf TEXT,
              notes TEXT
            );

            CREATE TABLE IF NOT EXISTS retrain_jobs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              status TEXT NOT NULL DEFAULT 'queued',
              created_at TEXT NOT NULL,
              started_at TEXT,
              finished_at TEXT,
              dataset_path TEXT,
              weights_out TEXT,
              log_path TEXT,
              error TEXT,
              num_images INTEGER DEFAULT 0,
              num_points INTEGER DEFAULT 0,
              epochs INTEGER DEFAULT 30,
              imgsz INTEGER DEFAULT 640,
              batch INTEGER DEFAULT 2,
              radius_ratio REAL DEFAULT 0.04,
              base_weights TEXT
            );
            """
        )

        migrate_schema(conn)

        project_exists = conn.execute("SELECT COUNT(*) AS total FROM projects").fetchone()["total"]
        if project_exists == 0:
            conn.execute(
                "INSERT INTO projects(name, notes, point_count, conf, iou, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "CoArrCP IA",
                    "Proyecto Flask inspirado en el flujo de CoArrCP con segmentación YOLO-seg.",
                    13,
                    0.25,
                    0.50,
                    now_iso(),
                ),
            )
        conn.commit()


def load_model(weights_path: str):
    global model, class_names, weights_loaded
    resolved = resolve_weights_path(weights_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"No existe el archivo de pesos: {weights_path}")
    model = YOLO(str(resolved))
    names = model.names if isinstance(model.names, dict) else {}
    class_names = [names[i] for i in sorted(names.keys())]
    weights_loaded = str(resolved)
    sync_components_with_model()


def sync_components_with_model():
    if not class_names:
        return
    with db_connect() as conn:
        for name in class_names:
            conn.execute(
                """
                INSERT INTO components(class_name, component_name, ggmf, gmf, notes)
                VALUES (?, ?, '', '', '')
                ON CONFLICT(class_name) DO NOTHING
                """,
                (name, name),
            )
        conn.commit()


def is_image_file(path: str) -> bool:
    return Path(path.lower()).suffix in SUPPORTED_EXTENSIONS


def decode_image(raw: bytes):
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def compute_coverage(result, names: List[str]) -> Dict[str, float]:
    coverage_pixels = {i: 0 for i in range(len(names))}

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return {names[i]: 0.0 for i in range(len(names))}

    masks = result.masks.data.cpu().numpy() > 0.5
    _, h, w = masks.shape
    total_pixels = h * w
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    order = np.argsort(-confs)
    occupied = np.zeros((h, w), dtype=bool)

    for idx in order:
        cls_id = classes[idx]
        if cls_id < 0 or cls_id >= len(names):
            continue
        mask = masks[idx]
        new_pixels = mask & (~occupied)
        coverage_pixels[cls_id] += int(new_pixels.sum())
        occupied |= mask

    return {names[i]: (100.0 * coverage_pixels[i] / total_pixels) for i in range(len(names))}


def component_map() -> Dict[str, str]:
    with db_connect() as conn:
        rows = conn.execute("SELECT class_name, component_name FROM components").fetchall()
    return {row["class_name"]: row["component_name"] for row in rows}


def generate_points(width: int, height: int, count: int = 13):
    base_norm = [
        (0.20, 0.20), (0.50, 0.20), (0.80, 0.20),
        (0.32, 0.35), (0.68, 0.35),
        (0.20, 0.50), (0.50, 0.50), (0.80, 0.50),
        (0.32, 0.65), (0.68, 0.65),
        (0.20, 0.80), (0.50, 0.80), (0.80, 0.80),
    ]

    if count <= len(base_norm):
        selected = base_norm[:count]
    else:
        cols = int(np.ceil(np.sqrt(count)))
        rows = int(np.ceil(count / cols))
        selected = []
        for r in range(rows):
            for c in range(cols):
                if len(selected) >= count:
                    break
                selected.append(((c + 1) / (cols + 1), (r + 1) / (rows + 1)))

    return [
        {
            "id": idx + 1,
            "point_name": f"P{idx + 1}",
            "x": round(nx * width, 1),
            "y": round(ny * height, 1),
        }
        for idx, (nx, ny) in enumerate(selected)
    ]


def assign_points_to_masks(result, points, image_width: int, image_height: int):
    predictions = []

    if not points:
        return predictions

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        for p in points:
            predictions.append({
                "id": p.get("id"),
                "point_name": p.get("point_name") or f"P{p.get('id')}",
                "x": p.get("x"),
                "y": p.get("y"),
                "class_name": None,
                "confidence": 0.0,
                "status": "sin_deteccion",
                "manual_label": int(p.get("manual_label") or 0),
            })
        return predictions

    masks = result.masks.data.cpu().numpy() > 0.5
    _, mask_h, mask_w = masks.shape

    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)

    for p in points:
        x_original = float(p.get("x", 0))
        y_original = float(p.get("y", 0))
        point_name = p.get("point_name") or f"P{p.get('id')}"
        manual_label = int(p.get("manual_label") or 0)

        # Etiqueta manual: solo actualiza coords, conserva clase del usuario.
        if manual_label and p.get("class_name"):
            predictions.append({
                "id": p.get("id"),
                "point_name": point_name,
                "x": x_original,
                "y": y_original,
                "class_name": p.get("class_name"),
                "confidence": float(p.get("confidence") or 1.0),
                "status": "manual",
                "manual_label": 1,
                "component_name": p.get("component_name"),
            })
            continue

        x_mask = int(round(x_original * mask_w / max(image_width, 1)))
        y_mask = int(round(y_original * mask_h / max(image_height, 1)))

        x_mask = min(max(x_mask, 0), mask_w - 1)
        y_mask = min(max(y_mask, 0), mask_h - 1)

        assigned = None

        for idx in order:
            if masks[idx, y_mask, x_mask]:
                cls_id = int(classes[idx])
                label = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
                assigned = {
                    "id": p.get("id"),
                    "point_name": point_name,
                    "x": x_original,
                    "y": y_original,
                    "class_name": label,
                    "confidence": round(float(confs[idx]), 4),
                    "status": "detectado",
                    "manual_label": 0,
                }
                break

        if assigned is None:
            assigned = {
                "id": p.get("id"),
                "point_name": point_name,
                "x": x_original,
                "y": y_original,
                "class_name": None,
                "confidence": 0.0,
                "status": "sin_deteccion",
                "manual_label": 0,
            }

        predictions.append(assigned)

    return predictions


# Colores fijos por clase (BGR). Sin morados de la paleta por defecto de YOLO.
CLASS_PALETTE_BGR = [
    (72, 187, 120),   # algas
    (80, 127, 255),   # corales
    (255, 200, 64),   # almejas
    (0, 165, 255),    # esponjas (naranja, no morado)
    (160, 160, 160),  # arena
]


def masks_at_image_size(result, height: int, width: int) -> np.ndarray:
    """Escala máscaras de inferencia al tamaño real de la fotografía."""
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, height, width), dtype=bool)

    scaled = ops.scale_masks(result.masks.data[None].float(), (height, width))[0]
    return (scaled.cpu().numpy() > 0.5)


def plot_segmentation_clean(result, frame: np.ndarray) -> np.ndarray:
    """Solo máscaras semitransparentes, sin cajas ni etiquetas de YOLO (evita puntos morados)."""
    out = frame.copy()
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return out

    h, w = out.shape[:2]
    masks = masks_at_image_size(result, h, w)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)
    occupied = np.zeros((h, w), dtype=bool)

    for idx in order:
        if idx >= masks.shape[0]:
            continue
        cls_id = int(classes[idx])
        if cls_id < 0:
            continue
        color = CLASS_PALETTE_BGR[cls_id % len(CLASS_PALETTE_BGR)]
        mask = masks[idx]
        new_pixels = mask & (~occupied)
        if not int(new_pixels.sum()):
            continue
        for channel in range(3):
            out[:, :, channel] = np.where(
                new_pixels,
                (out[:, :, channel].astype(np.float32) * 0.45 + color[channel] * 0.55).astype(np.uint8),
                out[:, :, channel],
            )
        occupied |= mask

    return out


def draw_points_on_result(image, point_predictions):
    out = image.copy()
    h, w = image.shape[:2]
    radius = max(14, int(min(w, h) * 0.014))
    border = radius + 4
    font_scale = max(0.7, min(w, h) / 1200)

    for p in point_predictions:
        x = int(round(float(p["x"])))
        y = int(round(float(p["y"])))
        point_id = p.get("point_name") or p.get("id", p.get("point_index"))
        status = p.get("status") or "sin_deteccion"

        if status == "detectado":
            color = (34, 197, 94)
        elif status == "manual":
            color = (14, 165, 233)
        else:
            color = (244, 63, 94)

        cv2.circle(out, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), border, (15, 23, 42), 3, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), border + 2, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        label = str(point_id)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        label_x = min(max(x + radius + 6, 4), w - label_size[0] - 4)
        label_y = max(y - radius - 6, label_size[1] + 8)
        cv2.rectangle(
            out,
            (label_x - 4, label_y - label_size[1] - 6),
            (label_x + label_size[0] + 4, label_y + 4),
            (15, 23, 42),
            -1,
        )
        cv2.putText(
            out,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def image_to_data_url(bgr_image):
    ok, encoded = cv2.imencode(".jpg", bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def process_frame(
    frame,
    conf: float,
    iou: float,
    point_count: int,
    result_abs_path: Optional[Path] = None,
    external_points=None,
    return_image_url: bool = False,
):
    if model is None:
        raise RuntimeError("Modelo no cargado. Ejecuta la app con --weights apuntando a best.pt.")

    start = time.perf_counter()

    result = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]
    height, width = frame.shape[:2]

    coverage = compute_coverage(result, class_names)

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            cid = int(cls[i])
            label = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            detections.append({
                "class_name": label,
                "confidence": round(float(scores[i]), 4),
                "bbox": [round(float(v), 1) for v in boxes[i].tolist()],
            })

    if external_points is not None:
        generated_points = external_points
    else:
        generated_points = generate_points(width, height, point_count)

    point_predictions = assign_points_to_masks(result, generated_points, width, height)

    plotted = plot_segmentation_clean(result, frame)
    image_url = None
    if result_abs_path is not None:
        cv2.imwrite(str(result_abs_path), plotted)
    if return_image_url:
        image_url = image_to_data_url(draw_points_on_result(plotted.copy(), point_predictions))

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "width": width,
        "height": height,
        "coverage": [
            {"class_name": name, "percent": round(float(coverage.get(name, 0.0)), 2)}
            for name in class_names
        ],
        "detections": detections,
        "points": point_predictions,
        "point_predictions": point_predictions,
        "num_detections": len(detections),
        "processing_ms": elapsed_ms,
        "image_url": image_url,
    }


def unique_name(original_name: str, suffix: str = "") -> str:
    cleaned = secure_filename(Path(original_name).name)
    if not cleaned:
        cleaned = "imagen.jpg"
    stem = Path(cleaned).stem
    ext = Path(cleaned).suffix.lower() or ".jpg"
    return f"{stem}_{uuid.uuid4().hex[:8]}{suffix}{ext}"


def infer_and_save(frame, original_name: str, group_name: str, conf: float, iou: float, point_count: int):
    original_filename = unique_name(original_name)
    result_filename = unique_name(original_name, suffix="_ia")

    original_abs = UPLOAD_DIR / original_filename
    result_rel = f"results/{result_filename}"
    result_abs = STATIC_RESULTS_DIR / result_filename

    cv2.imwrite(str(original_abs), frame)

    inference = process_frame(frame, conf, iou, point_count, result_abs)

    cmap = component_map()

    with db_connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO images(filename, original_name, group_name, original_path, result_path, width, height, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pendiente', ?)
            """,
            (
                original_filename,
                original_name,
                group_name,
                str(original_abs),
                result_rel,
                inference["width"],
                inference["height"],
                now_iso(),
            ),
        )
        image_id = cur.lastrowid

        cur = conn.execute(
            """
            INSERT INTO inference_runs(image_id, model_weights, conf, iou, point_count, created_at, num_detections, processing_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                image_id,
                weights_loaded,
                conf,
                iou,
                point_count,
                now_iso(),
                inference["num_detections"],
                inference["processing_ms"],
            ),
        )
        run_id = cur.lastrowid

        for row in inference["coverage"]:
            conn.execute(
                "INSERT INTO coverage(run_id, class_name, percent) VALUES (?, ?, ?)",
                (run_id, row["class_name"], row["percent"]),
            )

        for d in inference["detections"]:
            bbox = d["bbox"]
            class_name = d["class_name"]
            conn.execute(
                """
                INSERT INTO detections(run_id, class_name, confidence, x1, y1, x2, y2, component_name, accepted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    run_id,
                    class_name,
                    d["confidence"],
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    cmap.get(class_name, class_name),
                ),
            )

        for p in inference["points"]:
            class_name = p["class_name"]
            point_name = p.get("point_name") or f"P{p['id']}"
            conn.execute(
                """
                INSERT INTO points(
                  run_id, point_index, point_name, x, y, class_name, confidence,
                  component_name, status, edited_manual, manual_label
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
                """,
                (
                    run_id,
                    p["id"],
                    point_name,
                    p["x"],
                    p["y"],
                    class_name,
                    p["confidence"],
                    cmap.get(class_name, class_name) if class_name else None,
                    p["status"],
                ),
            )

        conn.commit()

    return image_id


def empty_result_item(filename=None, path=None, error=None):
    return {
        "path": path,
        "filename": filename,
        "error": error,
        "image_url": None,
        "coverage": [],
        "detections": [],
        "point_predictions": [],
        "num_detections": 0,
    }


def process_image_from_path(
    image_path: str,
    conf: float,
    iou: float,
    points=None,
    return_image: bool = True,
    point_count: int = 13,
):
    if not image_path or not os.path.exists(image_path):
        return empty_result_item(
            filename=os.path.basename(image_path) if image_path else None,
            path=image_path,
            error="La imagen no existe.",
        )

    if not is_image_file(image_path):
        return empty_result_item(
            filename=os.path.basename(image_path),
            path=image_path,
            error="El archivo no es una imagen soportada.",
        )

    frame = cv2.imread(image_path)
    if frame is None:
        return empty_result_item(
            filename=os.path.basename(image_path),
            path=image_path,
            error="No se pudo abrir la imagen.",
        )

    external_points = points if points else None
    item = process_frame(
        frame,
        conf,
        iou,
        point_count,
        result_abs_path=None,
        external_points=external_points,
        return_image_url=return_image,
    )
    item["path"] = image_path
    item["filename"] = os.path.basename(image_path)
    item["error"] = None
    return item


def get_overview():
    with db_connect() as conn:
        counts = conn.execute(
            """
            SELECT
              COUNT(*) AS total_images,
              SUM(CASE WHEN status = 'pendiente' THEN 1 ELSE 0 END) AS pending_images,
              SUM(CASE WHEN status = 'revisado' THEN 1 ELSE 0 END) AS reviewed_images
            FROM images
            """
        ).fetchone()

        dets = conn.execute("SELECT COUNT(*) AS total FROM detections").fetchone()["total"]

        latest = conn.execute(
            """
            SELECT i.*, r.id AS run_id, r.num_detections, r.processing_ms
            FROM images i
            LEFT JOIN inference_runs r ON r.id = (
              SELECT id FROM inference_runs WHERE image_id = i.id ORDER BY id DESC LIMIT 1
            )
            ORDER BY i.id DESC LIMIT 8
            """
        ).fetchall()

        avg_coverage = conn.execute(
            """
            SELECT class_name, ROUND(AVG(percent), 2) AS avg_percent
            FROM coverage
            GROUP BY class_name
            ORDER BY avg_percent DESC
            """
        ).fetchall()

        top_classes = conn.execute(
            """
            SELECT class_name, COUNT(*) AS total
            FROM detections
            WHERE class_name IS NOT NULL
            GROUP BY class_name
            ORDER BY total DESC
            LIMIT 10
            """
        ).fetchall()

    return {
        "total_images": counts["total_images"] or 0,
        "pending_images": counts["pending_images"] or 0,
        "reviewed_images": counts["reviewed_images"] or 0,
        "total_detections": dets or 0,
        "latest_images": latest,
        "avg_coverage": avg_coverage,
        "top_classes": top_classes,
    }


def get_images(limit: int = 60):
    with db_connect() as conn:
        return conn.execute(
            """
            SELECT i.*, r.id AS run_id, r.num_detections, r.processing_ms, r.conf, r.iou, r.point_count
            FROM images i
            LEFT JOIN inference_runs r ON r.id = (
              SELECT id FROM inference_runs WHERE image_id = i.id ORDER BY id DESC LIMIT 1
            )
            ORDER BY i.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def get_components():
    with db_connect() as conn:
        return conn.execute(
            "SELECT * FROM components ORDER BY class_name COLLATE NOCASE"
        ).fetchall()


def rows_to_dicts(rows):
    return [dict(row) for row in rows]


def refresh_points_for_image(image_id: int, updated_coords: Optional[Dict[int, tuple]] = None):
    if model is None:
        raise RuntimeError("Modelo no cargado.")

    detail = get_image_detail(image_id)
    if not detail or not detail.get("image") or not detail.get("run"):
        raise ValueError("Imagen o inferencia no encontrada.")

    image = detail["image"]
    run = detail["run"]
    frame = cv2.imread(str(image["original_path"]))
    if frame is None:
        raise ValueError("No se pudo abrir la imagen original.")

    height, width = frame.shape[:2]
    external_points = []
    existing_by_index = {int(row["point_index"]): row for row in detail["points"]}
    for row in detail["points"]:
        point_index = int(row["point_index"])
        x = float(row["x"])
        y = float(row["y"])
        if updated_coords and point_index in updated_coords:
            x, y = updated_coords[point_index]
        external_points.append({
            "id": point_index,
            "point_name": row.get("point_name") or f"P{point_index}",
            "x": x,
            "y": y,
            "class_name": row.get("class_name"),
            "component_name": row.get("component_name"),
            "confidence": row.get("confidence") or 0,
            "manual_label": int(row.get("manual_label") or 0),
        })

    result = model.predict(frame, conf=float(run["conf"]), iou=float(run["iou"]), verbose=False)[0]
    point_predictions = assign_points_to_masks(result, external_points, width, height)
    cmap = component_map()

    with db_connect() as conn:
        for p in point_predictions:
            existing = existing_by_index.get(int(p["id"]), {})
            manual_label = int(p.get("manual_label") or existing.get("manual_label") or 0)
            class_name = p.get("class_name")
            if manual_label:
                component_name = p.get("component_name") or existing.get("component_name") or (
                    cmap.get(class_name, class_name) if class_name else None
                )
                status = "manual"
            else:
                component_name = cmap.get(class_name, class_name) if class_name else None
                status = p["status"]
            point_name = p.get("point_name") or existing.get("point_name") or f"P{p['id']}"
            conn.execute(
                """
                UPDATE points
                SET x = ?, y = ?, point_name = ?, class_name = ?, confidence = ?,
                    component_name = ?, status = ?, edited_manual = 1, manual_label = ?
                WHERE run_id = ? AND point_index = ?
                """,
                (
                    p["x"],
                    p["y"],
                    point_name,
                    class_name,
                    p["confidence"],
                    component_name,
                    status,
                    manual_label,
                    run["id"],
                    p["id"],
                ),
            )
        conn.commit()

    plotted = plot_segmentation_clean(result, frame)

    if image["result_path"]:
        result_abs = APP_ROOT / "static" / image["result_path"]
        result_abs.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(result_abs), plotted)

    return serialize_points_for_run(int(run["id"]))


def serialize_points_for_run(run_id: int) -> List[dict]:
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT * FROM points WHERE run_id = ? ORDER BY point_index",
            (run_id,),
        ).fetchall()
    return rows_to_dicts(rows)


def slugify_class_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
    return cleaned or f"clase_{uuid.uuid4().hex[:6]}"


def ensure_catalog_entry(class_name: str, component_name: Optional[str] = None) -> dict:
    class_key = class_name.strip()
    if not class_key:
        raise ValueError("El nombre de clase no puede estar vacío.")
    display = (component_name or class_key).strip() or class_key
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO components(class_name, component_name, ggmf, gmf, notes)
            VALUES (?, ?, '', '', '')
            ON CONFLICT(class_name) DO NOTHING
            """,
            (class_key, display),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM components WHERE class_name = ?",
            (class_key,),
        ).fetchone()
    return dict(row)


def next_point_index(conn, run_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(point_index), 0) AS max_idx FROM points WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return int(row["max_idx"]) + 1


def get_latest_retrain_job() -> Optional[dict]:
    with db_connect() as conn:
        row = conn.execute(
            "SELECT * FROM retrain_jobs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def get_retrain_job(job_id: int) -> Optional[dict]:
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM retrain_jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def get_image_detail(image_id: Optional[int]):
    if not image_id:
        return None

    with db_connect() as conn:
        image = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        if not image:
            return None

        run = conn.execute(
            "SELECT * FROM inference_runs WHERE image_id = ? ORDER BY id DESC LIMIT 1",
            (image_id,),
        ).fetchone()
        if not run:
            return {"image": dict(image), "run": None, "coverage": [], "detections": [], "points": []}

        coverage = conn.execute(
            "SELECT * FROM coverage WHERE run_id = ? ORDER BY percent DESC",
            (run["id"],),
        ).fetchall()

        detections = conn.execute(
            "SELECT * FROM detections WHERE run_id = ? ORDER BY confidence DESC",
            (run["id"],),
        ).fetchall()

        points = conn.execute(
            "SELECT * FROM points WHERE run_id = ? ORDER BY point_index",
            (run["id"],),
        ).fetchall()

    return {
        "image": dict(image),
        "run": dict(run) if run else None,
        "coverage": rows_to_dicts(coverage),
        "detections": rows_to_dicts(detections),
        "points": rows_to_dicts(points),
    }


def get_review_nav(limit: int = 200):
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, original_name, status
            FROM images
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return rows_to_dicts(rows)


@app.route("/", methods=["GET"])
def index():
    tab = request.args.get("tab", "inicio")
    selected_image_id = request.args.get("image_id", type=int)

    if tab == "revision" and selected_image_id and model is not None:
        try:
            ensure_clean_preview(selected_image_id)
        except Exception:
            pass

    images = get_images()

    preview_cache = ""
    if tab == "revision" and selected_image_id:
        preview_cache = str(int(time.time() * 1000))

    return render_template(
        "index.html",
        tab=tab,
        model_loaded=model is not None,
        classes=class_names,
        weights_loaded=weights_loaded,
        db_path=str(DB_PATH),
        overview=get_overview(),
        images=images,
        review_nav=get_review_nav(),
        components=get_components(),
        detail=get_image_detail(selected_image_id),
        selected_image_id=selected_image_id,
        preview_cache=preview_cache,
        latest_retrain_job=get_latest_retrain_job(),
        correction_stats=_correction_stats(),
        sam3=sam3_status(),
    )


def _correction_stats() -> dict:
    with db_connect() as conn:
        return count_trainable_corrections(conn)


@app.route("/procesar", methods=["POST"])
def procesar():
    if model is None:
        flash("Modelo no cargado. Ejecuta la app con --weights apuntando a best.pt.", "error")
        return redirect(url_for("index", tab="cargar"))

    conf = float(request.form.get("conf", 0.25))
    iou = float(request.form.get("iou", 0.5))
    point_count = int(request.form.get("point_count", 13))

    files = request.files.getlist("files")
    valid_files = [f for f in files if f and f.filename]

    if not valid_files:
        flash("Selecciona una imagen, varias imágenes o una carpeta con imágenes.", "error")
        return redirect(url_for("index", tab="cargar"))

    processed = 0
    errors = []

    for f in valid_files:
        original_browser_name = f.filename
        pure = PurePosixPath(original_browser_name)
        original_name = pure.name
        parent = str(pure.parent)
        group_name = parent.split("/")[0] if parent and parent != "." else "general"

        if not is_image_file(original_name):
            continue

        raw = f.read()
        frame = decode_image(raw)

        if frame is None:
            errors.append(f"No se pudo decodificar: {original_browser_name}")
            continue

        try:
            infer_and_save(frame, original_name, group_name, conf, iou, point_count)
            processed += 1
        except Exception as exc:
            errors.append(f"{original_browser_name}: {exc}")

    if processed:
        flash(f"Procesamiento terminado: {processed} imagen/es.", "success")
    if errors:
        flash("Algunas imágenes no se procesaron: " + " | ".join(errors[:5]), "error")

    return redirect(url_for("index", tab="inferencia"))


@app.route("/procesar_carpeta_servidor", methods=["POST"])
def procesar_carpeta_servidor():
    if model is None:
        flash("Modelo no cargado. Ejecuta la app con --weights apuntando a best.pt.", "error")
        return redirect(url_for("index", tab="cargar"))

    folder_path = request.form.get("folder_path", "").strip()
    recursive = request.form.get("recursive") == "on"
    conf = float(request.form.get("conf", 0.25))
    iou = float(request.form.get("iou", 0.5))
    point_count = int(request.form.get("point_count", 13))

    if not folder_path or not Path(folder_path).is_dir():
        flash("La carpeta del servidor no existe.", "error")
        return redirect(url_for("index", tab="cargar"))

    paths = []
    if recursive:
        iterator = Path(folder_path).rglob("*")
    else:
        iterator = Path(folder_path).glob("*")

    for path in iterator:
        if path.is_file() and is_image_file(str(path)):
            paths.append(path)

    if not paths:
        flash("No se encontraron imágenes compatibles en la carpeta.", "error")
        return redirect(url_for("index", tab="cargar"))

    processed = 0
    errors = []

    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            errors.append(f"No se pudo abrir: {path.name}")
            continue

        try:
            group_name = path.parent.name
            infer_and_save(frame, path.name, group_name, conf, iou, point_count)
            processed += 1
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    flash(f"Carpeta procesada: {processed} imagen/es.", "success")
    if errors:
        flash("Errores: " + " | ".join(errors[:5]), "error")

    return redirect(url_for("index", tab="inferencia"))


@app.route("/componentes/guardar", methods=["POST"])
def guardar_componentes():
    rows = zip(
        request.form.getlist("class_name"),
        request.form.getlist("component_name"),
        request.form.getlist("ggmf"),
        request.form.getlist("gmf"),
        request.form.getlist("notes"),
    )

    with db_connect() as conn:
        for class_name, component_name, ggmf, gmf, notes in rows:
            if not class_name:
                continue
            conn.execute(
                """
                UPDATE components
                SET component_name = ?, ggmf = ?, gmf = ?, notes = ?
                WHERE class_name = ?
                """,
                (component_name or class_name, ggmf, gmf, notes, class_name),
            )
        conn.commit()

    flash("Componentes actualizados.", "success")
    return redirect(url_for("index", tab="componentes"))


@app.route("/catalogo/crear", methods=["POST"])
def crear_catalogo():
    data = request.get_json(force=True, silent=True) or {}
    raw_name = (data.get("class_name") or data.get("name") or "").strip()
    component_name = (data.get("component_name") or raw_name).strip()
    if not raw_name:
        return jsonify({"error": "Indica un nombre para la etiqueta."}), 400

    class_key = slugify_class_name(raw_name)
    if re.fullmatch(r"[A-Za-z0-9_\-]+", raw_name.strip()):
        class_key = raw_name.strip()

    try:
        entry = ensure_catalog_entry(class_key, component_name or class_key)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"ok": True, "component": entry})


@app.route("/catalogo/crear-form", methods=["POST"])
def crear_catalogo_form():
    raw_name = (request.form.get("class_name") or "").strip()
    component_name = (request.form.get("component_name") or raw_name).strip()
    if not raw_name:
        flash("Indica un nombre para la etiqueta.", "error")
        return redirect(url_for("index", tab="componentes"))
    class_key = slugify_class_name(raw_name)
    if re.fullmatch(r"[A-Za-z0-9_\-]+", raw_name.strip()):
        class_key = raw_name.strip()
    try:
        ensure_catalog_entry(class_key, component_name or class_key)
        flash(f"Etiqueta '{class_key}' agregada al catálogo.", "success")
    except Exception as exc:
        flash(str(exc), "error")
    return redirect(url_for("index", tab="componentes"))


@app.route("/imagen/<int:image_id>/puntos", methods=["POST"])
def actualizar_puntos(image_id: int):
    data = request.get_json(force=True, silent=True) or {}
    items = data.get("points") or []
    if not items:
        return jsonify({"error": "No se recibieron puntos para actualizar.", "points": []}), 400

    detail = get_image_detail(image_id)
    if not detail or not detail.get("run"):
        return jsonify({"error": "Imagen o inferencia no encontrada.", "points": []}), 404

    run_id = int(detail["run"]["id"])
    coord_updates: Dict[int, Tuple[float, float]] = {}

    # Resolver catálogo fuera de la transacción de puntos
    for item in items:
        if item.get("class_name"):
            ensure_catalog_entry(str(item["class_name"]).strip(), item.get("component_name"))

    cmap = component_map()

    with db_connect() as conn:
        for item in items:
            point_index = int(item["point_index"])
            row = conn.execute(
                "SELECT * FROM points WHERE run_id = ? AND point_index = ?",
                (run_id, point_index),
            ).fetchone()
            if not row:
                continue

            point_name = item.get("point_name", row["point_name"] or f"P{point_index}")
            class_name = row["class_name"]
            if "class_name" in item:
                class_name = str(item["class_name"]).strip() if item["class_name"] else None

            manual_label = int(item.get("manual_label", row["manual_label"] or 0))
            if "class_name" in item and class_name:
                manual_label = 1

            component_name = item.get("component_name", row["component_name"])
            if class_name and (manual_label or not component_name):
                component_name = component_name or cmap.get(class_name, class_name)

            status = row["status"]
            if manual_label and class_name:
                status = "manual"
            elif not class_name:
                status = "sin_deteccion"

            has_coords = "x" in item and "y" in item
            x = float(item["x"]) if has_coords else float(row["x"])
            y = float(item["y"]) if has_coords else float(row["y"])

            # Etiqueta manual o solo metadatos: no pisar con YOLO
            if manual_label or not has_coords or model is None:
                conn.execute(
                    """
                    UPDATE points
                    SET x = ?, y = ?, point_name = ?, class_name = ?, component_name = ?,
                        status = ?, edited_manual = 1, manual_label = ?,
                        confidence = CASE WHEN ? = 1 THEN 1.0 ELSE confidence END
                    WHERE run_id = ? AND point_index = ?
                    """,
                    (
                        x,
                        y,
                        point_name,
                        class_name,
                        component_name,
                        status if manual_label else (status if not has_coords else row["status"]),
                        manual_label,
                        manual_label,
                        run_id,
                        point_index,
                    ),
                )
            else:
                conn.execute(
                    "UPDATE points SET point_name = ? WHERE run_id = ? AND point_index = ?",
                    (point_name, run_id, point_index),
                )
                coord_updates[point_index] = (x, y)
        conn.commit()

    try:
        if coord_updates:
            points = refresh_points_for_image(image_id, coord_updates)
        else:
            points = serialize_points_for_run(run_id)
    except Exception as exc:
        return jsonify({"error": str(exc), "points": []}), 400

    return jsonify({"ok": True, "points": points})


@app.route("/imagen/<int:image_id>/puntos/agregar", methods=["POST"])
def agregar_punto(image_id: int):
    data = request.get_json(force=True, silent=True) or {}
    detail = get_image_detail(image_id)
    if not detail or not detail.get("run") or not detail.get("image"):
        return jsonify({"error": "Imagen o inferencia no encontrada."}), 404

    image = detail["image"]
    run_id = int(detail["run"]["id"])
    width = float(image["width"] or 1)
    height = float(image["height"] or 1)
    x = float(data.get("x", width / 2))
    y = float(data.get("y", height / 2))
    x = min(max(x, 0), width)
    y = min(max(y, 0), height)

    class_name = (data.get("class_name") or "").strip() or None
    point_name = (data.get("point_name") or "").strip()
    manual_label = 1 if class_name else 0
    status = "manual" if class_name else "sin_deteccion"
    component_name = None
    if class_name:
        ensure_catalog_entry(class_name, data.get("component_name") or class_name)
        component_name = component_map().get(class_name, class_name)

    with db_connect() as conn:
        point_index = next_point_index(conn, run_id)
        if not point_name:
            point_name = f"P{point_index}"
        conn.execute(
            """
            INSERT INTO points(
              run_id, point_index, point_name, x, y, class_name, confidence,
              component_name, status, edited_manual, manual_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                run_id,
                point_index,
                point_name,
                x,
                y,
                class_name,
                1.0 if manual_label else 0.0,
                component_name,
                status,
                manual_label,
            ),
        )
        conn.execute(
            "UPDATE inference_runs SET point_count = (SELECT COUNT(*) FROM points WHERE run_id = ?) WHERE id = ?",
            (run_id, run_id),
        )
        conn.commit()

    return jsonify({"ok": True, "points": serialize_points_for_run(run_id)})


@app.route("/imagen/<int:image_id>/puntos/eliminar", methods=["POST"])
def eliminar_punto(image_id: int):
    data = request.get_json(force=True, silent=True) or {}
    point_index = data.get("point_index")
    if point_index is None:
        return jsonify({"error": "Falta point_index."}), 400

    detail = get_image_detail(image_id)
    if not detail or not detail.get("run"):
        return jsonify({"error": "Imagen o inferencia no encontrada."}), 404

    run_id = int(detail["run"]["id"])
    with db_connect() as conn:
        row = conn.execute(
            "SELECT mask_path FROM points WHERE run_id = ? AND point_index = ?",
            (run_id, int(point_index)),
        ).fetchone()
        conn.execute(
            "DELETE FROM points WHERE run_id = ? AND point_index = ?",
            (run_id, int(point_index)),
        )
        conn.execute(
            "UPDATE inference_runs SET point_count = (SELECT COUNT(*) FROM points WHERE run_id = ?) WHERE id = ?",
            (run_id, run_id),
        )
        conn.commit()

    if row and row["mask_path"]:
        try:
            Path(row["mask_path"]).unlink(missing_ok=True)
        except Exception:
            pass

    return jsonify({"ok": True, "points": serialize_points_for_run(run_id)})


@app.route("/api/sam3/status", methods=["GET"])
def api_sam3_status():
    return jsonify({"ok": True, **sam3_status()})


@app.route("/api/sam3/load", methods=["POST"])
def api_sam3_load():
    try:
        status = load_sam3(force=True)
        return jsonify({"ok": True, **status})
    except Exception as exc:
        return jsonify({"ok": False, **sam3_status(), "error": str(exc)}), 503


@app.route("/imagen/<int:image_id>/puntos/sam3", methods=["POST"])
def generar_mascara_sam3(image_id: int):
    data = request.get_json(force=True, silent=True) or {}
    point_index = data.get("point_index")
    if point_index is None:
        return jsonify({"error": "Falta point_index."}), 400

    detail = get_image_detail(image_id)
    if not detail or not detail.get("image") or not detail.get("run"):
        return jsonify({"error": "Imagen o inferencia no encontrada."}), 404

    image = detail["image"]
    run_id = int(detail["run"]["id"])
    point_index = int(point_index)

    with db_connect() as conn:
        row = conn.execute(
            "SELECT * FROM points WHERE run_id = ? AND point_index = ?",
            (run_id, point_index),
        ).fetchone()
    if not row:
        return jsonify({"error": "Punto no encontrado."}), 404

    x = float(data.get("x", row["x"]))
    y = float(data.get("y", row["y"]))

    frame = cv2.imread(str(image["original_path"]))
    if frame is None:
        return jsonify({"error": "No se pudo abrir la imagen original."}), 400

    try:
        if not sam3_status()["loaded"]:
            load_sam3()
        result = predict_mask_from_point(
            frame,
            x,
            y,
            cache_key=f"{image_id}:{image.get('original_path')}",
            multimask=True,
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), **sam3_status()}), 503

    mask = result["mask"]
    mask_path = save_mask_png(mask, image_id, point_index)

    with db_connect() as conn:
        conn.execute(
            """
            UPDATE points
            SET x = ?, y = ?, mask_path = ?, mask_score = ?, mask_source = 'sam3',
                edited_manual = 1
            WHERE run_id = ? AND point_index = ?
            """,
            (x, y, str(mask_path), float(result["score"]), run_id, point_index),
        )
        conn.commit()

    # Preview: overlay on original for UI
    overlay = mask_overlay_bgr(frame, mask)
    ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    overlay_url = None
    if ok:
        import base64

        overlay_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    points = serialize_points_for_run(run_id)
    return jsonify(
        {
            "ok": True,
            "point_index": point_index,
            "mask_score": result["score"],
            "mask_path": str(mask_path),
            "mask_png": mask_to_data_url(mask),
            "overlay_url": overlay_url,
            "points": points,
            "sam3": sam3_status(),
        }
    )


def _run_retrain_job(job_id: int):
    job = get_retrain_job(job_id)
    if not job:
        return

    log_path = Path(job["log_path"])
    try:
        with db_connect() as conn:
            conn.execute(
                "UPDATE retrain_jobs SET status = ?, started_at = ?, error = NULL WHERE id = ?",
                ("running", now_iso(), job_id),
            )
            conn.commit()

            dataset_dir = RETRAIN_DATASETS_DIR / f"job_{job_id}"
            info = build_dataset_from_corrections(
                conn,
                dataset_dir,
                base_classes=class_names,
                radius_ratio=float(job["radius_ratio"] or 0.04),
            )
            conn.execute(
                """
                UPDATE retrain_jobs
                SET dataset_path = ?, num_images = ?, num_points = ?
                WHERE id = ?
                """,
                (info["dataset_yaml"], info["num_images"], info["num_points"], job_id),
            )
            conn.commit()

        base_weights = job["base_weights"] or weights_loaded or str(DEFAULT_WEIGHTS)
        run_name = f"job_{job_id}"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "train_seg.py"),
            "--data",
            info["dataset_yaml"],
            "--model",
            str(base_weights),
            "--epochs",
            str(int(job["epochs"] or 30)),
            "--imgsz",
            str(int(job["imgsz"] or 640)),
            "--batch",
            str(int(job["batch"] or 2)),
            "--project",
            str(RETRAIN_RUNS_DIR),
            "--name",
            run_name,
            "--patience",
            "15",
            "--workers",
            "2",
        ]

        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"CMD: {' '.join(cmd)}\n\n")
            logf.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )

        weights_out = RETRAIN_RUNS_DIR / run_name / "weights" / "best.pt"
        if proc.returncode != 0 or not weights_out.is_file():
            raise RuntimeError(
                f"Entrenamiento falló (code={proc.returncode}). Revisa el log: {log_path}"
            )

        with db_connect() as conn:
            conn.execute(
                """
                UPDATE retrain_jobs
                SET status = ?, finished_at = ?, weights_out = ?
                WHERE id = ?
                """,
                ("done", now_iso(), str(weights_out.resolve()), job_id),
            )
            conn.commit()
    except Exception as exc:
        with db_connect() as conn:
            conn.execute(
                """
                UPDATE retrain_jobs
                SET status = ?, finished_at = ?, error = ?
                WHERE id = ?
                """,
                ("error", now_iso(), str(exc), job_id),
            )
            conn.commit()
        try:
            with open(log_path, "a", encoding="utf-8") as logf:
                logf.write(f"\nERROR: {exc}\n")
        except Exception:
            pass


@app.route("/api/retrain/start", methods=["POST"])
def api_retrain_start():
    if model is None and not weights_loaded:
        return jsonify({"error": "No hay modelo/pesos cargados para fine-tuning."}), 500

    data = request.get_json(force=True, silent=True) or {}
    epochs = int(data.get("epochs", 30))
    imgsz = int(data.get("imgsz", 640))
    batch = int(data.get("batch", 2))
    radius_ratio = float(data.get("radius_ratio", 0.04))

    with db_connect() as conn:
        running = conn.execute(
            "SELECT id FROM retrain_jobs WHERE status IN ('queued', 'running') LIMIT 1"
        ).fetchone()
        if running:
            return jsonify({"error": "Ya hay un reentrenamiento en curso.", "job_id": running["id"]}), 409

        stats = count_trainable_corrections(conn)
        if stats["num_points"] <= 0:
            return jsonify({"error": "No hay puntos corregidos/etiquetados para reentrenar."}), 400

        cur = conn.execute(
            """
            INSERT INTO retrain_jobs(
              status, created_at, epochs, imgsz, batch, radius_ratio,
              base_weights, num_images, num_points, log_path
            ) VALUES ('queued', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso(),
                epochs,
                imgsz,
                batch,
                radius_ratio,
                weights_loaded,
                stats["num_images"],
                stats["num_points"],
                "",
            ),
        )
        job_id = cur.lastrowid
        log_path = RETRAIN_LOGS_DIR / f"job_{job_id}.log"
        log_path.write_text(f"Job {job_id} creado {now_iso()}\n", encoding="utf-8")
        conn.execute("UPDATE retrain_jobs SET log_path = ? WHERE id = ?", (str(log_path), job_id))
        conn.commit()

    thread = threading.Thread(target=_run_retrain_job, args=(job_id,), daemon=True)
    thread.start()
    return jsonify({"ok": True, "job_id": job_id, "stats": stats})


@app.route("/api/retrain/status", methods=["GET"])
def api_retrain_status():
    job_id = request.args.get("job_id", type=int)
    job = get_retrain_job(job_id) if job_id else get_latest_retrain_job()
    with db_connect() as conn:
        stats = count_trainable_corrections(conn)
    return jsonify({"ok": True, "job": job, "stats": stats})


@app.route("/api/retrain/activate", methods=["POST"])
def api_retrain_activate():
    data = request.get_json(force=True, silent=True) or {}
    job_id = data.get("job_id")
    job = get_retrain_job(int(job_id)) if job_id else get_latest_retrain_job()
    if not job or job.get("status") != "done" or not job.get("weights_out"):
        return jsonify({"error": "No hay pesos nuevos listos para activar."}), 400

    weights_path = Path(job["weights_out"])
    if not weights_path.is_file():
        return jsonify({"error": f"No existe el archivo de pesos: {weights_path}"}), 404

    try:
        load_model(str(weights_path))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"ok": True, "weights": weights_loaded, "classes": class_names, "job_id": job["id"]})


def ensure_clean_preview(image_id: int):
    """Regenera la vista de revisión: solo máscaras de color, sin puntos morados de YOLO."""
    if model is None:
        return

    detail = get_image_detail(image_id)
    if not detail or not detail.get("image") or not detail.get("run"):
        return

    image = detail["image"]
    run = detail["run"]
    if not image.get("result_path"):
        return

    frame = cv2.imread(str(image["original_path"]))
    if frame is None:
        return

    result = model.predict(frame, conf=float(run["conf"]), iou=float(run["iou"]), verbose=False)[0]
    plotted = plot_segmentation_clean(result, frame)
    result_abs = APP_ROOT / "static" / image["result_path"]
    result_abs.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(result_abs), plotted)


def render_image_with_points(image_id: int):
    if model is None:
        raise RuntimeError("Modelo no cargado.")

    detail = get_image_detail(image_id)
    if not detail or not detail.get("image") or not detail.get("run"):
        raise ValueError("Imagen o inferencia no encontrada.")

    image = detail["image"]
    run = detail["run"]
    frame = cv2.imread(str(image["original_path"]))
    if frame is None:
        raise ValueError("No se pudo abrir la imagen original.")

    result = model.predict(frame, conf=float(run["conf"]), iou=float(run["iou"]), verbose=False)[0]
    plotted = plot_segmentation_clean(result, frame)

    point_predictions = []
    for row in detail["points"]:
        point_predictions.append({
            "id": row.get("point_index"),
            "point_name": row.get("point_name") or f"P{row.get('point_index')}",
            "x": float(row["x"]),
            "y": float(row["y"]),
            "status": row.get("status") or "sin_deteccion",
            "class_name": row.get("class_name"),
        })
    return draw_points_on_result(plotted, point_predictions)


@app.route("/imagen/<int:image_id>/descargar", methods=["GET"])
def descargar_resultado(image_id: int):
    detail = get_image_detail(image_id)
    if not detail or not detail.get("image"):
        return jsonify({"error": "No hay resultado para descargar."}), 404

    try:
        import tempfile

        rendered = render_image_with_points(image_id)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, rendered)
        download_name = Path(detail["image"]["original_name"]).stem + "_etiquetada.jpg"
        return send_file(tmp.name, as_attachment=True, download_name=download_name)
    except Exception as exc:
        result_path = detail["image"].get("result_path")
        if result_path:
            result_abs = APP_ROOT / "static" / result_path
            if result_abs.is_file():
                download_name = Path(detail["image"]["original_name"]).stem + "_etiquetada.jpg"
                return send_file(result_abs, as_attachment=True, download_name=download_name)
        return jsonify({"error": str(exc)}), 400


@app.route("/api/imagen/<int:image_id>/estado", methods=["POST"])
def api_actualizar_estado(image_id: int):
    data = request.get_json(force=True, silent=True) or {}
    status = data.get("status", "").strip().lower()
    if status not in {"revisado", "pendiente"}:
        return jsonify({"error": "Estado inválido. Use revisado o pendiente."}), 400

    with db_connect() as conn:
        cur = conn.execute("SELECT id FROM images WHERE id = ?", (image_id,))
        if not cur.fetchone():
            return jsonify({"error": "Imagen no encontrada."}), 404
        conn.execute("UPDATE images SET status = ? WHERE id = ?", (status, image_id))
        conn.commit()

    return jsonify({"ok": True, "image_id": image_id, "status": status})


@app.route("/imagen/<int:image_id>/revisado", methods=["POST"])
def marcar_revisado(image_id: int):
    with db_connect() as conn:
        conn.execute("UPDATE images SET status = 'revisado' WHERE id = ?", (image_id,))
        conn.commit()

    flash("Imagen marcada como revisada.", "success")
    return redirect(url_for("index", tab="revision", image_id=image_id))


@app.route("/imagen/<int:image_id>/pendiente", methods=["POST"])
def marcar_pendiente(image_id: int):
    with db_connect() as conn:
        conn.execute("UPDATE images SET status = 'pendiente' WHERE id = ?", (image_id,))
        conn.commit()

    flash("Imagen marcada como pendiente.", "success")
    return redirect(url_for("index", tab="revision", image_id=image_id))


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify(
        {
            "ok": model is not None,
            "classes": class_names,
            "weights": weights_loaded,
            "db_path": str(DB_PATH),
            "message": "Modelo cargado" if model is not None else "Modelo no cargado",
        }
    )


@app.route("/api/predict_paths", methods=["POST"])
def api_predict_paths():
    if model is None:
        return jsonify({"error": "Modelo no cargado.", "results": []}), 500

    data = request.get_json(force=True, silent=True) or {}
    conf = float(data.get("conf", 0.25))
    iou = float(data.get("iou", 0.5))
    return_image = bool(data.get("return_image", True))
    point_count = int(data.get("point_count", 13))
    save_to_db = bool(data.get("save_to_db", False))
    images = data.get("images", [])

    if not images:
        return jsonify({"error": "No se recibieron imágenes.", "results": []}), 400

    results = []
    for img in images:
        image_path = img.get("path")
        points = img.get("points", [])
        item = process_image_from_path(
            image_path,
            conf,
            iou,
            points=points,
            return_image=return_image,
            point_count=point_count,
        )

        if save_to_db and item.get("error") is None and image_path:
            frame = cv2.imread(image_path)
            if frame is not None:
                group_name = Path(image_path).parent.name
                infer_and_save(frame, Path(image_path).name, group_name, conf, iou, point_count)
                item["saved_to_db"] = True

        results.append(item)

    return jsonify(
        {
            "error": None,
            "conf": conf,
            "iou": iou,
            "classes": class_names,
            "results": results,
        }
    )


@app.route("/api/predict_folder", methods=["POST"])
def api_predict_folder():
    if model is None:
        return jsonify({"error": "Modelo no cargado.", "results": []}), 500

    data = request.get_json(force=True, silent=True) or {}
    folder_path = data.get("folder_path")
    conf = float(data.get("conf", 0.25))
    iou = float(data.get("iou", 0.5))
    recursive = bool(data.get("recursive", False))
    return_image = bool(data.get("return_image", False))
    point_count = int(data.get("point_count", 13))
    save_to_db = bool(data.get("save_to_db", False))

    if not folder_path or not os.path.isdir(folder_path):
        return jsonify({"error": "La carpeta no existe.", "results": []}), 400

    image_paths = []
    if recursive:
        for root, _, files in os.walk(folder_path):
            for filename in files:
                path = os.path.join(root, filename)
                if is_image_file(path):
                    image_paths.append(path)
    else:
        for path in Path(folder_path).glob("*"):
            if path.is_file() and is_image_file(str(path)):
                image_paths.append(str(path))

    results = []
    for image_path in image_paths:
        item = process_image_from_path(
            image_path,
            conf,
            iou,
            points=[],
            return_image=return_image,
            point_count=point_count,
        )

        if save_to_db and item.get("error") is None:
            frame = cv2.imread(image_path)
            if frame is not None:
                group_name = Path(image_path).parent.name
                infer_and_save(frame, Path(image_path).name, group_name, conf, iou, point_count)
                item["saved_to_db"] = True

        results.append(item)

    return jsonify(
        {
            "error": None,
            "folder_path": folder_path,
            "total_images": len(image_paths),
            "conf": conf,
            "iou": iou,
            "classes": class_names,
            "results": results,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="CoArrCP IA Flask: interfaz web + SQLite + API para inferencia YOLO-seg."
    )
    parser.add_argument(
        "--weights",
        default=os.getenv("BENTHIC_WEIGHTS", str(DEFAULT_WEIGHTS)),
        help="Ruta al best.pt entrenado.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host Flask.")
    parser.add_argument("--port", type=int, default=9001, help="Puerto Flask.")
    parser.add_argument("--no-model", action="store_true", help="Arranca la interfaz sin cargar YOLO.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Recarga automática al cambiar código (solo desarrollo).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_db()

    if not args.no_model:
        load_model(args.weights)
        print(f"Modelo cargado: {weights_loaded}")
        print(f"Clases: {class_names}")
    else:
        print("Modo interfaz: modelo no cargado.")

    print(f"Interfaz web: http://{args.host}:{args.port}/")
    print(f"API health:   http://{args.host}:{args.port}/api/health")
    print(f"API paths:    POST http://{args.host}:{args.port}/api/predict_paths")
    print(f"API folder:   POST http://{args.host}:{args.port}/api/predict_folder")
    print(f"SQLite:       {DB_PATH}")
    if args.reload:
        print("Modo desarrollo: recarga automática activada (--reload)")
    app.run(host=args.host, port=args.port, debug=args.reload, use_reloader=args.reload)
