import argparse
import base64
import os
import time
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DATA_DIR = APP_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "coarcp_ia.sqlite"
STATIC_RESULTS_DIR = APP_ROOT / "static" / "results"
DEFAULT_WEIGHTS = PROJECT_ROOT / "experimentos_seg3/runs/seg3_r012/weights/best.pt"

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
            """
        )

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
                "x": p.get("x"),
                "y": p.get("y"),
                "class_name": None,
                "confidence": 0.0,
                "status": "sin_deteccion",
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
                    "x": x_original,
                    "y": y_original,
                    "class_name": label,
                    "confidence": round(float(confs[idx]), 4),
                    "status": "detectado",
                }
                break

        if assigned is None:
            assigned = {
                "id": p.get("id"),
                "x": x_original,
                "y": y_original,
                "class_name": None,
                "confidence": 0.0,
                "status": "sin_deteccion",
            }

        predictions.append(assigned)

    return predictions


def draw_points_on_result(image, point_predictions):
    out = image.copy()
    h, w = image.shape[:2]
    radius = max(14, int(min(w, h) * 0.014))
    border = radius + 4
    font_scale = max(0.7, min(w, h) / 1200)

    for p in point_predictions:
        x = int(round(float(p["x"])))
        y = int(round(float(p["y"])))
        point_id = p.get("id", p.get("point_index"))

        if p["status"] == "detectado":
            color = (34, 197, 94)
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

    plotted = result.plot()
    plotted = draw_points_on_result(plotted, point_predictions)

    image_url = None
    if result_abs_path is not None:
        cv2.imwrite(str(result_abs_path), plotted)
    if return_image_url:
        image_url = image_to_data_url(plotted)

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
            conn.execute(
                """
                INSERT INTO points(run_id, point_index, x, y, class_name, confidence, component_name, status, edited_manual)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    run_id,
                    p["id"],
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
    for row in detail["points"]:
        point_index = int(row["point_index"])
        x = float(row["x"])
        y = float(row["y"])
        if updated_coords and point_index in updated_coords:
            x, y = updated_coords[point_index]
        external_points.append({"id": point_index, "x": x, "y": y})

    result = model.predict(frame, conf=float(run["conf"]), iou=float(run["iou"]), verbose=False)[0]
    point_predictions = assign_points_to_masks(result, external_points, width, height)
    cmap = component_map()

    with db_connect() as conn:
        for p in point_predictions:
            class_name = p.get("class_name")
            conn.execute(
                """
                UPDATE points
                SET x = ?, y = ?, class_name = ?, confidence = ?, component_name = ?, status = ?, edited_manual = 1
                WHERE run_id = ? AND point_index = ?
                """,
                (
                    p["x"],
                    p["y"],
                    class_name,
                    p["confidence"],
                    cmap.get(class_name, class_name) if class_name else None,
                    p["status"],
                    run["id"],
                    p["id"],
                ),
            )
        conn.commit()

    plotted = result.plot()
    plotted = draw_points_on_result(plotted, point_predictions)

    if image["result_path"]:
        result_abs = APP_ROOT / "static" / image["result_path"]
        result_abs.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(result_abs), plotted)

    enriched = []
    for p in point_predictions:
        class_name = p.get("class_name")
        enriched.append({
            "point_index": p["id"],
            "x": p["x"],
            "y": p["y"],
            "class_name": class_name,
            "component_name": cmap.get(class_name, class_name) if class_name else None,
            "confidence": p["confidence"],
            "status": p["status"],
            "edited_manual": 1,
        })
    return enriched


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


@app.route("/", methods=["GET"])
def index():
    tab = request.args.get("tab", "inicio")
    selected_image_id = request.args.get("image_id", type=int)

    return render_template(
        "index.html",
        tab=tab,
        model_loaded=model is not None,
        classes=class_names,
        weights_loaded=weights_loaded,
        db_path=str(DB_PATH),
        overview=get_overview(),
        images=get_images(),
        components=get_components(),
        detail=get_image_detail(selected_image_id),
        selected_image_id=selected_image_id,
    )


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


@app.route("/imagen/<int:image_id>/puntos", methods=["POST"])
def actualizar_puntos(image_id: int):
    if model is None:
        return jsonify({"error": "Modelo no cargado.", "points": []}), 500

    data = request.get_json(force=True, silent=True) or {}
    updates = {}
    for item in data.get("points", []):
        point_index = int(item.get("point_index"))
        updates[point_index] = (float(item.get("x")), float(item.get("y")))

    if not updates:
        return jsonify({"error": "No se recibieron puntos para actualizar.", "points": []}), 400

    try:
        points = refresh_points_for_image(image_id, updates)
    except Exception as exc:
        return jsonify({"error": str(exc), "points": []}), 400

    return jsonify({"ok": True, "points": points})


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
    app.run(host=args.host, port=args.port, debug=False)
