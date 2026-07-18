#!/usr/bin/env python3
"""
Prototipo v1 — Mini Software de Segmentación Bentónica (Evaluación heurística 1).

Interfaz simple: subir imagen, conf/iou visibles, resultados en la misma página.
No modifica CoArrCP IA. Usar puerto 9002 mientras la app nueva corre en 9001.

  python3 run_prototipo_v1.py --port 9002
"""
import argparse
import base64
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, render_template, request
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "experimentos_seg3/runs/seg3_r012/weights/best.pt"

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))
model: Optional[YOLO] = None
class_names: List[str] = []


def load_model(weights_path: str):
    global model, class_names
    path = Path(weights_path)
    if not path.is_file():
        path = PROJECT_ROOT / weights_path
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de pesos: {weights_path}")
    model = YOLO(str(path))
    names = model.names if isinstance(model.names, dict) else {}
    class_names = [names[i] for i in sorted(names.keys())]


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


def image_to_data_url(bgr_img) -> str:
    ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def process_file(raw: bytes, filename: str, conf: float, iou: float):
    if model is None:
        return {"filename": filename, "error": "Modelo no cargado."}

    frame = decode_image(raw)
    if frame is None:
        return {"filename": filename, "error": "No se pudo decodificar la imagen."}

    result = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]
    coverage = compute_coverage(result, class_names)

    coverage_rows = [
        {"class_name": name, "percent": f"{coverage.get(name, 0.0):.2f}"}
        for name in class_names
    ]

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            c = int(cls[i])
            label = class_names[c] if 0 <= c < len(class_names) else str(c)
            detections.append({
                "class_name": label,
                "confidence": f"{float(scores[i]):.3f}",
                "bbox": ", ".join(f"{v:.1f}" for v in boxes[i].tolist()),
            })

    plotted = result.plot()
    return {
        "filename": filename,
        "image_url": image_to_data_url(plotted),
        "coverage": coverage_rows,
        "detections": detections,
        "num_detections": len(detections),
        "error": None,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    conf = 0.25
    iou = 0.5
    error = None
    results = []

    if request.method == "POST":
        try:
            conf = float(request.form.get("conf", 0.25))
            iou = float(request.form.get("iou", 0.5))
        except ValueError:
            error = "conf e iou deben ser números."
            return render_template(
                "index.html",
                classes=class_names,
                conf=conf,
                iou=iou,
                results=[],
                error=error,
            )

        if model is None:
            error = "Modelo no cargado. Ejecuta con --weights apuntando a best.pt."
        else:
            files = request.files.getlist("file")
            if not files or all(not f.filename for f in files):
                error = "Selecciona al menos una imagen."
            else:
                for f in files:
                    if not f.filename:
                        continue
                    raw = f.read()
                    results.append(process_file(raw, f.filename, conf, iou))

    return render_template(
        "index.html",
        classes=class_names,
        conf=conf,
        iou=iou,
        results=results,
        error=error,
    )


def main():
    parser = argparse.ArgumentParser(description="Prototipo v1 — segmentación bentónica (capturas DCU)")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()

    load_model(args.weights)
    print(f"Prototipo v1 (Eval. heurística original)")
    print(f"Modelo: {args.weights}")
    print(f"Clases: {class_names}")
    print(f"Abrir: http://{args.host}:{args.port}/")
    print("(CoArrCP IA nuevo sigue en http://127.0.0.1:9001/ si está activo)")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
