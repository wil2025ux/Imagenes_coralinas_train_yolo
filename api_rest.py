import argparse
import os
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO


app = FastAPI(title="Benthic Segmentation API", version="1.0.0")
model = None
class_names: List[str] = []


def load_model(weights_path: str):
    global model, class_names
    model = YOLO(weights_path)
    names = model.names if isinstance(model.names, dict) else {}
    class_names = [names[i] for i in sorted(names.keys())]


def decode_image(raw: bytes):
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": class_names,
    }


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...), conf: float = 0.25, iou: float = 0.5):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    raw = await file.read()
    img = decode_image(raw)
    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen enviada.")

    result = model.predict(img, conf=conf, iou=iou, verbose=False)[0]
    coverage = compute_coverage(result, class_names)

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            c = int(cls[i])
            label = class_names[c] if 0 <= c < len(class_names) else str(c)
            detections.append(
                {
                    "class_id": c,
                    "class_name": label,
                    "confidence": float(scores[i]),
                    "bbox_xyxy": [float(v) for v in boxes[i].tolist()],
                }
            )

    return {
        "coverage_percent": coverage,
        "detections": detections,
        "num_detections": len(detections),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="API REST local para inferencia bentonica.")
    parser.add_argument(
        "--weights",
        default=os.getenv("BENTHIC_WEIGHTS", "runs/segment/benthos_yolo11n_seg/weights/best.pt"),
        help="Ruta a pesos entrenados.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host del servidor.")
    parser.add_argument("--port", type=int, default=8000, help="Puerto del servidor.")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    load_model(args.weights)
    uvicorn.run(app, host=args.host, port=args.port)
