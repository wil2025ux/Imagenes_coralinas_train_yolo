import argparse

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_NAMES = ["algas", "corales", "almejas", "esponjas", "arena"]


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia en webcam para segmentacion bentonica.")
    parser.add_argument("--weights", required=True, help="Ruta a pesos entrenados (best.pt).")
    parser.add_argument("--source", default="0", help='Fuente de video: "0" webcam o ruta de video.')
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza.")
    parser.add_argument("--iou", type=float, default=0.5, help="Umbral IoU NMS.")
    parser.add_argument("--width", type=int, default=1240, help="Ancho de captura.")
    parser.add_argument("--height", type=int, default=1080, help="Alto de captura.")
    return parser.parse_args()


def resolve_source(source):
    if source.isdigit():
        return int(source)
    return source


def compute_coverage_percent(result, class_names):
    coverage_pixels = {idx: 0 for idx in range(len(class_names))}

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return {class_names[i]: 0.0 for i in range(len(class_names))}

    masks = result.masks.data.cpu().numpy() > 0.5
    _, h, w = masks.shape
    total_pixels = h * w
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    order = np.argsort(-confs)
    occupied = np.zeros((h, w), dtype=bool)

    for idx in order:
        cls_id = classes[idx]
        if cls_id < 0 or cls_id >= len(class_names):
            continue
        mask = masks[idx]
        new_pixels = mask & (~occupied)
        px = int(new_pixels.sum())
        coverage_pixels[cls_id] += px
        occupied |= mask

    coverage_pct = {
        class_names[i]: (100.0 * coverage_pixels[i] / total_pixels) for i in range(len(class_names))
    }
    return coverage_pct


def main():
    args = parse_args()
    model = YOLO(args.weights)
    source = resolve_source(args.source)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la fuente de video.")

    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(DEFAULT_NAMES)}
    class_names = [names[i] for i in sorted(names.keys())]

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
        result = results[0]
        vis = result.plot()
        coverage = compute_coverage_percent(result, class_names)

        y = 30
        for name in class_names:
            txt = f"{name}: {coverage.get(name, 0.0):.2f}%"
            cv2.putText(vis, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            y += 26

        cv2.putText(vis, "Presiona q para salir", (20, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("Segmentacion bentonica YOLO", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
