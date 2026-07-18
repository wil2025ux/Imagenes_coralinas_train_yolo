import argparse
import csv
from pathlib import Path

from ultralytics import YOLO
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analiza errores por imagen comparando clases GT vs predichas "
            "en un split (train/val/test)."
        )
    )
    parser.add_argument("--weights", required=True, help="Ruta a best.pt")
    parser.add_argument("--data", required=True, help="Ruta a dataset.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--output-csv",
        default="analisis_errores_split.csv",
        help="CSV de salida con detalle por imagen",
    )
    parser.add_argument(
        "--output-misses-txt",
        default="imagenes_no_detectadas.txt",
        help="TXT de imágenes con GT pero sin predicción",
    )
    return parser.parse_args()


def resolve_split_dir(data_yaml: Path, split: str):
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg.get("path", data_yaml.parent))
    split_entry = cfg.get(split)
    if split_entry is None:
        raise ValueError(f"El split '{split}' no existe en {data_yaml}")
    split_path = Path(split_entry)
    if not split_path.is_absolute():
        split_path = base / split_path
    return split_path.resolve()


def image_to_label_path(image_path: Path):
    p = str(image_path)
    if "/images/" in p:
        p = p.replace("/images/", "/labels/")
    elif "\\images\\" in p:
        p = p.replace("\\images\\", "\\labels\\")
    return Path(p).with_suffix(".txt")


def read_gt_classes(label_path: Path):
    if not label_path.exists():
        return set()
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return set()
    classes = set()
    for line in txt.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            classes.add(int(parts[0]))
        except Exception:
            continue
    return classes


def main():
    args = parse_args()
    model = YOLO(args.weights)
    split_dir = resolve_split_dir(Path(args.data), args.split)

    if not split_dir.exists():
        raise FileNotFoundError(f"No existe el directorio del split: {split_dir}")

    image_files = sorted([p for p in split_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not image_files:
        raise ValueError(f"No se encontraron imágenes en: {split_dir}")

    rows = []
    misses = []
    total_gt_obj = 0
    total_no_pred = 0
    total_mismatch = 0
    total_ok = 0

    results = model.predict(
        source=[str(p) for p in image_files],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False,
    )

    for r in results:
        img_path = Path(r.path).resolve()
        label_path = image_to_label_path(img_path)
        gt = read_gt_classes(label_path)
        pred = set()
        if r.boxes is not None and len(r.boxes) > 0:
            pred = set(r.boxes.cls.int().cpu().numpy().tolist())

        names = model.names if isinstance(model.names, dict) else {}
        gt_names = [names.get(i, str(i)) for i in sorted(gt)]
        pred_names = [names.get(i, str(i)) for i in sorted(pred)]
        missed_cls = sorted(list(gt - pred))
        extra_cls = sorted(list(pred - gt))
        missed_names = [names.get(i, str(i)) for i in missed_cls]
        extra_names = [names.get(i, str(i)) for i in extra_cls]

        if gt and not pred:
            status = "no_detectada"
            total_no_pred += 1
            misses.append(str(img_path))
        elif gt == pred:
            status = "ok"
            total_ok += 1
        elif gt:
            status = "desajuste_clases"
            total_mismatch += 1
        else:
            status = "sin_objeto_gt"

        if gt:
            total_gt_obj += 1

        rows.append(
            {
                "image": str(img_path),
                "label_file": str(label_path),
                "status": status,
                "gt_class_ids": ",".join(map(str, sorted(gt))),
                "gt_class_names": ",".join(gt_names),
                "pred_class_ids": ",".join(map(str, sorted(pred))),
                "pred_class_names": ",".join(pred_names),
                "missed_class_ids": ",".join(map(str, missed_cls)),
                "missed_class_names": ",".join(missed_names),
                "extra_class_ids": ",".join(map(str, extra_cls)),
                "extra_class_names": ",".join(extra_names),
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output_misses = Path(args.output_misses_txt)
    output_misses.parent.mkdir(parents=True, exist_ok=True)
    output_misses.write_text("\n".join(misses) + ("\n" if misses else ""), encoding="utf-8")

    print(f"Split analizado: {args.split}")
    print(f"Imagenes analizadas: {len(rows)}")
    print(f"Imagenes con objetos GT: {total_gt_obj}")
    print(f"OK (coinciden clases): {total_ok}")
    print(f"Sin deteccion con GT: {total_no_pred}")
    print(f"Desajuste de clases: {total_mismatch}")
    print(f"CSV: {output_csv.resolve()}")
    print(f"No detectadas (txt): {output_misses.resolve()}")


if __name__ == "__main__":
    main()
