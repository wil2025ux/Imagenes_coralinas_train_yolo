import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_CLASS_ORDER = ["algas", "corales", "almejas", "esponjas", "arena"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convierte un dataset COCO con poligonos a YOLO-seg y genera split train/val/test."
    )
    parser.add_argument("--coco-json", default="result_coco.json", help="Ruta a JSON COCO.")
    parser.add_argument("--images-dir", default="images", help="Carpeta con imagenes originales.")
    parser.add_argument(
        "--output-dir",
        default="benthic_yolo_seg",
        help="Carpeta de salida para dataset YOLO-seg.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Proporcion train.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Proporcion val.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Proporcion test.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para split reproducible.")
    parser.add_argument(
        "--class-order",
        nargs="+",
        default=DEFAULT_CLASS_ORDER,
        help="Orden final de clases en YOLO (ejemplo: algas corales almejas esponjas arena).",
    )
    return parser.parse_args()


def validate_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Las proporciones deben sumar 1.0, pero suman {total:.6f}.")


def to_yolo_polygon(points, width, height):
    if len(points) < 6 or len(points) % 2 != 0:
        return None
    out = []
    for i in range(0, len(points), 2):
        x = points[i] / width
        y = points[i + 1] / height
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        out.append((x, y))
    return out


def find_image_path(images_dir, file_name):
    candidate = images_dir / file_name
    if candidate.exists():
        return candidate
    candidate = images_dir / Path(file_name).name
    if candidate.exists():
        return candidate
    # Compatibilidad con exports de Label Studio: <uuid>-NOMBRE_ORIGINAL.JPG
    basename = Path(file_name).name
    if "-" in basename:
        stripped = basename.split("-", 1)[1]
        candidate = images_dir / stripped
        if candidate.exists():
            return candidate
    return None


def ensure_structure(output_dir):
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(output_dir, class_names):
    yaml_path = output_dir / "dataset.yaml"
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    coco_json = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if not coco_json.exists():
        raise FileNotFoundError(f"No existe JSON COCO: {coco_json}")
    if not images_dir.exists():
        raise FileNotFoundError(f"No existe carpeta de imagenes: {images_dir}")

    data = json.loads(coco_json.read_text(encoding="utf-8"))

    categories = data.get("categories", [])
    if not categories:
        raise ValueError("No hay categorias en el JSON COCO.")

    category_by_name = {c["name"]: c for c in categories}
    unknown = [c for c in args.class_order if c not in category_by_name]
    if unknown:
        raise ValueError(
            f"Estas clases no existen en COCO y no se pueden mapear: {unknown}. "
            f"Disponibles: {[c['name'] for c in categories]}"
        )

    coco_id_to_yolo_id = {}
    for yolo_id, class_name in enumerate(args.class_order):
        coco_id_to_yolo_id[category_by_name[class_name]["id"]] = yolo_id

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    valid_images = []
    missing_images = []
    for img in images:
        file_name = img.get("file_name")
        path = find_image_path(images_dir, file_name)
        if path is None:
            missing_images.append(file_name)
            continue
        valid_images.append((img, path))

    if not valid_images:
        raise ValueError("No se encontro ninguna imagen valida para procesar.")

    random.seed(args.seed)
    random.shuffle(valid_images)

    n = len(valid_images)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    splits = {
        "train": valid_images[:n_train],
        "val": valid_images[n_train : n_train + n_val],
        "test": valid_images[n_train + n_val :],
    }

    # Ajuste por seguridad si alguna proporcion da 0 en datasets pequenos.
    if n >= 3:
        empty_splits = [k for k, v in splits.items() if not v]
        if empty_splits:
            pool = [item for group in splits.values() for item in group]
            splits = {"train": [], "val": [], "test": []}
            splits["train"] = pool[: max(1, int(0.7 * n))]
            splits["val"] = pool[max(1, int(0.7 * n)) : max(2, int(0.9 * n))]
            splits["test"] = pool[max(2, int(0.9 * n)) :]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_structure(output_dir)

    written_labels = 0
    empty_labels = 0
    skipped_polygons = 0

    for split_name, split_items in splits.items():
        for img, src_path in split_items:
            width = img["width"]
            height = img["height"]
            image_id = img["id"]
            dst_image = output_dir / "images" / split_name / src_path.name
            dst_label = output_dir / "labels" / split_name / f"{src_path.stem}.txt"

            shutil.copy2(src_path, dst_image)

            rows = []
            for ann in anns_by_image.get(image_id, []):
                if ann.get("iscrowd", 0) == 1:
                    continue
                class_id = coco_id_to_yolo_id.get(ann["category_id"])
                if class_id is None:
                    continue

                segmentation = ann.get("segmentation", [])
                if not isinstance(segmentation, list):
                    continue
                for polygon in segmentation:
                    if not isinstance(polygon, list):
                        continue
                    yolo_polygon = to_yolo_polygon(polygon, width, height)
                    if yolo_polygon is None:
                        skipped_polygons += 1
                        continue
                    coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in yolo_polygon])
                    rows.append(f"{class_id} {coords}")

            if rows:
                dst_label.write_text("\n".join(rows) + "\n", encoding="utf-8")
                written_labels += 1
            else:
                dst_label.write_text("", encoding="utf-8")
                empty_labels += 1

    write_dataset_yaml(output_dir, args.class_order)

    print("=== Conversion completada ===")
    print(f"Imagenes COCO: {len(images)}")
    print(f"Imagenes procesadas: {n}")
    print(f"Imagenes faltantes en disco: {len(missing_images)}")
    print(f"Split train/val/test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"Etiquetas con objetos: {written_labels}")
    print(f"Etiquetas vacias: {empty_labels}")
    print(f"Poligonos omitidos por formato: {skipped_polygons}")
    print(f"Dataset YOLO-seg en: {output_dir.resolve()}")
    print(f"YAML: {output_dir.resolve() / 'dataset.yaml'}")

    if missing_images:
        print("\nAviso: algunas imagenes del JSON no se encontraron en la carpeta images.")
        for name in missing_images[:10]:
            print(f" - {name}")
        if len(missing_images) > 10:
            print(f" ... y {len(missing_images) - 10} mas")


if __name__ == "__main__":
    main()
