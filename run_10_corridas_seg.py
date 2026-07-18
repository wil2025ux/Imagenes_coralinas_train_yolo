import argparse
import csv
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO


RUN_CONFIGS = [
    # run_id, seed, train, val, test, epochs, imgsz, batch, lr0, lrf, weight_decay
    ("seg3_r01", 101, 0.70, 0.20, 0.10, 250, 640, 2, 0.0100, 0.010, 0.00050),
    ("seg3_r02", 202, 0.68, 0.22, 0.10, 220, 640, 2, 0.0080, 0.010, 0.00050),
    ("seg3_r03", 303, 0.72, 0.18, 0.10, 280, 640, 2, 0.0120, 0.010, 0.00050),
    ("seg3_r04", 404, 0.70, 0.15, 0.15, 300, 768, 2, 0.0100, 0.020, 0.00050),
    ("seg3_r05", 505, 0.75, 0.15, 0.10, 260, 640, 2, 0.0060, 0.010, 0.00070),
    ("seg3_r06", 606, 0.65, 0.20, 0.15, 320, 640, 2, 0.0100, 0.015, 0.00050),
    ("seg3_r07", 707, 0.70, 0.20, 0.10, 240, 512, 2, 0.0100, 0.010, 0.00050),
    ("seg3_r08", 808, 0.73, 0.17, 0.10, 280, 640, 1, 0.0080, 0.015, 0.00050),
    ("seg3_r09", 909, 0.67, 0.23, 0.10, 260, 768, 1, 0.0070, 0.020, 0.00080),
    ("seg3_r10", 1001, 0.70, 0.20, 0.10, 350, 640, 2, 0.0100, 0.010, 0.00050),
]


SAMPLE_IMAGES = [
    "DSCN3101.JPG",
    "DSCN3092.JPG",
    "DSCN3098.JPG",
    "DSCN3093.JPG",
    "DSCN3095.JPG",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta 10 corridas YOLO-seg con splits e hiperparametros diferentes."
    )
    parser.add_argument("--coco-json", default="result_coco.json")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--base-model", default="yolo11n-seg.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--project-root", default="experimentos_seg3")
    parser.add_argument(
        "--run-sample-inference",
        action="store_true",
        help="Si se activa, genera inferencia en 5 imagenes de control por cada corrida.",
    )
    return parser.parse_args()


def run_cmd(command: list[str], cwd: Path):
    proc = subprocess.run(command, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Comando fallo ({proc.returncode}): {' '.join(command)}")


def parse_results_csv(results_csv: Path):
    if not results_csv.exists():
        return {
            "best_epoch": "",
            "mask_map50": "",
            "mask_map5095": "",
            "box_map50": "",
            "box_map5095": "",
            "precision_mask": "",
            "recall_mask": "",
        }

    with results_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {
            "best_epoch": "",
            "mask_map50": "",
            "mask_map5095": "",
            "box_map50": "",
            "box_map5095": "",
            "precision_mask": "",
            "recall_mask": "",
        }

    # Columnas comunes en Ultralytics recientes.
    key_map = {
        "mask_map50": ["metrics/mAP50(M)", "metrics/seg(mAP50)", "metrics/mAP50(Masks)"],
        "mask_map5095": ["metrics/mAP50-95(M)", "metrics/seg(mAP50-95)"],
        "box_map50": ["metrics/mAP50(B)", "metrics/box(mAP50)"],
        "box_map5095": ["metrics/mAP50-95(B)", "metrics/box(mAP50-95)"],
        "precision_mask": ["metrics/precision(M)", "metrics/seg(precision)"],
        "recall_mask": ["metrics/recall(M)", "metrics/seg(recall)"],
    }

    def get_value(row, keys):
        for k in keys:
            if k in row and row[k] != "":
                return row[k]
        return ""

    best_idx = 0
    best_val = -1.0
    for i, row in enumerate(rows):
        v = get_value(row, key_map["mask_map50"])
        try:
            vf = float(v)
        except Exception:
            vf = -1.0
        if vf > best_val:
            best_val = vf
            best_idx = i

    best_row = rows[best_idx]
    return {
        "best_epoch": best_row.get("epoch", str(best_idx)),
        "mask_map50": get_value(best_row, key_map["mask_map50"]),
        "mask_map5095": get_value(best_row, key_map["mask_map5095"]),
        "box_map50": get_value(best_row, key_map["box_map50"]),
        "box_map5095": get_value(best_row, key_map["box_map5095"]),
        "precision_mask": get_value(best_row, key_map["precision_mask"]),
        "recall_mask": get_value(best_row, key_map["recall_mask"]),
    }


def main():
    args = parse_args()
    root = Path(".").resolve()
    project_root = root / args.project_root
    datasets_root = project_root / "datasets"
    runs_root = project_root / "runs"
    infer_root = project_root / "inferencia_5imgs"
    summary_csv = project_root / "resumen_corridas.csv"
    configs_csv = project_root / "configuraciones_corridas.csv"

    datasets_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    infer_root.mkdir(parents=True, exist_ok=True)

    # Guardar configuraciones para trazabilidad.
    with configs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "seed",
                "train_ratio",
                "val_ratio",
                "test_ratio",
                "epochs",
                "imgsz",
                "batch",
                "lr0",
                "lrf",
                "weight_decay",
            ]
        )
        for cfg in RUN_CONFIGS:
            writer.writerow(cfg)

    summary_rows = []
    for cfg in RUN_CONFIGS:
        (
            run_id,
            seed,
            train_ratio,
            val_ratio,
            test_ratio,
            epochs,
            imgsz,
            batch,
            lr0,
            lrf,
            weight_decay,
        ) = cfg

        print(f"\n=== Ejecutando {run_id} ===")
        dataset_out = datasets_root / run_id

        cmd_convert = [
            sys.executable,
            "prepare_coco_to_yolo_seg.py",
            "--coco-json",
            args.coco_json,
            "--images-dir",
            args.images_dir,
            "--output-dir",
            str(dataset_out),
            "--train-ratio",
            str(train_ratio),
            "--val-ratio",
            str(val_ratio),
            "--test-ratio",
            str(test_ratio),
            "--seed",
            str(seed),
            "--class-order",
            "algas",
            "corales",
            "almejas",
            "esponjas",
            "arena",
        ]
        run_cmd(cmd_convert, root)

        model = YOLO(args.base_model)
        model.train(
            task="segment",
            data=str(dataset_out / "dataset.yaml"),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=args.device,
            project=str(runs_root),
            name=run_id,
            patience=args.patience,
            workers=args.workers,
            pretrained=True,
            lr0=lr0,
            lrf=lrf,
            weight_decay=weight_decay,
            verbose=True,
        )

        run_dir = runs_root / run_id
        results_csv = run_dir / "results.csv"
        metrics = parse_results_csv(results_csv)
        best_pt = run_dir / "weights" / "best.pt"

        if args.run_sample_inference and best_pt.exists():
            sample_paths = [str((root / "images" / n).resolve()) for n in SAMPLE_IMAGES if (root / "images" / n).exists()]
            if sample_paths:
                inf_model = YOLO(str(best_pt))
                inf_model.predict(
                    source=sample_paths,
                    conf=0.05,
                    iou=0.5,
                    imgsz=imgsz,
                    device=args.device,
                    save=True,
                    project=str(infer_root),
                    name=run_id,
                    exist_ok=True,
                    verbose=False,
                )

        summary_rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "lr0": lr0,
                "lrf": lrf,
                "weight_decay": weight_decay,
                "best_epoch": metrics["best_epoch"],
                "mask_map50": metrics["mask_map50"],
                "mask_map5095": metrics["mask_map5095"],
                "box_map50": metrics["box_map50"],
                "box_map5095": metrics["box_map5095"],
                "precision_mask": metrics["precision_mask"],
                "recall_mask": metrics["recall_mask"],
                "best_pt": str(best_pt),
            }
        )

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "run_id",
            "seed",
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "epochs",
            "imgsz",
            "batch",
            "lr0",
            "lrf",
            "weight_decay",
            "best_epoch",
            "mask_map50",
            "mask_map5095",
            "box_map50",
            "box_map5095",
            "precision_mask",
            "recall_mask",
            "best_pt",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== Proceso completado ===")
    print(f"Resumen: {summary_csv}")
    print(f"Configuraciones: {configs_csv}")
    print(f"Corridas: {runs_root}")
    if args.run_sample_inference:
        print(f"Inferencias 5 imagenes: {infer_root}")


if __name__ == "__main__":
    main()
