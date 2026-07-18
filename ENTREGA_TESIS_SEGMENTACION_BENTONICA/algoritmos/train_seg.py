import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena un modelo YOLO de segmentacion bentonica.")
    parser.add_argument(
        "--data",
        default="benthic_yolo_seg/dataset.yaml",
        help="Ruta al dataset.yaml generado por prepare_coco_to_yolo_seg.py",
    )
    parser.add_argument(
        "--model",
        default="yolo11n-seg.pt",
        help="Modelo base para fine-tuning (ej. yolo11n-seg.pt, yolo11s-seg.pt).",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Numero de epocas.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Resolucion de entrenamiento.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--device",
        default="auto",
        help='Dispositivo: "auto", "mps", "cpu", "0", "0,1", etc.',
    )
    parser.add_argument("--project", default="runs/segment", help="Directorio de resultados.")
    parser.add_argument("--name", default="benthos_yolo11n_seg", help="Nombre de corrida.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--workers", type=int, default=4, help="Numero de workers.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    # Prioriza GPU CUDA, luego Apple Metal (MPS), y finalmente CPU.
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"No existe dataset YAML: {data_path}")

    device = resolve_device(args.device)
    print(f"Usando device: {device}")

    model = YOLO(args.model)
    model.train(
        task="segment",
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        pretrained=True,
        verbose=True,
    )

    print("\nEntrenamiento finalizado.")
    print("Pesos esperados:")
    print(f" - {args.project}/{args.name}/weights/best.pt")
    print(f" - {args.project}/{args.name}/weights/last.pt")


if __name__ == "__main__":
    main()
