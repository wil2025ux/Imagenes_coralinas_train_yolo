"""Exporta correcciones de puntos CoArrCP a un dataset YOLO-seg.

Prioridad de máscara:
1) máscara SAM3 guardada (mask_path)
2) disco circular débil alrededor del punto
"""
from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np


def circle_polygon_norm(cx: float, cy: float, radius: float, width: int, height: int, n_sides: int = 16) -> List[float]:
    """Polígono circular normalizado (coords 0–1) para label YOLO-seg."""
    pts: List[float] = []
    for i in range(n_sides):
        angle = 2.0 * math.pi * i / n_sides
        x = (cx + radius * math.cos(angle)) / max(width, 1)
        y = (cy + radius * math.sin(angle)) / max(height, 1)
        pts.append(round(min(max(x, 0.0), 1.0), 6))
        pts.append(round(min(max(y, 0.0), 1.0), 6))
    return pts


def mask_file_to_polygons(mask_path: Path, min_area: int = 20) -> List[List[float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    h, w = mask.shape[:2]
    binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            approx = cnt.reshape(-1, 2)
        coords: List[float] = []
        for px, py in approx:
            coords.append(round(float(px) / max(w, 1), 6))
            coords.append(round(float(py) / max(h, 1), 6))
        if len(coords) >= 6:
            polys.append(coords)
    return polys


def fetch_correction_rows(conn) -> List[dict]:
    """Imágenes con puntos etiquetados (manual o editados) en el último run."""
    rows = conn.execute(
        """
        SELECT
          i.id AS image_id,
          i.original_path,
          i.original_name,
          i.width,
          i.height,
          r.id AS run_id,
          p.point_index,
          p.point_name,
          p.x,
          p.y,
          p.class_name,
          p.component_name,
          p.manual_label,
          p.edited_manual,
          p.status,
          p.mask_path,
          p.mask_source,
          p.mask_score
        FROM images i
        JOIN inference_runs r ON r.id = (
          SELECT id FROM inference_runs WHERE image_id = i.id ORDER BY id DESC LIMIT 1
        )
        JOIN points p ON p.run_id = r.id
        WHERE p.class_name IS NOT NULL
          AND TRIM(p.class_name) != ''
          AND (p.manual_label = 1 OR p.edited_manual = 1 OR p.status = 'manual' OR p.mask_path IS NOT NULL)
        ORDER BY i.id, p.point_index
        """
    ).fetchall()
    return [dict(row) for row in rows]


def count_trainable_corrections(conn) -> Dict[str, int]:
    rows = fetch_correction_rows(conn)
    image_ids = {row["image_id"] for row in rows}
    with_mask = sum(1 for row in rows if row.get("mask_path"))
    return {"num_images": len(image_ids), "num_points": len(rows), "num_sam3_masks": with_mask}


def resolve_class_list(rows: Sequence[dict], base_classes: Optional[Sequence[str]] = None) -> List[str]:
    names: List[str] = []
    seen = set()
    for name in list(base_classes or []) + [row["class_name"] for row in rows]:
        key = str(name).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(key)
    return names


def build_dataset_from_corrections(
    conn,
    output_dir: Path,
    base_classes: Optional[Sequence[str]] = None,
    radius_ratio: float = 0.04,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict:
    rows = fetch_correction_rows(conn)
    if not rows:
        raise ValueError("No hay correcciones etiquetadas para reentrenar.")

    class_names = resolve_class_list(rows, base_classes)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    by_image: Dict[int, List[dict]] = {}
    for row in rows:
        by_image.setdefault(int(row["image_id"]), []).append(row)

    image_ids = sorted(by_image.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    if len(image_ids) == 1:
        train_ids, val_ids = image_ids, image_ids
    else:
        n_val = max(1, int(round(len(image_ids) * val_ratio)))
        n_val = min(n_val, len(image_ids) - 1)
        val_ids = image_ids[:n_val]
        train_ids = image_ids[n_val:]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    used_sam3 = 0
    used_circle = 0

    def write_split(split: str, ids: Sequence[int]) -> int:
        nonlocal used_sam3, used_circle
        written = 0
        for image_id in ids:
            points = by_image[image_id]
            src = Path(points[0]["original_path"])
            if not src.is_file():
                continue
            width = int(points[0]["width"] or 0)
            height = int(points[0]["height"] or 0)
            if width <= 0 or height <= 0:
                continue

            radius = max(8.0, float(min(width, height)) * float(radius_ratio))
            stem = f"img_{image_id}_{src.stem}"
            ext = src.suffix.lower() or ".jpg"
            dst_img = output_dir / "images" / split / f"{stem}{ext}"
            dst_lbl = output_dir / "labels" / split / f"{stem}.txt"
            shutil.copy2(src, dst_img)

            lines = []
            for p in points:
                cls_id = class_to_id[p["class_name"]]
                mask_path = p.get("mask_path")
                polys: List[List[float]] = []
                if mask_path and Path(mask_path).is_file():
                    polys = mask_file_to_polygons(Path(mask_path))
                    if polys:
                        used_sam3 += 1
                if not polys:
                    polys = [
                        circle_polygon_norm(float(p["x"]), float(p["y"]), radius, width, height)
                    ]
                    used_circle += 1
                for poly in polys:
                    lines.append(" ".join([str(cls_id)] + [str(v) for v in poly]))
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
            written += 1
        return written

    n_train = write_split("train", train_ids)
    n_val = write_split("val", val_ids)
    if n_train == 0:
        raise ValueError("No se pudieron copiar imágenes de entrenamiento.")

    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    yaml_text = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_yaml}\n"
    )
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    return {
        "dataset_yaml": str(yaml_path),
        "output_dir": str(output_dir),
        "class_names": class_names,
        "num_images": n_train + (0 if train_ids == val_ids else n_val),
        "num_train": n_train,
        "num_val": n_val,
        "num_points": len(rows),
        "used_sam3_masks": used_sam3,
        "used_circle_masks": used_circle,
    }
