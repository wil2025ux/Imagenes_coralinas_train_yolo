"""Asistente SAM3: punto (x,y) → máscara de instancia (modo interactivo SAM1)."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
SAM3_REPO = Path(os.getenv("SAM3_REPO", "/Users/arath/Sam3_prueba/sam3"))
MASKS_DIR = APP_ROOT / "data" / "masks"

_lock = threading.Lock()
_model = None
_processor = None
_device: Optional[str] = None
_load_error: Optional[str] = None
_image_cache_key: Optional[str] = None
_inference_state = None


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sam3_status() -> Dict:
    return {
        "loaded": _model is not None and _processor is not None,
        "device": _device,
        "error": _load_error,
        "repo": str(SAM3_REPO),
    }


def ensure_sam3_on_path():
    import sys

    repo = str(SAM3_REPO)
    if SAM3_REPO.is_dir() and repo not in sys.path:
        sys.path.insert(0, repo)


def load_sam3(force: bool = False) -> Dict:
    """Carga perezosa del modelo interactivo. Puede fallar si HF gated no está aprobado."""
    global _model, _processor, _device, _load_error, _image_cache_key, _inference_state

    with _lock:
        if _model is not None and _processor is not None and not force:
            return sam3_status()

        _load_error = None
        try:
            ensure_sam3_on_path()
            import torch
            from PIL import Image  # noqa: F401
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            device = default_device()
            bpe = SAM3_REPO / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            kwargs = {
                "enable_inst_interactivity": True,
                "device": device,
            }
            if bpe.is_file():
                kwargs["bpe_path"] = str(bpe)

            model = build_sam3_image_model(**kwargs)
            processor = Sam3Processor(model, device=device)
            _model = model
            _processor = processor
            _device = device
            _image_cache_key = None
            _inference_state = None
        except Exception as exc:
            _model = None
            _processor = None
            _device = None
            _image_cache_key = None
            _inference_state = None
            msg = str(exc)
            if "GatedRepoError" in type(exc).__name__ or "403" in msg or "gated" in msg.lower():
                _load_error = (
                    "Sin acceso al checkpoint gated de Hugging Face (facebook/sam3). "
                    "Pide acceso en https://huggingface.co/facebook/sam3 y vuelve a intentar."
                )
            else:
                _load_error = msg
            raise RuntimeError(_load_error) from exc

        return sam3_status()


def _set_image(image_bgr: np.ndarray, cache_key: str):
    global _inference_state, _image_cache_key

    if _model is None or _processor is None:
        load_sam3()

    if _image_cache_key == cache_key and _inference_state is not None:
        return _inference_state

    from PIL import Image

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    _inference_state = _processor.set_image(pil)
    _image_cache_key = cache_key
    return _inference_state


def predict_mask_from_point(
    image_bgr: np.ndarray,
    x: float,
    y: float,
    cache_key: str,
    multimask: bool = True,
) -> Dict:
    """
    Genera la mejor máscara SAM3 para un punto foreground (x,y) en coords de imagen.
    """
    state = _set_image(image_bgr, cache_key)
    point_coords = np.array([[float(x), float(y)]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    masks, scores, logits = _model.predict_inst(
        state,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=multimask,
    )
    scores = np.asarray(scores).reshape(-1)
    masks = np.asarray(masks)
    if masks.ndim == 4:
        # (1, K, H, W) or (K, 1, H, W)
        if masks.shape[0] == 1:
            masks = masks[0]
        else:
            masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None, ...]

    best_idx = int(np.argmax(scores)) if len(scores) else 0
    best_mask = (masks[best_idx] > 0.5).astype(np.uint8)
    best_score = float(scores[best_idx]) if len(scores) else 0.0

    return {
        "mask": best_mask,
        "score": best_score,
        "num_candidates": int(len(scores)),
        "scores": [float(s) for s in scores.tolist()],
    }


def mask_to_yolo_polygons(mask: np.ndarray, min_area: int = 20) -> List[List[float]]:
    """Convierte máscara binaria HxW a polígonos normalizados YOLO-seg (varios si hay blobs)."""
    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        pts = cnt.reshape(-1, 2)
        if len(pts) < 3:
            continue
        # Simplificar un poco
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            approx = pts
        coords: List[float] = []
        for px, py in approx:
            coords.append(round(float(px) / max(w, 1), 6))
            coords.append(round(float(py) / max(h, 1), 6))
        if len(coords) >= 6:
            polys.append(coords)
    return polys


def save_mask_png(mask: np.ndarray, image_id: int, point_index: int) -> Path:
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    path = MASKS_DIR / f"img{image_id}_p{point_index}.png"
    cv2.imwrite(str(path), (mask > 0).astype(np.uint8) * 255)
    return path


def mask_overlay_bgr(image_bgr: np.ndarray, mask: np.ndarray, color=(14, 165, 233), alpha=0.45) -> np.ndarray:
    out = image_bgr.copy()
    m = mask.astype(bool)
    if m.shape[:2] != out.shape[:2]:
        m = cv2.resize(mask.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    for c in range(3):
        channel = out[:, :, c].astype(np.float32)
        channel[m] = channel[m] * (1 - alpha) + color[c] * alpha
        out[:, :, c] = channel.astype(np.uint8)
    return out


def mask_to_data_url(mask: np.ndarray) -> str:
    import base64

    ok, buf = cv2.imencode(".png", (mask > 0).astype(np.uint8) * 255)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"
