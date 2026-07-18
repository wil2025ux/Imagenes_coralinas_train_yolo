#!/usr/bin/env python3
"""
Punto de entrada único para CoArrCP IA Flask.

Uso:
  python3 run.py
  python3 run.py --weights experimentos_seg3/runs/seg3_r012/weights/best.pt --port 9001
"""
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APP_PATH = PROJECT_ROOT / "coarcp_ia" / "app.py"


def main():
    if not APP_PATH.is_file():
        raise FileNotFoundError(f"No se encontró la app: {APP_PATH}")

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.argv[0] = str(APP_PATH)
    runpy.run_path(str(APP_PATH), run_name="__main__")


if __name__ == "__main__":
    main()
