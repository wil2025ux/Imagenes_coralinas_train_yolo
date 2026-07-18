#!/usr/bin/env python3
"""Arranque del prototipo v1 (vista anterior) en puerto 9002."""
import runpy
import sys
from pathlib import Path

LEGACY = Path(__file__).resolve().parent / "prototipo_v1" / "app_legacy.py"


def main():
    if not LEGACY.is_file():
        raise FileNotFoundError(f"No se encontró: {LEGACY}")
    sys.argv[0] = str(LEGACY)
    runpy.run_path(str(LEGACY), run_name="__main__")


if __name__ == "__main__":
    main()
