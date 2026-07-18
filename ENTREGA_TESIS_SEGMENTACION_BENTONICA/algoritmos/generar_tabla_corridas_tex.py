import csv
from pathlib import Path


ROOT = Path("/Users/arath/Desktop/Imagenes_coralinas_train_yolo")
EXPERIMENTS_DIR = ROOT / "experimentos_seg3"
CONFIGS_CSV = EXPERIMENTS_DIR / "configuraciones_corridas.csv"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
SUMMARY_CSV = EXPERIMENTS_DIR / "resumen_corridas.csv"
TABLE_TEX = ROOT / "tabla_corridas_auto.tex"


def fmt_float(value: str, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "N/D"


def read_configs():
    rows = []
    if not CONFIGS_CSV.exists():
        return rows
    with CONFIGS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def read_best_metrics(results_csv: Path):
    if not results_csv.exists():
        return None
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None

    best = None
    best_val = -1.0
    for row in rows:
        try:
            current = float(row.get("metrics/mAP50(M)", ""))
        except Exception:
            current = -1.0
        if current > best_val:
            best_val = current
            best = row
    return best


def build_summary(config_rows):
    summary = []
    for cfg in config_rows:
        run_id = cfg["run_id"]
        run_dir = RUNS_DIR / run_id
        results_csv = run_dir / "results.csv"
        best = read_best_metrics(results_csv)

        if best is None:
            summary.append(
                {
                    "run_id": run_id,
                    "epochs": cfg["epochs"],
                    "imgsz": cfg["imgsz"],
                    "batch": cfg["batch"],
                    "split": f"{cfg['train_ratio']}/{cfg['val_ratio']}/{cfg['test_ratio']}",
                    "mask_map50": "N/D",
                    "mask_map5095": "N/D",
                    "precision_mask": "N/D",
                    "recall_mask": "N/D",
                }
            )
            continue

        summary.append(
            {
                "run_id": run_id,
                "epochs": cfg["epochs"],
                "imgsz": cfg["imgsz"],
                "batch": cfg["batch"],
                "split": f"{cfg['train_ratio']}/{cfg['val_ratio']}/{cfg['test_ratio']}",
                "mask_map50": fmt_float(best.get("metrics/mAP50(M)", "")),
                "mask_map5095": fmt_float(best.get("metrics/mAP50-95(M)", "")),
                "precision_mask": fmt_float(best.get("metrics/precision(M)", "")),
                "recall_mask": fmt_float(best.get("metrics/recall(M)", "")),
            }
        )
    return summary


def write_summary_csv(summary_rows):
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "epochs",
                "imgsz",
                "batch",
                "split",
                "mask_map50",
                "mask_map5095",
                "precision_mask",
                "recall_mask",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def split_percent(split: str):
    try:
        t, v, s = [float(x) for x in split.split("/")]
        return f"{int(round(t * 100))}/{int(round(v * 100))}/{int(round(s * 100))}"
    except Exception:
        return split


def write_table_tex(summary_rows):
    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\renewcommand{\\arraystretch}{1.2}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{lcccccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Corrida} & \\textbf{Epochs} & \\textbf{imgsz} & \\textbf{Batch} & \\textbf{Split} & \\textbf{Mask mAP50} & \\textbf{Mask mAP50-95} & \\textbf{Prec.} & \\textbf{Recall} \\\\")
    lines.append("\\midrule")

    for r in summary_rows:
        row = " & ".join(
            [
                r["run_id"].replace("_", "\\_"),
                str(r["epochs"]),
                str(r["imgsz"]),
                str(r["batch"]),
                split_percent(r["split"]),
                r["mask_map50"],
                r["mask_map5095"],
                r["precision_mask"],
                r["recall_mask"],
            ]
        )
        lines.append(f"{row} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\caption{Matriz de corridas con resultados automáticos disponibles. N/D indica corrida no ejecutada o incompleta.}")
    lines.append("\\label{tab:matriz_corridas_seg3_auto}")
    lines.append("\\end{table}")

    TABLE_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    configs = read_configs()
    if not configs:
        raise FileNotFoundError(
            f"No existe o esta vacio: {CONFIGS_CSV}. Ejecuta primero run_10_corridas_seg.py."
        )

    summary = build_summary(configs)
    write_summary_csv(summary)
    write_table_tex(summary)

    print(f"Resumen generado: {SUMMARY_CSV}")
    print(f"Tabla LaTeX generada: {TABLE_TEX}")


if __name__ == "__main__":
    main()
