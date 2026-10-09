import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evalúa un modelo sobre el set de validación")
    parser.add_argument("--weights", required=True, help="Pesos a evaluar (ej. best.pt)")
    parser.add_argument("--data", required=True, help="Ruta al data.yaml del dataset")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output", default="runs/metrics.json")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split)

    resultados = {
        "weights": args.weights,
        "split": args.split,
        "precision": float(metrics.box.mp),   # de lo que detecta, cuánto es correcto
        "recall": float(metrics.box.mr),      # de lo que existe, cuánto encuentra
        "mAP50": float(metrics.box.map50),    # mAP con IoU >= 0.5
        "mAP50-95": float(metrics.box.map),   # mAP promediado en IoU 0.5 a 0.95 (más exigente)
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(resultados, indent=2))

    print("\nResultados:")
    for clave, valor in resultados.items():
        print(f"  {clave}: {valor}")
    print(f"Guardado en {args.output}")


if __name__ == "__main__":
    main()