import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning de YOLO con dataset propio")
    parser.add_argument("--data", required=True, help="Ruta al data.yaml del dataset")
    parser.add_argument("--model", default="yolo11n.pt", help="Pesos de partida (preentrenados)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="squirrel", help="Nombre de la corrida")
    parser.add_argument("--device", default=None, help="'0' para GPU, 'cpu' para CPU (default: auto)")
    args = parser.parse_args()

    # Partimos de pesos preentrenados en COCO
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        seed=42,  # reproducibilidad: misma corrida -> mismos resultados (aprox.)
    )

    print(f"\nMejores pesos guardados en: {model.trainer.best}")


if __name__ == "__main__":
    main()