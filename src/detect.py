import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Detección de objetos en video con YOLO")
    parser.add_argument("--source", required=True, help="Ruta al video de entrada")
    parser.add_argument("--output", default="runs/salida.mp4", help="Ruta del video de salida")
    parser.add_argument("--model", default="yolo11n.pt", help="Modelo base (COCO)")
    parser.add_argument(
        "--custom-model", default=None,
        help="Pesos del fine-tuning (ej. best.pt). Si se pasa, se corren los dos modelos",
    )
    parser.add_argument(
        "--base-classes", type=int, nargs="+", default=None,
        help="Clases COCO a conservar del modelo base (0=person, 2=car...). "
             "Con --custom-model, por defecto solo personas",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Confianza mínima")
    args = parser.parse_args()

    base = YOLO(args.model)
    custom = YOLO(args.custom_model) if args.custom_model else None

    # Con dos modelos, el base se limita a las clases indicadas (por defecto personas).
    # Si no se filtrara, seguiría llamando "dog" a las ardillas.
    if args.base_classes:
        base_classes = args.base_classes
    elif custom is not None:
        base_classes = [0]
    else:
        base_classes = None  # sin modelo propio: todas las clases, como antes

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = base(frame, conf=args.conf, classes=base_classes, verbose=False)[0]
        salida = result.plot()  # frame con las cajas del modelo base

        if custom is not None:
            result_custom = custom(frame, conf=args.conf, verbose=False)[0]
            salida = result_custom.plot(img=salida)  # suma las cajas del modelo propio

        writer.write(salida)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frames procesados: {frame_count}")

    cap.release()
    writer.release()
    print(f"Listo. {frame_count} frames. Salida en {args.output}")


if __name__ == "__main__":
    main()