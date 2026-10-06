import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Detección de objetos en video con YOLO")
    parser.add_argument("--source", required=True, help="Ruta al video de entrada")
    parser.add_argument("--output", default="runs/salida.mp4", help="Ruta al video de salida")
    parser.add_argument("--model", default="yolo11m.pt", help="Pesos del modelo")
    parser.add_argument("--config", type=float, default=0.3, help="Confianza mínima")

    args = parser.parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        results = model(frame, conf=args.config, verbose=False)[0]
        writer.write(results.plot())

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Procesadas {frame_count} frames")

    cap.release()
    writer.release()
    print(f"Listo. {frame_count} frames procesadas. Video de salida guardado en: {args.output}")

if __name__ == "__main__":
    main()