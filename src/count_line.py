import argparse
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Conteo por cruce de línea")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="runs/conteo.mp4")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument(
        "--line-pos", type=float, default=0.5,
        help="Posición horizontal de la línea (0 = izquierda, 1 = derecha)",
    )
    args = parser.parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_x = int(width * args.line_pos)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    ultimo_lado = {}  # id -> lado de la línea (-1 izquierda, 1 derecha)
    contados = set()  # ids que ya cruzaron (evita doble conteo por jitter)
    conteo = defaultdict(lambda: {"derecha": 0, "izquierda": 0})

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=args.conf, verbose=False,
        )[0]
        boxes = result.boxes

        if boxes.id is not None:
            ids = boxes.id.int().tolist()
            clases = boxes.cls.int().tolist()
            centros_x = boxes.xywh[:, 0].tolist()  # centro horizontal de cada caja

            for track_id, cls, cx in zip(ids, clases, centros_x):
                lado = 1 if cx > line_x else -1
                previo = ultimo_lado.get(track_id)

                if previo is not None and previo != lado and track_id not in contados:
                    nombre = result.names[cls]
                    if lado == 1:
                        conteo[nombre]["derecha"] += 1
                    else:
                        conteo[nombre]["izquierda"] += 1
                    contados.add(track_id)

                ultimo_lado[track_id] = lado

        salida = result.plot()
        cv2.line(salida, (line_x, 0), (line_x, height), (0, 0, 255), 2)
        for i, (clase, c) in enumerate(sorted(conteo.items())):
            texto = f"{clase}: {c['derecha']} derecha / {c['izquierda']} izquierda"
            cv2.putText(
                salida, texto, (10, 30 + 30 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
        writer.write(salida)

    cap.release()
    writer.release()

    print("\nCruces de línea:")
    for clase, c in sorted(conteo.items()):
        print(f"  {clase}: {c['derecha']} derecha, {c['izquierda']} izquierda")
    print(f"Video en {args.output}")


if __name__ == "__main__":
    main()