import argparse
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Tracking y conteo de objetos únicos")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="runs/tracking_ardilla.mp4")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.3)
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
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    ids_por_clase = defaultdict(set)  # clase -> IDs únicos vistos

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=args.conf, verbose=False,
        )[0]

        if result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            clases = result.boxes.cls.int().tolist()
            for track_id, cls in zip(ids, clases):
                ids_por_clase[result.names[cls]].add(track_id)

        writer.write(result.plot())

    cap.release()
    writer.release()

    print("\nObjetos únicos detectados:")
    for clase, ids in sorted(ids_por_clase.items()):
        print(f"  {clase}: {len(ids)}")
    print(f"Video en {args.output}")


if __name__ == "__main__":
    main()