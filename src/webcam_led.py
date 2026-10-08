import argparse
import time

import cv2
import serial
from ultralytics import YOLO

FRAMES_CONFIRMATION = 3


def mano_levantada(kp_xy, kp_conf, umbral=0.5):
    # índices COCO: (hombro, muñeca) izq y der
    for hombro, muneca in ((5, 9), (6, 10)):
        if kp_conf[hombro] > umbral and kp_conf[muneca] > umbral:
            # y menor = más arriba en la imagen
            if kp_xy[muneca][1] < kp_xy[hombro][1]:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Mano levantada -> LED en Arduino")
    parser.add_argument("--port", default=None, help="Puerto del Arduino, ej. COM3")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    arduino = None
    if args.port:
        arduino = serial.Serial(args.port, 9600, timeout=1)
        time.sleep(2)  # el Arduino se resetea al abrir el puerto, hay que esperar a que arranque

    model = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # DSHOW abre mucho más rápido que el backend default en Windows
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    estado = False
    contador = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, verbose=False)[0]

        detectada = False
        kps = result.keypoints
        if kps is not None and kps.conf is not None:
            for xy, conf in zip(kps.xy, kps.conf):
                if mano_levantada(xy.tolist(), conf.tolist()):
                    detectada = True
                    break

        # debounce: solo cambia de estado si la detección se sostiene
        # FRAMES_CONFIRMATION frames seguidos (evita parpadeo del LED)
        if detectada != estado:
            contador += 1
            if contador >= FRAMES_CONFIRMATION:
                estado = detectada
                contador = 0
                if arduino:
                    arduino.write(b"1" if estado else b"0")
        else:
            contador = 0

        salida = result.plot()
        texto = "MANO LEVANTADA" if estado else "..."
        color = (0, 0, 255) if estado else (200, 200, 200)
        cv2.putText(salida, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("webcam", salida)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.write(b"0")
        arduino.close()


if __name__ == "__main__":
    main()