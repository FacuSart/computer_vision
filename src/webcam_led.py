# ============================================================
# 1. IMPORTS Y CONSTANTES
# ============================================================
import argparse  # para leer argumentos desde la terminal (--port, --camera)
import time      # para hacer una pausa (sleep) al abrir el puerto serie

import cv2       # OpenCV: abre la cámara, dibuja texto y muestra la ventana
import serial    # pyserial: comunicación por USB con el Arduino
from ultralytics import YOLO  # el modelo de detección/pose

# Cuántos frames seguidos tiene que repetirse una detección para que
# cambiemos de estado. Sirve para filtrar falsos positivos de un solo frame
# (si la detección parpadea, el LED no parpadea con ella).
FRAMES_CONFIRMATION = 3


# ============================================================
# 2. FUNCIÓN QUE DECIDE SI UNA PERSONA TIENE LA MANO LEVANTADA
# ============================================================
def mano_levantada(kp_xy, kp_conf, umbral=0.5):
    # kp_xy:   lista de 17 puntos (x, y) del cuerpo de UNA persona
    # kp_conf: lista de 17 valores de confianza (0 a 1), uno por punto
    #
    # Los índices de los puntos siguen el formato COCO:
    #   5 = hombro izquierdo,  6 = hombro derecho
    #   9 = muñeca izquierda, 10 = muñeca derecha
    #
    # Se revisan los dos brazos: (hombro, muñeca) de cada lado.
    for hombro, muneca in ((5, 9), (6, 10)):
        # Solo evaluamos si el modelo está razonablemente seguro de AMBOS
        # puntos; si no los ve bien, no sacamos conclusiones.
        if kp_conf[hombro] > umbral and kp_conf[muneca] > umbral:
            # kp_xy[...][1] es la coordenada y. En imágenes y = 0 es el borde
            # SUPERIOR, así que "más arriba" significa un valor de y MENOR.
            # Si la muñeca está más arriba que el hombro: mano levantada.
            if kp_xy[muneca][1] < kp_xy[hombro][1]:
                return True
    return False  # ningún brazo cumplió la condición


# ============================================================
# 3. PROGRAMA PRINCIPAL
# ============================================================
def main():
    # --- 3.1 Argumentos de la terminal ---
    parser = argparse.ArgumentParser(description="Mano levantada -> LED en Arduino")
    parser.add_argument("--port", default=None, help="Puerto del Arduino, ej. COM3")
    parser.add_argument("--camera", type=int, default=0)  # 0 = cámara por defecto
    args = parser.parse_args()

    # --- 3.2 Conexión con el Arduino (opcional) ---
    # Si no pasás --port, arduino queda en None y el script corre igual,
    # solo con la ventana de la cámara. Útil para probar sin hardware.
    arduino = None
    if args.port:
        arduino = serial.Serial(args.port, 9600, timeout=1)  # 9600 = velocidad en baudios
        time.sleep(2)  # al abrir el puerto el Arduino se reinicia; esperamos a que arranque

    # --- 3.3 Carga del modelo y apertura de la cámara ---
    model = YOLO("yolo11n-pose.pt")  # versión "pose": devuelve 17 puntos del cuerpo
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # DSHOW abre mucho más rápido que MSMF en Windows
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    # --- 3.4 Estado inicial ---
    estado = False   # estado "oficial": ¿hay mano levantada confirmada?
    contador = 0     # cuántos frames seguidos lleva la detección "contradiciendo" al estado

    # --------------------------------------------------------
    # 3.5 LOOP PRINCIPAL: se repite una vez por frame de la cámara
    # --------------------------------------------------------
    while True:
        ok, frame = cap.read()  # lee un frame; ok=False si falla la cámara
        if not ok:
            break

        # Corre el modelo sobre el frame. [0] porque devuelve un resultado por imagen.
        result = model(frame, verbose=False)[0]

        # --- Decidir si EN ESTE FRAME hay alguna mano levantada ---
        detectada = False
        kps = result.keypoints  # los puntos del cuerpo de todas las personas detectadas
        if kps is not None and kps.conf is not None:
            # Recorremos cada persona (cada una trae sus 17 puntos y confianzas)
            for xy, conf in zip(kps.xy, kps.conf):
                if mano_levantada(xy.tolist(), conf.tolist()):
                    detectada = True
                    break  # con una persona alcanza; no seguimos buscando

        # --- Filtro anti-parpadeo (debounce) ---
        # Si lo detectado en este frame DIFIERE del estado oficial, sumamos al
        # contador. Solo cuando la diferencia se sostiene FRAMES_CONFIRMATION
        # frames seguidos, cambiamos de estado y avisamos al Arduino.
        if detectada != estado:
            contador += 1
            if contador >= FRAMES_CONFIRMATION:
                estado = detectada
                contador = 0
                if arduino:
                    # b"1" = encender, b"0" = apagar (el sketch del Arduino lee ese byte)
                    arduino.write(b"1" if estado else b"0")
        else:
            # Coincide con el estado oficial: reiniciamos el contador, porque la
            # discrepancia no fue sostenida.
            contador = 0

        # --- Dibujar y mostrar ---
        salida = result.plot()  # frame con el esqueleto dibujado
        texto = "MANO LEVANTADA" if estado else "..."
        color = (0, 0, 255) if estado else (200, 200, 200)  # OpenCV usa BGR: (0,0,255) = rojo
        cv2.putText(salida, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("webcam", salida)

        # waitKey(1) espera 1 ms por una tecla y además permite que la ventana se
        # refresque. Si se apretó "q", salimos del loop.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- 3.6 Limpieza al terminar ---
    cap.release()            # libera la cámara
    cv2.destroyAllWindows()  # cierra la ventana
    if arduino:
        arduino.write(b"0")  # deja el LED apagado
        arduino.close()      # libera el puerto serie (si no, queda ocupado)


# ============================================================
# 4. PUNTO DE ENTRADA
# ============================================================
# Este bloque hace que main() corra solo cuando ejecutás el archivo directamente
# (python src/webcam_led.py) y no cuando lo importás desde otro script.
if __name__ == "__main__":
    main()