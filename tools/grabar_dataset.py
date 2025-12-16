import cv2
import mediapipe as mp
import os
import numpy as np

# --- CONFIGURACIÓN ---
NOMBRE_DATASET = 'Dataset_Videos'
LETRA_ACTUAL = 'A'  # Cambia esto manualmente o haz un input() al inicio
FRAMES_POR_GRABACION = 30 # 1 segundo aprox a 30fps

# Crear carpeta si no existe
RUTA_CARPETA = os.path.join(NOMBRE_DATASET, LETRA_ACTUAL)
if not os.path.exists(RUTA_CARPETA):
    os.makedirs(RUTA_CARPETA)

# Configuración MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Variables de control de grabación
grabando = False
contador_frames = 0
video_writer = None
numero_video = 0 # Para no sobrescribir archivos (video_0.mp4, video_1.mp4...)

# Buscar el siguiente número de video disponible en la carpeta para no sobrescribir
existing_files = os.listdir(RUTA_CARPETA)
while f"video_{numero_video}.mp4" in existing_files:
    numero_video += 1

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. EFECTO ESPEJO (UX)
        # Invertimos horizontalmente para que sea natural para el usuario
        frame = cv2.flip(frame, 1)

        # 2. PROCESAMIENTO
        # Hacemos una COPIA para dibujar. El 'frame' original se mantiene LIMPIO.
        image_para_mostrar = frame.copy()
        
        # Convertir a RGB para MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # 3. DIBUJAR EN LA COPIA (FEEDBACK VISUAL)
        mp_drawing.draw_landmarks(
            image_para_mostrar, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image_para_mostrar, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image_para_mostrar, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 4. LÓGICA DE GRABACIÓN
        k = cv2.waitKey(1) & 0xFF

        # Iniciar grabación con 'R'
        if k == ord('r') and not grabando:
            grabando = True
            contador_frames = 0
            nombre_archivo = os.path.join(RUTA_CARPETA, f"video_{numero_video}.mp4")
            
            # Configurar el escritor de video (MJPG es seguro para .avi/.mp4 en opencv)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            h, w, _ = frame.shape
            video_writer = cv2.VideoWriter(nombre_archivo, fourcc, 30.0, (w, h))
            print(f"Iniciando grabación: {nombre_archivo}")

        if grabando:
            # IMPORTANTE: Guardamos 'frame' (limpio), NO 'image_para_mostrar' (dibujado)
            video_writer.write(frame)
            
            # Indicador Visual (REC) en la pantalla
            cv2.circle(image_para_mostrar, (30, 30), 20, (0, 0, 255), -1) # Círculo Rojo
            cv2.putText(image_para_mostrar, "REC", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            contador_frames += 1
            
            # Detener grabación al llegar a los frames deseados
            if contador_frames >= FRAMES_POR_GRABACION:
                grabando = False
                video_writer.release()
                print(f"Grabación finalizada. Video guardado.")
                numero_video += 1 # Aumentar índice para el próximo

        # Mostrar la imagen con dibujos
        cv2.imshow('Recolector de Datos LSM', image_para_mostrar)

        # Salir con 'ESC'
        if k == 27:
            break

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()