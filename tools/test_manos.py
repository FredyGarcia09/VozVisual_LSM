import cv2
import mediapipe as mp
import time

# Configuración de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializar webcam
cap = cv2.VideoCapture(0)

# Variables para calcular FPS
tiempo_anterior = 0
tiempo_actual = 0

# Configurar modelo
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    print("Iniciando cámara... presiona 'Esc' para salir.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se pudo obtener el video.")
            continue

        # Preparar imagen
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predicción
        results = hands.process(image_rgb)

        # Dibujar resultados
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

        # FPS
        tiempo_actual = time.time()
        # segundo / la diferencia de tiempo entre cuadros
        fps = 1 / (tiempo_actual - tiempo_anterior)
        tiempo_anterior = tiempo_actual

        # Mostrar FPS en pantalla
        cv2.putText(image, f'FPS: {int(fps)}', (10, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Mostrar
        cv2.imshow('VozVisual LSM - Prueba de Camara', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()