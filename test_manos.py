import cv2
import mediapipe as mp

# 1. Configuración de MediaPipe (El "Cerebro" que ve manos)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 2. Inicializar la webcam (0 suele ser la cámara default)
cap = cv2.VideoCapture(0)

# 3. Configurar el modelo de detección
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,             # Queremos detectar hasta 2 manos
    min_detection_confidence=0.5, # Qué tan seguro debe estar para decir "es una mano"
    min_tracking_confidence=0.5) as hands:

    print("Iniciando cámara... presiona 'Esc' para salir.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se pudo obtener el video de la cámara.")
            continue

        # 4. Preparar la imagen
        # MediaPipe usa color RGB, pero OpenCV usa BGR. Hay que convertirlo.
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 5. ¡Hacer la predicción! (Aquí ocurre la magia de IA)
        results = hands.process(image_rgb)

        # 6. Dibujar los resultados
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujamos los puntos (nudos) y las conexiones (huesos)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

        # 7. Mostrar en pantalla
        cv2.imshow('VozVisual LSM - Prueba de Camara', image)

        # Salir si presionas la tecla ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()