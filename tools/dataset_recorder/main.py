import cv2
import mediapipe as mp
import os
import time
import math
import csv
import datetime

# --- CONFIGURACIÓN DEL PROYECTO ---
DATASET_PATH = "output_videos"
METADATA_FILE = "metadata.csv"
LETRA_ACTUAL = "A"  # Cambiar según la seña a grabar
UMBRAL_MOVIMIENTO = 0.15  # Sensibilidad (0.0 a 1.0). Ajustar si detecta muy rápido.
TIEMPO_ESPERA_FINAL = 1.0  # Segundos quieto para cortar la grabación
TIEMPO_CALIBRACION = 3  # Segundos iniciales para medir tu posición neutra

# --- ESTADOS DE LA MÁQUINA ---
ESTADO_CALIBRANDO = "CALIBRANDO"  # Midiendo posición neutra
ESTADO_IDLE = "ESPERANDO"         # Esperando que te muevas
ESTADO_GRABANDO = "GRABANDO"      # Guardando video
ESTADO_FINALIZANDO = "FINALIZANDO" # Detectó quietud, contando tiempo para cortar

class DataRecorder:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        
        # Variables de Lógica
        self.estado_actual = ESTADO_CALIBRANDO
        self.posicion_neutra = None # Guardará (x, y) de las muñecas
        self.inicio_quietud = 0
        self.video_writer = None
        self.frames_buffer = [] # Buffer para no perder el inicio del movimiento
        
        # Configurar carpetas
        self.setup_directories()

    def setup_directories(self):
        # Crear carpeta de videos
        ruta_letra = os.path.join(DATASET_PATH, LETRA_ACTUAL)
        if not os.path.exists(ruta_letra):
            os.makedirs(ruta_letra)
        
        # Iniciar CSV de metadatos si no existe
        if not os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Archivo", "Etiqueta", "Fecha", "Duracion_Frames"])

    def obtener_distancia_promedio(self, results):
        """Calcula qué tan lejos están las muñecas de la posición neutra."""
        if not results.pose_landmarks:
            return 0
            
        # Obtenemos coordenadas de muñecas (Left=15, Right=16)
        wrist_l = results.pose_landmarks.landmark[15]
        wrist_r = results.pose_landmarks.landmark[16]
        
        # Promedio de posición actual (x, y)
        curr_x = (wrist_l.x + wrist_r.x) / 2
        curr_y = (wrist_l.y + wrist_r.y) / 2
        
        if self.posicion_neutra is None:
            return (curr_x, curr_y)
            
        # Distancia Euclidiana simple vs la referencia
        dist = math.hypot(curr_x - self.posicion_neutra[0], curr_y - self.posicion_neutra[1])
        return dist

    def guardar_metadata(self, filename, frames):
        with open(METADATA_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            timestamp = datetime.datetime.now().isoformat()
            writer.writerow([filename, LETRA_ACTUAL, timestamp, frames])
            print(f"✅ Guardado: {filename}")

    def run(self):
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            start_time = time.time()
            
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break

                # 1. UX: Espejo y Copia Limpia
                frame = cv2.flip(frame, 1)
                debug_image = frame.copy() # Dibujamos aquí, guardamos 'frame'
                
                # Procesar MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                
                # Dibujar esqueleto (Solo visual)
                self.mp_drawing.draw_landmarks(debug_image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                
                # --- LÓGICA DE ESTADOS ---
                distancia = self.obtener_distancia_promedio(results)

                # A. CALIBRACIÓN (Primeros 3 segundos)
                if self.estado_actual == ESTADO_CALIBRANDO:
                    elapsed = time.time() - start_time
                    cv2.putText(debug_image, f"CALIBRANDO: Quedate quieto... {3 - int(elapsed)}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    if elapsed > TIEMPO_CALIBRACION:
                        if isinstance(distancia, tuple): # Si detectó manos
                            self.posicion_neutra = distancia
                            self.estado_actual = ESTADO_IDLE
                            print(f"Posición Neutra Calibrada: {self.posicion_neutra}")
                        else:
                            start_time = time.time() # Reiniciar si no te ve

                # B. ESPERANDO (IDLE)
                elif self.estado_actual == ESTADO_IDLE:
                    cv2.putText(debug_image, f"Listo. Haz la seña: {LETRA_ACTUAL}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # TRIGGER INICIO: Si te mueves más allá del umbral
                    if distancia > UMBRAL_MOVIMIENTO:
                        self.estado_actual = ESTADO_GRABANDO
                        # Generar nombre de archivo único
                        ts = int(time.time())
                        filename = f"{DATASET_PATH}/{LETRA_ACTUAL}/video_{ts}.mp4"
                        
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        h, w, _ = frame.shape
                        self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))
                        self.frames_buffer = [] # Reset
                        print("🔴 Iniciando Grabación por movimiento")

                # C. GRABANDO
                elif self.estado_actual == ESTADO_GRABANDO:
                    cv2.circle(debug_image, (30, 30), 20, (0, 0, 255), -1)
                    cv2.putText(debug_image, "GRABANDO...", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    self.video_writer.write(frame) # Guarda FRAME LIMPIO
                    self.frames_buffer.append(frame)

                    # TRIGGER FIN: Si vuelves a estar cerca de la neutra
                    if distancia < UMBRAL_MOVIMIENTO:
                        self.estado_actual = ESTADO_FINALIZANDO
                        self.inicio_quietud = time.time()

                # D. FINALIZANDO (Confirmación de parada)
                elif self.estado_actual == ESTADO_FINALIZANDO:
                    cv2.putText(debug_image, "Mantente quieto para terminar...", (50, 450), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # Seguimos grabando por si fue una falsa alarma
                    self.video_writer.write(frame)
                    self.frames_buffer.append(frame)

                    # Si te vuelves a mover, cancelamos el fin y seguimos grabando
                    if distancia > UMBRAL_MOVIMIENTO:
                        self.estado_actual = ESTADO_GRABANDO
                    
                    # Si pasaste quieto el tiempo suficiente -> CORTAR
                    elif (time.time() - self.inicio_quietud) > TIEMPO_ESPERA_FINAL:
                        self.video_writer.release()
                        frames_totales = len(self.frames_buffer)
                        self.guardar_metadata(f"video_{int(time.time())}.mp4", frames_totales)
                        
                        self.estado_actual = ESTADO_IDLE
                        # Feedback visual verde rápido
                        cv2.rectangle(debug_image, (0,0), (640,480), (0,255,0), 10)
                        cv2.imshow('LSM Recolector', debug_image)
                        cv2.waitKey(500) # Pausa visual

                # Mostrar UI
                # Barra de distancia para debug
                if self.posicion_neutra and isinstance(distancia, float):
                    largo_barra = int(distancia * 500)
                    color = (0, 255, 0) if distancia < UMBRAL_MOVIMIENTO else (0, 0, 255)
                    cv2.rectangle(debug_image, (50, 400), (50 + largo_barra, 420), color, -1)
                    cv2.rectangle(debug_image, (50 + int(UMBRAL_MOVIMIENTO*500), 390), (52 + int(UMBRAL_MOVIMIENTO*500), 430), (255, 255, 0), 2)

                cv2.imshow('LSM Recolector', debug_image)

                if cv2.waitKey(1) & 0xFF == 27: # ESC para salir
                    break
            
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DataRecorder()
    app.run()