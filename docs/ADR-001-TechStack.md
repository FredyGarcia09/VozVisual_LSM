# ADR-001: Definición del Stack Tecnológico (Versión inicial)

* **Estado:** Aceptado
* **Fecha:** 2023-10-27
* **Decisores:** Alfredo García Olmedo
* **Contexto Técnico:** Proyecto VozVisual LSM (Primer avance hacia un traductor LSM).

## Contexto y Problema
El proyecto requiere una arquitectura capaz de soportar:
1. Recolección de datos de video masiva (Dataset).
2. Entrenamiento de modelos de IA (Deep Learning).
3. Despliegue en múltiples plataformas (Web, Desktop, Móvil) con recursos limitados (Estudiante/Startup).
4. Escalabilidad futura para funcionalidades de chat y gamificación.

## Decisión
Se ha decidido utilizar el siguiente stack tecnológico:

### 1. Frontend
* **Tecnología:** React.js (Vite).
* **Justificación:** Estándar de la industria, ecosistema masivo, permite migración sencilla a aplicaciones de escritorio (Electron) y móvil (React Native).

### 2. Backend (API & Lógica)
* **Tecnología:** Python con FastAPI.
* **Justificación:**
    * Python es el lenguaje nativo de la IA (PyTorch/TensorFlow/MediaPipe).
    * FastAPI ofrece alto rendimiento (asíncrono) y documentación automática (Swagger UI).
    * Evita la complejidad de mantener dos lenguajes distintos.

### 3. Base de Datos
* **Tecnología:** PostgreSQL.
* **Justificación:** Base de datos relacional robusta, open-source y soportada nativamente en Azure. Ideal para gestionar usuarios, progreso y metadatos.

### 4. Infraestructura (Cloud)
* **Proveedor:** Microsoft Azure.
* **Servicios:**
    * Azure App Service (Hosting API).
    * Azure Blob Storage (Almacenamiento Dataset).
    * Azure Database for PostgreSQL.
* **Justificación:** Aprovechamiento de créditos "Microsoft for Startups Founders Hub" (se busca conseguirlos) y de los creditos "Azure for Students".

### 5. Herramientas de IA (Local & Edge)
* **Librerías:** MediaPipe, OpenCV.
* **Estrategia:** "Edge Computing". La inferencia (detección de señas) correrá en el dispositivo del usuario (Browser/PC) usando TensorFlow.js o ONNX Runtime para reducir latencia y costos de servidor.

## Consecuencias
* **Positivas:** Reducción de costos operativos, unificación de lenguaje (Python), arquitectura moderna y modular.
* **Negativas:** Python puede ser menos estricto en tipos que Java/C#, requiere disciplina en el código. Dependencia de que el dispositivo del usuario soporte WebGL/WASM para la IA.