# LocalTranscription v2.0

**Infraestructura para transcripción local por voz en tiempo real con soporte multi-micrófono, clasificación por VAD, diarización de hablantes y asistente técnico de preguntas y respuestas mediante NVIDIA NIM.**

Este proyecto ha sido mejorado para ofrecer un entorno interactivo y web que facilita la grabación de clases, conferencias o reuniones.

---

## Características principales

*   **Soporte Multi-Micrófono**: Selecciona de manera dinámica múltiples dispositivos de entrada (por ejemplo, micrófono interno de la computadora y micrófono de webcam) para sumarlos y normalizarlos en tiempo real.
*   **Transcripción Local Ultrarrápida**: Utiliza el modelo local **NVIDIA Parakeet TDT 0.6B** en CPU.
*   **VAD en Tiempo Real**: Clasificación automatizada de cada segundo de audio en habla estructurada (`oratoria`) o `ruido/silencio` utilizando **Silero VAD** (con fallback local).
*   **Diarización en Vivo y Post-Procesamiento**: Identificación automatizada de hablantes (Hablante 1, 2, 3, etc.) tanto en vivo como al finalizar la grabación con **pyannote.audio**.
*   **Inteligencia y Q&A con NVIDIA NIM**: Integración con las APIs de NVIDIA NIM para realizar preguntas técnicas avanzadas basadas en el contexto exacto de la oratoria. Ajustes de RPM (por defecto 30 RPM), temperatura y tamaño de la ventana de contexto modificables en tiempo real desde la web.
*   **Panel de Control Web**: Interfaz oscura de alto contraste premium (estática, sin transiciones ni animaciones lentas) para un control completo de micrófonos, transcripción en vivo, guardado en carpetas personalizadas y chat.

---

## Requisitos previos

*   Python **3.10**
*   FFmpeg instalado en el sistema (requerido por `pydub` para el manejo de WAV).
*   [Hugging Face Account & Token](https://huggingface.co/) (para poder descargar el modelo de diarización `pyannote/speaker-diarization-3.1`). Debes aceptar los términos del modelo en Hugging Face.
*   [NVIDIA NIM API Key](https://build.nvidia.com/) (gratuita al registrarse, prefijo `nvapi-`).

---

## Instalación y Configuración

### 1. Clonar el repositorio e instalar dependencias

Puedes usar **pip** o **conda** para configurar tu entorno:

#### Opción A: Usando Pip (Recomendado)
```bash
pip install -r requirements.txt
```

#### Opción B: Usando Conda
```bash
conda env create -f environment.yml
conda activate localtranscription
```

### 2. Descargar el modelo ASR (Parakeet TDT)

1.  Descarga el archivo del modelo desde Hugging Face:  
    👉 [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
2.  Descarga la versión `.nemo`: `parakeet-tdt-0.6b-v3.nemo`
3.  Coloca el archivo `.nemo` en la raíz de este repositorio.

### 3. Configurar variables de entorno

Copia la plantilla de variables de entorno y renómbrala a `.env`:
```bash
cp .env.example .env
```
Edita `.env` con tus tokens y rutas preferidas:
```env
NIM_API_KEY=nvapi-tu-api-key-aqui
HF_TOKEN=hf_tu-token-huggingface-aqui
RECORDINGS_DIR=C:\ruta\de\guardado\grabaciones
```
*Nota: También puedes cambiar o ingresar estos valores directamente desde la página web en la sección de ajustes.*

---

## Cómo Ejecutar el Proyecto

### 🌐 Interfaz Web Dashboard (Recomendado)

Levanta el servidor Flask local:
```bash
python app.py
```
El servidor cargará el modelo local en segundo plano (puedes ver el estado de carga en la parte superior derecha de la web). 

Abre tu navegador en:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

**Flujo en la Web:**
1.  **Selecciona tus micrófonos** en el listado lateral.
2.  Configura la **Carpeta de Guardado** donde desees almacenar el archivo WAV final, el archivo SRT de subtítulos y el TXT con la transcripción.
3.  Haz clic en **Iniciar Grabación**. La transcripción se mostrará cada 1 segundo en el área de texto identificando el hablante en tiempo real.
4.  Usa el panel de **Análisis e Inteligencia** para formular preguntas técnicas a NVIDIA NIM sobre los temas expuestos por el orador.
5.  Haz clic en **Detener** para consolidar el archivo de audio y generar los archivos finales `.txt` y `.srt` diarizados en tu carpeta especificada.

---

### 💻 Uso mediante Consola (CLI original)

Si deseas realizar la transcripción offline de un archivo de audio WAV pre-grabado en formato SRT o TXT:

```bash
python transcribe_nemo.py path/to/audio.wav --model parakeet-tdt-0.6b-v3.nemo --textformat both
```

#### Parámetros opcionales del CLI:
*   `--model PATH`: Ruta al modelo `.nemo`.
*   `--chunk-seconds N`: Duración en segundos de procesamiento (por defecto: 20).
*   `--threads N`: Cantidad de hilos de CPU (por defecto: 4).
*   `--textformat`: `txt`, `srt` o `both` (por defecto: `srt`).