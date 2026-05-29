import time
import requests
import threading
from config import Config

class RateLimiter:
    """
    Thread-safe Rate Limiter to enforce Requests Per Minute (RPM) limits.
    """
    def __init__(self, rpm):
        self.rpm = rpm
        self.lock = threading.Lock()
        self.request_times = []

    def set_rpm(self, rpm):
        with self.lock:
            self.rpm = rpm

    def wait_if_needed(self):
        if self.rpm <= 0:
            return
            
        with self.lock:
            now = time.time()
            # Remove times older than 60 seconds
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            if len(self.request_times) >= self.rpm:
                # Calculate sleep time until the oldest request in the window is 60s old
                sleep_time = 60.0 - (now - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # Re-clean up after sleeping
                now = time.time()
                self.request_times = [t for t in self.request_times if now - t < 60]
                
            self.request_times.append(time.time())

class NIMClient:
    def __init__(self, api_key=None, model=None, base_url=None, temperature=None, max_rpm=None):
        self.api_key = api_key or Config.NIM_API_KEY
        self.model = model or Config.NIM_MODEL
        self.base_url = base_url or Config.NIM_BASE_URL
        self.temperature = temperature if temperature is not None else Config.NIM_TEMPERATURE
        
        rpm = max_rpm if max_rpm is not None else Config.NIM_MAX_RPM
        self.rate_limiter = RateLimiter(rpm)

    def update_config(self, api_key=None, model=None, temperature=None, max_rpm=None):
        if api_key is not None:
            self.api_key = api_key
        if model is not None:
            self.model = model
        if temperature is not None:
            self.temperature = float(temperature)
        if max_rpm is not None:
            self.rate_limiter.set_rpm(int(max_rpm))

    def ask(self, question, transcript_text, context_window_chars=4000):
        """
        Sends the question along with a contextualized subset of the transcript (up to context_window_chars).
        """
        if not self.api_key:
            return "Error: NVIDIA NIM API Key not set. Please update in Settings."

        # Enforce rate limiting
        self.rate_limiter.wait_if_needed()

        # Contextualize transcript: extract the latest or most relevant part of the transcript
        # If the transcript is longer than context_window_chars, take the last context_window_chars characters.
        context = transcript_text
        if len(transcript_text) > context_window_chars:
            context = "..." + transcript_text[-context_window_chars:]

        # Custom prompt for technical students
        system_prompt = (
            "Eres una IA experta de IT que asiste a estudiantes técnicos de IT en conferencias y exposiciones.\n"
            "Tu objetivo es analizar la transcripción de la oratoria proporcionada y responder a la pregunta del usuario. "
            "Si el usuario pide preguntas técnicas, debes formular preguntas profundas, desafiantes y constructivas "
            "que los estudiantes puedan hacerle al orador basadas exactamente en los temas que ha expuesto.\n"
            "Mantén un tono académico, técnico y profesional en español.\n"
            "Tambien puedes responder preguntas que el profesor haga para ayudar a los estudiantes a entender mejor el tema.\n"
            "No inventes información, solo basate en la transcripción y en lo que conozcas del tema.\n"
            "Si no sabes la respuesta a una pregunta, responde que no puedes responderla, no intentes adivinar."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcripción Contextualizada:\n\"\"\"\n{context}\n\"\"\"\n\nPregunta/Instrucción: {question}"}
            ],
            "temperature": self.temperature,
            "max_tokens": 1024,
            "stream": False
        }

        try:
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                return f"Error de NVIDIA NIM API ({response.status_code}): {response.text}"
        except Exception as e:
            return f"Excepción durante la conexión con NVIDIA NIM: {str(e)}"
            
    def generate_student_questions(self, transcript_text, context_window_chars=4000):
        """
        Helper method to specifically ask for technical questions to challenge/ask the speaker.
        """
        prompt = (
            "Basándote en la transcripción, formula una lista de 3 preguntas técnicas profundas "
            "y bien estructuradas que un estudiante técnico podría hacerle al orador "
            "de su charla para profundizar en el tema o indagar sobre limitaciones de su enfoque."
        )
        return self.ask(prompt, transcript_text, context_window_chars)
