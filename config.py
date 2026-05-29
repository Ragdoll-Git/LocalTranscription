import os
from dotenv import load_dotenv

# Load optional .env file
load_dotenv()

class Config:
    # ASR model settings
    ASR_MODEL_PATH = os.environ.get("ASR_MODEL_PATH", "parakeet-tdt-0.6b-v3.nemo")
    SAMPLE_RATE = 16000
    CHUNK_SECONDS = 10  # Process in 10-second chunks for better live feedback
    CPU_THREADS = int(os.environ.get("CPU_THREADS", "4"))
    
    # VAD settings
    VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
    
    # Speaker Diarization
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    
    # NVIDIA NIM settings
    NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    NIM_API_KEY = os.environ.get("NIM_API_KEY", "")
    NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.3-70b-instruct")
    
    # Customizable context/NIM limits
    NIM_TEMPERATURE = 0.5
    NIM_CONTEXT_WINDOW_TOKENS = 2048
    NIM_MAX_RPM = 15  # Requests per minute rate limiting
    
    # Default folder to save mixed recording
    RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", os.path.join(os.getcwd(), "recordings"))
