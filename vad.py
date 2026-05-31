import logging
import numpy as np
import torch
import warnings
from config import Config

try:
    from logger_setup import get_logger
    log = get_logger("vad")
except Exception:
    log = logging.getLogger("vad")


class VoiceActivityDetector:
    def __init__(self):
        self.model = None
        self.utils = None
        self.fallback = False

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model, self.utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
            log.info("Silero VAD loaded")
        except Exception as e:
            log.warning("Silero VAD failed (%s) — using energy-based fallback", e)
            self.fallback = True

    def is_speech_chunk(self, audio_data, sample_rate=16000, threshold=None):
        """
        Detects if a numpy float32 array contains speech.
        audio_data: np.array of float32 values.
        """
        if threshold is None:
            threshold = Config.VAD_THRESHOLD
            
        if len(audio_data) == 0:
            return False

        if self.fallback:
            # Energy-based VAD fallback
            # Calculate Root Mean Square (RMS) energy
            rms = np.sqrt(np.mean(audio_data**2))
            # Define an empirical threshold (0.015 for normalized float32 audio)
            return rms > (threshold * 0.03)
        else:
            try:
                # Silero expects torch tensor
                tensor = torch.from_numpy(audio_data).float()
                # Run the model
                speech_prob = self.model(tensor, sample_rate).item()
                return speech_prob >= threshold
            except Exception:
                # If Silero fails on chunk size (it expects specific sizes like 512, 1024, 1536 samples)
                # Let's run energy-based VAD as secondary fallback
                rms = np.sqrt(np.mean(audio_data**2))
                return rms > (threshold * 0.03)

    def classify_speech_vs_noise(self, audio_data, sample_rate=16000, chunk_size=512):
        """
        Classifies whether the audio has structured speech ('oratoria')
        or conversational noise / background noise ('ruido').
        It scans the audio in small frames and calculates the ratio of active speech.
        """
        if len(audio_data) < chunk_size:
            return "ruido"
            
        speech_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_data) - chunk_size, chunk_size):
            frame = audio_data[i:i+chunk_size]
            total_frames += 1
            if self.is_speech_chunk(frame, sample_rate):
                speech_frames += 1
                
        if total_frames == 0:
            return "ruido"
            
        speech_ratio = speech_frames / total_frames
        
        # If speech is present for more than 25% of the time, we classify it as speech (oratoria).
        # Otherwise, if it's mostly silent or has sporadic clicks/noise, we classify it as noise.
        if speech_ratio > 0.25:
            return "oratoria"
        else:
            return "ruido"
