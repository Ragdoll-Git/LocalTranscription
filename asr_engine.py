import os
import sys
import logging
import torch
import warnings
from contextlib import redirect_stdout, redirect_stderr
import nemo.collections.asr as nemo_asr
from config import Config

class ASREngine:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model = None
        self.verbose = False

    def load_model(self, model_path=None, verbose=False):
        if self.model is not None:
            return self.model
            
        self.verbose = verbose
        if not verbose:
            # Suppress NeMo logging and warnings
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for logger_name in logging.root.manager.loggerDict:
                logging.getLogger(logger_name).setLevel(logging.CRITICAL)
            logging.getLogger("nemo").setLevel(logging.CRITICAL)
            logging.getLogger("nemo.collections.asr").setLevel(logging.CRITICAL)

        path = model_path or Config.ASR_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"ASR model file not found at: {path}. Please place it in the directory or specify in config.")

        torch.set_num_threads(Config.CPU_THREADS)
        torch.set_grad_enabled(False)

        # Redirect output if verbose is False
        stdout_ctx = redirect_stdout(sys.stdout) if verbose else redirect_stdout(open(os.devnull, "w"))
        stderr_ctx = redirect_stderr(sys.stderr) if verbose else redirect_stderr(open(os.devnull, "w"))

        with stdout_ctx, stderr_ctx:
            self.model = nemo_asr.models.ASRModel.restore_from(
                restore_path=path,
                map_location="cpu"
            )
            self.model.eval()
            self.model = self.model.to(torch.device("cpu"))
            self.model.cfg.decoding.strategy = "greedy"
        
        return self.model

    def transcribe(self, audio_paths):
        """
        Transcribe a list of audio file paths.
        """
        if self.model is None:
            self.load_model()
            
        stdout_ctx = redirect_stdout(sys.stdout) if self.verbose else redirect_stdout(open(os.devnull, "w"))
        stderr_ctx = redirect_stderr(sys.stderr) if self.verbose else redirect_stderr(open(os.devnull, "w"))
        
        with stdout_ctx, stderr_ctx:
            results = self.model.transcribe(audio_paths, batch_size=len(audio_paths))
            
        # Parakeet returns transcription objects, extract text
        return [res.text.strip() if hasattr(res, 'text') else str(res).strip() for res in results]
