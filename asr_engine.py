import os
import sys
import logging
# pyrefly: ignore [missing-import]
import torch
import warnings
from contextlib import redirect_stdout, redirect_stderr
from types import ModuleType

# Mock missing nv_one_logger and its submodules to prevent import crashes in NeMo on local systems
if "nv_one_logger" not in sys.modules:
    def _make_pkg(name):
        m = ModuleType(name)
        m.__path__ = []  # Required so Python treats it as a package and allows submodule imports
        m.__package__ = name
        return m

    mock_module = _make_pkg("nv_one_logger")
    api_module = _make_pkg("nv_one_logger.api")
    config_module = ModuleType("nv_one_logger.api.config")
    telemetry_module = _make_pkg("nv_one_logger.training_telemetry")
    telemetry_api_module = _make_pkg("nv_one_logger.training_telemetry.api")
    callbacks_module = ModuleType("nv_one_logger.training_telemetry.api.callbacks")
    telemetry_config_module = ModuleType("nv_one_logger.training_telemetry.api.config")
    telemetry_provider_module = ModuleType("nv_one_logger.training_telemetry.api.training_telemetry_provider")
    integration_module = _make_pkg("nv_one_logger.training_telemetry.integration")
    ptl_module = ModuleType("nv_one_logger.training_telemetry.integration.pytorch_lightning")

    class DummyConfig:
        def __init__(self, *args, **kwargs):
            pass

    class DummyProvider:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    class DummyTimeEventCallback:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    def dummy_on_app_start(*args, **kwargs):
        pass

    config_module.OneLoggerConfig = DummyConfig
    callbacks_module.on_app_start = dummy_on_app_start
    telemetry_config_module.TrainingTelemetryConfig = DummyConfig
    telemetry_provider_module.TrainingTelemetryProvider = DummyProvider
    ptl_module.TimeEventCallback = DummyTimeEventCallback

    api_module.config = config_module
    mock_module.api = api_module

    telemetry_api_module.callbacks = callbacks_module
    telemetry_api_module.config = telemetry_config_module
    telemetry_api_module.training_telemetry_provider = telemetry_provider_module
    telemetry_module.api = telemetry_api_module
    integration_module.pytorch_lightning = ptl_module
    telemetry_module.integration = integration_module
    mock_module.training_telemetry = telemetry_module

    sys.modules["nv_one_logger"] = mock_module
    sys.modules["nv_one_logger.api"] = api_module
    sys.modules["nv_one_logger.api.config"] = config_module
    sys.modules["nv_one_logger.training_telemetry"] = telemetry_module
    sys.modules["nv_one_logger.training_telemetry.api"] = telemetry_api_module
    sys.modules["nv_one_logger.training_telemetry.api.callbacks"] = callbacks_module
    sys.modules["nv_one_logger.training_telemetry.api.config"] = telemetry_config_module
    sys.modules["nv_one_logger.training_telemetry.api.training_telemetry_provider"] = telemetry_provider_module
    sys.modules["nv_one_logger.training_telemetry.integration"] = integration_module
    sys.modules["nv_one_logger.training_telemetry.integration.pytorch_lightning"] = ptl_module

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
