import os
import wave
import time
import queue
import threading
import logging
import numpy as np
import sounddevice as sd
from config import Config

try:
    from logger_setup import get_logger
    log = get_logger("audio")
except Exception:
    log = logging.getLogger("audio")

_HOST_PRIORITY = {
    # MME is the most permissive Windows API. Empirically the only one that opens
    # reliably with callback-mode streams on this hardware.  Higher-fidelity hosts
    # (WASAPI/DirectSound) intermittently return GLE 0x490 from WdmSyncIoctl even
    # when the device is otherwise free.
    "MME": 0,
    "Windows DirectSound": 1,
    "Windows WASAPI": 2,
}

# WDM-KS requires exclusive kernel access; almost always fails in shared setups.
_SKIPPED_HOSTS = ("Windows WDM-KS",)

_NON_MIC_KEYWORDS = (
    "mapper",                       # Microsoft Sound Mapper
    "primary sound capture",        # EN generic driver alias
    "primary capture driver",
    "controlador primario",         # ES Primary Sound Capture Driver
    "controlador de captura",
    "asignador de sonido",          # ES Sound Mapper
    "stereo mix",                   # EN system-loopback (not a real mic)
    "mezcla est",                   # ES Mezcla estereo / estéreo
    "what u hear",                  # SoundBlaster loopback
)


def _is_real_mic(name):
    lower = name.lower().strip()
    if not lower:
        return False
    return not any(kw in lower for kw in _NON_MIC_KEYWORDS)


def list_microphones():
    """
    List input audio devices, deduplicated by name across host APIs.
    Prefers WASAPI > WDM-KS > DirectSound > MME, drops virtual mappers.
    """
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as e:
        log.exception("list_microphones: sd.query_devices failed")
        return []

    candidates = []
    for idx, d in enumerate(devices):
        if d.get('max_input_channels', 0) <= 0:
            continue
        name = (d.get('name') or '').strip()
        if not _is_real_mic(name):
            continue
        host_idx = d.get('hostapi', -1)
        host_name = hostapis[host_idx]['name'] if 0 <= host_idx < len(hostapis) else ''
        if host_name in _SKIPPED_HOSTS:
            continue
        candidates.append({
            'id': idx,
            'name': name,
            'channels': d['max_input_channels'],
            'default_sr': d.get('default_samplerate'),
            'host': host_name,
            '_prio': _HOST_PRIORITY.get(host_name, 99),
        })

    # Group by prefix-equivalence: MME truncates names to ~30 chars while
    # WASAPI/DirectSound keep the full name. Two devices belong to the same
    # physical mic if the shorter name is a prefix of the longer one.
    groups = []
    for c in candidates:
        cname = c['name'].lower()
        matched = None
        for g in groups:
            gname = g['_display_name'].lower()
            short, long = (cname, gname) if len(cname) <= len(gname) else (gname, cname)
            if long.startswith(short):
                matched = g
                break
        if matched is None:
            groups.append({**c, '_display_name': c['name'], '_members': [c]})
        else:
            matched['_members'].append(c)
            if len(c['name']) > len(matched['_display_name']):
                matched['_display_name'] = c['name']
            if c['_prio'] < matched['_prio']:
                # Promote: take this candidate's id+host (better access), keep long name
                matched['id'] = c['id']
                matched['host'] = c['host']
                matched['_prio'] = c['_prio']
                matched['channels'] = c['channels']
                matched['default_sr'] = c['default_sr']

    result = []
    for g in groups:
        result.append({
            'id': g['id'],
            'name': g['_display_name'],
            'channels': g['channels'],
            'default_sr': g['default_sr'],
            'host': g['host'],
        })
    result.sort(key=lambda x: x['name'].lower())
    log.debug("list_microphones: %d raw -> %d after sanitization", len(candidates), len(result))
    return result

class MicrophoneStream:
    def __init__(self, device_id, sample_rate=16000, block_size=1024):
        self.device_id = device_id
        self.sample_rate = sample_rate        # target rate (16k for ASR)
        self.block_size = block_size
        self.queue = queue.Queue()
        self.stream = None
        self.active = False
        self.actual_rate = sample_rate        # set by start()

    def callback(self, indata, frames, time_info, status):
        if status:
            pass
        # Resample to target rate only when the device couldn't open at 16k
        if self.actual_rate != self.sample_rate:
            try:
                from scipy.signal import resample_poly
                mono = indata[:, 0] if indata.ndim > 1 else indata
                out = resample_poly(mono, up=self.sample_rate, down=self.actual_rate)
                out = out.astype(np.float32).reshape(-1, 1)
                self.queue.put(out)
            except Exception:
                # If scipy isn't available, just push raw — better than nothing
                self.queue.put(indata.copy())
        else:
            self.queue.put(indata.copy())

    def start(self):
        """
        Opens the device. Strategy:
          1) Query the device's default sample rate (what WASAPI/Windows actually expects).
          2) Try opening at that rate without any extra_settings — this is what worked
             in our standalone diagnostic for all WASAPI/DirectSound mics.
          3) If we got something other than 16 kHz, resample in the callback.
        Hand-set blocksize=0 lets PortAudio pick a safe value (avoids huge buffers).
        """
        self.active = True

        info = sd.query_devices(self.device_id)
        default_rate = int(round(info.get('default_samplerate') or 48000))

        # Build the ordered list of rates to try
        candidates = []
        if default_rate not in candidates:
            candidates.append(default_rate)
        for r in (48000, 44100, 32000, 16000):
            if r not in candidates:
                candidates.append(r)

        last_err = None
        for rate in candidates:
            try:
                # Mirror exactly what worked in the standalone diagnostic:
                # no blocksize, no extra_settings.
                self.stream = sd.InputStream(
                    device=self.device_id,
                    channels=1,
                    samplerate=rate,
                    callback=self.callback,
                    dtype='float32',
                )
                self.actual_rate = rate
                self.stream.start()
                if rate == self.sample_rate:
                    log.info("MicrophoneStream %s started @ %d Hz", self.device_id, rate)
                else:
                    log.info("MicrophoneStream %s started @ %d Hz (resample -> %d)",
                             self.device_id, rate, self.sample_rate)
                return
            except Exception as e:
                last_err = e
                self.stream = None
                log.warning("Mic %s: rate %d failed (%s)", self.device_id, rate, e)

        # All rates failed — give up loudly
        raise RuntimeError(
            f"Mic {self.device_id}: ningún sample rate funcionó. Último error: {last_err}"
        )

    def stop(self):
        self.active = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


class MultiMicMixer:
    def __init__(self, sample_rate=16000, block_size=1024):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.streams = {} # device_id -> MicrophoneStream
        self.is_recording = False
        
        self.mix_thread = None
        self.wave_file = None
        self.output_path = None
        self.recording_buffer = []  # Store whole mixed recording as float32 for transcription chunking
        
        # Lock for stream management
        self.lock = threading.Lock()

    def add_microphone(self, device_id):
        with self.lock:
            if device_id in self.streams:
                log.debug("add_microphone: %s already added", device_id)
                return
            stream = MicrophoneStream(device_id, self.sample_rate, self.block_size)
            self.streams[device_id] = stream
            if self.is_recording:
                stream.start()
            log.info("add_microphone: device %s added (total=%d)", device_id, len(self.streams))

    def remove_microphone(self, device_id):
        with self.lock:
            if device_id in self.streams:
                stream = self.streams[device_id]
                stream.stop()
                del self.streams[device_id]
                log.info("remove_microphone: device %s removed (remaining=%d)", device_id, len(self.streams))

    def start_recording(self, output_dir, file_name="live_recording.wav"):
        with self.lock:
            if self.is_recording:
                log.warning("start_recording called but already recording")
                return

            os.makedirs(output_dir, exist_ok=True)
            self.output_path = os.path.join(output_dir, file_name)
            log.info("start_recording: %s with %d mic(s)", self.output_path, len(self.streams))

            self.wave_file = wave.open(self.output_path, 'wb')
            self.wave_file.setnchannels(1)
            self.wave_file.setsampwidth(2)
            self.wave_file.setframerate(self.sample_rate)

            self.recording_buffer = []

            # Try to start each stream; if ALL fail, roll back cleanly so the caller
            # can retry and the UI isn't stuck thinking we're recording.
            started = []
            for dev_id, stream in self.streams.items():
                try:
                    stream.start()
                    started.append(dev_id)
                except Exception as e:
                    log.error("Mic %s failed to start: %s", dev_id, e)

            if not started:
                # Roll back: close wav, reset state, raise so endpoint can return 500
                try:
                    self.wave_file.close()
                except Exception:
                    pass
                self.wave_file = None
                try:
                    os.remove(self.output_path)
                except Exception:
                    pass
                self.output_path = None
                self.is_recording = False
                raise RuntimeError(
                    "Ningun microfono pudo abrirse. Verificá permisos de Windows "
                    "(Configuración → Privacidad → Micrófono) y que no esté en uso por otra app."
                )

            self.is_recording = True
            self.mix_thread = threading.Thread(target=self._mix_loop, daemon=True)
            self.mix_thread.start()

    def stop_recording(self):
        with self.lock:
            if not self.is_recording:
                log.warning("stop_recording called but not recording")
                return None
            self.is_recording = False

        if self.mix_thread:
            self.mix_thread.join()

        with self.lock:
            for stream in self.streams.values():
                stream.stop()

            if self.wave_file:
                self.wave_file.close()
                self.wave_file = None

            path = self.output_path
            self.output_path = None
            samples = sum(len(c) for c in self.recording_buffer)
            log.info("stop_recording: wrote %s (%.1fs of audio)", path, samples / self.sample_rate)
            return path

    def _mix_loop(self):
        """
        The background thread that pulls from all active microphone queues,
        mixes them, and writes the output to the wave file and recording_buffer.
        """
        while self.is_recording:
            # We want to pull one chunk from each active stream
            chunks = []
            active_ids = []
            
            with self.lock:
                active_streams = list(self.streams.items())
                
            if not active_streams:
                time.sleep(0.01)
                continue
                
            # Try to fetch from each queue
            for dev_id, stream in active_streams:
                try:
                    # Non-blocking get with a tiny timeout or check empty
                    chunk = stream.queue.get(timeout=0.1)
                    chunks.append(chunk)
                    active_ids.append(dev_id)
                except queue.Empty:
                    # If empty, we can skip or pad with zeros
                    pass
            
            if not chunks:
                continue
                
            # Find the minimum chunk size to prevent mismatch issues
            min_len = min(len(c) for c in chunks)
            
            # Mix the chunks (summation)
            mixed_chunk = np.zeros((min_len, 1), dtype=np.float32)
            for c in chunks:
                mixed_chunk += c[:min_len]
                
            # Normalize to avoid clipping (divide by number of mixed channels, or apply soft limiter)
            num_mics = len(chunks)
            if num_mics > 0:
                mixed_chunk = mixed_chunk / num_mics
                
            # Store in float32 buffer for the ASR processing
            self.recording_buffer.append(mixed_chunk.flatten())
            
            # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767] for WAV file
            int_data = (mixed_chunk * 32767.0).astype(np.int16)
            
            # Write to wave file
            if self.wave_file:
                try:
                    self.wave_file.writeframes(int_data.tobytes())
                except Exception:
                    pass

    def get_audio_since(self, start_sample, end_sample=None):
        """
        Retrieves a slice of the recorded float32 audio.
        """
        if not self.recording_buffer:
            return np.array([], dtype=np.float32)
            
        full_audio = np.concatenate(self.recording_buffer)
        if end_sample is None:
            return full_audio[start_sample:]
        return full_audio[start_sample:end_sample]

    def get_current_length_samples(self):
        if not self.recording_buffer:
            return 0
        return sum(len(c) for c in self.recording_buffer)
