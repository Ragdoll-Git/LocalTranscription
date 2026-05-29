import os
import wave
import time
import queue
import threading
import numpy as np
import sounddevice as sd
from config import Config

def list_microphones():
    """
    List all available input audio devices.
    Returns a list of dicts with device info.
    """
    devices = sd.query_devices()
    input_devices = []
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            input_devices.append({
                'id': idx,
                'name': d['name'],
                'channels': d['max_input_channels'],
                'default_sr': d['default_samplerate']
            })
    return input_devices

class MicrophoneStream:
    def __init__(self, device_id, sample_rate=16000, block_size=1024):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.queue = queue.Queue()
        self.stream = None
        self.active = False

    def callback(self, indata, frames, time_info, status):
        if status:
            pass # print(f"Device {self.device_id} status: {status}", flush=True)
        # Put a copy of the input data into the queue
        self.queue.put(indata.copy())

    def start(self):
        self.active = True
        self.stream = sd.InputStream(
            device=self.device_id,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self.callback,
            dtype='float32'
        )
        self.stream.start()

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
                return
            stream = MicrophoneStream(device_id, self.sample_rate, self.block_size)
            self.streams[device_id] = stream
            if self.is_recording:
                stream.start()

    def remove_microphone(self, device_id):
        with self.lock:
            if device_id in self.streams:
                stream = self.streams[device_id]
                stream.stop()
                del self.streams[device_id]

    def start_recording(self, output_dir, file_name="live_recording.wav"):
        with self.lock:
            if self.is_recording:
                return
            
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            self.output_path = os.path.join(output_dir, file_name)
            
            # Setup wave file
            self.wave_file = wave.open(self.output_path, 'wb')
            self.wave_file.setnchannels(1)
            self.wave_file.setsampwidth(2) # 16-bit PCM
            self.wave_file.setframerate(self.sample_rate)
            
            self.recording_buffer = []
            self.is_recording = True
            
            # Start all streams
            for stream in self.streams.values():
                stream.start()
                
            # Start mixing thread
            self.mix_thread = threading.Thread(target=self._mix_loop, daemon=True)
            self.mix_thread.start()

    def stop_recording(self):
        with self.lock:
            if not self.is_recording:
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
