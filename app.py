import os
import time
import threading
import tempfile
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv, set_key

from config import Config
from audio_engine import list_microphones, MultiMicMixer
from asr_engine import ASREngine
from vad import VoiceActivityDetector
from diarization import SpeakerDiarizer, assign_speakers_to_transcript_chunks
from nim_qa import NIMClient

app = Flask(__name__)
app.config['SECRET_KEY'] = 'localtranscription_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# State holders
mixer = MultiMicMixer()
vad_detector = VoiceActivityDetector()
nim_client = NIMClient()
asr_engine = ASREngine.get_instance()
diarizer = SpeakerDiarizer()

# Live recording state
recording_state = {
    "is_recording": False,
    "output_dir": Config.RECORDINGS_DIR,
    "file_name": "grabacion.wav",
    "chunks": [],  # list of dicts: {start, end, text, classification}
    "last_processed_sample": 0,
    "model_loaded": False
}

# Lock for live processing
processing_lock = threading.Lock()

# Load ASR model in a background thread on startup
def load_asr_background():
    try:
        print("Starting background ASR model load...")
        asr_engine.load_model()
        recording_state["model_loaded"] = True
        print("ASR Model loaded successfully in background.")
        socketio.emit('status_update', {'status': 'idle', 'model_loaded': True})
    except Exception as e:
        print(f"Error loading ASR model: {e}")
        socketio.emit('status_update', {'status': 'error', 'message': f"Failed to load model: {str(e)}"})

threading.Thread(target=load_asr_background, daemon=True).start()

# Helper loop to process chunks during live recording
def live_transcription_loop():
    global mixer
    chunk_samples = Config.CHUNK_SECONDS * Config.SAMPLE_RATE
    
    while True:
        time.sleep(0.5)
        
        with processing_lock:
            if not recording_state["is_recording"]:
                break
                
            current_samples = mixer.get_current_length_samples()
            available_samples = current_samples - recording_state["last_processed_sample"]
            
            if available_samples >= chunk_samples:
                # We have enough audio to process a chunk
                start_sample = recording_state["last_processed_sample"]
                end_sample = start_sample + chunk_samples
                
                audio_chunk = mixer.get_audio_since(start_sample, end_sample)
                recording_state["last_processed_sample"] = end_sample
                
                start_sec = start_sample / Config.SAMPLE_RATE
                end_sec = end_sample / Config.SAMPLE_RATE
                
                # 1. Run VAD
                classification = vad_detector.classify_speech_vs_noise(audio_chunk, Config.SAMPLE_RATE)
                
                text = ""
                speaker = "Hablante 1"
                if classification == "oratoria":
                    # Save temporary wav chunk for ASR
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        temp_path = tf.name
                        
                    try:
                        import wave
                        with wave.open(temp_path, 'wb') as w:
                            w.setnchannels(1)
                            w.setsampwidth(2)
                            w.setframerate(Config.SAMPLE_RATE)
                            int_data = (audio_chunk * 32767.0).astype(np.int16)
                            w.writeframes(int_data.tobytes())
                            
                        # 2. Transcribe
                        results = asr_engine.transcribe([temp_path])
                        text = results[0] if results else ""
                    except Exception as e:
                        print(f"Error transcribing live chunk: {e}")
                        text = "[Error de transcripción]"
                    finally:
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                            
                    # 3. Real-time speaker diarization on accumulated audio
                    if not diarizer.fallback:
                        try:
                            accumulated_audio = mixer.get_audio_since(0)
                            if len(accumulated_audio) > 0:
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf_acc:
                                    acc_path = tf_acc.name
                                try:
                                    import wave
                                    with wave.open(acc_path, 'wb') as w_acc:
                                        w_acc.setnchannels(1)
                                        w_acc.setsampwidth(2)
                                        w_acc.setframerate(Config.SAMPLE_RATE)
                                        int_data_acc = (accumulated_audio * 32767.0).astype(np.int16)
                                        w_acc.writeframes(int_data_acc.tobytes())
                                    
                                    # Diarize accumulated audio
                                    diar_segs = diarizer.diarize(acc_path)
                                    
                                    # Find dominant speaker in this chunk's time window
                                    overlap_times = {}
                                    for seg in diar_segs:
                                        overlap_start = max(start_sec, seg['start'])
                                        overlap_end = min(end_sec, seg['end'])
                                        overlap = overlap_end - overlap_start
                                        if overlap > 0:
                                            overlap_times[seg['speaker']] = overlap_times.get(seg['speaker'], 0.0) + overlap
                                    if overlap_times:
                                        speaker = max(overlap_times, key=overlap_times.get)
                                except Exception as e:
                                    print(f"Error in live diarization: {e}")
                                finally:
                                    try:
                                        os.remove(acc_path)
                                    except Exception:
                                        pass
                        except Exception as e:
                            print(f"Error fetching accumulated audio for live diarization: {e}")
                else:
                    text = "[Ruido / Silencio]"
                    speaker = "Ruido/Silencio"
                
                chunk_data = {
                    "start": start_sec,
                    "end": end_sec,
                    "text": text,
                    "classification": classification,
                    "speaker": speaker
                }
                
                recording_state["chunks"].append(chunk_data)
                
                # Emit live update to front-end
                socketio.emit('new_chunk', chunk_data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/mics', methods=['GET'])
def get_mics():
    all_mics = list_microphones()
    active_mics = list(mixer.streams.keys())
    return jsonify({
        'all': all_mics,
        'active': active_mics
    })

@app.route('/api/mics/add', methods=['POST'])
def add_mic():
    data = request.json
    device_id = int(data.get('id'))
    mixer.add_microphone(device_id)
    return jsonify({'status': 'ok', 'active': list(mixer.streams.keys())})

@app.route('/api/mics/remove', methods=['POST'])
def remove_mic():
    data = request.json
    device_id = int(data.get('id'))
    mixer.remove_microphone(device_id)
    return jsonify({'status': 'ok', 'active': list(mixer.streams.keys())})

@app.route('/api/recording/start', methods=['POST'])
def start_rec():
    if not recording_state["model_loaded"]:
        return jsonify({'status': 'error', 'message': 'ASR model is still loading. Please wait.'}), 400
        
    data = request.json or {}
    output_dir = data.get('output_dir', Config.RECORDINGS_DIR)
    file_name = data.get('file_name', 'live_recording.wav')
    
    if not file_name.endswith('.wav'):
        file_name += '.wav'
        
    recording_state["output_dir"] = output_dir
    recording_state["file_name"] = file_name
    recording_state["chunks"] = []
    recording_state["last_processed_sample"] = 0
    recording_state["is_recording"] = True
    
    # Start mixer
    mixer.start_recording(output_dir, file_name)
    
    # Start live transcription worker thread
    threading.Thread(target=live_transcription_loop, daemon=True).start()
    
    return jsonify({
        'status': 'recording',
        'output_path': os.path.join(output_dir, file_name)
    })

@app.route('/api/recording/stop', methods=['POST'])
def stop_rec():
    if not recording_state["is_recording"]:
        return jsonify({'status': 'error', 'message': 'Not recording'}), 400
        
    recording_state["is_recording"] = False
    
    # Stop mixer and save final WAV
    wav_path = mixer.stop_recording()
    
    # Post-processing: Speaker Diarization on the saved WAV file
    diarization_segments = diarizer.diarize(wav_path)
    
    # Assign speakers
    final_chunks = assign_speakers_to_transcript_chunks(
        recording_state["chunks"], 
        diarization_segments
    )
    recording_state["chunks"] = final_chunks
    
    # Save text & srt outputs in the same custom folder
    base_name = os.path.splitext(wav_path)[0]
    txt_path = f"{base_name}_transcript.txt"
    srt_path = f"{base_name}_transcript.srt"
    
    # Save TXT
    with open(txt_path, 'w', encoding='utf-8') as f:
        for chunk in final_chunks:
            speaker = chunk.get('speaker', 'Hablante 1')
            text = chunk.get('text', '')
            if chunk.get('classification') == 'ruido':
                f.write(f"[{speaker}]: (Ruido / Silencio)\n")
            else:
                f.write(f"[{speaker}]: {text}\n")
                
    # Save SRT
    def format_ts(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        s_int = int(s % 60)
        ms = int((s - int(s)) * 1000)
        return f"{h:02}:{m:02}:{s_int:02},{ms:03}"
        
    with open(srt_path, 'w', encoding='utf-8') as f:
        for idx, chunk in enumerate(final_chunks):
            start = format_ts(chunk['start'])
            end = format_ts(chunk['end'])
            speaker = chunk['speaker']
            text = chunk['text']
            f.write(f"{idx+1}\n{start} --> {end}\n[{speaker}]: {text}\n\n")

    return jsonify({
        'status': 'idle',
        'wav_path': wav_path,
        'txt_path': txt_path,
        'srt_path': srt_path,
        'chunks': final_chunks
    })

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'GET':
        return jsonify({
            'api_key': nim_client.api_key,
            'model': nim_client.model,
            'temperature': nim_client.temperature,
            'max_rpm': nim_client.rate_limiter.rpm,
            'context_window': Config.NIM_CONTEXT_WINDOW_TOKENS,
            'hf_token': Config.HF_TOKEN,
            'output_dir': Config.RECORDINGS_DIR,
            'model_loaded': recording_state["model_loaded"]
        })
    else:
        data = request.json
        
        # Update config values in memory
        if 'api_key' in data:
            nim_client.update_config(api_key=data['api_key'])
            Config.NIM_API_KEY = data['api_key']
        if 'model' in data:
            nim_client.update_config(model=data['model'])
            Config.NIM_MODEL = data['model']
        if 'temperature' in data:
            nim_client.update_config(temperature=float(data['temperature']))
            Config.NIM_TEMPERATURE = float(data['temperature'])
        if 'max_rpm' in data:
            nim_client.update_config(max_rpm=int(data['max_rpm']))
            Config.NIM_MAX_RPM = int(data['max_rpm'])
        if 'context_window' in data:
            Config.NIM_CONTEXT_WINDOW_TOKENS = int(data['context_window'])
        if 'hf_token' in data:
            global diarizer
            Config.HF_TOKEN = data['hf_token']
            diarizer = SpeakerDiarizer(hf_token=data['hf_token'])
        if 'output_dir' in data:
            Config.RECORDINGS_DIR = data['output_dir']
            
        # Try to save to a local .env file if it exists, to persist settings
        try:
            env_path = '.env'
            if not os.path.exists(env_path):
                with open(env_path, 'w') as f:
                    pass
            set_key(env_path, "NIM_API_KEY", Config.NIM_API_KEY)
            set_key(env_path, "NIM_MODEL", Config.NIM_MODEL)
            set_key(env_path, "HF_TOKEN", Config.HF_TOKEN)
            set_key(env_path, "RECORDINGS_DIR", Config.RECORDINGS_DIR)
            set_key(env_path, "NIM_TEMPERATURE", str(Config.NIM_TEMPERATURE))
            set_key(env_path, "NIM_CONTEXT_WINDOW_TOKENS", str(Config.NIM_CONTEXT_WINDOW_TOKENS))
            set_key(env_path, "NIM_MAX_RPM", str(Config.NIM_MAX_RPM))
        except Exception as e:
            print(f"Could not persist to .env: {e}")
            
        return jsonify({'status': 'ok'})

@app.route('/api/nim/ask', methods=['POST'])
def nim_ask():
    data = request.json
    question = data.get('question')
    
    # Get current transcript text
    text_parts = []
    for chunk in recording_state["chunks"]:
        speaker = chunk.get('speaker', 'Hablante 1')
        txt = chunk.get('text', '')
        if chunk.get('classification') == 'oratoria':
            text_parts.append(f"[{speaker}]: {txt}")
        else:
            text_parts.append(f"[{speaker}]: (Ruido / Silencio)")
            
    transcript_text = "\n".join(text_parts)
    
    # Context window setting in characters
    # Map context window tokens to an approximate character length (~4 chars per token)
    context_chars = Config.NIM_CONTEXT_WINDOW_TOKENS * 4
    
    response = nim_client.ask(question, transcript_text, context_window_chars=context_chars)
    return jsonify({'response': response})

@app.route('/api/nim/suggest_questions', methods=['POST'])
def suggest_questions():
    # Build current transcript text
    text_parts = []
    for chunk in recording_state["chunks"]:
        if chunk.get('classification') == 'oratoria':
            text_parts.append(f"[{chunk.get('speaker')}]: {chunk.get('text')}")
            
    transcript_text = "\n".join(text_parts)
    context_chars = Config.NIM_CONTEXT_WINDOW_TOKENS * 4
    
    response = nim_client.generate_student_questions(transcript_text, context_window_chars=context_chars)
    return jsonify({'response': response})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)
