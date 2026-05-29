import os
import torch
from config import Config

class SpeakerDiarizer:
    def __init__(self, hf_token=None):
        self.hf_token = hf_token or Config.HF_TOKEN
        self.pipeline = None
        self.fallback = False
        
        if not self.hf_token:
            print("No Hugging Face token provided in config. Diarization will fall back to single-speaker mode.")
            self.fallback = True
            return

        try:
            # Import dynamically to prevent crashes if pyannote is not installed
            from pyannote.audio import Pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            if self.pipeline is not None:
                # Run on GPU if available, else CPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.pipeline.to(device)
                print(f"Loaded pyannote speaker-diarization-3.1 model on {device}.")
            else:
                print("Failed to load pretrained pyannote pipeline. Using single-speaker fallback.")
                self.fallback = True
        except Exception as e:
            print(f"Failed to initialize pyannote.audio: {e}. Diarization will use fallback.")
            self.fallback = True

    def diarize(self, wav_path):
        """
        Process the WAV file and return a list of segments with speaker labels.
        Format: [{'start': start_sec, 'end': end_sec, 'speaker': 'Hablante X'}]
        """
        if self.fallback or not self.pipeline:
            # Simple fallback: return one big segment or basic classification
            import wave
            try:
                with wave.open(wav_path, 'rb') as w:
                    frames = w.getnframes()
                    rate = w.getframerate()
                    duration = frames / float(rate)
                return [{'start': 0.0, 'end': duration, 'speaker': 'Hablante 1'}]
            except Exception:
                return [{'start': 0.0, 'end': 3600.0, 'speaker': 'Hablante 1'}]

        try:
            # Run the diarization pipeline
            diarization = self.pipeline(wav_path)
            
            # Map pyannote's internal speaker labels (e.g. SPEAKER_00) to clean numbers (Hablante 1, Hablante 2, etc.)
            speaker_map = {}
            speaker_counter = 1
            
            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_map:
                    speaker_map[speaker] = f"Hablante {speaker_counter}"
                    speaker_counter += 1
                    
                results.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker_map[speaker]
                })
            
            # Merge adjacent segments of the same speaker
            if not results:
                return []
                
            merged_results = []
            current = results[0]
            for next_seg in results[1:]:
                if next_seg['speaker'] == current['speaker'] and next_seg['start'] - current['end'] < 1.0:
                    current['end'] = next_seg['end']
                else:
                    merged_results.append(current)
                    current = next_seg
            merged_results.append(current)
            
            return merged_results
        except Exception as e:
            print(f"Error during speaker diarization: {e}")
            # Safe fallback
            return [{'start': 0.0, 'end': 3600.0, 'speaker': 'Hablante 1'}]

def assign_speakers_to_transcript_chunks(chunks, diarization_segments):
    """
    Given a list of transcript chunks with their start/end timestamps,
    assign the dominant speaker for that duration.
    """
    assigned_chunks = []
    
    for chunk in chunks:
        start = chunk.get('start', 0.0)
        end = chunk.get('end', start + 10.0)
        text = chunk.get('text', '')
        classification = chunk.get('classification', 'oratoria')
        
        if classification == 'ruido':
            assigned_chunks.append({
                'start': start,
                'end': end,
                'speaker': 'Ruido/Silencio',
                'text': text,
                'classification': 'ruido'
            })
            continue

        # Find dominant speaker in this time range
        overlap_times = {}
        for seg in diarization_segments:
            # Calculate overlap duration
            overlap_start = max(start, seg['start'])
            overlap_end = min(end, seg['end'])
            overlap = overlap_end - overlap_start
            if overlap > 0:
                speaker = seg['speaker']
                overlap_times[speaker] = overlap_times.get(speaker, 0.0) + overlap
                
        if overlap_times:
            dominant_speaker = max(overlap_times, key=overlap_times.get)
        else:
            dominant_speaker = 'Hablante 1'
            
        assigned_chunks.append({
            'start': start,
            'end': end,
            'speaker': dominant_speaker,
            'text': text,
            'classification': 'oratoria'
        })
        
    return assigned_chunks
