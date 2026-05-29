import argparse
import logging
import os
import tempfile
import math
import sys
import time
import warnings
from contextlib import redirect_stdout, redirect_stderr
from threading import Thread, Event

from pydub import AudioSegment
from tqdm import tqdm
import torch
import nemo.collections.asr as nemo_asr

# -----------------------------
# ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser(description="CPU-optimized Parakeet transcription")
parser.add_argument("audio", type=str, help="Path to input WAV file (required)")
parser.add_argument("--model", type=str, default="parakeet-tdt-0.6b-v3.nemo",
                    help="Path to local .nemo model file")
parser.add_argument("--chunk-seconds", type=int, default=20, help="Chunk length in seconds (default: 20)")
parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads to use (default: 4)")
parser.add_argument("--debug", action="store_true", help="Keep intermediate WAV files for debugging")
parser.add_argument("--textformat", type=str, default="srt", choices=["txt", "srt", "both"],
                    help="Output format: txt, srt, or both (default: srt)")
parser.add_argument("--verbose", action="store_true", help="Show NeMo/PyDub messages in terminal")
args = parser.parse_args()

# -----------------------------
# VERBOSE & WARNINGS
# -----------------------------
if not args.verbose:
    # suppress PyDub ffmpeg warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # suppress all NeMo loggers
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger("nemo").setLevel(logging.CRITICAL)
    logging.getLogger("nemo.collections.asr").setLevel(logging.CRITICAL)

torch.set_num_threads(args.threads)
torch.set_grad_enabled(False)

# -----------------------------
# LOAD MODEL WITH SPINNER
# -----------------------------
spinner_chars = "|/-\\"
spinner_event = Event()

def spinner_task(event):
    idx = 0
    while not event.is_set():
        print(f"\rLoading local model... {spinner_chars[idx % len(spinner_chars)]}", end="", flush=True)
        idx += 1
        time.sleep(0.1)
    print("\rModel loaded!             ")

spinner_thread = Thread(target=spinner_task, args=(spinner_event,))
spinner_thread.start()

from asr_engine import ASREngine
asr_engine = ASREngine.get_instance()
asr_engine.load_model(model_path=args.model, verbose=args.verbose)

# stop spinner
spinner_event.set()
spinner_thread.join()

# -----------------------------
# AUDIO PREP
# -----------------------------
audio = AudioSegment.from_wav(args.audio)
audio = audio.set_channels(1)
audio = audio.set_frame_rate(16000)
chunk_ms = args.chunk_seconds * 1000

# -----------------------------
# TEMP DIR (auto-clean unless debug)
# -----------------------------
temp_ctx = tempfile.TemporaryDirectory() if not args.debug else None
work_dir = temp_ctx.name if temp_ctx else os.getcwd()

mono_path = os.path.join(work_dir, "audio_mono.wav")
audio.export(mono_path, format="wav")

# -----------------------------
# SPLIT INTO CHUNKS
# -----------------------------
chunk_paths = []

for i in range(0, len(audio), chunk_ms):
    chunk = audio[i:i + chunk_ms]
    if args.debug:
        chunk_path = os.path.join(work_dir, f"chunk_{i // chunk_ms}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    else:
        fd, chunk_path = tempfile.mkstemp(suffix=".wav", dir=work_dir)
        os.close(fd)
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)

# -----------------------------
# TRANSCRIBE
# -----------------------------
all_text = []

for idx, chunk_path in enumerate(tqdm(chunk_paths, desc="Transcribing", unit="chunk")):
    result = asr_engine.transcribe([chunk_path])
    all_text.append(result[0])

# -----------------------------
# CLEANUP TEMP FILES
# -----------------------------
if temp_ctx:
    try:
        temp_ctx.cleanup()
    except Exception:
        for path in chunk_paths + [mono_path]:
            try:
                os.remove(path)
            except Exception:
                pass

# -----------------------------
# SAVE OUTPUT
# -----------------------------
base_name = os.path.splitext(os.path.basename(args.audio))[0]

if args.textformat in ["txt", "both"]:
    txt_file = os.path.join(os.getcwd(), f"{base_name}_transcript.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))
    print(f"✅ TXT saved to {txt_file}")

if args.textformat in ["srt", "both"]:
    srt_file = os.path.join(os.getcwd(), f"{base_name}_transcript.srt")

    def format_ts(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        s_int = int(s % 60)
        ms = int((s - int(s)) * 1000)
        return f"{h:02}:{m:02}:{s_int:02},{ms:03}"

    with open(srt_file, "w", encoding="utf-8") as f:
        for i, text in enumerate(all_text):
            start_sec = i * args.chunk_seconds
            end_sec = min((i + 1) * args.chunk_seconds, len(audio) / 1000)
            f.write(f"{i+1}\n{format_ts(start_sec)} --> {format_ts(end_sec)}\n{text}\n\n")
    print(f"✅ SRT saved to {srt_file}")

print(f"✅ Done! Transcript(s) ready: {base_name}_transcript")
