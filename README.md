# LocalTranscription

**Infrastructure for running local speech-to-text (STT) transcription using NVIDIA’s Parakeet TDT 0.6B (via NeMo).**

This repository provides a simple, reliable pipeline for offline transcription of WAV audio using NVIDIA’s **Parakeet TDT 0.6B** model. It is optimized for **CPU-only environments** and designed to work consistently across common platforms (Windows, Linux, macOS).

The core script, `transcribe_nemo.py`, handles audio preprocessing, chunking, and greedy decoding — all locally, with no cloud or API dependencies.

Model homepage:
[https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

---

## Features

* ✔️ Local transcription (no internet or API keys)
* ✔️ CPU-only operation
* ✔️ WAV audio → TXT and SRT subtitle output
* ✔️ Chunked processing for long audio files
* ✔️ High transcription accuracy (≈90%+ on clean speech)
* ✔️ Simple CLI interface

Typical performance on a modern desktop CPU (e.g., Intel i5): **~3× real-time**
(10 minutes of audio transcribes in ~3–4 minutes).

---

## Requirements

* Conda (Miniconda or Anaconda)
* Python **3.10**
* ffmpeg (required by PyDub)

---

## Setup (Recommended)

This project uses a **Conda environment file** to ensure consistent, cross-platform installs.

### 1️⃣ Create the environment

From the repository root:

```bash
conda env create -f environment.yml
```

### 2️⃣ Activate the environment

```bash
conda activate localtranscription
```

All Python dependencies (PyTorch, NeMo, etc.) are installed automatically.

---

## Model Download (Required)

This repository does **not** include the Parakeet model due to size limitations.

1. Download the model from Hugging Face:
   👉 [https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

2. Download the file:

   ```
   parakeet-tdt-0.6b-v3.nemo
   ```

3. Place the `.nemo` file in the repository root directory.

By default, `transcribe_nemo.py` looks for the model in the current working directory.
A custom path may be provided with `--model`.

---

## Usage

Basic transcription:

```bash
python transcribe_nemo.py path/to/audio.wav
```

### Optional Flags

```
--model MODEL_PATH        Path to the .nemo model file
--chunk-seconds N         Chunk length in seconds (default: 20)
--threads N               Number of CPU threads (default: 4)
--debug                   Keep intermediate WAV chunks
--textformat FORMAT       txt, srt, or both (default: srt)
--verbose                 Show detailed logs
```

Example:

```bash
python transcribe_nemo.py audio.wav \
  --model parakeet-tdt-0.6b-v3.nemo \
  --threads 6 \
  --textformat both
```

---

## Output

For an input file named `meeting.wav`:

* `meeting_transcript.txt` — Plain text transcription
* `meeting_transcript.srt` — Subtitle file with timestamps

Files are written to the current working directory.

---

## How It Works

The transcription pipeline follows these steps:

1. **Load the NeMo ASR model** from a local `.nemo` file
2. **Normalize audio** to mono, 16 kHz WAV
3. **Split long audio** into fixed-duration chunks
4. **Transcribe each chunk** using greedy decoding
5. **Aggregate results** and write TXT and/or SRT output

All processing is performed locally on CPU.

---

## Performance & Accuracy

On typical desktop CPUs (e.g., Intel i5 class):

* **Speed:** ~3× real-time
* **Accuracy:** >90% on clean, conversational speech

Results vary depending on audio quality, speaker clarity, and background noise.

---

## License & Model Terms

The transcription **infrastructure** in this repository is covered under its own license.
The Parakeet TDT model is distributed under the terms specified on its Hugging Face page.