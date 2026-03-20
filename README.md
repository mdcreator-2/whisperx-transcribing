# WhisperX Transcribing

A robust audio transcription pipeline using [WhisperX](https://github.com/m-bain/whisperx) for word-level timestamps, speaker diarization (optional), and high-accuracy alignment. This setup is optimized for Windows 11 with NVIDIA RTX 40-series GPUs (e.g., RTX 4050).

## Features
- **High-Speed Transcription**: Powered by Faster-Whisper.
- **Precision Alignment**: Word-level timestamps using Wav2Vec2.
- **GPU Optimized**: Configured for CUDA 12.6 and RTX 4050 performance.
- **Easy Configuration**: Managed via environment variables.

## Prerequisites
- **Python 3.12**: Strictly required (WhisperX is incompatible with 3.13+).
- **FFmpeg**: Must be installed and available in your system PATH.
- **NVIDIA GPU**: Recommended with CUDA 12.x drivers installed.

## Installation

Follow these steps to set up the environment exactly as required for the RTX 4050 / Windows environment.

### 1. Create a Python 3.12 Virtual Environment
```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install WhisperX
Install the base library from GitHub. This will initially install some CPU-only dependencies which we will overwrite in the next step.
```bash
pip install git+https://github.com/m-bain/whisperx.git
```

### 3. Install GPU-Accelerated PyTorch
Force-reinstall the specific PyTorch version compatible with CUDA 12.6 and WhisperX.
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

### 4. Optimize Dependencies
Lock Transformers and HuggingFace Hub to versions known to be stable with WhisperX.
```bash
pip install "transformers>=4.48.0,<5.0.0" "huggingface-hub<1.0.0"
pip install python-dotenv
```

## Configuration

The application uses a `.env` file for configuration. Create a file named `.env` in the root directory:

```env
DEVICE=cuda
BATCH_SIZE=16
COMPUTE_TYPE=float16
```

- `DEVICE`: `cuda` or `cpu`.
- `BATCH_SIZE`: Adjust based on VRAM (16 is a good starting point for RTX 4050).
- `COMPUTE_TYPE`: `float16` for GPU speed, `int8` or `float32` for CPU.

## Usage

1. Place your audio file in the project directory.
2. Update the `audio_file` variable in `main.py` with your filename.
3. Run the script:
   ```bash
   python main.py
   ```

The script will generate a `transcription_output.json` file containing the full transcription with word-level timestamps.

## Output
The final word-level timestamps will be printed to the console and saved to `transcription_output.json`.