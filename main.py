import warnings
import torch

# Mute the torchcodec warning
warnings.filterwarnings("ignore")

# Mute the TF32 warning
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import whisperx
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

device = os.getenv("DEVICE", "cuda")
batch_size = int(os.getenv("BATCH_SIZE", "16"))
compute_type = os.getenv("COMPUTE_TYPE", "float16")


def transcribe_audio_segments(audio_file):
    print("1. Loading VAD and Faster-Whisper...")
    model = whisperx.load_model("base", device, compute_type=compute_type)

    print("2. Transcribing...")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language="en")
    print(f"Detected language: {result['language']}")

    print("3. Loading Forced Alignment Model (Wav2Vec2)...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

    print("4. Aligning Words to Audio Waveform...")
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    return result['segments']


audio_file = "your_audio_file.wav"
transcribed_segments = transcribe_audio_segments(audio_file)

# Save to JSON
output_path = "transcription_output.json"
with open(output_path, "w") as f:
    json.dump({"segments": transcribed_segments}, f, indent=4)

print(f"Successfully saved word-level timestamps to {output_path}")

print("\n--- FINAL WORD-LEVEL TIMESTAMPS ---")
for segment in transcribed_segments:
    for word_info in segment["words"]:
        start = word_info.get('start', '?.??')
        end = word_info.get('end', '?.??')
        word = word_info['word']
        print(f"[{start} -> {end}] {word}")