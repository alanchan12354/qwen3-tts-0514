import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

try:
    print(f"Attempting to load {MODEL_ID} with Flash Attention 2...")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
except Exception as e:
    print(f"Flash Attention 2 loading failed: {e}")
    print("Falling back to standard attention (Eager/SDPA)...")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16 # Better compatibility for fallback
    )

print("Synthesizing audio...")
wavs, sr = model.generate_custom_voice(
    text="The PowerShell script is now executing. System status: Operational.",
    language="English",
    speaker="Ryan"
)

sf.write("output.wav", wavs[0], sr)
print("Success. Audio saved to 'output.wav'.")
