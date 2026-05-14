import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16
)

wavs, sr = model.generate_voice_design(
    text="喂，你食咗飯未呀？今晚去邊度打邊爐？", # Use Traditional Chinese
    language="Chinese",
    instruct="A female voice speaking in fluent Cantonese with a natural Hong Kong accent."
)
sf.write("cantonese_output.wav", wavs[0], sr)