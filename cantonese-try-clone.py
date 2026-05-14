import torch
import torchaudio
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 1. Initialize Model (Base version is required for cloning)
model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
model = Qwen3TTSModel.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)

# 2. Define your inputs
# For Cantonese, use Traditional Chinese characters
target_text = "你好，我而家正係度測試緊廣東話語音複製功能，聽落去係咪好似呢？"
ref_audio_path = "canto-sample.wav"
ref_transcript = "即係譬如就算你播歌喇。我唔知啊，首先佢係咪想針對一樣嘢呢？就係哩陣時，警察呢，就喺哩個移交儀式嗰時候呢，就攪咗一啲古典音樂。好大聲嘅。噉佢係咪" # What is said in the wav

# 3. Perform Cloning (use correct API and parameter names)
# The model 'continues' the style of the prompt
wavs, sr = model.generate_voice_clone(
    text=target_text,
    ref_audio=ref_audio_path,
    ref_text=ref_transcript,
    language="chinese"  # Standard Chinese tag, model infers Cantonese from audio/text
)

# 4. Save Output (handle numpy arrays and torch tensors)
out_wav = wavs[0]
if hasattr(out_wav, "cpu"):
    out_wav = out_wav.cpu().numpy()
sf.write("cloned_cantonese.wav", out_wav, sr)