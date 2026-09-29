import io
import soundfile as sf
from kokoro import KPipeline
import torch
from rvc_python.infer import RVCInference # Pipeline wrapper

class PerfectSpongeBobTTS:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline = KPipeline(lang_code = 'a', device = self.device)
        
        # Initialize the local RVC model with pre-made weights
        self.rvc = RVCInference(device=self.device)
        self.rvc.load_model("model.pth") 

    def get_speech(self, text: str) -> io.BytesIO:
        # 1. Generate clean, dry baseline speech using a default voice
        generator = self.pipeline(text, voice = "am_adam", speed = 1.0)
        audio_segments = [audio for _, _, audio in generator]
        base_audio = torch.cat(audio_segments).numpy()
        
        # Save temporary baseline file
        sf.write("temp_base.wav", base_audio, 24000)
        
        # 2. Instantly convert the baseline audio into SpongeBob locally
        # pitch_shift +12 raises a male baseline voice up an octave to sound like him
        self.rvc.infer_file("temp_base.wav", "temp_spongebob.wav", pitch_shift=12)
        
        # 3. Read it back into your buffer structure
        with open("temp_spongebob.wav", "rb") as f:
            buffer = io.BytesIO(f.read())
            
        return buffer
