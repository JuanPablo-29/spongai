import io
import torch
import soundfile as sf
from kokoro import KPipeline

class TTS:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            torch.backends.cudnn.enabled = False
        self.pipeline = KPipeline(lang_code = 'a', device = self.device)
        v1 = self.pipeline.load_voice('af_heart')
        v2 = self.pipeline.load_voice('am_adam')
        spongebob_blend = (v1 * 0.6) + (v2 * 0.4)
        self.pipeline.voices['spongebob_style'] = spongebob_blend

    def get_speech(self, text: str, speed: float = 1.0) -> io.BytesIO:
        generator = self.pipeline(text, voice = "spongebob_style", speed = speed, split_pattern = r'\n+')
        audio_segments = []
        for _, _, audio in generator:
            audio_segments.append(audio)
        final_audio = torch.cat(audio_segments).numpy()
        buffer = io.BytesIO()
        sf.write(buffer, final_audio, 24000, format='WAV')
        buffer.seek(0)
        return buffer

if __name__ == "__main__":
    tts = TTS()
    audio = tts.get_speech("Poop")
    print("Success!")