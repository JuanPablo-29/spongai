import io
import os
import soundfile as sf
from kokoro import KPipeline
import torch
from rvc_python.infer import RVCInference # Pipeline wrapper

class SpongeTTS:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline = KPipeline(lang_code = 'a', device = self.device)
        
        # Initialize the local RVC model with pre-made weights
        self.rvc = RVCInference(device=self.device)
        self.rvc.load_model("weights/model.pth") 

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

if __name__ == "__main__":
    try:
        print("Initializing SpongeTTS pipeline...")
        tts = SpongeTTS()
        
        test_phrase = "I'm ready! I'm ready! Order up! One perfect Krabby Patty coming right up!"
        print(f"\nProcessing test phrase: \"{test_phrase}\"")
        
        # Generate the audio stream buffer
        audio_buffer = tts.get_speech(test_phrase)
        
        # Save the buffer stream contents locally to a permanent WAV file
        output_filename = "test_spongebob.wav"
        with open(output_filename, "wb") as f:
            f.write(audio_buffer.getbuffer())
            
        print(f"\n🎉 Success! Open your directory and play: {output_filename}")
        
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        
    finally:
        # Clean up temporary disk files used by the inference engine loop
        for temp_file in ["temp_base.wav", "temp_spongebob.wav"]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass