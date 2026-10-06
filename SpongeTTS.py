import torch
import fairseq
# Tell PyTorch to trust the fairseq dictionary class when loading voice models
torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])

import io
import os
import re
import soundfile as sf
from kokoro import KPipeline
from rvc_python.infer import RVCInference 

class SpongeTTS:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Initialize local Kokoro baseline generator
        self.pipeline = KPipeline(lang_code='a', device=self.device)
        
        # 2. Initialize and configure the local RVC engine
        self.rvc = RVCInference(device=self.device)
        self.rvc.load_model("weights/model.pth") 
        
        # --- CARTOON CHARACTER OPTIMIZATIONS ---
        self.rvc.f0_up_key = 13            # Shifts pitch up slightly higher than an octave (+13 semitones)
        self.rvc.f0_method = "rmvpe"       # Premium pitch tracking to follow expressive speech cleanly
        self.rvc.file_index = "weights/model.index" # Blends in SpongeBob's exact accent maps
        self.rvc.index_rate = 0.65         # Balance weight (0.6 - 0.7 gives optimal inflection variance)

    def _spongebobify_text(self, text: str) -> str:
        """
        Helper that forces the local text-to-speech engine out of 'monotone reading mode'
        by transforming punctuation, adding exclamation points, and emphasizing text.
        """
        # Clean basic text formatting
        processed = text.strip()
        
        # Turn periods into exclamation points to force enthusiastic speech inflection
        processed = processed.replace(".", "!")
        
        # Make the words UPPERCASE so the underlying TTS voice engine shouts with energy
        processed = processed.upper()
        
        # Add signature catchphrases if the sentence looks standard
        if not processed.endswith("!"):
            processed += "!"
            
        return processed

    def get_speech(self, text: str) -> io.BytesIO:
        # Step A: Transform the incoming text to make the baseline speaker sound highly energetic
        animated_text = self._spongebobify_text(text)
        print(f"Feeding animated text to TTS engine: \"{animated_text}\"")
        
        # Step B: Generate the baseline voice with a speed booster (1.15) to match his fast cadence
        generator = self.pipeline(animated_text, voice="am_adam", speed=1.15)
        audio_segments = [audio for _, _, audio in generator]
        base_audio = torch.cat(audio_segments).numpy()
        
        # Save temporary baseline file
        sf.write("temp_base.wav", base_audio, 24000)
        
        # Step C: Locally convert the baseline audio into SpongeBob using the active index file
        self.rvc.infer_file("temp_base.wav", "temp_spongebob.wav")
        
        # Step D: Read it back into your server-compatible stream buffer structure
        with open("temp_spongebob.wav", "rb") as f:
            buffer = io.BytesIO(f.read())
            
        return buffer

# --- TEST ROUTINE BLOCK ---
if __name__ == "__main__":
    try:
        print("Initializing Optimized SpongeTTS pipeline...")
        tts = SpongeTTS()
        
        # Test input string - notice it's standard, but the internal formatter will hype it up!
        test_phrase = input("Spongebob: ")
        print(f"\nIncoming text stream: \"{test_phrase}\"")
        
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
