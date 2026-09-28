from faster_whisper import WhisperModel
class STT:
    def __init__(self):
        self.model_size = "large-v3"
        self.model = WhisperModel(self.model_size, device = "cuda", compute_type = "float16")
    def get(self, file: str):
        segments, info = self.model.transcribe(file)
        result = ""
        for segment in segments:
            result += segment.text
        return result

if __name__ == "__main__":
    stt = STT()
    text = stt.get("doom.mp3")
    print(text)