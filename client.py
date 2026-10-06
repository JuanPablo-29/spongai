import sounddevice as sd
import io
import soundfile as sf
import numpy as np
import requests

SERVER_URL = "http://100.93.202.120:8000"

def record(samplerate=16000, chunk_duration=0.1, silence_threshold=500, silence_duration=1.5, max_duration=15):
    silence_chunks_needed = int(silence_duration / chunk_duration)
    max_chunks = int(max_duration/chunk_duration)
    frames = []
    silent_count = 0
    for _ in range(max_chunks):
        chunk = sd.rec(int(chunk_duration*samplerate),samplerate=samplerate, channels=1, dtype="int16")
        sd.wait()
        frames.append(chunk)

        volume = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

        print(volume)

        if volume < silence_threshold:
            silent_count += 1
        else:
            silent_count = 0

        if silent_count >= silence_chunks_needed:
            break
    audio = np.concatenate(frames)
    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate, format="WAV")
    buffer.seek(0)
    return buffer

def transcribe(buffer):
    files = {"audio": ("recording.wav", buffer, "audio/wav")}
    response = requests.post(f"{SERVER_URL}/transcribe", files = files)
    response.raise_for_status()
    return response.json()["text"]

def chat(text):
    data = {"text": text}
    response = requests.post(f"{SERVER_URL}/chat", data=data)
    response.raise_for_status()
    return response.json()["response"]

def speak(text):
    data = {"text": text}
    response = requests.post(f"{SERVER_URL}/speak", data=data)
    response.raise_for_status()
    return response.content


def play_audio(audio_bytes):
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    sd.play(data, samplerate)
    sd.wait

if __name__ == "__main__":
    buffer = record()
    text = transcribe(buffer)
    print(text)
    reply = chat(text)
    print(reply)
    audio_bytes = speak(reply)
    play_audio(audio_bytes)
    