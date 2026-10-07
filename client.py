import sounddevice as sd
import io
import soundfile as sf
import numpy as np
import requests
import queue

SERVER_URL = "http://100.93.202.120:8000"

def record(samplerate=16000, chunk_duration=0.5, silence_threshold=55, silence_duration=1.5, max_duration=15):
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    silence_chunks_needed = int(silence_duration / chunk_duration)
    max_chunks = int(max_duration / chunk_duration)
    blocksize = int(chunk_duration * samplerate)

    frames = []
    silent_count = 0

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="int16",
                         blocksize=blocksize, callback=callback):
        print("Recording... Speak now.")
        for _ in range(max_chunks):
            chunk = q.get()  # blocks until the stream delivers the next full block
            frames.append(chunk)
            volume = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))  # same RMS line you already have
            print(f"Volume: {volume:.2f}")
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
    return io.BytesIO(response.content)


def play_audio(audio_bytes):
    data, samplerate = sf.read(audio_bytes)
    sd.play(data, samplerate)
    sd.wait()

if __name__ == "__main__":
    while True:
        buffer = record()
        with open("debug_recording.wav", "wb") as f:
            f.write(buffer.read())
        buffer.seek(0)
        text = transcribe(buffer)
        print(text)
        reply = chat(text)
        print(reply)
        audio_bytes = speak(reply)
        play_audio(audio_bytes)
    