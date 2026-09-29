import sounddevice as sd
import io
import soundfile as sf
import numpy as np


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