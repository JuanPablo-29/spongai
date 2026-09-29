import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from speech_to_text import STT

app = FastAPI()
stt = STT()

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    contents = await audio.read()
    tmp = tempfile.NamedTemporaryFile(delete= False)
    tmp.write(contents)
    tmp.close()
    try:
        text = stt.get(tmp.name)
    finally:
        os.remove(tmp.name)
    return {"text": text}