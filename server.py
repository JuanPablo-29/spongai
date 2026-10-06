import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from speech_to_text import STT
from SpongeTTS import SpongeTTS
from query import Query

app = FastAPI()
stt = STT()
tts = SpongeTTS()
llm = Query()

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

@app.post("/speak")
async def speak(text: str = Form(...)):
    audio = tts.get_speech(text)
    return StreamingResponse(audio, media_type="audio/wav")

@app.post("/chat")
async def ask(text: str = Form(...)):
    response = llm.main(text)
    return {"response": response}