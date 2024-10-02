import webbrowser
import sys
import argparse
from os.path import join, dirname
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ASR engine path
sys.path.append(join(dirname(__file__), 'Automatic_Speech_Recognition'))
from engine import SpeechRecognitionEngine

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Global variable for ASR engine
global asr_engine

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/start_asr")
async def start():
    action = DemoAction()
    asr_engine.run(action)
    return {"message": "Speech recognition started successfully!"}

@app.get("/get_audio")
async def get_audio():
    with open('transcript.txt', 'r') as f:
        transcript = f.read()
    return {"transcript": transcript}

class DemoAction:
    def __init__(self):
        self.asr_results = ""
        self.current_beam = ""
    
    def __call__(self, x):
        results, current_context_length = x
        self.current_beam = results
        transcript = " ".join(self.asr_results.split() + results.split())
        self.save_transcript(transcript)
        if current_context_length > 10:
            self.asr_results = transcript

    def save_transcript(self, transcript):
        with open("transcript.txt", 'w+') as f:
            # print(transcript)
            f.write(transcript)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of Automatic Speech Recognition")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the model file")
    parser.add_argument("--kenlm_file", type=str, default=None, help="Path to the KenLM file")
    args = parser.parse_args()

    asr_engine = SpeechRecognitionEngine(args.model_file, args.kenlm_file)
    webbrowser.open_new('http://127.0.0.1:3000/')
    
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)