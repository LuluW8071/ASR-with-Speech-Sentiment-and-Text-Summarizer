import webbrowser
import sys
import argparse

from os.path import join, dirname
from flask import Flask, render_template, jsonify

# ASR engine path
sys.path.append(join(dirname(__file__), 'Automatic_Speech_Recognition'))
from engine import SpeechRecognitionEngine

global asr_engine
app = Flask(__name__,static_folder='static', static_url_path='/static')


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/start_asr")
def start():
    action = DemoAction()
    asr_engine.run(action)
    return jsonify("speechrecognition start success!")


@app.route("/get_audio")
def get_audio():
    with open('transcript.txt', 'r') as f:
        transcript = f.read()
    return jsonify(transcript)


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
            f.write(transcript)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of Automatic Speech Recognition")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the model file")
    parser.add_argument("--kenlm_file", type=str, default=None, help="Path to the KenLM file")
    args = parser.parse_args()

    asr_engine = SpeechRecognitionEngine(args.model_file, args.kenlm_file)
    webbrowser.open_new('http://127.0.0.1:3000/')
    app.run(port=3000)
    