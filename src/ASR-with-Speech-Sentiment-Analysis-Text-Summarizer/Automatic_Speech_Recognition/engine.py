import pyaudio
import threading
import time
import argparse
import wave
import torchaudio
import torch
import sys
import numpy as np
from neuralnet.dataset import get_featurizer
from decoder import DecodeGreedy, CTCBeamDecoder
from threading import Event


class Listener:
    def __init__(self, sample_rate=16000, record_seconds=2):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk)

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nSpeech Recognition engine is now listening... \n")


class SpeechRecognitionEngine:
    def __init__(self, model_file, ken_lm_file=None, context_length=10):
        self.listener = Listener(sample_rate=16000)
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')
        self.featurizer = get_featurizer(16000)
        self.audio_q = list()
        self.out_args = None
        self.beam_search = CTCBeamDecoder(beam_size=100, kenlm_path=ken_lm_file)
        self.context_length = context_length * 50

    def save(self, waveforms, fname="audio_temp"):
        wf = wave.open(fname, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b"".join(waveforms))
        wf.close()
        return fname

    def predict(self, audio):
        with torch.inference_mode():
            fname = self.save(audio)
            waveform, _ = torchaudio.load(fname)
            log_mel = self.featurizer(waveform).transpose(1, 2)
            # print("Log Mel Shape", log_mel.shape)
            encoder_out = self.model.encoder(log_mel)
            # print("Encoder Shape", encoder_out.shape)
            decoder_out = self.model.decoder(encoder_out)
            # print("Decoder Shape", decoder_out.shape)

            out = torch.nn.functional.softmax(decoder_out, dim=-1)
            # print("Output Shape", out.shape)

            self.out_args = out if self.out_args is None else torch.cat((self.out_args, out), dim=1)
            # print("Self Args Shape", self.out_args.shape)

            results = self.beam_search(self.out_args)
            current_context_length = self.out_args.shape[1] / 50
            if self.out_args.shape[1] > self.context_length:
                self.out_args = None
            return results, current_context_length

    def inference_loop(self, action):
        while True:
            if len(self.audio_q) < 5:
                continue
            else:
                pred_q = self.audio_q.copy()
                self.audio_q.clear()
                action(self.predict(pred_q))
            time.sleep(0.05)

    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop, args=(action,), daemon=True)
        thread.start()


class DemoAction:
    def __init__(self):
        self.asr_results = ""
        self.current_beam = ""

    def __call__(self, x):
        results, current_context_length = x
        self.current_beam = results
        transcript = " ".join(self.asr_results.split() + results.split())
        print(transcript)
        if current_context_length > 10:
            self.asr_results = transcript


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demoing the speech recognition engine.")
    parser.add_argument('--model_file', type=str, required=True, help='Path to the optimized Conformer model file.')
    parser.add_argument('--kenlm_file', type=str, help='Path to the KenLM file (optional).')

    args = parser.parse_args()

    asr_engine = SpeechRecognitionEngine(args.model_file, args.kenlm_file)
    action = DemoAction()

    asr_engine.run(action)
    threading.Event().wait()
