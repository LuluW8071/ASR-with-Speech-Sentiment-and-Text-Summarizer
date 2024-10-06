import numpy as np
import joblib
import librosa
import torch

from sklearn.preprocessing import StandardScaler
from neuralnet.model import HybridModel
from feature import getMELspectrogram, splitIntoChunks 


EMOTIONS = {
    1: 'neutral', 
    2: 'calm', 
    3: 'happy', 
    4: 'sad', 
    5: 'angry', 
    6: 'fear', 
    7: 'disgust', 
    0: 'surprise'
}

scaler = StandardScaler()
model = HybridModel(len(EMOTIONS))
model.load_state_dict(torch.load("model/speech_sentiment.pt", map_location=torch.device('cpu')))
SAMPLE_RATE = 48000
scaler = joblib.load('model/scaler.pkl')

def process_audio(audio_file_path):
    global scaler
    chunked_spec = []

    # Load audio file
    audio, sample_rate = librosa.load(audio_file_path, sr=SAMPLE_RATE, duration=3)
    signal = np.zeros((int(SAMPLE_RATE * 3),))
    signal[:len(audio)] = audio
    mel_spectrogram = getMELspectrogram(signal, SAMPLE_RATE)
    chunks = splitIntoChunks(mel_spectrogram, win_size=128, stride=64)

    chunked_spec.append(chunks)
    chunks = np.stack(chunked_spec, axis=0)
    chunks = np.expand_dims(chunks, axis=2)

    # Reshape the chunks
    chunks = np.reshape(chunks, newshape=(1, -1)) 
    chunks_scaled = scaler.transform(chunks)
    chunks_scaled = np.reshape(chunks_scaled, newshape=(1, 7, 1, 128, 128))  

    # Convert to tensor for model input
    chunks_tensor = torch.tensor(chunks_scaled).float()

    # Model inference
    with torch.inference_mode():
        model.eval()
        _, output_softmax, _ = model(chunks_tensor)
        predictions = torch.argmax(output_softmax, dim=1)
        print(predictions)
        predicted_emotion = EMOTIONS[predictions.item()]

        print(f"Predicted Emotion: {predicted_emotion}")
        return predicted_emotion

file_path = "fear.wav"
process_audio(file_path)

