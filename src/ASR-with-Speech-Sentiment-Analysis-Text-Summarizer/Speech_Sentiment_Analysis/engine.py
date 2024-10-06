import numpy as np
import joblib
import argparse
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

# Initialize scaler
scaler = StandardScaler()

def process_audio(audio_file_path, sample_rate=48000):
    global scaler
    chunked_spec = []

    # Load audio file
    audio, _ = librosa.load(audio_file_path, sr=sample_rate, duration=3)
    signal = np.zeros((int(sample_rate * 3),))
    signal[:len(audio)] = audio
    mel_spectrogram = getMELspectrogram(signal, sample_rate)
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
        predicted_emotion = EMOTIONS[predictions.item()]

        print(f"Predicted Emotion: {predicted_emotion}")
        return predicted_emotion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demoing the speech sentiment recognition engine.")
    parser.add_argument('--model_file', type=str, default="model/speech_sentiment.pt", required=True, help='Path to SER model file.')
    parser.add_argument('--scaler_file', type=str, default="model/scaler.pkl", required=True, help='Path to SER model file.')
    parser.add_argument('--file_path', type=str, default="fear.wav", help='Input File')

    args = parser.parse_args()

    model = HybridModel(len(EMOTIONS))
    model.load_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))
    scaler = joblib.load('model/scaler.pkl')
    process_audio(audio_file_path=args.file_path, sample_rate=48000)

