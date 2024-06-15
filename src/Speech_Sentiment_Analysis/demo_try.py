import torch
import torchaudio
import pyaudio
import numpy as np
from neuralnet.model import neuralnet
from utils import extract_features
import wave 

# Load the trained model
model = neuralnet(input_size=1, output_shape=6)
checkpoint = torch.load("model/sentiment-model-19-0.07.ckpt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

# Set up PyAudio
CHUNK = 2048
RATE = 18500
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Set the path where you want to save the audio file
output_file = "recorded_audio.wav"

# Create PyAudio object and open stream
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

try:
    print("Recording...")
    while True:
        audio_data = np.array([], dtype=np.int16)
        
        # Record audio for 3 seconds
        for _ in range(int(RATE / CHUNK * 3)):
            data = stream.read(CHUNK)
            audio_data = np.append(audio_data, np.frombuffer(data, dtype=np.float32))
            # print('Raw:', np.isnan(audio_data).sum())
            audio_data = np.nan_to_num(audio_data, nan=0.0)
            # print('Fill:', np.isnan(audio_data).sum())
            # print(audio_data)
        print("Recording finished.")



        # Extract features from the recorded audio
        features = extract_features(audio_data, RATE)
        # Save the recorded audio as a WAV file
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_data.tobytes())

        # Load the recorded audio into a tensor
        waveform, sample_rate = torchaudio.load(output_file)
        # Extract features from the loaded audio
        features = extract_features(waveform.numpy().flatten(), sample_rate)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        # print(input_tensor)

        # Perform inference
        with torch.inference_mode():
            output = model(input_tensor)

        # Convert output to probabilities and get predicted class
        probabilities = torch.softmax(output, dim=1)
        labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Print predicted class
        print("Predicted class:", labels[predicted_class])

except KeyboardInterrupt:
    print("Stopped by user.")

# Close the stream and terminate PyAudio
stream.stop_stream()
stream.close()
audio.terminate()