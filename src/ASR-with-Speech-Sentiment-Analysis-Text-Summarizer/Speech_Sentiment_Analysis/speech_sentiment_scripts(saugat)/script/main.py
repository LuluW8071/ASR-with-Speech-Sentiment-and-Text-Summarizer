import librosa
import numpy as np
import torch
from IPython.display import Audio, display
import os
import joblib
from architecture import HybridModel  # Import your model architecture
from chunk_and_spectogram import getMELspectrogram, splitIntoChunks  # Import the functions

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

# Load your trained model
LOAD_PATH = os.path.join(os.getcwd(), 'models')
model = HybridModel(len(EMOTIONS))

# UNCOMMENT THE CODE LINE 1 TO 2 AND COMMENT THE CODE BELOW LINE 3 TO 4 IF YOU PLAN ON RUNNING THE MODEL ON GPU

# 1
# Load model weights and move to the appropriate device
# model.load_state_dict(torch.load(os.path.join(LOAD_PATH, '/content/speech_sentiment_asr.pt')))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)  # Move the model to the GPU or keep it on CPU
# print('Model is loaded from {}'.format(os.path.join(LOAD_PATH, 'speech_sentiment_asr.pt')))
# 2

# 3
# Load model weights and move to the appropriate device (CPU version)
model.load_state_dict(torch.load(os.path.join(LOAD_PATH, '/content/speech_sentiment_asr.pt'), map_location=torch.device('cpu')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the GPU or keep it on CPU
print('Model is loaded from {}'.format(os.path.join(LOAD_PATH, 'speech_sentiment_asr.pt')))
# 4

SAMPLE_RATE = 48000  
DURATION = 3  
NUM_MEL_BINS = 128  

# Load your fitted scaler
scaler = joblib.load('/content/scaler.pkl')

def process_audio(audio_file_path):
    """
    Process the audio file, convert to MEL spectrogram, split into chunks, scale and make predictions.
    """
    # Load audio file
    audio, sample_rate = librosa.load(audio_file_path, sr=SAMPLE_RATE)

    # Ensure the audio length is the desired target length
    target_length = SAMPLE_RATE * DURATION
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)), 'constant')

    # Compute MEL spectrogram
    mel_spectrogram = getMELspectrogram(audio, SAMPLE_RATE)
    print(f"Mel Spectrogram Shape: {mel_spectrogram.shape}")

    # Split into chunks
    chunks = splitIntoChunks(mel_spectrogram, win_size=128, stride=64)
    print(f"Chunks Shape Before Scaling: {chunks.shape}")

    # Pad or truncate to 7 chunks
    num_chunks = chunks.shape[0]
    print(f"Number of Chunks: {num_chunks}")
    if num_chunks < 7:
        padding = np.zeros((7 - num_chunks, 128, 128))
        chunks = np.concatenate((chunks, padding), axis=0)
    elif num_chunks > 7:
        chunks = chunks[:7]

    # Prepare chunks for model input
    chunks = chunks[np.newaxis, :]  # Add batch dimension
    chunks = np.expand_dims(chunks, axis=1)  # Add channel dimension (for CNN)
    chunks_reshaped = chunks.reshape(1, 7, 1, 128, 128)
    print(f"Chunks Shape After Reshaping: {chunks_reshaped.shape}")

    # Scale the chunks
    chunks_scaled = scaler.transform(chunks_reshaped.reshape(1, -1))
    chunks_scaled = chunks_scaled.reshape(1, 7, 1, 128, 128)
    print(f"Chunks Shape After Scaling: {chunks_scaled.shape}")

    # Convert to tensor for model input
    chunks_tensor = torch.tensor(chunks_scaled, device=device).float()

    # Make predictions with the model
    with torch.no_grad():
        model.eval()
        _, output_softmax, _ = model(chunks_tensor)
        predictions = torch.argmax(output_softmax, dim=1)
        predicted_emotion = EMOTIONS[predictions.item()]

    # Display the audio
    display(Audio(audio_file_path))

    # Print the predicted emotion
    print(f"Predicted Emotion: {predicted_emotion}")

    return predicted_emotion

# Take input audio file from user
file_path = input("Enter the path to your .wav file: ")

# Process the audio and predict emotion
process_audio(file_path)
