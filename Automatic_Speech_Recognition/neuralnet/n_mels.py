import torchaudio
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt

# Load an example audio file
audio_path = "scripts/converted_dataset/clips/common_voice_en_38096741.wav"
waveform, sample_rate = torchaudio.load(audio_path)

# Define the number of Mel bands
n_mels = 128//2  # You can adjust this value if you need to calculate with different n_mels

# Define the MelSpectrogram transform
mel_spectrogram_transform = transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=n_mels
)

# Apply the transform to the waveform
mel_spectrogram = mel_spectrogram_transform(waveform)

# Inspect the shape of the resulting Mel spectrogram
print(f'Mel Spectrogram Shape: {mel_spectrogram.shape}')

# Plot the Mel spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(mel_spectrogram.log2()[0, :, :].detach().numpy(), cmap='viridis', aspect='auto')
plt.title('Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')

# Save the plot as an image file
output_path = 'mel_spectrogram.png'
plt.savefig(output_path)
print(f'Mel spectrogram saved to {output_path}')
