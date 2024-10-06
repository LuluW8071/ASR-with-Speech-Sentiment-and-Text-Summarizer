import librosa
import numpy as np

def getMELspectrogram(audio, sample_rate):
    """
    Compute the MEL spectrogram of an audio signal.

    Parameters:
    - audio: Audio time series
    - sample_rate: Sampling rate of the audio

    Returns:
    - mel_spec_db: MEL spectrogram in decibel scale
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sample_rate, 
        n_fft=1024, 
        win_length=512, 
        window='hamming', 
        hop_length=256, 
        n_mels=128, 
        fmax=sample_rate / 2
    )
    
    # Convert power spectrogram to decibel scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def splitIntoChunks(mel_spec, win_size, stride):
    """
    Split the MEL spectrogram into chunks.

    Parameters:
    - mel_spec: MEL spectrogram
    - win_size: Window size for each chunk
    - stride: Step size to move the window (overlap control)

    Returns:
    - A stack of chunks along the time axis
    """
    t = mel_spec.shape[1]
    
    # Calculate number of chunks based on stride
    num_of_chunks = int(t / stride)

    chunks = []

    # Create chunks from the spectrogram
    for i in range(num_of_chunks):
        chunk = mel_spec[:, i * stride:i * stride + win_size]
        
        # Only append chunks of the correct size
        if chunk.shape[1] == win_size:
            chunks.append(chunk)

    return np.stack(chunks, axis=0)
