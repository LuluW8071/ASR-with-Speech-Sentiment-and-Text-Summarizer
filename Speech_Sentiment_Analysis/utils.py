""" Function Script for preprocessing audio data and extract features """

import librosa
import numpy as np


# Zero Crossing Rate
# Reference: https://librosa.org/doc/latest/generated/librosa.feature.zero_crossing_rate.html#librosa.feature.zero_crossing_rate
def zcr(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

# RMS Energy
# Reference: https://librosa.org/doc/latest/generated/librosa.feature.rms.html#librosa.feature.rms
def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

# MFCC
# Reference: https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)

# Feature Extraction of ZCR, RMS, MFCC
def extract_features(data, sr, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                                    ))
    return result


""" Original """
def get_features(path):
    data, sampling_rate = librosa.load(path, duration=2.5, offset=0.6)
    # print('Data and sampling rate:', data.shape, sampling_rate)
    result = extract_features(data, sampling_rate)
    # print(result.shape)
    return result
