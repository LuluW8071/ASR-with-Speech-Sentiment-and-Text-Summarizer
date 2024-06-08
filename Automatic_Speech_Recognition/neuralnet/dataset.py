import pytorch_lightning as pl
import torch
import torchaudio

import pandas as pd 
import numpy as np

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from utils import TextTransform 

# Spectrogram Augmentation
class SpecAugment(nn.Module):
    def __init__(self, rate=22050, freq_mask=30, time_mask=100):
        super(SpecAugment, self).__init__()
        self.rate = rate

        self.train_specaug = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=rate, n_mels=64),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        self.valid_specaug = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=rate, n_mels=64)
        )

    def forward(self, x, train:bool):
        if train:
            print(x)
            return self.train_specaug(x)
        else:
            return self.valid_specaug(x)
        

def data_processing(data, data_type:bool):
    text_transform = TextTransform()        # From utils.py

    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    spec_augment = SpecAugment()
    for (waveform, _, utterance, _, _, _) in data:
        if data_type:
            spec = spec_augment(waveform, train=True).squeeze(0).transpose(0, 1)
            print('train', spec)
        else:
            spec = spec_augment(waveform, train=False).squeeze(0).transpose(0, 1)
            print('val', spec)
        spectrograms.append(spec)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    print(spectrograms, labels)
    return spectrograms, labels, input_lengths, label_lengths


# Loading Datasets
class Data(Dataset):

    def __init__(self, json_path, train=True):
        print(f'Loading json data from {json_path}')
        self.data = pd.read_json(json_path, encoding="utf-8")
        self.train = train
        self.spec_augment = SpecAugment()

        print(self.data.iloc[2])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        waveform = torch.tensor(row['waveform'])
        print(waveform)
        utterance = row['utterance']
        return waveform, utterance, self.train
    

# Collate function
def collate_fn(batch):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    text_transform = TextTransform()
    spec_augment = SpecAugment()
    for waveform, utterance, train in batch:
        spec = spec_augment(waveform, train=train).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

# Example usage
dataset = Data(json_path="scripts/converted_dataset/test.json", train=True)
