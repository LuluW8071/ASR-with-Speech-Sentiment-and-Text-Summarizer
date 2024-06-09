import pytorch_lightning as pl
# import pandas as pd
import json
import torchaudio
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from utils import TextTransform 

# Custom Dataset Class
class CustomAudioDataset(Dataset):
    def __init__(self, json_path, transform=None, log_ex=True):
        print(f'Loading json data from {json_path}')
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        # print(self.data)
        self.text_process = TextTransform()
        self.log_ex = log_ex
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # print(item)
        
        try:
            waveform, _ = torchaudio.load(item['key'])    # Point to location of audio data 
            utterance = item['text'].lower()              # Point to sentence of audio data
            # print(waveform, sample_rate) 

            label = self.text_process.text_to_int(utterance)
            spectrogram = self.audio_transforms(waveform) # (channel, feature, time)

            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)

            if spec_len < label_len:
                    raise Exception('spectrogram len is bigger then label len')
            if spectrogram.shape[0] > 1:
                raise Exception('dual channel, skipping audio file %s'%item['key'])
            if spectrogram.shape[2] > 16000:
                raise Exception('spectrogram to big. size %s'%spectrogram.shape[2])
            if label_len == 0:
                raise Exception('label len is zero... skipping %s'%item['key'])
            return spectrogram, label, spec_len, label_len

        except Exception as e:
            if self.log_ex:
                print(str(e), item['key'])
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)  
        # print(label)
        

             
        # return waveform, sample_rate, utterance


# NOTE: Define train_audio_transforms and valid_audio_transforms
train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
    
""" Initialize TextProcess for text processing """
text_transform = TextTransform()


# Lightning Data Module
class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_json, test_json, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.train_json = train_json
        self.test_json = test_json
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = CustomAudioDataset(self.train_json, transform=train_audio_transforms)
        self.test_dataset = CustomAudioDataset(self.test_json, transform=valid_audio_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          collate_fn=lambda x: self.data_processing(x, 'train'), num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
                          collate_fn=lambda x: self.data_processing(x, 'valid'), num_workers=self.num_workers)

    def data_processing(self, data, data_type="train"):
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []

        for (waveform, utterance, input_length, label_length) in data:
            if data_type == 'train':
                spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            elif data_type == 'valid':
                spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            else:
                raise Exception('data_type should be train or valid')
            
            spectrograms.append(spec)
            label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
            labels.append(label)
            input_lengths.append(input_length)
            label_lengths.append(label_length)
        
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
        
        return spectrograms, labels, input_lengths, label_lengths


if __name__ == "__main__":
    # Define parameters
    batch_size = 64
    train_json = 'scripts/converted_dataset/test.json'
    test_json = 'scripts/converted_dataset/test.json'
    num_workers = 1

    # Create data module instance
    data_module = SpeechDataModule(batch_size, train_json, test_json, num_workers)

    # Set up data module (downloads data if necessary and prepares datasets)
    data_module.setup()

    # Load the data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Example of iterating through the training data loader
    for batch in train_loader:
        spectrograms, labels, input_lengths, label_lengths = batch
        print("Spectrograms shape:", spectrograms.shape)
        print("Labels shape:", labels.shape)
        print("-"*20)
        break  # Remove this line to iterate through the entire data loader

