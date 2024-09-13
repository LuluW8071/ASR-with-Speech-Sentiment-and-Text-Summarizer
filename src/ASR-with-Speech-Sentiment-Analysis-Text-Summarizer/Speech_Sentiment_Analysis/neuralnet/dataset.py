import pytorch_lightning as pl
import pandas as pd
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as transforms

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


# Custom Mel Spectrogram Transform Class
class LogMelSpec(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, hop_length=380, n_fft=1024):
        super(LogMelSpec, self).__init__()
        self.transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels,
                                                   hop_length=hop_length, n_fft=n_fft)

    def forward(self, x):
        x = self.transform(x)     # mel spectrogram
        x = torch.log(x + 1e-14)  # logarithmic, add small value to avoid inf
        return x


# Custom Dataset Class
class CustomAudioDataset(Dataset):
    def __init__(self, file_path, num_samples, target_length, valid=False):
        self.file_path = file_path
        self.num_samples = num_samples
        self.target_length = target_length
        print(f'Loading csv data from {file_path}')
        self.data = pd.read_csv(self.file_path)

        # Map emotions to numerical labels
        self.class_labels = pd.Categorical(self.data['Emotions']).codes
        self.valid = valid

        if valid:
            self.audio_transforms = nn.Sequential(
                LogMelSpec()
            )
        else:
            self.audio_transforms = nn.Sequential(
                LogMelSpec(),
                transforms.FrequencyMasking(freq_mask_param=30),
                transforms.TimeMasking(time_mask_param=50)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        file_path = item['Path']

        try:
            waveform, _ = torchaudio.load(file_path)

            class_label = self.class_labels[idx]
            label = torch.tensor(class_label).long()

            spectrogram = self.audio_transforms(waveform)           # (channel, feature, time)
            spectrogram = self._mix_down_if_necessary(spectrogram)  # make mono channel if dual channel is present 
            spectrogram = self._right_pad_spectrogram(spectrogram)  # right padd if necessary

            if spectrogram.shape[0] > 1:
                raise Exception('\ndual channel, skipping audio file %s' % file_path)
            if spectrogram.shape[2] > self.target_length:
                spectrogram = spectrogram[:, :, :self.target_length]

            return spectrogram, label

        # Returning the previous sample if an exception occurs
        except Exception as e:
            print(str(e), file_path)
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)
    
    def _right_pad_spectrogram(self, spectrogram):
        length = spectrogram.shape[2]
        if length < self.target_length:
            num_missing_samples = self.target_length - length
            last_dim_padding = (0, num_missing_samples)
            spectrogram = F.pad(spectrogram, last_dim_padding)
        return spectrogram

    # Make it mono-channel if clip is in dual-channel
    def _mix_down_if_necessary(self, spectrogram):
        if spectrogram.shape[0] > 1:
            spectrogram = torch.mean(spectrogram, dim=0, keepdim=True)
        return spectrogram
    
    def describe(self):
        return self.data.describe()


# Lightning Data Module
class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_csv, test_csv, num_workers, num_samples=16000, target_length=250):
        super().__init__()
        self.batch_size = batch_size
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.target_length = target_length

    def setup(self, stage=None):
        self.train_dataset = CustomAudioDataset(self.train_csv, num_samples=self.num_samples, target_length=self.target_length, valid=False)
        self.test_dataset = CustomAudioDataset(self.test_csv, num_samples=self.num_samples, target_length=self.target_length, valid=True)

    def data_processing(self, data):
        spectrograms = []
        labels = []
        for (spectrogram, label) in data:
            if spectrogram is None:
                continue

            spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
            labels.append(label.clone().detach().long())

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = torch.stack(labels)

        return spectrograms, labels

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=lambda x: self.data_processing(x),
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=lambda x: self.data_processing(x),
                          num_workers=self.num_workers,
                          pin_memory=True)
