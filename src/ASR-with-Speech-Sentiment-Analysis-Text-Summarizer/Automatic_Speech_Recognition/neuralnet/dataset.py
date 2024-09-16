import pytorch_lightning as pl
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils import TextTransform     # Comment this for ASR engine inference

class LogMelSpec(nn.Module):
    def __init__(self, sample_rate=16000,
                hop_length=160, n_mels=80):
        super(LogMelSpec, self).__init__()
        self.transform = transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_mels=n_mels,
            hop_length=hop_length
        )

    def forward(self, x):
        x = self.transform(x)
        x = torch.log(x + 1e-14)  # logarithmic, add small value to avoid inf
        return x


def get_featurizer(sample_rate, n_feats=80, hop_length=160):
    return LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, hop_length=hop_length)


class CustomAudioDataset(Dataset):
    def __init__(self, dataset, transform=None, log_ex=True, valid=False):
        self.dataset = dataset
        self.text_process = TextTransform()  # Initialize TextProcess for text processing
        self.log_ex = log_ex

        if valid:
            self.audio_transforms = nn.Sequential(LogMelSpec())
        else:
            time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=15, p=0.05) for _ in range(10)]
            self.audio_transforms = nn.Sequential(
                LogMelSpec(),
                transforms.FrequencyMasking(freq_mask_param=15),
                *time_masks,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            waveform, sample_rate, utterance, _, _, _ = self.dataset[idx]
            utterance = utterance.lower()
            label = self.text_process.text_to_int(utterance)

            # Apply audio transformations
            spectrogram = self.audio_transforms(waveform)  # (channel, feature, time)

            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)

            # Check if spectrogram or label length is valid
            if spec_len < label_len or spectrogram.shape[0] > 1 or label_len == 0:
                raise ValueError('Invalid spectrogram or label length.')

            return spectrogram, label, spec_len, label_len

        except FileNotFoundError as fnf_error:
            if self.log_ex:
                pass

            # Skip the file and move to the next available sample
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)

        except Exception as e:
            # Handle any other exceptions and retry with neighboring samples
            if self.log_ex:
                print(str(e))
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)

class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_url, test_url, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.train_url = train_url
        self.test_url = test_url
        self.num_workers = num_workers
        self.text_process = TextTransform() 

    def setup(self, stage=None):
        # Load multiple training and test URLs
        train_dataset = [torchaudio.datasets.LIBRISPEECH("./data", url=url, download=True) for url in self.train_url]
        test_dataset = [torchaudio.datasets.LIBRISPEECH("./data", url=url, download=True) for url in self.test_url]

        # Concatenate multiple datasets into one
        combined_train_dataset = torch.utils.data.ConcatDataset(train_dataset)
        combined_test_dataset = torch.utils.data.ConcatDataset(test_dataset)

        self.train_dataset = CustomAudioDataset(combined_train_dataset, valid=False)
        self.test_dataset = CustomAudioDataset(combined_test_dataset, valid=True)

    def data_processing(self, data):
        spectrograms, labels, references, input_lengths, label_lengths = [], [], [], [], []
        for (spectrogram, label, input_length, label_length) in data:
            if spectrogram is None:
                continue
            spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
            labels.append(torch.Tensor(label))
            input_lengths.append(((spectrogram.shape[-1] - 1) // 2 - 1) // 2)
            label_lengths.append(label_length)
            references.append(self.text_process.int_to_text(label))  # Convert label back to text

        # Pad the spectrograms to have the same width (time dimension)
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        # Convert input_lengths and label_lengths to tensors
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)

        mask = torch.ones(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[1])
        for i, l in enumerate(input_lengths):
            mask[i, :, :l] = 0

        return spectrograms, labels, input_lengths, label_lengths, references, mask.bool()
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.data_processing,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=self.data_processing,
                          num_workers=self.num_workers,
                          pin_memory=True)
