import pytorch_lightning as pl
import json
import torchaudio
import torch
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
        self.text_process = TextTransform()                 # Initialize TextProcess for text processing
        self.log_ex = log_ex
        self.audio_transforms = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        file_path = item['key']

        try:
            waveform, _ = torchaudio.load(file_path)        # Point to location of audio data
            utterance = item['text'].lower()                # Point to sentence of audio data
            # print(waveform, sample_rate)
            # print('Sentences:', utterance)
            label = self.text_process.text_to_int(utterance)
            spectrogram = self.audio_transforms(waveform)   # (channel, feature, time)

            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)

            # print(f'SpecShape: {spectrogram.shape}')
            # print(f'SpecShape[-1]: {spectrogram.shape[-1]}\t Speclen: {spec_len}')

            if spec_len < label_len:
                raise Exception('spectrogram len is bigger then label len')
            if spectrogram.shape[0] > 1:
                raise Exception('dual channel, skipping audio file %s' % file_path)
            if spectrogram.shape[2] > 16000:
                raise Exception('spectrogram to big. size %s' %spectrogram.shape[2])
            if label_len == 0:
                raise Exception('label len is zero... skipping %s' % file_path)
            
            # print(f'{idx}. {utterance}')
            return spectrogram, label, spec_len, label_len

        except Exception as e:
            if self.log_ex:
                print(str(e), file_path)
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)
        
    def describe(self):
        return self.data.describe()


# NOTE: Define train_audio_transforms and valid_audio_transforms
train_audio_transforms = nn.Sequential(
    transforms.MelSpectrogram(sample_rate=16000, n_mels=64),
    transforms.FrequencyMasking(freq_mask_param=30),
    transforms.TimeMasking(time_mask_param=100)
)

valid_audio_transforms = transforms.MelSpectrogram(sample_rate=16000, n_mels=64)


# Lightning Data Module
class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_json, test_json, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.train_json = train_json
        self.test_json = test_json
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = CustomAudioDataset(self.train_json,
                                                transform=train_audio_transforms)
        self.test_dataset = CustomAudioDataset(self.test_json, 
                                               transform=valid_audio_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, 
                          collate_fn=lambda x: self.data_processing(x, 'train'), 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=lambda x: self.data_processing(x, 'valid'), 
                          num_workers=self.num_workers)

    def data_processing(self, data, data_type):
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        for (waveform, label, input_length, label_length) in data:
            if data_type == 'train':
                # print(f'SpecShape: {waveform.shape}')
                spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
                # print(f'SpecAugment: {spec.shape}\n')
            elif data_type == 'valid':
                # print('Val_waveform:', waveform.shape)
                spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            else:
                raise Exception('data_type should be train or valid')

            spectrograms.append(spec)
            # print(len(spectrograms))
            # print(f'Check1:{label}')
            label = torch.Tensor(label)
            # print(f'Check2:{label}\n')
            labels.append(label)
            input_lengths.append(input_length)
            label_lengths.append(label_length)

        # Print the shapes of spectrograms before padding
        # for spec in spectrograms:
        #     print("Spec before padding:", spec.shape)

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths


# Checking
if __name__ == "__main__":
    # Define parameters
    batch_size = 8
    train_json = 'scripts/converted_dataset/train.json'
    test_json = 'scripts/converted_dataset/test.json'
    num_workers = 0

    # Create data module instance
    data_module = SpeechDataModule(batch_size, train_json, test_json, num_workers)

    # Set up data module (downloads data if necessary and prepares datasets)
    data_module.setup()

    # Load the data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(len(train_loader), len(val_loader))