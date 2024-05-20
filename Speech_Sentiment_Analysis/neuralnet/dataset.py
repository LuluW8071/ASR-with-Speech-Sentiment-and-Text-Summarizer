import pytorch_lightning as pl 
import pandas as pd
import numpy as np

import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import Dataset, random_split, DataLoader

class EmotionDataset(Dataset):
    def __init__(self, file_path, max_width, max_height):
        self.audio_file_dict = pd.read_csv(file_path, index_col=0)
        self.audio_file_dict.dropna(inplace=True) # drop missing values
        
        self.max_width = max_width
        self.max_height = max_height
    
    def __getitem__(self, path):
        audio_path = self.audio_file_dict.index[path]
        audio, _ = torchaudio.load(audio_path)
        audio = torch.mean(audio, dim=0).unsqueeze(0)
        spectrogram = torchaudio.transforms.Spectrogram()(audio)
        spectrogram = F.pad(spectrogram, [0, self.max_width - spectrogram.size(2), 0, self.max_height - spectrogram.size(1)])
        
        label = pd.get_dummies(self.audio_file_dict.emotion).iloc[path].values
        label = torch.from_numpy(label).float()
        return (spectrogram, label)
    
    def __len__(self):
        return len(self.audio_file_dict)
    

    
class EmotionDataModule(pl.LightningDataModule):
    def __init__(self, file_path, max_width, max_height, batch_size, num_workers):
        super().__init__()
        self.file_path = file_path
        self.max_width = max_width
        self.max_height = max_height
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Create Dataset
        dataset = EmotionDataset(self.file_path, self.max_width, self.max_height)

        # RandomSplit the dataset [75:25]
        dataset_size = len(dataset)
        val_size = int(0.25*dataset_size)
        train_size = dataset_size - val_size
        self.train_data, self.val_data = random_split(dataset, [train_size, val_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        pass

    
