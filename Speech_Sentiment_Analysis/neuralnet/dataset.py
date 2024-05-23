import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

class EmotionDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self._prepare_data()

    def _prepare_data(self):
        """ Load Data and One Hot Encode the Labels """
        Emotions = pd.read_csv(self.file_path)
        # print(Emotions)
        Emotions = Emotions.fillna(0)      

        X = Emotions.iloc[:, :-1].values
        Y = Emotions['Emotions'].values

        # OneHotEncode labels
        encoder = OneHotEncoder()
        Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

        # Convert to PyTorch tensors
        # print(X.shape, Y.shape)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
       

    def __getitem__(self, index):
        # print(self.X[index].shape, self.Y[index].shape)
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

class EmotionDataModule(pl.LightningDataModule):
    def __init__(self, file_path, batch_size, num_workers):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # No need for explicit preparation
        # as it's done in EmotionDatasetc custom dataloader
        pass 

    def setup(self, stage=None):
        # Create Dataset
        dataset = EmotionDataset(self.file_path)

        # RandomSplit the dataset [80:20]
        dataset_size = len(dataset)
        val_size = int(0.20 * dataset_size)
        train_size = dataset_size - val_size
        self.train_data, self.val_data = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        # Using val_data as test_dataset in the end for final eval
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
