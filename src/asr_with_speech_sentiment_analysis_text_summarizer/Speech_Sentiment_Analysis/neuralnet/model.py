import pytorch_lightning as pl
from torch import nn


class neuralnet(pl.LightningModule):
    def __init__(self, input_size, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(input_size, 512, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4736, 512)
        self.fc2 = nn.Linear(512, output_shape)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

