import torch
import torch.nn as nn
from torch.nn import functional as F


class ActDropNormCNN1D(nn.Module):
    def __init__(self, n_feats):
        super(ActDropNormCNN1D, self).__init__()
        self.norm = nn.BatchNorm1d(n_feats)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.gelu(self.norm(x))
        return x


class CNN1D(nn.Module):
    def __init__(self, n_feats):
        super(CNN1D, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, kernel_size=5, stride=2, padding=5//2),
            ActDropNormCNN1D(n_feats),
        )

    def forward(self, x):
        x = x.squeeze(1)  # (batch_size, 1, n_feats, time)
        x = self.cnn1(x)  # (batch_size, n_feats, new_time1)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(16000, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.dense(x)
        return x


class SpeechEmotionRecognition(nn.Module):
    hyper_parameters = {
        "num_classes": 7,   # output_class
        "n_feats": 128,     # n_mels
        "dropout": 0.2
    }

    def __init__(self, num_classes, n_feats, dropout):
        super(SpeechEmotionRecognition, self).__init__()
        self.cnn = CNN1D(n_feats=n_feats)
        self.mlp = MLP()
        self.dropout = nn.Dropout(dropout)
        self.final_fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(1, -1)  # flatten to (batch_size, n_feats)
        # print(x.shape)
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.final_fc(x)
        return x

