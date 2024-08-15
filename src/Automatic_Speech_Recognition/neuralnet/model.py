import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualCNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(out_channels)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        residual = x
        out = self.gelu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # Residual connection
        out = self.gelu(out)
        return out

class SpeechRecognition(nn.Module):
    hyper_parameters = {
        "num_classes": 29,  # output_class
        "n_feats": 128,     # n_mels
        "dropout": 0.3,
        "hidden_size": 512,
        "num_layers": 2     # RNN Layers
    }

    def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Residual CNN Loop Block
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            *[nn.Sequential(
                ResidualCNNBlock(32, 32),
                nn.Dropout(dropout)
            ) for _ in range(3)]
        )

        self.gru = nn.GRU(input_size=32 * (n_feats // 2), hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout, bidirectional=True)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)  # Adjust for bidirectionality
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size * 2, num_classes)  # Adjust for bidirectionality

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers * 2, self.hidden_size  # num_layers * num_directions
        return torch.zeros(n, batch_size, hs)

    def forward(self, x, hidden):
        x = x.squeeze(2)                               # Remove extra dimension
        x = self.cnn(x)                                # Pass through CNN layers
        x = x.view(x.size(0), x.size(2), -1)           # Flatten to (batch, time, feature)
        x = x.transpose(0, 1)                          # (time, batch, feature)
        out, hn = self.gru(x, hidden)
        x = self.dropout2(F.gelu(self.layer_norm2(out)))
        return self.final_fc(x), hn
