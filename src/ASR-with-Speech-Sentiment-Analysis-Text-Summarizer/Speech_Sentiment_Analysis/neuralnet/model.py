import torch
import torch.nn as nn

# TimeDistributed layer for applying a module to each time step in a sequence
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        if len(x.size()) == 3:
            x_reshape = x.contiguous().view(-1, x.size(2)) 
        elif len(x.size()) == 4: 
            x_reshape = x.contiguous().view(-1, x.size(2), x.size(3))
        else: 
            x_reshape = x.contiguous().view(-1, x.size(2), x.size(3), x.size(4))
        
        # Apply the module to the reshaped tensor
        y = self.module(x_reshape)
        
        if len(x.size()) == 3:
            y = y.contiguous().view(x.size(0), -1, y.size(1)) 
        elif len(x.size()) == 4:
            y = y.contiguous().view(x.size(0), -1, y.size(1), y.size(2))  
        else:
            y = y.contiguous().view(x.size(0), -1, y.size(1), y.size(2), y.size(3))  
        return y


class HybridModel(nn.Module):
    def __init__(self, num_emotions):
        """
        Initialize the HybridModel with convolutional, LSTM, and attention layers.

        Parameters:
        - num_emotions: Number of emotion classes for classification.
        """
        super().__init__()

        # Convolutional Block
        self.conv2Dblock = nn.Sequential(
            # 1. Convolutional Block
            TimeDistributed(nn.Conv2d(in_channels=1,
                                   out_channels=16,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)),
            TimeDistributed(nn.BatchNorm2d(16)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)),
            TimeDistributed(nn.Dropout(p=0.3)),
            # 2. Convolutional Block
            TimeDistributed(nn.Conv2d(in_channels=16,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)),
            TimeDistributed(nn.BatchNorm2d(32)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=4, stride=4)),
            TimeDistributed(nn.Dropout(p=0.3)),
            # 3. Convolutional Block
            TimeDistributed(nn.Conv2d(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)),
            TimeDistributed(nn.BatchNorm2d(64)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=4, stride=4)),
            TimeDistributed(nn.Dropout(p=0.3))
        )
        
        # LSTM Block
        hidden_size = 64
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(p=0.4)
        self.attention_linear = nn.Linear(2 * hidden_size, 1)  # 2 * hidden_size for bidirectional LSTM
        
        # Linear layer for output classification
        self.out_linear = nn.Linear(2 * hidden_size, num_emotions)
    
    def forward(self, x):  
        conv_embedding = self.conv2Dblock(x)
        
        conv_embedding = torch.flatten(conv_embedding, start_dim=2)
        
        # Apply LSTM layers
        lstm_embedding, (h, c) = self.lstm(conv_embedding)
        lstm_embedding = self.dropout_lstm(lstm_embedding)
        
        # Compute attention weights
        batch_size, T, _ = lstm_embedding.shape
        attention_weights = [None] * T
        for t in range(T):
            embedding = lstm_embedding[:, t, :]
            attention_weights[t] = self.attention_linear(embedding)
        
        # Normalize attention weights
        attention_weights_norm = nn.functional.softmax(torch.stack(attention_weights, -1), dim=-1)
        
        # Apply attention to LSTM outputs
        attention = torch.bmm(attention_weights_norm, lstm_embedding)  # (Bx1xT)*(B,T,hidden_size*2) = (B,1,2*hidden_size)
        attention = torch.squeeze(attention, 1)
        
        # Compute output logits and softmax probabilities
        output_logits = self.out_linear(attention)
        output_softmax = nn.functional.softmax(output_logits, dim=1)
        
        return output_logits, output_softmax, attention_weights_norm
    
    
def loss_fnc(predictions, targets):
    """
    Computes the cross-entropy loss between predictions and targets.

    Parameters:
    - predictions: Predicted output from the model (logits).
    - targets: Ground truth labels for the input data.
    
    Returns:
    - CrossEntropyLoss
    """
    loss_function = nn.CrossEntropyLoss()
    return loss_function(input=predictions, target=targets)
