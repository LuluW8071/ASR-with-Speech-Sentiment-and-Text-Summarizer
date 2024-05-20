import pytorch_lightning as pl
import torch
import torchaudio

import torchmetrics
from torch import nn 
from efficientnet_pytorch import EfficientNet

class neuralnet(pl.LightningModule):
    def __init__(self, input_size, num_classes, learning_rate):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size = 5, stride = 2, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2),
            nn.Dropout(0.2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2),
            # nn.Dropout(0.2)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2),
            nn.Dropout(0.2)
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(83968, 1000)
        self.fc2 = nn.Linear(1000, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes) 

        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def _common_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        # print(y.shape, y_pred.shape)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred, y
    
    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        y_pred = torch.argmax(y_pred, dim=1)
        y = torch.argmax(y, dim=1)
        # print('Pred Labels:', y_pred
        # print('Labels:', y)
        accuracy = self.accuracy(y_pred, y)  
        self.log_dict({"loss": loss,  
                       "accuracy": accuracy}, 
                       on_step=True, on_epoch=False, 
                       prog_bar=True,
                       logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx) 
        y_pred = torch.argmax(y_pred, dim=1)
        y = torch.argmax(y, dim=1)
        accuracy = self.accuracy(y_pred, y)  
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        X = batch  
        # X = X.reshape(X.shape[0], -1) 
        y_pred = self.forward(X)
        preds = torch.argmax(y_pred, dim=1) 
        return preds 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
              'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                      mode='min', 
                                                                      factor=0.4, 
                                                                      patience=3),
              'monitor': 'val_loss',
          }
        return [optimizer], [scheduler]



