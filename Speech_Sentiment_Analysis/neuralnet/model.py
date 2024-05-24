import pytorch_lightning as pl
from comet_ml import Experiment
import torch

import torchmetrics
from torch import nn

# Load API
from dotenv import load_dotenv
import os
load_dotenv()

class neuralnet(pl.LightningModule):
    def __init__(self, input_size, output_shape, learning_rate, steps):
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

        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=output_shape)

        self.learning_rate = learning_rate
        self.log_steps = steps
        self.experiment = Experiment(api_key=os.getenv('API_KEY'),
                                     project_name=os.getenv('PROJECT_NAME'))

        # For Confusion Matrix logger
        self.predictions = []
        self.targets = []

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

    def _common_step(self, batch, batch_idx):
        X, y = batch
        # print(f'Original X shape: {X.shape}')
        X = X.unsqueeze(1)  # Added channel dimension
        # print(f'Reshaped X shape: {X.shape}')
        y_pred = self.forward(X)
        loss = self.loss_fn(y_pred, y)
        y_pred = torch.argmax(y_pred, dim=1)
        y = torch.argmax(y, dim=1)
        return loss, y_pred, y

    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        # print('Pred Labels:', y_pred
        # print('Labels:', y)
        accuracy = self.accuracy(y_pred, y)
        self.log_dict({"loss": loss,
                       "accuracy": accuracy},
                      on_step=True, on_epoch=False,
                      prog_bar=True,
                      logger=True)
        return loss

    def on_validation_epoch_start(self):
        # Reset the predictions and targets lists at the start of each validation epoch
        self.predictions = []
        self.targets = []

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)

        # Accumulate predictions and targets for confusion matrix
        self.predictions.append(y_pred.detach().to('cpu'))
        self.targets.append(y.detach().to('cpu'))
        
        # Log metrics to Comet.ml 
        self.experiment.log_metric('val_loss', loss, step=self.log_steps)
        self.experiment.log_metric('val_acc', accuracy, step=self.log_steps)
        return loss

    def on_validation_epoch_end(self):
        # Concatenate all accumulated predictions and targets
        all_preds = torch.cat(self.predictions, dim=0).numpy()
        all_targets = torch.cat(self.targets, dim=0).numpy()
        
        labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

        # Log the confusion matrix to Comet
        self.experiment.log_confusion_matrix(y_true=all_targets, 
                                             y_predicted=all_preds,
                                             labels=labels)

    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", accuracy, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        X = batch
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
