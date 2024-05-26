from comet_ml import Experiment         # You can comment this for freezing the model
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import torch
import torchmetrics
from torch import nn
import argparse

from model import neuralnet
from dataset import EmotionDataModule

# Load API
from dotenv import load_dotenv
import os
load_dotenv()


class SentimentTrainer(pl.LightningModule):
    def __init__(self, model, learning_rate, steps):
        super(SentimentTrainer, self).__init__()
        self.model = model

        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=6)

        self.learning_rate = learning_rate
        self.log_steps = steps
        self.experiment = Experiment(api_key=os.getenv('API_KEY'),
                                     project_name=os.getenv('PROJECT_NAME'))

        # For Confusion Matrix logger
        self.predictions = []
        self.targets = []

    def forward(self, x):
        return self.model(x)

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


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datamodule = EmotionDataModule(file_path=args.file_path,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers)
    
    # Call setup to initialize datasets
    datamodule.setup()
    
    # Create model
    num_classes = 6

    
    model = neuralnet(input_size=1,
                      output_shape=num_classes).to(device)
    
    sentiment_trainer = SentimentTrainer(model=model, learning_rate=args.lr, steps=args.steps)

    # Save the model periodically by monitoring a quantity
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                          dirpath="./saved_checkpoint/",
                                          filename="sentiment-model-{epoch:02d}-{val_loss:.2f}")   # Checkpoint filename

    # Trainer Instance
    trainer = pl.Trainer(accelerator=device,
                         devices=args.gpus,
                         min_epochs=1,
                         max_epochs=args.epochs,
                         precision=args.precision,
                         log_every_n_steps=args.steps,
                         callbacks=[EarlyStopping(monitor="val_loss"),
                                    checkpoint_callback]
                        )
    
    # Fit the model to the training data using the Trainer's fit method.
    trainer.fit(sentiment_trainer, datamodule)
    trainer.validate(sentiment_trainer, datamodule)
    trainer.test(sentiment_trainer, datamodule)

if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech Sentiment Model")

    # Train Device Hyperparameters
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=0, type=int,
                        help='n data loading workers, default 0 = main process only')

    # Audio File CSV
    parser.add_argument('--file_path', default=None, required=True, type=str,
                        help='Folder path to load training data')
    

    # General Train Hyperparameters
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, help='size of batch')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')
    
    # Params
    parser.add_argument('--steps', default=200, type=int, help='log every n steps')
    
    args = parser.parse_args()
    main(args)

