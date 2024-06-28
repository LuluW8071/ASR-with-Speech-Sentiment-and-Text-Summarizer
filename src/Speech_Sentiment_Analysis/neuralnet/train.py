import comet_ml
import pytorch_lightning as pl 
import os
import argparse
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Load API
from dotenv import load_dotenv
load_dotenv()

from dataset import SpeechDataModule
from model import SpeechEmotionRecognition


class SentimentTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super(SentimentTrainer, self).__init__()
        self.model = model
        self.args = args

        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=7)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.50, patience=6),
            'monitor': 'val_loss',
        }
        return [optimizer], [scheduler]

    def _common_step(self, batch, batch_idx):
        spectrograms, labels = batch  
        y_pred = self.forward(spectrograms)
        loss = self.loss_fn(y_pred, labels)
        y_pred = torch.argmax(y_pred, dim=1)
        return loss, y_pred, labels

    def training_step(self, batch, batch_idx):
        loss, y_pred, labels = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        # Reset the predictions and targets lists at the start of each validation epoch
        self.predictions = []
        self.targets = []

    def validation_step(self, batch, batch_idx):
        loss, y_pred, labels = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, labels)

        self.log("val_loss", loss, prog_bar=True, batch_size=self.args.batch_size)
        self.log("val_acc", accuracy, prog_bar=True)

        # Accumulate predictions and targets for confusion matrix
        self.predictions.append(y_pred.detach().to('cpu'))
        self.targets.append(labels.detach().to('cpu'))
        return loss

    def on_validation_epoch_end(self):
        # Concatenate all accumulated predictions and targets
        all_preds = torch.cat(self.predictions, dim=0).numpy()
        all_targets = torch.cat(self.targets, dim=0).numpy()

        labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

        # Log the confusion matrix to Comet
        self.logger.experiment.log_confusion_matrix(y_true=all_targets,
                                                    y_predicted=all_preds,
                                                    labels=labels)

    def predict_step(self, batch, batch_idx):
        X = batch
        y_pred = self.forward(X)
        preds = torch.argmax(y_pred, dim=1)
        return preds
        

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare dataset
    data_module = SpeechDataModule(batch_size=args.batch_size,
                                   train_csv=args.train_csv,
                                   test_csv=args.valid_csv,
                                   num_workers=args.num_workers)
    
    # Call setup to initialize datasets
    data_module.setup()
    
    # Log hyperparams of model and setup trainer
    h_params = SpeechEmotionRecognition.hyper_parameters
    model = SpeechEmotionRecognition(**h_params)
    sentiment_trainer = SentimentTrainer(model=model, args = args)

    # NOTE: Comet Logger
    comet_logger = CometLogger(api_key=os.getenv('API_KEY'),
                               project_name=('PROJECT_NAME'))


    ## NOTE: Define Trainer callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath="./saved_checkpoint/",
        filename='SentimentModel-{epoch:02d}',
        save_top_k=1,
        mode='min'
    )

    # Trainer Instance
    trainer = pl.Trainer(accelerator=device,
                         devices=args.gpus,
                         min_epochs=1,
                         max_epochs=args.epochs,
                         precision=args.precision,
                         log_every_n_steps=args.steps,
                         gradient_clip_val=1.0,
                         callbacks=[EarlyStopping(monitor="val_loss"), checkpoint_callback],
                         logger=comet_logger
                        )
    
    # Resuming to train checkpoint
    ckpt_path = args.checkpoint_path if args.checkpoint_path else None
    
    # Fit the model to the training data using the Trainer's fit method.
    trainer.fit(sentiment_trainer, data_module, ckpt_path=ckpt_path)
    trainer.validate(sentiment_trainer, data_module)


if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech Sentiment Model")

    # Train Device Hyperparameters
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=2, type=int,
                        help='n data loading workers, default 2')

    # Train and Valid File
    parser.add_argument('--train_csv', default=None, required=True, type=str, help='csv file to load training data')                   
    parser.add_argument('--valid_csv', default=None, required=True, type=str, help='csv file to load testing data')
    
    # Checkpoint path
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path to a checkpoint file to resume training')
    
    # General Train Hyperparameters
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help='size of batch')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')
    
    # Params
    parser.add_argument('--steps', default=50, type=int, help='log every n steps')
    
    args = parser.parse_args()
    main(args)
