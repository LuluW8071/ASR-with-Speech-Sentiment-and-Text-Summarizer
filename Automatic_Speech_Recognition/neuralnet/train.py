from comet_ml import Experiment, ExistingExperiment
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os 
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger

from dataset import SpeechDataModule
from model import SpeechRecognitionModel
from utils import GreedyDecoder
from scorer import wer, cer

# Load API
from dotenv import load_dotenv
import os
load_dotenv()

class ASRTrainer(pl.LightningModule):
    def __init__(self, model, args, train_loader_len):
        super(ASRTrainer, self).__init__()
        self.model = model
        self.args = args
        self.train_loader_len = train_loader_len

        # Metrics
        self.loss_fn = nn.CTCLoss(blank=28, zero_infinity=True)    # unique char_map_str = 28 in utils.py


    def forward(self, x):
        return self.model(x)
    

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)     # Referred to DeepSpeech2 Paper
        scheduler = {
            'scheduler': optim.lr_scheduler.OneCycleLR(optimizer, 
                                                       max_lr=self.args.learning_rate,
                                                       steps_per_epoch=self.train_loader_len,
                                                       epochs=self.args.epochs,
                                                       anneal_strategy='linear'),
            'monitor': 'val_loss',
        }   
        return [optimizer], [scheduler]
    

    def _common_step(self, batch, batch_idx):
        spectograms, labels, input_lengths, label_lengths = batch
        y_pred = self.forward(spectograms)  # (batch, time, n_class)
        y_pred = F.log_softmax(y_pred, dim=2)
        # y_pred = y_pred.transpose(0, 1)     # (time, batch, n_class)
        loss = self.loss_fn(y_pred, labels, input_lengths, label_lengths)
        return loss, y_pred, labels, label_lengths
    

    def training_step(self, batch, batch_idx):
        loss, y_pred, labels, _ = self._common_step(batch, batch_idx)

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        loss, y_pred, labels, label_lengths = self._common_step(batch, batch_idx)
        print(y_pred.shape)
        # Decode preds for WER and CER
        val_cer, val_wer = [], []
        decoded_preds, decoded_targets = GreedyDecoder(y_pred.transpose(0, 1), labels, label_lengths)
        print('Decoded predictions:', decoded_preds[0:2])
        print('Decoded targets:', decoded_targets[0:2])
        for j in range(len(decoded_preds)):
            val_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            val_wer.append(wer(decoded_targets[j], decoded_preds[j]))
        avg_cer = sum(val_cer)/len(val_cer)
        avg_wer = sum(val_wer)/len(val_wer)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_cer', avg_cer, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('val_wer', avg_wer, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        pass


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare dataset
    data_module = SpeechDataModule(batch_size=args.batch_size,
                                   train_json=args.train_json,
                                   test_json=args.valid_json, 
                                   num_workers=args.num_workers)
    data_module.setup()
    train_loader_len = len(data_module.train_dataloader())    # To pass on scheduler
    
    # Refer to DeepSpeech2 Paper
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 64,
        "stride":2,
        "dropout": 0.2,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs
    }

    model = SpeechRecognitionModel(hparams['n_cnn_layers'], 
                                   hparams['n_rnn_layers'], 
                                   hparams['rnn_dim'],
                                   hparams['n_class'], 
                                   hparams['n_feats'], 
                                   hparams['stride'], 
                                   hparams['dropout']).to(device)
    speech_trainer = ASRTrainer(model=model, args=args, train_loader_len=train_loader_len) 


    # NOTE: Comet Logger
    comet_logger = CometLogger(api_key=os.getenv('API_KEY'),
                               project_name=('PROJECT_NAME'))

    # NOTE: Define Trainer callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath="./saved_checkpoint/",
        filename='ASR-{epoch:02d}-{val_loss:.2f}',
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
                         callbacks=[EarlyStopping(monitor="val_loss"), checkpoint_callback],
                         logger=comet_logger
                        )
    
    # Fit the model to the training data using the Trainer's fit method.
    trainer.fit(speech_trainer, data_module)
    trainer.validate(speech_trainer, data_module)


if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="Train ASR Model")

    # Train Device Hyperparameters
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=0, type=int,
                        help='n data loading workers, default 0 = main process only')

    # Train and Valid File
    parser.add_argument('--train_json', default=None, required=True, type=str,
                        help='json file to load training data')                   
    parser.add_argument('--valid_json', default=None, required=True, type=str,
                        help='json file to load testing data')

    # General Train Hyperparameters
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, help='size of batch')
    parser.add_argument('-lr','--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')
    
    # Params
    parser.add_argument('--steps', default=200, type=int, help='log every n steps')
    
    args = parser.parse_args()
    main(args)

# !python3 train.py --train_json "dataset/train.json" --valid_json "dataset/test.json" -w 2 --epochs 3 --batch_size 2 -lr 5e-4