import comet_ml
import pytorch_lightning as pl
import os 
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger

# Load API
from dotenv import load_dotenv
load_dotenv()

from dataset import SpeechDataModule
from model import SpeechRecognition
from utils import GreedyDecoder
from scorer import wer, cer

class ASRTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super(ASRTrainer, self).__init__()
        self.model = model
        self.args = args

        # Metrics
        self.loss_fn = nn.CTCLoss(blank=28, zero_infinity=True)

    def forward(self, x, hidden):
        return self.model(x, hidden)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.50, patience=6),
            'monitor': 'val_loss',
        }
        return [optimizer], [scheduler]
    
    def _common_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths = batch
        bs = spectrograms.shape[0]
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self(spectrograms, (hn, c0))
        output = F.log_softmax(output, dim=2)
        
        loss = self.loss_fn(output, labels, input_lengths, label_lengths)
        return loss, output, labels, label_lengths
    
    def training_step(self, batch, batch_idx):
        loss, y_pred, labels, label_lengths = self._common_step(batch, batch_idx)

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        loss, y_pred, labels, label_lengths = self._common_step(batch, batch_idx)
        val_cer, val_wer = [], []

        decoded_preds, decoded_targets = GreedyDecoder(y_pred.transpose(0, 1), labels, label_lengths)
        
        # Log predictions
        log_targets = decoded_targets[-1]
        log_preds = {"Preds": decoded_preds[-1]}
        self.logger.experiment.log_text(text=log_targets, metadata=log_preds)

        for j in range(len(decoded_preds)):
            # print(len(decoded_preds), decoded_preds[j], decoded_preds[j])
            val_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            val_wer.append(wer(decoded_targets[j], decoded_preds[j]))
        avg_cer = sum(val_cer) / len(val_cer)
        avg_wer = sum(val_wer) / len(val_wer)

        self.log_dict({
        'val_cer': avg_cer,
        'val_wer': avg_wer,
    }, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_idx)
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

    # Log hyperparams of model and setup trainer
    h_params = SpeechRecognition.hyper_parameters
    model = SpeechRecognition(**h_params)
    speech_trainer = ASRTrainer(model=model, 
                                args=args) 

    # NOTE: Comet Logger
    comet_logger = CometLogger(api_key=os.getenv('API_KEY'),
                               project_name='PROJECT_NAME')

    # NOTE: Define Trainer callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath="./saved_checkpoint/",
        filename='ASR-{epoch:02d}',
        save_top_k=True,
        mode='min'
    )

    # Trainer Instance
    trainer = pl.Trainer(accelerator=device,
                         devices=args.gpus,
                         min_epochs=1,
                         max_epochs=args.epochs,
                         precision=args.precision,
                         val_check_interval=args.steps,
                         gradient_clip_val=1.0,
                         callbacks=[EarlyStopping(monitor="val_loss"), 
                                    checkpoint_callback
                         ],
                         logger=comet_logger
                        )
    
    # Fit the model to the training data using the Trainer's fit method.
    ckpt_path = args.checkpoint_path if args.checkpoint_path else None
    trainer.fit(speech_trainer, data_module, ckpt_path=ckpt_path)
    trainer.validate(speech_trainer, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASR Model")

    # Train Device Hyperparameters
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=0, type=int, help='n data loading workers, default 0 = main process only')

    # Train and Valid File
    parser.add_argument('--train_json', default=None, required=True, type=str, help='json file to load training data')                   
    parser.add_argument('--valid_json', default=None, required=True, type=str, help='json file to load testing data')

    # General Train Hyperparameters
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch')
    parser.add_argument('-lr','--learning_rate', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')
    
    # Params
    parser.add_argument('--steps', default=1000, type=int, help='val every n steps')
    
    # Checkpoint path
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path to a checkpoint file to resume training')

    args = parser.parse_args()
    main(args)