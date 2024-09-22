import comet_ml
import pytorch_lightning as pl
import os 
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger
# from collections import OrderedDict
from torchmetrics.text import WordErrorRate, CharErrorRate
# Load API

from dotenv import load_dotenv
load_dotenv()

from dataset import SpeechDataModule
from model import ConformerEncoder, LSTMDecoder
from utils import GreedyDecoder
# from scorer import wer, cer


class ConformerASR(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(ConformerASR, self).__init__()
        self.encoder = ConformerEncoder(**encoder_params)
        self.decoder = LSTMDecoder(**decoder_params)

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output


class ASRTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super(ASRTrainer, self).__init__()
        self.model = model
        self.args = args

        # Metrics
        self.losses = []
        self.val_cer = []
        self.val_wer = []
        self.char_error_rate = CharErrorRate()
        self.word_error_rate = WordErrorRate()
        self.loss_fn = nn.CTCLoss(blank=28, zero_infinity=True)

        # Precompute sync_dist for distributed GPUs train
        self.sync_dist = True if args.gpus > 1 else False

    def forward(self, x):
        return self.model(x)
    
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=1e-6
        )

        scheduler = {
            'scheduler': optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.args.lr_step_size,
                gamma=self.args.lr_gamma
            ),
            'monitor': 'val_loss'
        }

        return [optimizer], [scheduler]
    
    
    def _common_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths, _, _ = batch
        output = self(spectrograms)     # Directly calls forward method of conformer
        output = F.log_softmax(output, dim=-1).transpose(0, 1)
        
        loss = self.loss_fn(output, labels, input_lengths, label_lengths)
        return loss, output, labels, label_lengths
    
    def training_step(self, batch, batch_idx):
        loss,_,_,_ = self._common_step(batch, batch_idx)

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return loss

    
    def validation_step(self, batch, batch_idx):
        loss, y_pred, labels, label_lengths = self._common_step(batch, batch_idx)
        self.losses.append(loss)

        # Greedy decoding
        decoded_preds, decoded_targets = GreedyDecoder(y_pred.transpose(0, 1), labels, label_lengths)
        
        # Calculate metrics
        cer_batch = self.char_error_rate(decoded_preds, decoded_targets)
        wer_batch = self.word_error_rate(decoded_preds, decoded_targets)
        
        # Append batch metrics to lists
        self.val_cer.append(cer_batch)
        self.val_wer.append(wer_batch)

        # Log final predictions
        if batch_idx%50==0:
            log_targets = decoded_targets[-1]
            log_preds = {"Preds": decoded_preds[-1]}
            self.logger.experiment.log_text(text=log_targets, metadata=log_preds)

        return {'val_loss': loss}


    def on_validation_epoch_end(self):
        # Calculate averages of metrics over the entire epoch
        avg_loss = torch.stack(self.losses).mean()
        avg_cer = torch.stack(self.val_cer).mean()
        avg_wer = torch.stack(self.val_wer).mean()

        # Log averaged metrics
        self.log('val_cer', avg_cer, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)
        self.log('val_wer', avg_wer, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)
        self.log('val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)

        # Clear the lists for the next epoch
        self.losses.clear()
        self.val_cer.clear()
        self.val_wer.clear()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare dataset
    data_module = SpeechDataModule(batch_size=args.batch_size,
                                   train_json=args.train_json,
                                   test_json=args.valid_json, 
                                   num_workers=args.num_workers)
    data_module.setup()

    # Define model hyperparameters
    # https://arxiv.org/pdf/2005.08100 : Table 1 for conformer parameters
    encoder_params = {
        'd_input': 80,                        # Input features: n-mels
        'd_model': 144,                       # Encoder Dims
        'num_layers': 16,                     # Encoder Layers
        'conv_kernel_size': 31,
        'feed_forward_residual_factor': 0.5,
        'feed_forward_expansion_factor': 4,
        'num_heads': 4,                       # Relative MultiHead Attetion Heads
        'dropout': 0.1,
    }

    decoder_params = {
        'd_encoder': 144,    # Match with Encoder layer
        'd_decoder': 320,    # Decoder Dim
        'num_layers': 1,     # Deocder Layer
        'num_classes': 29,   # Output Classes
    }

    # Model Instance
    model = ConformerASR(encoder_params, decoder_params)

    # Adjust epochs if checkpoint path is provided
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        args.epochs += checkpoint['epoch'] 

    speech_trainer = ASRTrainer(model=model, args=args)

    # NOTE: Comet Logger
    comet_logger = CometLogger(api_key=os.getenv('API_KEY'),
                               project_name=os.getenv('PROJECT_NAME'))

    # NOTE: Define Trainer callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath="./saved_checkpoint/",
        filename='ASR-{epoch:02d}-{val_loss:.2f}-{val_wer:.2f}',
        save_top_k=3,        # 3 Checkpoints
        mode='min'
    )

    # Trainer Instance
    trainer_args = {
        'accelerator': device,
        'devices': args.gpus,
        'min_epochs': 1,
        'max_epochs': args.epochs,
        'precision': args.precision,
        'check_val_every_n_epoch': 1, 
        'gradient_clip_val': 1.0,
        'callbacks': [LearningRateMonitor(logging_interval='epoch'),
                      EarlyStopping(monitor="val_loss"), 
                      checkpoint_callback],
        'logger': comet_logger
    }
    
    if args.gpus > 1:
        trainer_args['strategy'] = args.dist_backend
        
    trainer = pl.Trainer(**trainer_args)
    
    # Fit the model to the training data using the Trainer's fit method.
    ckpt_path = args.checkpoint_path if args.checkpoint_path else None
    trainer.fit(speech_trainer, data_module, ckpt_path=ckpt_path)   
    trainer.validate(speech_trainer, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASR Model")

    # Train Device Hyperparameters
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=4, type=int, help='n data loading workers')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str,
                        help='which distributed backend to use for aggregating multi-gpu train')

    # Train and Valid File
    parser.add_argument('--train_json', default=None, required=True, type=str, help='json file to load training data')                   
    parser.add_argument('--valid_json', default=None, required=True, type=str, help='json file to load testing data')

    # General Train Hyperparameters
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch')
    parser.add_argument('-lr', '--learning_rate', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')
    parser.add_argument('--lr_step_size', type=int, default=10, help='Number of epochs for step decay') 
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Decay factor') 
    
    # Checkpoint path
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path to a checkpoint file to resume training')

    args = parser.parse_args()
    main(args)
