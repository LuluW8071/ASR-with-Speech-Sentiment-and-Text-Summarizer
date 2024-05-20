import pytorch_lightning as pl 
import torch

from model import neuralnet
from dataset import EmotionDataModule

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datamodule = EmotionDataModule(file_path=args.file_path,
                                   max_width=1266,
                                   max_height=201,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers)
    
    # Call setup to initialize datasets
    datamodule.setup()
    
    # Create model
    num_classes = 7
    model = neuralnet(input_size=1,
                      num_classes=num_classes,
                      learning_rate=args.lr).to(device)
    
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
                         log_every_n_steps=50,
                         callbacks=[EarlyStopping(monitor="val_loss"),
                                    checkpoint_callback]
                        )
    
    # Fit the model to the training data using the Trainer's fit method.
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)

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
    parser.add_argument('--batch_size', default=16, type=int, help='size of batch')
    parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')
    
    args = parser.parse_args()
    main(args)


