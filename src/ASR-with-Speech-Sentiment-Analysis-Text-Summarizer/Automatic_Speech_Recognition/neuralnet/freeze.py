""" Freezes and optimize the model. Use after training"""
import argparse
import torch

from torch import nn
from model import ConformerEncoder, LSTMDecoder
from collections import OrderedDict

class ConformerASR(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(ConformerASR, self).__init__()
        self.encoder = ConformerEncoder(**encoder_params)
        self.decoder = LSTMDecoder(**decoder_params)

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

def trace(model):
    model.eval()
    x = torch.rand(1, 128, 80)  # (Batch_size, seq_length, input_feat)
    traced = torch.jit.trace(model, x)
    return traced

encoder_params = {
        'd_input': 80,  # input features
        'd_model': 144,
        'num_layers': 16,
        'conv_kernel_size': 31,
        'feed_forward_residual_factor': 0.5,
        'feed_forward_expansion_factor': 4,
        'num_heads': 4,
        'dropout': 0.1,
    }
    

decoder_params = {
        'd_encoder': 144,  # Should match d_model of encoder
        'd_decoder': 320,
        'num_layers': 1,
        'num_classes': 29,  # Adjust based on your output classes
    }


def main(args):
    print("loading model from", args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))
    
    model = ConformerASR(encoder_params, decoder_params)
    model_state_dict = checkpoint['state_dict']
    encoder_weights = {k: v for k, v in model_state_dict.items() if k.startswith("encoder.")}
    decoder_weights = {k: v for k, v in model_state_dict.items() if k.startswith("decoder.")}

    print("tracing model...")
    traced_model = trace(model)
    print("saving to", args.save_path)
    traced_model.save(args.save_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freezes and optimize the model.")
    parser.add_argument('--model_checkpoint', type=str, default=None, required=True,
                        help='Checkpoint of model to optimize')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='path to save optmized model')

    args = parser.parse_args()
    main(args)
