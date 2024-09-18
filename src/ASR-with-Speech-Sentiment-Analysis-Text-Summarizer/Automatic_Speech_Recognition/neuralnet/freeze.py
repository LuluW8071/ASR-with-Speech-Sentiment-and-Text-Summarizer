import argparse
import torch
from torch import nn
from collections import OrderedDict

from model import ConformerEncoder, LSTMDecoder  # Import your model definitions

# Define the Conformer ASR model class
class ConformerASR(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(ConformerASR, self).__init__()
        self.encoder = ConformerEncoder(**encoder_params)
        self.decoder = LSTMDecoder(**decoder_params)

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

# Define tracing functions
def trace_encoder(encoder):
    encoder.eval()
    x = torch.rand(1, 300, 80)  # (Batch_size, seq_length, input_feat)
    return torch.jit.trace(encoder, x)

def trace_decoder(decoder):
    decoder.eval()
    x = torch.rand(1, 300, 29)  # (Batch_size, seq_length, output_feat)
    return torch.jit.trace(decoder, x)

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

# Define a class for the traced model
class TracedConformerASR(nn.Module):
    def __init__(self, traced_encoder, traced_decoder):
        super(TracedConformerASR, self).__init__()
        self.encoder = traced_encoder
        self.decoder = traced_decoder

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

# Main function to load, trace, and save the model
def main(args):
    print("Loading model from", args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))
    
    # Initialize the ConformerASR model
    model = ConformerASR(encoder_params, decoder_params)
    model_state_dict = checkpoint['state_dict']

    # Prepare the state dicts for encoder and decoder
    encoder_state_dict = OrderedDict()
    decoder_state_dict = OrderedDict()

    # Split the state dict into encoder and decoder parts
    for k, v in model_state_dict.items():
        if k.startswith('model.encoder.'):
            name = k.replace('model.encoder.', '')  # Remove 'encoder.' prefix
            encoder_state_dict[name] = v
        elif k.startswith('model.decoder.'):
            name = k.replace('model.decoder.', '')   # Remove 'decoder.' prefix
            decoder_state_dict[name] = v

    # Load state dicts into the model
    model.encoder.load_state_dict(encoder_state_dict)
    model.decoder.load_state_dict(decoder_state_dict)

    # Trace the encoder
    print("Tracing encoder...")
    traced_encoder = trace_encoder(model.encoder)

    # Trace the decoder
    print("Tracing decoder...")
    traced_decoder = trace_decoder(model.decoder)

    # Create a traced model with traced encoder and decoder
    print("Creating traced model...")
    traced_model = TracedConformerASR(traced_encoder, traced_decoder)
    
    # Trace the entire model
    dummy_input = torch.rand(1, 300, 80)  # (Batch_size, seq_length, input_feat)
    traced_full_model = torch.jit.trace(traced_model, dummy_input)
    
    # Save the traced full model
    print("Saving to", args.save_path)
    torch.jit.save(traced_full_model, args.save_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freezes and optimizes the Conformer model.")
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Checkpoint of model to optimize')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save optimized model')

    args = parser.parse_args()
    main(args)
