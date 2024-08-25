from neuralnet.utils import TextTransform
import torch
import ctcdecode

# Initialize TextTransform for text processing
textprocess = TextTransform()

# Define labels for the vocabulary
labels = ["'",  # 0
          " ",  # 1
          "a",  # 2
          "b",
          "c",
          "d",
          "e",
          "f",
          "g",
          "h",
          "i",
          "j",
          "k",
          "l",
          "m",
          "n",
          "o",
          "p",
          "q",
          "r",
          "s",
          "t",
          "u",
          "v",
          "w",
          "x",
          "y",
          "z",  # 27
          "_"]  # 28, blank


def DecodeGreedy(output, blank_label=28, collapse_repeated=True):
    """ Decodes the output of the neural network using greedy decoding """
    arg_maxes = torch.argmax(output, dim=2).squeeze(1) 
    decode = []
    for i, index in enumerate(arg_maxes):
    	if index != blank_label:
    		if collapse_repeated and i != 0 and index == arg_maxes[i-1]:
    			continue
    		decode.append(index.item())
    return textprocess.int_to_text_sequence(decode)

class CTCBeamDecoder:
    """
    Class for performing CTC beam decoding with a language model.

    Args:
        beam_size (int): Beam size for beam search decoding.
        blank_id (int): Index of the blank label in the vocabulary.
        kenlm_path (str): Path to the KenLM language model file.
    """

    def __init__(self, beam_size=100, blank_id=labels.index('_'), kenlm_path=None):
        """
        Initializes the CTCBeamDecoder.

        Args:
            beam_size (int): Beam size for beam search decoding.
            blank_id (int): Index of the blank label in the vocabulary.
            kenlm_path (str): Path to the KenLM language model file.
            alpha (float): Scaling factor for language model score.
            beta (float): Scaling factor for length normalization.
        """
        self.decoder = ctcdecode.CTCBeamDecoder(
            labels,                     # Labels for the vocabulary
            alpha=0.522729216841,       # Weight associated with the language model score
            beta=0.96506699808,         # Weight associated with the number of words within our beam
            beam_width=beam_size,       # Beam size for beam search decoding; higher = better accuracy but slower
            blank_id=labels.index('_'), # Index of the blank label in the vocabulary
            model_path=kenlm_path)      # Path to the KenLM language model file
        print("Finished loading beam search")

    def __call__(self, output):
        """
        Perform beam search decoding on the given output.

        Args:
            output (torch.Tensor): Output tensor from the neural network.

        Returns:
            str: Decoded text sequence.
        """
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)

        # Convert result to string
        return self.convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])

    def convert_to_string(self, tokens, vocab, seq_len):
        """
        Convert token sequence to string using the vocabulary.

        Args:
            tokens (list): List of token indices.
            vocab (list): Vocabulary list.
            seq_len (int): Length of the sequence.

        Returns:
            str: Decoded text sequence.
        """
        return ''.join([vocab[x] for x in tokens[0:seq_len]])