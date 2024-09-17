import os
import argparse
import random
from tqdm import tqdm
from pathlib import Path
import sox
from multiprocessing.pool import ThreadPool
from sox.core import SoxiError

def read_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                filename, transcript, _ = parts
                data.append((filename, transcript))
    return data

def convert_audio(input_path, output_path, output_format="flac"):
    try:
        tfm = sox.Transformer()
        tfm.set_output_format(file_type=output_format)
        tfm.rate(samplerate=16000)
        tfm.build(input_filepath=input_path, output_filepath=output_path)
        return True
    except SoxiError as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False

def process_file(args):
    filename, transcript, input_audio_dir, output_directory, output_format = args
    input_path = os.path.join(input_audio_dir, filename)
    output_filename = os.path.splitext(filename)[0] + '.' + output_format
    output_path = os.path.join(output_directory, output_filename)
    
    if not os.path.exists(input_path):
        print(f"Warning: Audio file not found: {input_path}")
        return None
    
    if convert_audio(input_path, output_path, output_format):
        return os.path.splitext(output_filename)[0], transcript
    return None

def process_dataset(dataset, output_directory, input_audio_dir, output_format, num_workers):
    os.makedirs(output_directory, exist_ok=True)

    # No upsampling during audio processing
    args_list = [(filename, transcript, input_audio_dir, output_directory, output_format) 
                 for filename, transcript in dataset]

    with ThreadPool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_file, args_list), 
                            total=len(args_list), 
                            desc=f"Processing {os.path.basename(output_directory)}"))

    results = [result for result in results if result is not None]

    return results  # Return processed results without upsampling

def write_transcript(results, output_directory, upsample=10):
    # Now upsample only in the transcription file
    with open(os.path.join(output_directory, 'trans.txt'), 'w', encoding='utf-8') as f:
        for filename, transcript in results:
            for _ in range(upsample):
                f.write(f"{filename} {transcript}\n")  # Upsample only in the transcript file

def main(args):
    input_audio_dir = os.path.dirname(args.input_file)
    data = read_data(args.input_file)
    random.seed(42)  # For reproducibility
    random.shuffle(data)

    split_index = int(len(data) * (1 - args.percent / 100))
    train_data = data[:split_index]
    test_data = data[split_index:]

    train_directory = os.path.join(args.output_dir, 'mimic-train')
    test_directory = os.path.join(args.output_dir, 'mimic-test')

    # Process datasets without upsampling
    train_results = process_dataset(train_data, train_directory, input_audio_dir, "flac", args.num_workers)
    test_results = process_dataset(test_data, test_directory, input_audio_dir, "flac", args.num_workers)

    # Upsample only when writing to the trans.txt file
    write_transcript(train_results, train_directory, args.upsample)
    write_transcript(test_results, test_directory, 1)  # No upsampling in test

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     """
                                     Utility script to convert mimic record WAV to FLAC
                                     split train and test data into separate directories with trans.txt files.
                                     """
                                     )
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input text file with custom format')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory where train and test subdirectories will be created')
    parser.add_argument('--percent', type=float, default=10,
                        help='Percentage of data to use for testing (default: 10)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of worker threads for processing (default: 2)')
    parser.add_argument('--upsample', type=int, default=10,
                        help='Number of times to upsample the train dataset (default: 10)')

    args = parser.parse_args()
    main(args)
