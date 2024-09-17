import os
import argparse
import random
import csv
import sox
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from pathlib import Path
from sox.core import SoxiError


def clean_text(text):
    characters_to_remove = ':!,"'';—?'
    translator = str.maketrans('', '', characters_to_remove)
    return text.translate(translator)

def process_file(row, output_directory, input_directory, output_format):
    file_name = row['path']
    output_name = file_name.rpartition('.')[0] + '.' + output_format 
    text = clean_text(row['sentence'])
    input_audio_path = os.path.join(input_directory, 'clips', file_name)
    output_audio_path = os.path.join(output_directory, output_name)

    # Skip conversion if the output file already exists
    if os.path.exists(output_audio_path):
        return os.path.splitext(output_name)[0], text

    # Check if the input file is valid before processing
    try:
        tfm = sox.Transformer()
        tfm.rate(samplerate=16000)
        tfm.build(input_filepath=input_audio_path, output_filepath=output_audio_path)
    except SoxiError:
        print(f"Skipping file due to SoxiError: {input_audio_path}")
        return None

    return os.path.splitext(output_name)[0], text

def main(args):
    input_directory = args.file_path.rpartition('/')[0]
    percent = args.percent
    train_directory = os.path.abspath(os.path.join(args.save_path, 'common-voice-train'))
    test_directory = os.path.abspath(os.path.join(args.save_path, 'common-voice-test'))

    for directory in [train_directory, test_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    with open(args.file_path, encoding="utf-8") as f:
        length = sum(1 for _ in f) - 1

    with open(args.file_path, newline='', encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        data_to_process = list(reader)

    random.shuffle(data_to_process)
    split_index = int(length * (1 - percent / 100))
    train_data = data_to_process[:split_index]
    test_data = data_to_process[split_index:]

    def process_dataset(dataset, output_directory):
        if args.convert:
            print(f"Processing {len(dataset)} files for {os.path.basename(output_directory)}. Converting to {args.output_format.upper()} using {args.num_workers} workers.")
            with ThreadPool(args.num_workers) as pool:
                results = list(tqdm(pool.imap(lambda x: process_file(x, output_directory, input_directory, args.output_format), dataset), total=len(dataset)))
        else:
            results = [(row['path'].rpartition('.')[0], clean_text(row['sentence'])) for row in dataset]

        results = [result for result in results if result is not None]
        
        with open(os.path.join(output_directory, 'trans.txt'), 'w', encoding='utf-8') as f:
            for filename, text in results:
                f.write(f"{filename} {text}\n")

    process_dataset(train_data, train_directory)
    process_dataset(test_data, test_directory)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     """
                                     Utility script to convert CommonVoice MP3 to FLAC/WAV and
                                     split train and test data into separate directories with trans.txt files.
                                     """
                                    )
    parser.add_argument('--file_path', type=str, required=True,
                        help='path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--save_path', type=str, required=True,
                        help='path to the directory where train and test subdirectories will be created')
    parser.add_argument('--percent', type=int, default=10,
                        help='percent of clips put into test set instead of train set')
    parser.add_argument('--convert', action='store_true',
                        help='indicates that the script should convert mp3 to the specified output format')
    parser.add_argument('--not-convert', dest='convert', action='store_false',
                        help='indicates that the script should not convert mp3 to the specified output format')
    parser.add_argument('-w', '--num_workers', type=int, default=2,
                        help='number of worker threads for processing')
    parser.add_argument('--output_format', type=str, default='flac', choices=['flac', 'wav'],
                        help='output audio format (flac or wav)')

    args = parser.parse_args()
    main(args)