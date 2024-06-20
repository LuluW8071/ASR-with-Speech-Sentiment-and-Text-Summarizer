import os
import argparse
import random
import json
import csv
import sox
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from pathlib import Path

# Function to clean text by removing specified characters
def clean_text(text):
    characters_to_remove =':!,-"‘’;—?'
    translator = str.maketrans('', '', characters_to_remove)
    return text.translate(translator)

def process_file(row, clips_directory, directory, output_format):
    file_name = row['path']  # Original file location
    clips_name = file_name.rpartition('.')[0] + '.' + output_format  # Converted file extension: FLAC
    text = clean_text(row['sentence'])  # Clean the sentence
    audio_path = os.path.join(directory, 'clips', file_name)
    output_audio_path = os.path.join(clips_directory, clips_name)

    # Convert MP3 to FLAC using Sox
    tfm = sox.Transformer()
    tfm.rate(samplerate=16000)
    tfm.build(input_filepath=audio_path, output_filepath=output_audio_path)

    return {'key': clips_directory + '/' + clips_name, 'text': text}

def main(args):
    data = []  # Empty list to store clips and sentences
    directory = args.file_path.rpartition('/')[0]
    percent = args.percent

    # Create a 'clips' directory inside defined save_json_path
    clips_directory = os.path.abspath(os.path.join(args.save_json_path, 'clips'))

    if not os.path.exists(clips_directory):
        os.makedirs(clips_directory)

    with open(args.file_path, encoding="utf-8") as f:
        length = sum(1 for _ in f) - 1

    with open(args.file_path, newline='', encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        data_to_process = [(row, clips_directory, directory, args.output_format) for row in reader]

    if args.convert:
        print(f"{length} files found. Converting MP3 to {args.output_format.upper()} using {args.num_workers} workers.")
        with ThreadPool(args.num_workers) as pool:
            data = list(tqdm(pool.imap(lambda x: process_file(*x), data_to_process), total=length))
    else:
        for row in data_to_process:
            file_name = row[0]['path']
            clips_name = file_name.rpartition('.')[0] + '.' + args.output_format
            text = clean_text(row[0]['sentence'])
            data.append({'key': clips_directory + '/' + clips_name, 'text': text})

    # Splitting data into train and test set and saving into JSON file
    random.shuffle(data)
    print("Creating train and test JSON sets")

    train_data = data[:int(length * (1 - percent / 100))]
    test_data = data[int(length * (1 - percent / 100)):]

    with open(os.path.join(args.save_json_path, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(os.path.join(args.save_json_path, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     """
                                        Utility script to convert CommonVoice MP3 to FLAC and
                                        split train and test JSON files for training ASR model. 
                                     """
                                    )
    parser.add_argument('--file_path', type=str, default=None, required=True,
                        help='path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='path to the dir where the json files are supposed to be saved')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='percent of clips put into test.json instead of train.json')
    parser.add_argument('--convert', default=True, action='store_true',
                        help='indicates that the script should convert mp3 to flac')
    parser.add_argument('--not-convert', dest='convert', action='store_false',
                        help='indicates that the script should not convert mp3 to flac')
    parser.add_argument('-w','--num_workers', type=int, default=2,
                        help='number of worker threads for processing')
    parser.add_argument('--output_format', type=str, default='flac',
                        help='output audio format (flac or wav)')

    args = parser.parse_args()
    main(args)
