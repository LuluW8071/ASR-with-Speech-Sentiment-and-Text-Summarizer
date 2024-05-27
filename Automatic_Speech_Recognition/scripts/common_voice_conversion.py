""" 
Utility script to convert commonvoice MP3 to WAV and
split train and test JSON files for training ASR model.
"""

import os 
import argparse
import random
import json 
import csv
from pydub import AudioSegment

def main(args):
    data = []   # Empty list to store clips and sentence
    directory = args.file_path.rpartition('/')[0]
    percent = args.percent

    # Create a 'clips' directory inside defined save_json_path
    clips_directory = os.path.abspath(os.path.join(args.save_json_path, 'clips'))

    if not os.path.exists(clips_directory):
        os.makedirs(clips_directory)

    with open(args.file_path) as f:
        length = sum(1 for _ in f)
    
    with open(args.file_path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        index = 1
        if args.convert:
            print(str(length-1) + " files were found")

        for row in reader:
            file_name = row['path']                             # Converted file location
            clips_name = file_name.rpartition('.')[0] + '.wav'  # Converted file extenstion: wav
            text = row['sentence']                              # Extract Sentences
            if args.convert:
                data.append({'key': clips_directory + '/' + clips_name,
                             'text': text})
                print(f"Converting file {index}/{length-1} to WAV --- ({(index/(length-1))*100:.3f}%)", end="\r")

                # Defining Source and Destination for conversion
                src = directory + '/clips/' + file_name
                dest = os.path.join(clips_directory, clips_name)
                # print(f'Source Location: {src}\nDestination Location: {dest}')
                
                # Conversion: MP3 ---> WAV
                sound = AudioSegment.from_mp3(src)      # Take audio clips from source
                sound = sound.set_frame_rate(16000)     # Set FrameRate/SampleRate to 16000Hz
                sound.export(dest, format="wav")        # Export audio clips to destination
                index += 1

            else:
                data.append({'key': clips_directory + '/' + clips_name,
                             'text': text})
                index += 1

    # Splitting data into train and test set and saving into JSON file
    random.shuffle(data)
    print("Creating JSON's train and test set")

    train_data = data[:int(length * (1 - percent / 100))]
    test_data = data[int(length * (1 - percent / 100)):]

    with open(os.path.join(args.save_json_path, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(os.path.join(args.save_json_path, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     """
                                        Utility script to convert commonvoice MP3 to WAV and
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
                        help='says that the script should convert mp3 to wav')
    parser.add_argument('--not-convert', dest='convert', action='store_false',
                        help='says that the script should not convert mp3 to wav')

    
    args = parser.parse_args()
    main(args) 