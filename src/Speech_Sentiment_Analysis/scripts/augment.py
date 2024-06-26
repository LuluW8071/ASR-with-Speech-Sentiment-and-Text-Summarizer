import os
import argparse
import csv
import librosa
import numpy as np
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import random

# Function to apply SoX effect for converting sample rate
def apply_sox_effect(input_file, output_file):
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)

    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        return False

    cmd = ['sox', input_file, '-r', '16000', output_file]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing file: {input_file}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error output: {e.stderr}")
        return False

# Noise function using numpy and librosa
def noise(audio_data, sr, noise_rate=0.035):
    noise_amp = noise_rate * np.random.uniform() * np.amax(audio_data)
    audio_data = audio_data + noise_amp * np.random.normal(size=audio_data.shape[0])
    return audio_data, sr

# Stretch function using librosa
def stretch(audio_data, sr, rate=0.8):
    return librosa.effects.time_stretch(y=audio_data, rate=rate), sr

# Shift function using numpy
def shift(audio_data, sr):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1300)
    return np.roll(audio_data, shift_range), sr

# Pitch function using librosa
def pitch(audio_data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=audio_data, sr=sr, n_steps=pitch_factor), sr

# Process file function for multi-threading
def process_file(row, base_directory, output_directory):
    file_name = row['path']
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    audio_path = os.path.join(base_directory, file_name)
    
    # Ensure original sample is converted to 16 kHz and saved as is
    original_output_path = os.path.join(output_directory, f"{base_name}.flac")
    if not apply_sox_effect(audio_path, original_output_path):
        print(f"Failed to process original for file: {audio_path}")
        return None
    
    # Load audio using librosa
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    # Prepare the list of processed data with the original file
    processed_data = [{
        'Path': os.path.abspath(original_output_path),
        'Emotions': row['emotions']
    }]

    # Apply noise, stretch, shift, pitch
    augmentations = {
        'noise': noise,
        'stretch': stretch,
        'shift': shift,
        'pitch': pitch
    }

    for aug_name, aug_func in augmentations.items():
        try:
            augmented_data, sr = aug_func(audio_data, sr)
            output_name = f"{base_name}_{aug_name}.flac"
            output_path = os.path.join(output_directory, output_name)
            apply_sox_effect(audio_path, output_path)  # Use sox to write FLAC files
            processed_data.append({
                'Path': os.path.abspath(output_path),
                'Emotions': row['emotions']
            })
        except Exception as e:
            print(f"Error applying {aug_name} augmentation to {audio_path}: {e}")

    return processed_data

# Main function
def main(args):
    base_directory = os.path.dirname(args.file_path)
    clips_directory = os.path.abspath(os.path.join(args.save_csv_path, 'clips'))

    if not os.path.exists(clips_directory):
        os.makedirs(clips_directory)

    with open(args.file_path, newline='', encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        data = list(reader)

    print(f"{len(data)} files found. Processing using {args.num_workers} workers.")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(lambda x: process_file(x, base_directory, clips_directory), data), total=len(data)))

    # Flatten the list of results
    all_data = [item for sublist in results for item in sublist if sublist]

    # Splitting data into train and test set
    random.shuffle(all_data)
    length = len(all_data)
    percent = args.percent
    print(f"Creating train and test sets with {percent}% for testing")

    train_data = all_data[:int(length * (1 - percent / 100))]
    test_data = all_data[int(length * (1 - percent / 100)):]

    fieldnames = ['Path', 'Emotions']

    with open(os.path.join(args.save_csv_path, 'train.csv'), 'w', newline='', encoding='utf-8') as csv_file:  
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in train_data:
            writer.writerow(result)

    with open(os.path.join(args.save_csv_path, 'test.csv'), 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in test_data:
            writer.writerow(result)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility script to process audio files using noise, stretch, shift, pitch augmentations and save as CSV file with file path and emotions.")
    parser.add_argument('--file_path', type=str, default=None, required=True, help='Path to the CSV file containing audio file paths')
    parser.add_argument('--save_csv_path', type=str, default=None, required=True, help='Path to the directory where the CSV file will be saved')
    parser.add_argument('--percent', type=int, default=10, required=False, help='Percentage of clips put into test set')
    parser.add_argument('-w', '--num_workers', type=int, default=2, help='Number of worker threads for processing')

    args = parser.parse_args()
    main(args)
