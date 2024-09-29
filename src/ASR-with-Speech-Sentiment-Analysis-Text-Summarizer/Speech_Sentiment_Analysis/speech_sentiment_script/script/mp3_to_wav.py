from pydub import AudioSegment
import os

def convert_to_wav(input_file, output_file, sample_rate=48000, bitrate="768k"):
    """
    Converts an MP3 file to WAV format with specified sample rate and bitrate.

    :param input_file: Path to the input MP3 file
    :param output_file: Path to save the output WAV file
    :param sample_rate: Desired sample rate for the output WAV file
    :param bitrate: Desired bitrate for the output WAV file
    """
    try:
        # Load audio using pydub
        audio = AudioSegment.from_file(input_file)
        
        # Set sample rate and export as WAV
        audio = audio.set_frame_rate(sample_rate)
        
        # Export as WAV file
        audio.export(output_file, format="wav", bitrate=bitrate)
        print(f"Successfully converted {input_file} to {output_file} with sample rate {sample_rate} and bitrate {bitrate}.")
    except Exception as e:
        print(f"Error converting file: {e}")

if __name__ == "__main__":
    # Ask the user for input and output file paths
    input_file = input("Enter the path to the MP3 file: ")
    output_file = input("Enter the path for the output WAV file: ")

    # Convert the MP3 file to WAV
    convert_to_wav(input_file, output_file, sample_rate=48000, bitrate="768k")
