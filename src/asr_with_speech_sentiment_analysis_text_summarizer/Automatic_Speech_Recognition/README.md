# Automatic Speech Recognition

## Dependencies

- [FFmpeg](https://www.ffmpeg.org/)

    - **For Windows:**
        ```bash
        # Extract the archive
        ffmpeg/bin/
        # Edit environment variables to insert path 
        path/to/ffmpeg/bin/
        
        # Open terminal and verify installation
        ffmpeg -version
        ```
    - **For Linux:**
        ```bash
        # Update package list and install FFmpeg
        sudo apt update
        sudo apt install ffmpeg

        # Verify installation
        ffmpeg -version
        ```
</br>

## 1. Data Collection

To convert audio files to the required WAV format and create JSON files for training and testing data from [CommonVoice](https://commonvoice.mozilla.org/en/datasets) by Mozilla, use the following command:

```bash
python scripts/common_voice_conversion.py --file_path "path/to/validated.tsv" --save_json_path "path/to/save_json_path" --percent 20 --convert
```

Alternatively, if conversion isn't needed, you can use the following command:

```bash
python scripts/common_voice_conversion.py --file_path "path/to/validated.tsv" --save_json_path "path/to/save_json_path" --percent 20 --not-convert
```

### JSON Structure

The following JSON structure will be created for both training and testing:

```json
[
    {   
        "key": "/path/to/audio/speech.wav", 
        "text": "This is a sentence of converted speech audio."
    },
    ...
]
```