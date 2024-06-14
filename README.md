# ASR-with-Speech-Sentiment-Analysis-Text-Summarizer

<div align="center">

![Code in Progress](https://img.shields.io/badge/status-in_progress-yellow.svg) ![License](https://img.shields.io/github/license/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer) ![Closed Issues](https://img.shields.io/github/issues-closed/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer) ![Open PRs](https://img.shields.io/github/issues-pr/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer) ![Contributors](https://img.shields.io/github/contributors/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer)

</div>

## Introduction

This project aims to develop an advanced system that integrates Automatic Speech Recognition (ASR), speech sentiment analysis, and text summarization. The system will address challenges in accurate speech recognition across diverse accents and noisy environments, providing real-time emotional tone interpretation (sentiment analysis), and generating summaries to retain essential information. Targeting applications such as customer service, business meetings, media, and education, this project seeks to enhance documentation, understanding, and emotional context in communication.

## Goals

- [ ] Accurate ASR system for diverse accents and operable in noisy environments.
- [ ] Emotion Analysis through tone of speech.
- [ ] Meaningful Text Summarizer without loss of critical information.
- [ ] Integrating each component into one cohesive system provides real-time transcription and summaries.

## Contributors <img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="25px" />
<a href="https://github.com/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer">
</a>


## Project Architecture


## High Level Next Steps


# Usage
## Installation
<!--To begin this project, use the included `Makefile`

#### Creating Virtual Environment

This package is built using `python-3.8`. 
We recommend creating a virtual environment and using a matching version to ensure compatibility.

#### pre-commit

`pre-commit` will automatically format and lint your code. You can install using this by using
`make use-pre-commit`. It will take effect on your next `git commit`

#### pip-tools

The method of managing dependencies in this package is using `pip-tools`. To begin, run `make use-pip-tools` to install. 

Then when adding a new package requirement, update the `requirements.in` file with 
the package name. You can include a specific version if desired but it is not necessary. 

To install and use the new dependency you can run `make deps-install` or equivalently `make`

If you have other packages installed in the environment that are no longer needed, you can you `make deps-sync` to ensure that your current development environment matches the `requirements` files.  -->

#### 1. Install Required Dependencies

Before installing dependencies from `requirements.txt`, make sure you have installed 
- [**CUDA ToolKit v11.8/12.1**](https://developer.nvidia.com/cuda-toolkit-archive)
- [**PyTorch**](https://pytorch.org/)


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

```bash
pip install -r requirements.txt
```

#### 2. Configure Comet-ML Integration

To enable real-time loss curve plotting, system metrics tracking, and confusion matrix visualization, follow these steps:

1. **Sign Up for Comet-ML**: Visit [Comet](https://www.comet.com/site/) to sign up and obtain your API key.
2. **Edit `.env` File**: Replace `dummy_key` with your actual Comet-ML API key and project name in the `.env` file.

    ```python
    API_KEY = "dummy_key"
    PROJECT_NAME = "dummy_key"
    ```

## Usage Instructions

### ASR

1. Audio Conversion

    ```bash
    py common_voice.py --file_path "file_path/to/validated.tsv" --save_json_path "file_path/to/save/json" --percent 10
    ```

2. Train Model

    ```bash
    py train.py --train_json "dataset/train.json" --valid_json "dataset/test.json" -w 2 --epochs 20 --batch_size 128 -lr 5e-4
    ```

### Speech Sentiment

1.  Train Model

    Run the `Speech_Sentiment.ipynb` first to get the *audio_path* and *emotions* table in csv format

    ```bash
    py train.py --file_path "speech_emotion.csv" --epochs 20 --batch_size 64 -w 2 --steps 400
    ```

# Data Source

| Project            | Dataset Source                            |
|--------------------|-------------------------------------------|
| ASR                | [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)                     |
| Speech Sentiment   | [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio), [CremaD](https://www.kaggle.com/datasets/ejlok1/cremad)                       |
| Text Summarizer                |                     |

## Code Structure

## Artifacts Location

# Results

## Metrics Used

## Evaluation Results