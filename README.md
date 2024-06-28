# ASR-with-Speech-Sentiment-Analysis-Text-Summarizer

<div align="center">

![Code in Progress](https://img.shields.io/badge/status-in_progress-yellow.svg) ![License](https://img.shields.io/github/license/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer) ![Closed Issues](https://img.shields.io/github/issues-closed/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer) ![Open PRs](https://img.shields.io/github/issues-pr/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer) ![Contributors](https://img.shields.io/github/contributors/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer)

</div>

## Introduction

This project aims to develop an advanced system that integrates Automatic Speech Recognition (ASR), speech sentiment analysis, and text summarization. The system will address challenges in accurate speech recognition across diverse accents and noisy environments, providing real-time emotional tone interpretation (sentiment analysis), and generating summaries to retain essential information. Targeting applications such as customer service, business meetings, media, and education, this project seeks to enhance documentation, understanding, and emotional context in communication.

## Intermediate Goals

- [x] Baseline Model for ASR
- [ ] Baseline Model for SER
- [ ] Baseline Model for Text Summarizer

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

### 1. ASR(Automatic Speech Recognition)

| Base Model </br>(1CNN1D - Dense_Layers - 2Bi_LSTM)              | Final Model </br>  |
|---------------------|-----------------------------------------------|
| ![Base Model](docs/asr_base_model.jpeg)          |         ![Code in Progress](https://img.shields.io/badge/status-in_progress-red.svg)                         |

### 2. SER(Speech Emotion Recognition)

| Base Model </br>            | Final Model </br>  |
|---------------------|-----------------------------------------------|
| ![Code in Progress](https://img.shields.io/badge/status-in_progress-red.svg)          |         ![Code in Progress](https://img.shields.io/badge/status-in_progress-red.svg) 

### 3. Text Summarizer

| Base Model </br>            | Final Model </br>  |
|---------------------|-----------------------------------------------|
| ![Code in Progress](https://img.shields.io/badge/status-in_progress-red.svg)          |         ![Code in Progress](https://img.shields.io/badge/status-in_progress-red.svg) 

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
- [**SOX**](https://sourceforge.net/projects/sox/)
    - **For Linux:**
        ```bash
        # Update package list and install SOX
        sudo apt update
        sudo apt install sox libsox-fmt-all

        # Verify installation
        sox --version
        ```

```bash
pip install -r requirements.txt
```

#### 2. Configure [**Comet-ML**](https://www.comet.com/site/) Integration

Replace `dummy_key` with your actual Comet-ML API key and project name in the `.env` file to enable real-time loss curve plotting, system metrics tracking, and confusion matrix visualization.

```python
API_KEY = "dummy_key"
PROJECT_NAME = "dummy_key"
```

## Usage Instructions

### ASR

1. Audio Conversion
    > - `--not-convert` if you don't want audio conversion
    ```bash
    py common_voice.py --file_path "file_path/to/validated.tsv" 
                       --save_json_path "file_path/to/save/json" 
                       -w 4
                       --percent 10 
                       --output_format 'wav' or 'flac'
    ```

2. Train Model
    > - `--checkpoint_path "path/to/checkpoint_file"` to load pre-trained model and fine tune on it.
    ```bash
    py train.py --train_json "path/to/train.json" 
                --valid_json "path/to/test.json" 
                -w 4 
                --batch_size 128 
                -lr 2e-4
                --epochs 20 
    ```

### Speech Sentiment

1.  Audio Downsample and Augment

    Run the `Speech_Sentiment.ipynb` first to get the *path* and *emotions* table in csv format and downsample all clips.

    ```bash
    py downsample.py --file_path "path/to/audio_file.csv" 
                     --save_csv_path "output/path" 
                     -w 4
                     --output_format 'wav' or 'flac'
    ```

    ```bash
    py augment.py --file_path "path/to/emotion_dataset.csv" 
                  --save_csv_path "output/path" 
                  -w 4
                  --percent 20 
                  
    ```

2. Train the model

   ```bash
    py neuralnet/train.py --train_csv "path/to/train.csv" 
                          --test_csv "path/to/test.csv" 
                          -w 4 
                          --batch_size 256 
                          --epochs 25
                          -lr 1e-3
    ```

# Data Source

| Project            | Dataset Source                            |
|--------------------|-------------------------------------------|
| ASR                | [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)                     |
| Speech Sentiment   | [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio), [CremaD](https://www.kaggle.com/datasets/ejlok1/cremad), [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess), [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)                   |
| Text Summarizer                |                     |

## Code Structure

## Artifacts Location

# Results

## Metrics Used

## Evaluation Results