# Automatic Speech Recognition with Speech Sentiment Analysis & Text Summarizer

## Usage

1.  Install the required dependencies:
    > Before installing dependencies from `requirements.txt` you must have installed **torch** from official website of [PyTorch](https://pytorch.org/). And if you want to train your own model install **CUDA** supported version of torch.

    ```bash
    pip install -r requirements.txt
    ```

2.  Run the training script:

    Run the `Speech_Sentiment.ipynb` first to get the *audio_path* and *emotions* table in csv format

    ```bash
    cd Speech_Sentiment_Analysis
    python neuralnet/train.py --file_path "/path/to/file_path.csv" --epochs 20 --batch_size 64 -w 2
    ```