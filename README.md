# Automatic Speech Recognition with Speech Sentiment Analysis & Text Summarizer

## Usage

1.  Install the required dependencies:
    > Before installing dependencies from `requirements.txt` you must have installed **torch** from official website of [PyTorch](https://pytorch.org/). And if you want to train your own model install **CUDA** supported version of torch.

    ```bash
    pip install -r requirements.txt
    ```
2.  Load you Comet-ML API

    For real-time loss curve plot, system metrics, confusion-matrix, edit `.env` with your Comet.ml API key and project name. Click [here](https://www.comet.com/site/) to sign up and get your Comet-ML API key.

    ```python
    API_KEY = "YOUR_API_KEY"
    PROJECT_NAME = "YOUR_PROJECT_NAME"
    ```

3.  Run the training script:

    Run the `Speech_Sentiment.ipynb` first to get the *audio_path* and *emotions* table in csv format

    ```bash
    cd Speech_Sentiment_Analysis/
    python neuralnet/train.py --file_path "speech_emotion.csv" --epochs 20 --batch_size 32 -w 2 --steps 400
    ```