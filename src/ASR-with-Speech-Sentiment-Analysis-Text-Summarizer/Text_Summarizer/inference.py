from transformers import pipeline

# Initialize the single pipeline for text generation
bart_cnn_finetuned = pipeline("text2text-generation", model="404sau404/bart_samsum_finetuned_for_asr")


def get_samples_from_file(file_path):
    """
    Reads text samples from a text file where each sample is separated by a double newline.
    
    Parameters:
    - file_path (str): The path to the text file containing the samples.

    Returns:
    - list: A list of text samples.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # Split the text into samples by double newlines
        samples = content.split("\n\n")
        return samples
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []

def summarize_texts(text):
    """
    Summarizes texts using the provided pipeline.

    Parameters:
    - pipeline (Pipeline): The transformers pipeline for text generation.
    - texts (list): A list of texts to summarize.

    Returns:
    - list: A list of summaries.
    """
    global bart_cnn_finetuned
    try:
        summary = bart_cnn_finetuned(text, max_new_tokens=160)[0]['generated_text']
        print(f"Summary:\n", summary, end="\n\n")
    except Exception as e:
        print(f"Error processing text: {e}")
    
    return summary

# if __name__ == "__main__":
#     file_path = "sample/text_samples.txt"

#     # Fetch text samples from the file and summarize them
#     texts = get_samples_from_file(file_path)
#     summaries = summarize_texts(bart_cnn_finetuned, texts)
