import torch
from neuralnet.model import neuralnet   # Call model instance from neuralnet

from utils import get_features

# Load the trained model
model = neuralnet(input_size=1, output_shape=6)  
checkpoint = torch.load("model/sentiment-model-19-0.07.ckpt", map_location=torch.device('cpu'))

# Evaluate model
model.eval()
model.load_state_dict(checkpoint['state_dict'])

labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# Perform inference   
audio_path = ['sample/' + label + '.wav' for label in labels]   # just a sampling loop instead of taking one sample at a time

for audio in audio_path:
    features = get_features(audio)
    # print(features.shape, features.dtype)
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # Add extra batch dimension and channel dimension
    # print(input_tensor.shape, input_tensor.dtype)

    with torch.inference_mode():
        output = model(input_tensor)

    # Convert output to probabilities and get predicted class
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Calculate confidence
    confidence = probabilities[0, predicted_class].item()

    # Print predicted class
    print(f"Path: {audio}\nPredicted class: {labels[predicted_class]} --- Confidence: {confidence:.3f}\n")