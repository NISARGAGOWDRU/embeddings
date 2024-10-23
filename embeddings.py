import os
import torch
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image

# Load pre-trained models for text and image embeddings
text_model_name = "distilbert-base-uncased"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name)
text_model.eval()  # Set to evaluation mode

# Load pre-trained image model (ResNet in this case)
image_model = models.resnet50(pretrained=True)
image_model.eval()  # Set to evaluation mode

# Define a transformation for the input image
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_text_embeddings(text):
    """Generate embeddings for the input text."""
    inputs = text_tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outputs = text_model(**inputs)

    # Get the mean of the last hidden state for the embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def generate_image_embeddings(image_path):
    """Generate embeddings for the input image."""
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        embeddings = image_model(image)
    return embeddings

if __name__ == "__main__":
    # Example for text input
    text_input = "This is an example sentence."
    text_embeddings = generate_text_embeddings(text_input)
    print("Generated Text Embeddings:", text_embeddings)

    # Example for image input
    image_path = r"C:\Users\gowdr\OneDrive\Desktop\langchain\custom_chatbot\example.jpg"  # Replace with your image path
    image_embeddings = generate_image_embeddings(image_path)
    print("Generated Image Embeddings:", image_embeddings)
