import os
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPModel
from torchvision import transforms

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variable for lazy loading
_model = None
_preprocess = None

def get_model():
    global _model
    if _model is None:
        _model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16').to(device)
        _model.eval()
    return _model

def get_preprocess():
    global _preprocess
    if _preprocess is None:
        _preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return _preprocess

def transform_image(image):
    model = get_model()
    preprocess = get_preprocess()
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.get_image_features(pixel_values=image)
    return features.squeeze().cpu().numpy().astype(np.float32)

def load_index(index_file):
    return faiss.read_index(index_file)

def search_similar_images(query_image, index, top_k=5):
    query_embedding = transform_image(query_image)
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
    distances, indices = index.search(np.array([query_embedding_normalized]), top_k)
    return distances[0][:top_k], indices[0][:top_k] 