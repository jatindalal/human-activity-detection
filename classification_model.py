import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_model():
    feature_extractor = AutoFeatureExtractor.from_pretrained("hazardous/har_google_vit_finetuned")
    model = AutoModelForImageClassification.from_pretrained("hazardous/har_google_vit_finetuned")
    model.to(device).eval()

    return model, feature_extractor