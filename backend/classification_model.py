import torch
from transformers import ViTImageProcessor, AutoModelForImageClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_model():
    """
    Loads the huggingface model on gpu / cpu depending on availablity
    """
    feature_extractor = ViTImageProcessor.from_pretrained("hazardous/har_google_vit_finetuned")
    model = AutoModelForImageClassification.from_pretrained("hazardous/har_google_vit_finetuned")
    model.to(device).eval()

    return model, feature_extractor