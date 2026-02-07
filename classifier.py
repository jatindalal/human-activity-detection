import numpy as np
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor


class Classifier:
    def __init__(self):
        self.device = self._get_best_device()
        print("Classifier using device:", self.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained("hazardous/har_google_vit_finetuned")
        self.model = AutoModelForImageClassification.from_pretrained("hazardous/har_google_vit_finetuned")
        self.model.to(self.device)

    def _get_best_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")

        # Apple Silicon (macOS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def classify(self, image: np.ndarray):
        rgb_image = image[:, :, ::-1]  # BGR → RGB
        encoding = self.feature_extractor(rgb_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**encoding.to(self.device))
            logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()
        return self.model.config.id2label[predicted_class_idx]

if __name__ == "__main__":
    pass
