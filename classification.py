import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer_class(pil_image, model, feature_extractor):
    """
    This function takes in a PIL Image, reference to the classification model and its feature extractor and gives mack the predicted class for that image.
    """
    encoding = feature_extractor(pil_image.convert("RGB"), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding.to(device))
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]