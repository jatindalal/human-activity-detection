import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer_class(pil_image, model, feature_extractor):
    encoding = feature_extractor(pil_image.convert("RGB"), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding.to(device))
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]