import torch
import torchvision

def load_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    return model