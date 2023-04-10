import torch
import torchvision

def load_model():
    """
    Loads the segmentation model on gpu / cpu depending on availablity
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    return model