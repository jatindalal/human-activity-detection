import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

def load_model():
    """
    Loads the segmentation model on gpu / cpu depending on availablity
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    return model