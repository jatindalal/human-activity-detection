import torch
import numpy as np
import random
import cv2
from torchvision.transforms import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor()
])
 
def get_outputs(image, model, threshold=0.30):
    """
    Returns back the masks for the detected objects in the image passed,
    Threshold decides the minimum probablity required to mark a object
    """
    with torch.no_grad():
        outputs = model(image)
    
    scores = list(outputs[0]['scores'].detach().cpu().numpy())

    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)

    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]

    return masks

def get_masked_image(opencv_image, model):
    """
    Returns masked image for a given opencv_image
    """
    frame = transform(opencv_image)
    frame = frame.unsqueeze(0).to(device)
    masks = get_outputs(frame, model)

    masked_image = None

    try:
        masked_image = np.zeros(masks[0].shape).astype('uint8')
        for mask in masks:
            masked_image = cv2.add(masked_image, 255*mask.astype('uint8'))
        masked_image = cv2.bitwise_and(opencv_image, opencv_image, mask = masked_image)
    except:
        print("Couldn't detect objects, Number of masks detected: {}".format(len(masks)))
        masked_image = None

    return masked_image