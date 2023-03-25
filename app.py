from segmentation_model import load_model as load_segmenter
from classification_model import load_model as load_classifier
from segmentation import get_segmented_masked_images
from classification import infer_class

import cv2
from PIL import Image
import base64
import numpy as np

from fastapi import FastAPI, Form

app = FastAPI()

@app.on_event('startup')
def load_models():
    global segmentation_model, classification_model, feature_extractor
    
    segmentation_model = load_segmenter()
    classification_model, feature_extractor = load_classifier()

@app.post("/uploadimage/")
async def create_item(filedata: str = Form(...)):
    image_as_bytes = str.encode(filedata)
    recovered_image = base64.b64decode(image_as_bytes)

    nparr = np.fromstring(recovered_image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    masked_image, segmented_image = get_segmented_masked_images(image, segmentation_model)
    if masked_image is not None:
        image_to_classify = masked_image
    else:
        image_to_classify = image

    color_coverted = cv2.cvtColor(image_to_classify, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)

    inferred_class = infer_class(pil_image, classification_model, feature_extractor)

    retval, masked_buffer = cv2.imencode('.jpg', image_to_classify)
    masked_jpg_as_text = base64.b64encode(masked_buffer)

    retval, segmented_buffer = cv2.imencode('.jpg', segmented_image)
    segmented_jpg_as_text = base64.b64encode(segmented_buffer)

    return {"class": inferred_class, "masked_image": masked_jpg_as_text, "segmented_image": segmented_jpg_as_text}

