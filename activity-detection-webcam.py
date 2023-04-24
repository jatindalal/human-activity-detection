#!/usr/bin/env python3

from backend.segmentation_model import load_model as load_segmenter
from backend.classification_model import load_model as load_classifier
from backend.segmentation import get_masked_image
from backend.classification import infer_class

import cv2
from PIL import Image
import numpy as np

segmentation_model = load_segmenter()
classification_model, feature_extractor = load_classifier()

vid = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read()
    masked_image = get_masked_image(frame, segmentation_model)

    if masked_image is not None:
        image_to_classify = masked_image
    else:
        image_to_classify = frame

    color_coverted = cv2.cvtColor(image_to_classify, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    inferred_class = infer_class(pil_image, classification_model, feature_extractor)

    if masked_image is not None:
        cv2.imshow('Masked Feed', masked_image)
    cv2.imshow('Webcam Input', frame)

    print("Inferred Class: {}".format(inferred_class))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()