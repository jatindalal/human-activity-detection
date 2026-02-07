#!/usr/bin/env python3

import cv2
import numpy as np

from classifier import Classifier
from segmenter import Segmenter

vid = cv2.VideoCapture(0)
segmenter = Segmenter()
classifier = Classifier()

USE_MASKED = True # usually gives worse classification

while True:
    ret, frame = vid.read()
    mask = segmenter.segment(frame).astype(np.float32) / 255.0
    masked_frame = (frame.astype(np.float32) * mask[..., None]).astype(np.uint8)
    if USE_MASKED:
        inferred_class = classifier.classify(masked_frame)
    else:
        inferred_class = classifier.classify(frame)

    cv2.imshow("Input", frame)
    cv2.imshow("Masked", masked_frame)
    print("activity:", inferred_class, ' ' * 15, end='\r')
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print()
        break

vid.release()
cv2.destroyAllWindows()
