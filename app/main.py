# app/main.py

import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from classifier import Classifier
from segmenter import Segmenter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    app.state.classifier = Classifier()
    app.state.segmenter = Segmenter()

    try:
        yield
    finally:
        pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/infer/")
async def infer(filedata: str = Form(...)):
    image_bytes = base64.b64decode(filedata)
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data")

    mask = app.state.segmenter.segment(image).astype(np.float32) / 255.0
    masked_image = (image.astype(np.float32) * mask[..., None]).astype(np.uint8)
    inferred_class = app.state.classifier.classify(masked_image)

    # encode output image
    _, masked_buffer = cv2.imencode(".jpg", masked_image)
    masked_b64 = base64.b64encode(masked_buffer).decode()

    return {
        "class": inferred_class,
        "masked_image": f"data:image/jpg;base64,{masked_b64}",
    }
