import cv2
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from segmentation_model import load_model as load_segmenter
from classification_model import load_model as load_classifier
from segmentation import get_masked_image
from classification import infer_class

app = Flask(__name__, template_folder='./templates', static_folder='./static')
socketio = SocketIO(app,cors_allowed_origins='*' )

segmentation_model = load_segmenter()
classification_model, feature_extractor = load_classifier()

@app.route('/')
def index():
    return render_template('./index.html')

def base64_to_image(base64_string):
    image_bytes = base64.b64decode(base64_string)

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image

@socketio.on("image")
def receive_image(image):
    image = base64_to_image(image)

    masked_image= get_masked_image(image, segmentation_model)
    if masked_image is not None:
        image_to_classify = masked_image
    else:
        image_to_classify = image
    
    color_coverted = cv2.cvtColor(image_to_classify, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)

    inferred_class = infer_class(pil_image, classification_model, feature_extractor)

    retval, masked_buffer = cv2.imencode('.jpg', image_to_classify)
    masked_jpg_as_text = base64.b64encode(masked_buffer).decode()

    b64_src = "data:image/jpg;base64,"
    processed_img_data = b64_src + masked_jpg_as_text

    emit("response_back", processed_img_data)
    emit("class_response", inferred_class)


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')
