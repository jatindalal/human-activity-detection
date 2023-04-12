FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \ 
    apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgl1-mesa-glx

RUN pip install fastapi numpy opencv-python transformers \ 
    python-multipart transformers gunicorn uvicorn

WORKDIR /app
COPY ./* ./

ENTRYPOINT [ "gunicorn", "app:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:80" , "--timeout", "240"]