FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 安裝 Python 3.10 + distutils (確保能正常執行 setup.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils python3.10-venv \
    git ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 用 ensurepip 給 Python3.10 裝 pip
RUN python3.10 -m ensurepip

# 升級 pip、setuptools 等
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 之後所有安裝都寫 python3.10 -m pip install ...
RUN python3.10 -m pip install --no-cache-dir 'numpy<2.0.0' packaging 'cython<3.0' 'setuptools<75.0.0'
RUN python3.10 -m pip install --no-cache-dir audiocraft
RUN python3.10 -m pip install --no-cache-dir git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
RUN python3.10 -m pip install --no-cache-dir 'torch==2.1.0'
RUN python3.10 -m pip install --no-cache-dir ninja flash-attn
RUN python3.10 -m pip install --no-cache-dir openai open_clip_torch 'accelerate>=0.26.0'

WORKDIR /app
COPY . /app

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# 最後確保執行時也用 Python 3.10
CMD ["python3.10", "gui.py"]