ARG TORCH="torch==2.15.0 torchvision==0.29.2"
ARG CUDA_VER=cu134


# ------------------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------------------
FROM continuumio/miniconda3 AS build-py
WORKDIR /root
RUN printf "name: py\n\ndependencies:\n  - python=3.13\n  - pip\n" > environment.yml
RUN conda env create -f environment.yml && \
    conda install -c conda-forge conda-pack


# ------------------------------------------------------------------------------
# Builder for CUDA
# ------------------------------------------------------------------------------
FROM build-py AS build-py-cuda
ARG TORCH
ARG CUDA_VER
# CUDA wheels first, so requirements.txt finds torch/torchvision already satisfied
RUN /opt/conda/envs/py/bin/pip3 install --no-cache-dir $TORCH \
    --index-url https://download.pytorch.org/whl/$CUDA_VER
ADD requirements.txt /tmp/requirements.txt
RUN /opt/conda/envs/py/bin/pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    conda list -n py | grep numpy

RUN conda-pack -n py -o /tmp/env.tar && \
    mkdir /venv && cd /venv && \
    tar xf /tmp/env.tar && \
    conda env remove -n py -y && \
    conda clean --all -y || true && \
    rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache /root/.cache/pip


# ------------------------------------------------------------------------------
# Final slim image
# ------------------------------------------------------------------------------
FROM nvidia/cuda:13.4.3-runtime-ubuntu26.04 AS cuda-python
ENV TZ=Etc/UTC \
    PYTHONUNBUFFERED=1
# libxrender1/libxext6: needed by rdkit
RUN apt-get update && apt-get install -y --no-install-recommends git rsync curl wget unzip unrar bzip2 tar \
    p7zip-full libxrender1 libxext6 && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build-py-cuda /venv /venv
ENV PATH="/venv/bin:$PATH"
ENV PY="/venv/bin/python"
ENV SSL_CERT_FILE="/venv/ssl/cacert.pem"

WORKDIR /project

CMD [ "/bin/bash" ]
