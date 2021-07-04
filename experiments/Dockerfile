# Base image
FROM nvidia/cuda:11.0-runtime-ubuntu20.04

WORKDIR /home/user

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python3.7 python3.7-dev python3.7-distutils python3-pip \
    git \
    poppler-utils \
    pkg-config \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# Set default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN update-alternatives --set python3 /usr/bin/python3.7

# Copy source code
COPY ./experiments/src /home/user/experiments/src
COPY ./experiments/setup.py /home/user/experiments/setup.py

# Install necessary package dependencies
COPY ./experiments/requirements.txt /home/user/experiments/requirements.txt
RUN cd experiments && pip3 install -r requirements.txt
# Install sompy package here
RUN pip3 install -e git+https://github.com/DavidSilva98/SOMPY.git#egg=SOMPY

# Add Tini (What is advantage of Tini? https://github.com/krallin/tini/issues/8)
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# The container listens to port 9999
EXPOSE 9999

# Defaults for executing the container (will be executed under Tini)
CMD ["jupyter", "notebook", "--port=9999", "--no-browser", "--ip=0.0.0.0", "--allow-root"]