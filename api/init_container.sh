#!/bin/bash
# Download models if they aren't already
# TODO: Also download ms-marco-TinyBERT-L-6 (OSError: Unable to load weights from pytorch checkpoint
# file. If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True)
# if [ ! -d "data/external/ms-marco-TinyBERT-L-6" ]; then
#     GIT_LFS_SKIP_SMUDGE=1 && \
#     git clone https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-6 ./data/external/ms-marco-TinyBERT-L-6
# else
#     echo "ms-marco-TinyBERT-L-6 already exists."
# fi

if [ ! -d "data/external/msmarco-distilbert-base-v2" ]; then
    wget -O ./data/external/msmarco-distilbert-base-v2.zip http://sbert.net/models/msmarco-distilbert-base-v2.zip && \
    unzip ./data/external/msmarco-distilbert-base-v2.zip && \
    rm ./data/external/msmarco-distilbert-base-v2.zip
else
    echo "msmarco-distilbert-base-v2 already exists."
fi

# Load backups (wait 20 sec for ODES node initialization)
sleep 20 && python3 api/load_backups.py

# Launch application
gunicorn api.application:app -b 0.0.0.0 -k uvicorn.workers.UvicornWorker --workers 1 --timeout 180