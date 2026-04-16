#!/bin/bash
# IceIQ API Server Start Script

cd ~/iceiq-dev
eval "$(/opt/homebrew/bin/conda shell.bash hook)"
conda activate iceiq
pip install fastapi uvicorn python-multipart -q
python3 server.py
