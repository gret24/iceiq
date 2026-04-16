#!/bin/bash
set -e
if ! command -v conda &>/dev/null; then
  curl -fsSL -o /tmp/mf.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
  bash /tmp/mf.sh -b -p ~/miniforge3 && eval "$(~/miniforge3/bin/conda shell.bash hook)" && conda init bash && rm /tmp/mf.sh
  echo "Miniforge installed. Restart terminal, then run this again."; exit 0
fi
eval "$(conda shell.bash hook)"
conda create -n iceiq python=3.10 -y 2>/dev/null || true
conda activate iceiq
pip install -q --upgrade pip
pip install -q ultralytics torch torchvision torchaudio
pip install -q opencv-python-headless filterpy lap boxmot easyocr
pip install -q numpy scipy scikit-learn shapely matplotlib pandas
pip install -q pillow tqdm pyyaml requests moviepy
echo "=== DONE ==="
python3 -c "import torch;print(f'PyTorch {torch.__version__} MPS:{torch.backends.mps.is_available()}')"
python3 -c "import ultralytics;print(f'YOLO {ultralytics.__version__}')"
