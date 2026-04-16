#!/bin/bash
D="$HOME/iceiq-dev"; L="$D/logs"; T=$(date +%Y%m%d_%H%M%S)
eval "$(conda shell.bash hook)" 2>/dev/null; conda activate iceiq 2>/dev/null
case "$1" in
  status)
    echo "=== IceIQ Status ==="
    python3 --version 2>&1
    python3 -c "import torch;print(f'PyTorch {torch.__version__} MPS:{torch.backends.mps.is_available()}')" 2>&1
    python3 -c "import ultralytics;print(f'YOLO {ultralytics.__version__}')" 2>&1
    echo "Disk: $(df -h ~ | tail -1 | awk '{print $4}') free"
    echo "Videos: $(ls $D/data/videos/*.mp4 2>/dev/null | wc -l | tr -d ' ')"
    echo "Results: $(ls -d $D/data/results/*/ 2>/dev/null | wc -l | tr -d ' ')";;
  test)
    V="${2:-$(ls $D/data/videos/*.mp4 2>/dev/null|head -1)}"
    echo "Testing: $V"
    cd "$D" && python3 pipeline/run_pipeline.py --config configs/pipeline_config.yaml --video "$V" --output "data/results/$T" 2>&1 | tee "$L/test_$T.log";;
  batch)
    cd "$D"; for V in data/videos/*.mp4; do
      N=$(basename "$V" .mp4)
      echo "=== $N ==="; python3 pipeline/run_pipeline.py --config configs/pipeline_config.yaml --video "$V" --output "data/results/batch_$T/$N" 2>&1
    done | tee "$L/batch_$T.log";;
  log)
    tail -80 "$L/$(ls -t $L/ | head -1)";;
  *)
    echo "Commands: status | test [video] | batch | log";;
esac
echo "Done: $1 $(date)"
