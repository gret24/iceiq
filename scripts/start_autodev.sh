#!/bin/bash
export GEMINI_API_KEY="AIzaSyBH8GUZk_l4U7ubj72ec_c8WWi0VL1rtFE"
export TELEGRAM_BOT_TOKEN="AAEnqp7lEAnJdVV4Zrx51MylritPn4nLmAg"
export TELEGRAM_CHAT_ID="8740736142"
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate iceiq
cd ~/iceiq-dev
nohup python3 scripts/autodev.py > logs/gemini_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Gemini AutoDev PID: $!"
echo "Monitor: tail -f ~/iceiq-dev/logs/$(ls -t logs/gemini_*.log 2>/dev/null | head -1)"
echo "Stop: kill $!"
