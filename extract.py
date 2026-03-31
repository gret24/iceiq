#!/usr/bin/env python3
import subprocess
import os

input_file = "test.mp4"
output_dir = "frames"

os.makedirs(output_dir, exist_ok=True)

cmd = [
    "ffmpeg",
    "-i", input_file,
    "-vf", "fps=2",
    os.path.join(output_dir, "frame_%03d.jpg"),
    "-y"
]

subprocess.run(cmd, check=True)

count = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
print(f"총 {count}장 추출 완료.")
