FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Python 패키지
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY . .

# 포트
EXPOSE 8000

# 시작
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
