FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Python 패키지 (requirements만 먼저 복사 — 레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델 다운로드 (별도 레이어 — 코드 변경 시 캐시 재사용)
COPY download_models.py .
RUN python3 download_models.py

# 앱 코드 복사
COPY . .

# 런타임 데이터 디렉토리 생성
RUN mkdir -p data/uploads data/results data/highlights

# 포트
EXPOSE 8000

# 시작
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--limit-max-request-size", "2147483648"]
