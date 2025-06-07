FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# pip 업그레이드 후 requirements 설치
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
EXPOSE 5001

CMD ["python", "aiserver.py"]