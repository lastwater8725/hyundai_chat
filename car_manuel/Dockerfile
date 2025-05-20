FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 컨테이너 내 작업 디렉토리
WORKDIR /app

# 로컬 requirements.txt 파일을 컨테이너의 /app 디렉토리로 복사
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# torch GPU 버전 별도 설치
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 로컬 전체 프로젝트 파일을 컨테이너로 복사(여기서 로컬은 서버)
COPY . .

#stlimlit 기본 포트 
EXPOSE 8501

# 컨테이너 시작시 기본 실행 명령어 false는 외부 접속 불값
CMD ["streamlit", "run", "src/front/main.py", "--server.port=8501","--server.enableCORS=false"] 

