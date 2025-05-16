# 실행 편의 스크립트 입니다.

#!/bin/bash
# dockerfile 기준으로 이미지 빌드
docker build -t hyundai-rag .
# gpu지원 도커 컨테이너 실행 및 stramliti 8501로 실행
docker run --gpus all -p 8501:8501 hyundai-rag
