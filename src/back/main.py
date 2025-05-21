from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from back.app.api import router as query_router

app = FastAPI()

# 🔓 CORS 허용: 프론트엔드 (예: Streamlit) 요청 허용을 위한 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 전체 허용 (배포 시에는 도메인 제한 권장)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚏 API 라우터 등록
app.include_router(query_router)

