import uvicorn
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from routers import ai_router
from services.ai_service import AIService

# Khởi tạo ứng dụng FastAPI sạch sẽ
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API chuẩn Công nghiệp cho phát hiện Deepfake"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Hàm khởi động: Gắn AI Service vào trạng thái toàn cục của App"""
    print("Đang khởi động Server và tải AI Model...")
    try:
        # Lưu AIService vào app.state để các file router có thể gọi được
        app.state.ai_service = AIService()
    except Exception as e:
        print(f"Lỗi khởi tạo AI Service: {e}")

# Nhúng các bộ định tuyến (Routers) vào App chính
app.include_router(ai_router.router, prefix=settings.API_PREFIX)

@app.get("/")
def read_root():
    return {
        "message": f"Chào mừng đến với {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    print(f"Khởi động Server tại cổng 8000...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
