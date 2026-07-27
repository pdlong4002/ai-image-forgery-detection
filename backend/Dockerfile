# Sử dụng Python 3.10 mỏng nhẹ
FROM python:3.10-slim

# Ngăn Python tạo file .pyc và bật hiển thị log ngay lập tức
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cài đặt các thư viện hệ thống cần thiết (OpenCV cần libgl)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Sao chép và cài đặt thư viện
COPY backend/requirements.txt ./backend/requirements.txt
# Cài đặt bản CPU siêu nhẹ của PyTorch (150MB thay vì 2.5GB) để chạy demo web cực nhanh
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Cài đặt các thư viện nhẹ còn lại
RUN pip install --no-cache-dir -r backend/requirements.txt

# Sao chép mã nguồn backend và cả ai_core vào container
COPY backend ./backend
COPY ai_core ./ai_core

# Chuyển thư mục làm việc vào backend để chạy FastAPI
WORKDIR /app/backend

# Expose cổng 8000
EXPOSE 8000

# Khởi chạy server FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
