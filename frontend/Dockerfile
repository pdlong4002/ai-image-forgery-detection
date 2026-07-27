# Sử dụng Node 20 Alpine thay vì 18 để tương thích Vite 8
FROM node:20-alpine

WORKDIR /app

# Copy file cấu hình package để cài đặt trước (tối ưu cache Docker)
# Cố tình KHÔNG copy package-lock.json để tránh xung đột thư viện giữa Windows và Linux
COPY package.json ./

# Cài đặt Node modules
RUN npm install

# Copy toàn bộ mã nguồn frontend
COPY . .

# Expose cổng của Vite (mặc định là 5173)
EXPOSE 5173

# Chạy môi trường Dev và buộc bind IP (0.0.0.0) để host Windows có thể thấy cổng
CMD ["npm", "run", "dev", "--", "--host"]
