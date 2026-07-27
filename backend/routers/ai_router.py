from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Form
from schemas.ai_schema import PredictionResponse

# Tạo bộ định tuyến (router) riêng cho AI
router = APIRouter(
    prefix="/ai",
    tags=["AI Prediction"]
)

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...), model_name: str = Form("resnet")):
    """
    Nhận file ảnh upload từ người dùng, chạy qua model AI và trả về kết quả kèm Heatmap Base64.
    """
    ai_service = getattr(request.app.state, 'ai_service', None)
    
    if ai_service is None:
        raise HTTPException(status_code=503, detail="AI Service chưa sẵn sàng (có thể thiếu file weights).")
        
    # Kiểm tra định dạng file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File upload phải là hình ảnh (JPEG, PNG).")
        
    try:
        # Đọc dữ liệu thô
        image_bytes = await file.read()
        
        # Gửi sang AI Service để phân tích kèm theo tên model yêu cầu
        result = ai_service.predict_image(image_bytes, model_name=model_name)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-all")
async def predict_all(request: Request, file: UploadFile = File(...)):
    """
    Nhận file ảnh upload từ người dùng, chạy qua TẤT CẢ 3 model AI để phục vụ tính năng A/B Testing.
    """
    ai_service = getattr(request.app.state, 'ai_service', None)
    
    if ai_service is None:
        raise HTTPException(status_code=503, detail="AI Service chưa sẵn sàng (có thể thiếu file weights).")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File upload phải là hình ảnh (JPEG, PNG).")
        
    try:
        image_bytes = await file.read()
        result = ai_service.predict_all(image_bytes)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail="Lỗi tổng hợp A/B Testing")
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
