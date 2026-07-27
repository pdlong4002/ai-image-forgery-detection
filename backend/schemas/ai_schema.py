from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    success: bool
    label: str
    confidence: float
    real_probability: float
    fake_probability: float
    model_name: str
    image_size: str
    processing_time: float
    heatmap_base64: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "label": "FAKE",
                "confidence": 0.95,
                "real_probability": 0.05,
                "fake_probability": 0.95,
                "model_name": "RESNET50",
                "image_size": "1920x1080",
                "processing_time": 0.45,
                "heatmap_base64": "data:image/jpeg;base64,/9j/4AAQSk..."
            }
        }
