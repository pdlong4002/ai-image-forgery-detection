from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Deepfake Detection API"
    VERSION: str = "2.0.0" # Đánh dấu bản nâng cấp
    API_PREFIX: str = "/api/v1"
    
settings = Settings()
