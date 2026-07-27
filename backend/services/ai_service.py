import sys
import os
from pathlib import Path

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Thêm thư mục ai_core vào sys.path để có thể import các module từ ai_core
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_CORE_PATH = PROJECT_ROOT / "ai_core"
sys.path.append(str(AI_CORE_PATH))

import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
import cv2
import yaml
import time

from models.resnet_transfer import ResNetTransfer
from models.efficientnet_transfer import EfficientNetTransfer
from models.densenet_transfer import DenseNetTransfer
from xai.gradcam_runner.gradcam_runner import XAIAnalyzer

class AIService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Khởi tạo AI Service trên thiết bị: {self.device}")
        
        # Đọc cấu hình từ ai_core/config.yml
        config_path = AI_CORE_PATH / "config.yml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.img_size = self.config['training']['image_size']
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.current_model_name = self.config['model']['name'].lower()
        try:
            self._load_model(self.current_model_name)
        except Exception as e:
            print(f"Cảnh báo khởi tạo: {e}")
        
    def _load_model(self, model_name_str):
        print(f"\nYêu cầu nạp mô hình: {model_name_str.upper()}")
        num_classes = self.config['model']['num_classes']
        
        # Load mô hình tương ứng
        if model_name_str == "resnet":
            new_model = ResNetTransfer(num_classes=num_classes)
            new_target_layer = new_model.resnet.layer4[-1]
        elif model_name_str == "efficientnet":
            new_model = EfficientNetTransfer(num_classes=num_classes)
            new_target_layer = new_model.efficientnet.features[-1]
        elif model_name_str == "densenet":
            new_model = DenseNetTransfer(num_classes=num_classes)
            new_target_layer = new_model.densenet.features.denseblock4
        else:
            raise ValueError(f"Không hỗ trợ mô hình {model_name_str}")
            
        # Tìm đường dẫn file weights (best_model) trong từng thư mục riêng của model
        checkpoint_dir = (AI_CORE_PATH / self.config['paths']['checkpoint_dir']).resolve()
        best_model_path = checkpoint_dir / model_name_str.lower() / f"best_model_{model_name_str}.pth"
        
        if not best_model_path.exists():
            raise FileNotFoundError(f"Chưa có file weights cho {model_name_str.upper()}. Vui lòng huấn luyện mô hình này trước!")
            
        print(f"Đang tải trọng số từ: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
        
        if "model" in checkpoint:
            new_model.load_state_dict(checkpoint["model"])
        else:
            new_model.load_state_dict(checkpoint)
            
        new_model.to(self.device)
        new_model.eval()
        
        # Cập nhật vào self
        self.model = new_model
        self.target_layer = new_target_layer
        self.current_model_name = model_name_str
        
        # Khởi tạo XAI Analyzer
        self.xai_analyzer = XAIAnalyzer(model=self.model, target_layer=self.target_layer, device=self.device)
        print("Đã tải mô hình và XAI thành công!")

    def _convert_cv2_to_base64(self, cv2_img):
        """Chuyển đổi ảnh OpenCV (numpy array) sang chuỗi Base64"""
        _, buffer = cv2.imencode('.jpg', cv2_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

    def predict_image(self, image_bytes, model_name="resnet"):
        """Nhận byte ảnh, xử lý và trả về kết quả JSON"""
        
        model_name = model_name.lower()
        if getattr(self, 'current_model_name', None) != model_name:
            try:
                self._load_model(model_name)
            except Exception as e:
                return {"error": str(e)}
                
        if not hasattr(self, 'model'):
            return {"error": "Mô hình chưa được tải (Thiếu file best_model.pth)"}
            
        # Đọc ảnh từ byte stream
        start_time = time.time()
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_size_str = f"{img.width}x{img.height}"
        except Exception as e:
            return {"error": f"Lỗi đọc ảnh: {str(e)}"}
            
        # Chuẩn bị Tensor cho mô hình
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Chuẩn bị ảnh gốc cho XAI
        original_img_np = np.array(img.resize((self.img_size, self.img_size))).astype(np.float32) / 255.0
        
        # Dự đoán
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            fake_prob = probabilities[0].item()
            real_prob = probabilities[1].item()
            predicted_class = torch.argmax(probabilities).item()
            
        label = "REAL" if predicted_class == 1 else "FAKE"
        confidence = real_prob if predicted_class == 1 else fake_prob
        
        # Lấy tên Model
        model_name_str = self.current_model_name.upper()
        
        # Chạy XAI (Grad-CAM)
        heatmap_base64 = None
        if hasattr(self, 'xai_analyzer'):
            try:
                targets = [ClassifierOutputTarget(predicted_class)]
                grayscale_cam = self.xai_analyzer.cam(input_tensor=img_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                
                # Resize heatmap to match original image dimensions
                orig_width, orig_height = img.size
                grayscale_cam_resized = cv2.resize(grayscale_cam, (orig_width, orig_height))
                
                # Use original image for background overlay
                original_img_full_np = np.array(img).astype(np.float32) / 255.0
                
                visualization = show_cam_on_image(original_img_full_np, grayscale_cam_resized, use_rgb=True)
                visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                
                # Thay vì lưu file, chuyển nó thành Base64
                heatmap_base64 = self._convert_cv2_to_base64(visualization_bgr)
                
            except Exception as e:
                print(f"Lỗi XAI: {e}")
        else:
            print("Cảnh báo: Không thể sinh Heatmap vì XAI Analyzer chưa được khởi tạo (do thiếu file weights).")
            
        processing_time = round(time.time() - start_time, 3)
            
        return {
            "success": True,
            "label": label,
            "confidence": confidence,
            "real_probability": real_prob,
            "fake_probability": fake_prob,
            "model_name": model_name_str,
            "image_size": image_size_str,
            "processing_time": processing_time,
            "heatmap_base64": f"data:image/jpeg;base64,{heatmap_base64}" if heatmap_base64 else None
        }

    def predict_all(self, image_bytes):
        """
        Chạy suy luận song song trên cả 3 mô hình liên tiếp và tổng hợp kết quả (Parallel Inference).
        """
        models_to_test = ["resnet", "efficientnet", "densenet"]
        results = {}
        
        for m in models_to_test:
            try:
                print(f"\n[Parallel Inference] Đang chạy mô hình: {m}")
                res = self.predict_image(image_bytes, model_name=m)
                results[m] = res
            except Exception as e:
                print(f"Lỗi khi chạy {m}: {e}")
                results[m] = {"error": str(e), "model_name": m}
                
        return {"success": True, "results": results}
