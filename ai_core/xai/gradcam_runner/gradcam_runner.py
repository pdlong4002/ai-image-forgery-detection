import os
import cv2
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class XAIAnalyzer:
    def __init__(self, model, target_layer, device):
        """
        Khởi tạo Grad-CAM analyzer.
        """
        self.model = model
        self.device = device
        
        # SỬA LỖI TẠI ĐÂY: Bắt buộc bật requires_grad=True để tính được đạo hàm
        for param in self.model.parameters():
            param.requires_grad = True
            
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def generate_heatmap(self, image_tensor, original_image_np, output_path, target_class=None):
        """
        Sinh ra bản đồ nhiệt (Heatmap) dựa trên Gradient.
        """
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        
        # Sinh ra grayscale CAM
        grayscale_cam = self.cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :] # Lấy ảnh đầu tiên trong batch
        
        # Đè heatmap lên ảnh gốc
        visualization = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
        
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Chuyển đổi RGB sang BGR để lưu bằng OpenCV
        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), visualization_bgr)
        
        return str(output_path)
