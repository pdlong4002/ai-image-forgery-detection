import os
import yaml
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from xai.gradcam_runner.gradcam_runner import XAIAnalyzer

# Import model
from models.resnet_transfer import ResNetTransfer
from models.efficientnet_transfer import EfficientNetTransfer
from models.densenet_transfer import DenseNetTransfer

def predict_image(image_path):
    # 1. Load config
    PROJECT_ROOT = Path(__file__).resolve().parent
    with open(PROJECT_ROOT / "config.yml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Setup transforms (must match testing transforms in train.py)
    img_size = config['training']['image_size']
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Load Model & Weights
    # Tự động đọc file .pth dựa theo tên model đang cấu hình (tìm trong thư mục con tương ứng)
    model_name_str = config['model']['name'].lower()
    best_model_path = PROJECT_ROOT / config['paths']['checkpoint_dir'] / model_name_str / f"best_model_{model_name_str}.pth"
    
    # Fallback: nếu không tìm thấy trong thư mục con, thử tìm ở thư mục ngoài (tương thích ngược)
    if not best_model_path.exists():
        fallback_path = PROJECT_ROOT / config['paths']['checkpoint_dir'] / f"best_model_{model_name_str}.pth"
        if fallback_path.exists():
            best_model_path = fallback_path
    
    num_classes = config['model']['num_classes']
    if model_name_str == "resnet":
        model = ResNetTransfer(num_classes=num_classes)
    elif model_name_str == "efficientnet":
        model = EfficientNetTransfer(num_classes=num_classes)
    elif model_name_str == "densenet":
        model = DenseNetTransfer(num_classes=num_classes)
    else:
        model = ResNetTransfer(num_classes=num_classes)
    
    if not best_model_path.exists():
        print(f"Lỗi: Không tìm thấy file trọng số tại {best_model_path}")
        return
        
    # Load dictionary chứa thông tin checkpoint
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    # Lấy state_dict của mô hình ra
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        # Tương thích ngược với file pth cũ (chỉ có state_dict)
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    # 4. Load Image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Lỗi đọc ảnh: {e}")
        return
        
    img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension
    
    # 5. Predict
    with torch.no_grad():
        output = model(img_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # 0 = Fake, 1 = Real
        fake_prob = probabilities[0].item()
        real_prob = probabilities[1].item()
        
        predicted_class = torch.argmax(probabilities).item()
        
    print(f"\n--- KẾT QUẢ DỰ ĐOÁN ---")
    print(f"Ảnh: {image_path}")
    if predicted_class == 1:
        print(f"Nhãn: REAL (Ảnh Thật)")
    else:
        print(f"Nhãn: FAKE (Ảnh AI / Ngụy tạo)")
    print(f"Độ tự tin (Confidence): Real={real_prob*100:.2f}% | Fake={fake_prob*100:.2f}%")
    
    # 6. Giải thích bằng Grad-CAM (XAI)
    try:
        # Chọn target_layer (lớp chập cuối cùng) dựa theo mô hình
        if model_name_str == "resnet":
            target_layer = model.resnet.layer4[-1]
        elif model_name_str == "efficientnet":
            target_layer = model.efficientnet.features[-1]
        elif model_name_str == "densenet":
            target_layer = model.densenet.features.denseblock4
        else:
            target_layer = model.resnet.layer4[-1]
            
        xai_analyzer = XAIAnalyzer(model=model, target_layer=target_layer, device=device)
        
        # Chuẩn bị ảnh gốc cho Grad-CAM (Resize và đưa về [0, 1])
        original_img_np = np.array(img.resize((img_size, img_size))).astype(np.float32) / 255.0
        
        heatmap_name = f"heatmap_{Path(image_path).name}"
        heatmap_path = PROJECT_ROOT / "xai" / "heatmaps" / heatmap_name
        
        saved_path = xai_analyzer.generate_heatmap(img_tensor, original_img_np, str(heatmap_path), target_class=predicted_class)
        print(f"XAI: Đã xuất bản đồ nhiệt (Heatmap) tại: {saved_path}")
        
    except Exception as e:
        print(f"Lỗi khi chạy XAI (Grad-CAM): {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Sử dụng: python inference.py <đường_dẫn_ảnh>")
    else:
        img_path = sys.argv[1]
        predict_image(img_path)