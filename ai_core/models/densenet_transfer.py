import torch
import torch.nn as nn
from torchvision import models

class DenseNetTransfer(nn.Module):
    def __init__(self, num_classes=2, fine_tune=False):
        super().__init__()
        
        # 1. Tải mô hình DenseNet-121 đã được huấn luyện trước trên ImageNet
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # 2. Đóng băng trọng số nếu không tinh chỉnh (fine-tuning)
        if not fine_tune:
            for param in self.densenet.parameters():
                param.requires_grad = False
                
        # 3. Thay thế lớp phân loại (Classifier) cuối cùng
        # DenseNet lưu trữ lớp fully connected trong 'classifier' (một lớp Linear)
        # DenseNet-121 có chiều đặc trưng (feature dim) = 1024
        in_features = self.densenet.classifier.in_features  # Với DenseNet-121, giá trị này là 1024
        
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=0.4),                 # Dropout đầu vào (hơi cao vì dense connections đã có regularization tốt)
            nn.Linear(in_features, 512),       # Giảm chiều từ 1024 xuống 512
            nn.ReLU(),
            nn.Dropout(p=0.2),                 # Dropout nhẹ trước output
            nn.Linear(512, num_classes)        # Lớp đầu ra: Fake (0) hoặc Real (1)
        )
        
    def forward(self, x):
        return self.densenet(x)


# Mã kiểm thử (test code)
if __name__ == '__main__':
    model = DenseNetTransfer(num_classes=2, fine_tune=False)
    
    fake_images = torch.randn(8, 3, 224, 224)
    print(f"\nInput shape:  {fake_images.shape} (Batch=8, Channels=3, W=224, H=224)")

    output = model(fake_images)
    print(f"Output shape: {output.shape} (Batch=8, Classes=2)")
