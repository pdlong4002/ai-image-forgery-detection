import torch
import torch.nn as nn
from torchvision import models

class EfficientNetTransfer(nn.Module):
    def __init__(self, num_classes=2, fine_tune=False):
        super().__init__()
        
        # 1. Tải mô hình EfficientNet-B0 đã được huấn luyện trước
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # 2. Đóng băng trọng số nếu không tinh chỉnh (fine-tuning)
        if not fine_tune:
            for param in self.efficientnet.parameters():
                param.requires_grad = False
                
        # 3. Thay thế lớp phân loại (Classifier) cuối cùng
        # EfficientNet lưu trữ lớp fully connected trong 'classifier' (khác với ResNet dùng 'fc')
        # Lớp 'classifier' là một Sequential chứa Dropout và Linear.
        in_features = self.efficientnet.classifier[1].in_features  # Với B0, giá trị này là 1280
        
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.efficientnet(x)


# Mã kiểm thử
if __name__ == '__main__':
    model = EfficientNetTransfer(num_classes=2, fine_tune=False)
    
    fake_images = torch.randn(8, 3, 224, 224)
    print(f"\nInput shape:  {fake_images.shape} (Batch=8, Channels=3, W=224, H=224)")

    output = model(fake_images)
    print(f"Output shape: {output.shape} (Batch=8, Classes=2)")
